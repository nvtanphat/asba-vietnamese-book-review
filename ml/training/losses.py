from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits, labels, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
    """Multi-class focal loss (Lin et al. 2017): down-weights easy, well-classified examples
    by (1 - p_t)^gamma so training focuses on hard/rare examples instead of relying on
    class-weighting alone."""
    ce = F.cross_entropy(logits, labels, weight=weight, label_smoothing=label_smoothing, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def multitask_loss(logits, labels, weights=None, task_weights=None, loss_type: str = "ce", gamma: float = 2.0, label_smoothing: float = 0.0):
    """Average or task-weighted per-task loss across a multitask head list. `logits` is a list of [B, C_i]
    tensors (one per task), `labels` is a [B, T] tensor with one column per task."""
    weights = weights or [None] * len(logits)
    if loss_type == "focal":
        per_task = [focal_loss(x, labels[:, i], weight=weights[i], gamma=gamma, label_smoothing=label_smoothing) for i, x in enumerate(logits)]
    else:
        per_task = [F.cross_entropy(x, labels[:, i], weight=weights[i], label_smoothing=label_smoothing) for i, x in enumerate(logits)]
    if task_weights is not None:
        tw = torch.as_tensor(task_weights, dtype=torch.float32, device=logits[0].device)
        stacked = torch.stack(per_task)
        return (stacked * tw).sum() / tw.sum()
    return sum(per_task) / len(logits)



def class_balanced_weights(labels, num_classes: int, beta: float = 0.999, cap: float | None = None):
    """Effective-number-of-samples class weighting (Cui et al. 2019, "Class-Balanced Loss"):
    w_c = (1-beta)/(1-beta^n_c), normalized so the max weight is 1.0. Reflects diminishing
    marginal value of additional samples in an over-represented class better than plain
    inverse-frequency weighting, which is why the specialized legacy PhoBERT trainer (that
    reached f1_combined=0.795 vs. the unified pipeline's ~0.65) used it instead."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    nz = counts > 0
    weights[nz] = (1.0 - beta) / (1.0 - beta ** counts[nz])
    if cap is not None:
        weights = torch.clamp(weights, max=float(cap))
    if weights.max() > 0:
        weights = weights / weights.max()
    return weights


def downweight_class(weights, class_idx: int, scale: float):
    """Scale one class's weight (e.g. the dominant "absent" class) and renormalize by the mean
    of the remaining nonzero weights, so it stops drowning out the loss signal from rarer
    classes without needing a full weight re-derivation."""
    w = weights.clone()
    if 0 <= class_idx < len(w) and w[class_idx] > 0:
        w[class_idx] = w[class_idx] * scale
    nz = w > 0
    if nz.any():
        w = w / w[nz].mean()
    return w


def two_stage_aspect_loss(
    presence_logits,
    sentiment_logits,
    aspect_labels,
    *,
    presence_weight=None,
    sentiment_weight=None,
    absent_class: int = 3,
    stage1_weight: float = 0.25,
    stage2_weight: float = 0.75,
    loss_type: str = "focal",
    gamma: float = 2.5,
    label_smoothing: float = 0.1,
):
    """Two-stage aspect loss: presence detection (binary CE: absent vs. present) is scored
    separately from polarity classification (3-class, scored only on present-labeled
    samples). Ports the decomposition from the specialized legacy PhoBERT trainer
    (ml/legacy_phobert), which decoupling avoids the single-4-class-head failure mode where
    the dominant "absent" label drowns out polarity signal for sparse aspects (price,
    service) — exactly the aspects where the unified pipeline's single-head models score
    near zero. `presence_logits` is [B, 2] (index 0 = absent, index 1 = present),
    `sentiment_logits` is [B, 3], `aspect_labels` is [B] with values in {0, 1, 2, 3}."""
    present_target = (aspect_labels != absent_class).long()
    presence_loss = F.cross_entropy(presence_logits, present_target, weight=presence_weight)

    present_mask = aspect_labels != absent_class
    if present_mask.any():
        s2_logits = sentiment_logits[present_mask]
        s2_labels = aspect_labels[present_mask]
        if loss_type == "focal":
            sentiment_loss = focal_loss(s2_logits, s2_labels, weight=sentiment_weight, gamma=gamma, label_smoothing=label_smoothing)
        else:
            sentiment_loss = F.cross_entropy(s2_logits, s2_labels, weight=sentiment_weight, label_smoothing=label_smoothing)
    else:
        sentiment_loss = presence_logits.sum() * 0.0

    return stage1_weight * presence_loss + stage2_weight * sentiment_loss


def two_stage_multitask_weights(train_y, beta: float = 0.999, absent_scale: float = 0.2, sentiment_max_ratio: float = 3.0):
    """Per-aspect (presence_weight, sentiment_weight) CPU tensors derived from training
    labels (columns 1: of `train_y`, one per aspect task). Shared by every trainer that
    supports a two_stage head (transformer encoders, BiLSTM, TextCNN) so the weight
    derivation logic lives in one place.

    `sentiment_max_ratio` bounds the max/min ratio across the 3 present-only sentiment
    classes. Uncapped effective-number weighting blows this up to 10-15x on aspects where
    "neutral" is a rare present-only label (e.g. as_service: 49 neutral vs. 1368 negative
    present samples) — confirmed via error analysis to make the model hedge toward "neutral"
    on unambiguous positive/negative text (e.g. "giao hang nhanh" predicted neutral) because
    missing a neutral sample was penalized far more than missing a negative/positive one.
    """
    n_aspects = train_y.shape[1] - 1
    presence_weights, sentiment_weights = [], []
    for i in range(1, n_aspects + 1):
        aspect_col = train_y[:, i]
        present_target = (aspect_col != 3).astype(int)
        pw = downweight_class(class_balanced_weights(present_target, 2, beta=beta), 0, absent_scale)
        presence_weights.append(pw)
        present_values = aspect_col[aspect_col != 3]
        sw = class_balanced_weights(present_values, 3, beta=beta) if len(present_values) else torch.ones(3, dtype=torch.float32)
        sw = sw.clamp(min=sw.max() / float(sentiment_max_ratio))
        sentiment_weights.append(sw)
    return presence_weights, sentiment_weights


def two_stage_multitask_loss(
    logits,
    labels,
    task_head,
    weights,
    aspect_weights,
    *,
    task_weights=None,
    loss_type: str = "ce",
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    sentiment_loss_weight: float = 0.5,
    aspect_loss_weight: float = 0.5,
    stage1_weight: float = 0.25,
    stage2_weight: float = 0.75,
    focal_gamma_present: float = 2.5,
    aspect_label_smoothing: float = 0.1,
):
    """Combined loss for a multi-task head: standard loss on the overall-sentiment task
    (task 0) plus `two_stage_aspect_loss` on each aspect task, weighted and averaged.
    Falls back to plain `multitask_loss` (honoring `task_weights` if given) when `task_head`
    isn't a two_stage head (no cached presence/sentiment logits) or `aspect_weights` is None,
    so callers can invoke this unconditionally regardless of the configured head_type."""
    if not (aspect_weights is not None and getattr(task_head, "last_presence_logits", None)):
        return multitask_loss(logits, labels, weights=weights, task_weights=task_weights, loss_type=loss_type, gamma=gamma, label_smoothing=label_smoothing)
    presence_weights, sentiment_weights = aspect_weights
    device = logits[0].device
    sent_loss = multitask_loss([logits[0]], labels[:, :1], weights=[weights[0]], loss_type=loss_type, gamma=gamma, label_smoothing=label_smoothing)
    aspect_losses = [
        two_stage_aspect_loss(
            pres_logits, sent_logits, labels[:, i + 1],
            presence_weight=presence_weights[i].to(device), sentiment_weight=sentiment_weights[i].to(device),
            stage1_weight=stage1_weight, stage2_weight=stage2_weight,
            loss_type="focal" if loss_type == "focal" else "ce",
            gamma=focal_gamma_present, label_smoothing=aspect_label_smoothing,
        )
        for i, (pres_logits, sent_logits) in enumerate(zip(task_head.last_presence_logits, task_head.last_sentiment_logits))
    ]
    aspect_loss = sum(aspect_losses) / len(aspect_losses)
    sw, aw = sentiment_loss_weight, aspect_loss_weight
    return (sw * sent_loss + aw * aspect_loss) / (sw + aw)


def sequence_focal_loss(logits, labels, ignore_index: int = -100, gamma: float = 2.0, label_smoothing: float = 0.0):
    """Token-level focal loss for seq2seq LMs. `logits` is [B, T, V], `labels` is [B, T] with
    `ignore_index` marking padding positions (matches the HF convention used by ViT5)."""
    vocab = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, label_smoothing=label_smoothing, reduction="none")
    valid = flat_labels != ignore_index
    ce = ce[valid]
    if ce.numel() == 0:
        return logits.sum() * 0.0
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()
