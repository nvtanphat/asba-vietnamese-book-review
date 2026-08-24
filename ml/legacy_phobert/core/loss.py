"""Loss functions: Improved Focal, Asymmetric Focal, Supervised Contrastive."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ABSENT_ASPECT_CLASS


# ---------------------------------------------------------------------------
# Improved Focal Loss helpers
# ---------------------------------------------------------------------------

def _build_smoothed_targets(logits, targets, smoothing):
    n_cls = logits.size(-1)
    with torch.no_grad():
        t = torch.full_like(logits, smoothing / n_cls)
        t.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing + smoothing / n_cls)
    return t


def _resolve_gamma(targets, logits, gamma, gamma_by_class):
    if gamma_by_class is None:
        return torch.full((targets.size(0),), float(gamma), dtype=logits.dtype, device=logits.device)
    return torch.as_tensor(gamma_by_class, dtype=logits.dtype, device=logits.device)[targets]


def _filter_easy_absent(loss, targets, true_probs, ignore_easy_absent, ignore_threshold, absent_class):
    if not (ignore_easy_absent and absent_class is not None and ignore_threshold is not None):
        return loss.mean()
    keep = ~((targets == absent_class) & (true_probs > ignore_threshold))
    return loss[keep].mean() if keep.any() else loss.mean()


def improved_focal_loss(
    logits, targets, *, gamma=2.0, alpha=None, smoothing=0.0,
    gamma_by_class=None, ignore_easy_absent=False, ignore_threshold=None, absent_class=None,
):
    smoothed   = _build_smoothed_targets(logits, targets, smoothing)
    log_probs  = F.log_softmax(logits, dim=-1)
    true_probs = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
    focal_g    = _resolve_gamma(targets, logits, gamma, gamma_by_class)
    loss       = (1 - true_probs).pow(focal_g) * -(smoothed * log_probs).sum(dim=-1)
    if alpha is not None:
        loss = loss * alpha.to(logits.device)[targets]
    return _filter_easy_absent(loss, targets, true_probs, ignore_easy_absent, ignore_threshold, absent_class)


def asymmetric_focal_loss(logits, targets, *, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, class_weights=None):
    probs = torch.softmax(logits, dim=-1)
    log_p = torch.log(probs.clamp(min=1e-8))
    C     = logits.size(-1)
    oh    = F.one_hot(targets, C).float().to(logits.device)
    p_pos = (probs * oh).sum(-1)
    loss  = -(1 - p_pos).pow(gamma_pos) * (log_p * oh).sum(-1)
    pn_s  = (probs * (1 - oh) + clip).clamp(max=1.0)
    loss  = loss - (pn_s.pow(gamma_neg) * (1 - oh) * torch.log((1.0 - probs * (1 - oh)).clamp(min=1e-8))).sum(-1)
    if class_weights is not None:
        loss = loss * class_weights.to(logits.device)[targets]
    return loss.mean()


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

class SupervisedContrastiveLoss(nn.Module):
    """Pull CLS embeddings of same-class samples together → better Trung lập separability."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        B = features.size(0)
        if B < 2:
            return features.new_tensor(0.0)
        feat = F.normalize(features, dim=-1)
        sim  = feat @ feat.T / self.temperature
        pos  = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos.fill_diagonal_(0.0)
        denom = torch.log(
            (torch.exp(sim) * (1 - torch.eye(B, device=sim.device))).sum(-1).clamp(min=1e-8)
        )
        return (-((pos * sim).sum(-1) / pos.sum(-1).clamp(min=1) - denom)).mean()


scl_criterion = SupervisedContrastiveLoss(temperature=0.07)


# ---------------------------------------------------------------------------
# Unified loss dispatcher
# ---------------------------------------------------------------------------

def compute_label_loss(logits, targets, *, loss_name="ce", class_weights=None, gamma=2.0, alpha=None, focal_params=None):
    fp = dict(focal_params or {})
    if loss_name == "ce":
        return F.cross_entropy(logits, targets, weight=class_weights)
    if loss_name == "improved_focal":
        return improved_focal_loss(
            logits, targets,
            gamma=fp.get("gamma", gamma), alpha=alpha,
            smoothing=fp.get("smoothing", 0.0),
            gamma_by_class=fp.get("gamma_by_class"),
            ignore_easy_absent=fp.get("ignore_easy_absent", False),
            ignore_threshold=fp.get("ignore_threshold"),
            absent_class=fp.get("absent_class"),
        )
    if loss_name == "asl":
        return asymmetric_focal_loss(
            logits, targets,
            gamma_neg=fp.get("asl_gamma_neg", 4.0),
            gamma_pos=fp.get("asl_gamma_pos", 1.0),
            clip=fp.get("asl_clip", 0.05),
            class_weights=class_weights,
        )
    raise ValueError(f"Unknown loss_name: {loss_name!r}")
