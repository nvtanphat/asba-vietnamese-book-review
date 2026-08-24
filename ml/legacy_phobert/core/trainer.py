"""Class-weight helpers, joint sampler, and ABSAMultiTaskTrainer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer, TrainingArguments

from .config import (
    ABSENT_ASPECT_CLASS, ASPECT_COLS, ASPECT_MULT_CLIP_MAX, ASPECT_MULT_CLIP_MIN, BASE_FOCAL_GAMMA,
    CLASS_BALANCED_BETA, DEFAULT_ASPECT_LOSS_WEIGHT, DEFAULT_SENTIMENT_LOSS_WEIGHT,
    EPOCHS, BATCH_SIZE, FOCAL_CONFIG_STANDARD, LEARNING_RATE,
    N_ASPECTS, SAMPLER_TEMPERATURE, SAMPLER_WEIGHT_CAP,
    STAGE1_LOSS_WEIGHT, STAGE2_LOSS_WEIGHT,
)
from .loss import compute_label_loss, scl_criterion
from .model import parse_logits
from .seed import GLOBAL_GENERATOR, seed_worker

# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_class_balanced_weights(labels, classes, beta=0.999, normalize="mean", cap=None):
    """Class-balanced weights (Cui et al.), matching the notebook's helper 1:1.

    `normalize`: "mean" divides by the mean of the valid (non-zero) weights,
    "max" divides by the max weight, anything else (e.g. False) skips normalization.
    """
    values  = np.asarray(labels)
    weights = []
    for c in classes:
        n = int((values == c).sum())
        w = (1 - beta) / (1 - beta**n) if n > 0 else 0.0
        if cap is not None:
            w = min(w, cap)
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    valid = weights > 0
    if normalize == "mean" and valid.any():
        weights = weights / weights[valid].mean()
    elif normalize == "max" and weights.max() > 0:
        weights = weights / weights.max()
    return weights


def compute_aspect_loss_multipliers(train_df: pd.DataFrame) -> dict[str, float]:
    """Data-driven rare-aspect loss multiplier (as_price/as_service get more attention,
    normalized mean stays 1) — matches the notebook's `ASPECT_LOSS_MULTIPLIERS` exactly.
    """
    present_counts = np.array(
        [
            int((pd.to_numeric(train_df[col], errors="coerce").fillna(ABSENT_ASPECT_CLASS).astype(int)
                 != ABSENT_ASPECT_CLASS).sum())
            for col in ASPECT_COLS
        ],
        dtype=np.float32,
    )
    median_present = np.median(present_counts)
    raw_mult = np.sqrt(median_present / np.maximum(present_counts, 1.0))
    raw_mult = np.clip(raw_mult, ASPECT_MULT_CLIP_MIN, ASPECT_MULT_CLIP_MAX)
    raw_mult = raw_mult / raw_mult.mean()
    return {col: float(raw_mult[i]) for i, col in enumerate(ASPECT_COLS)}


def build_weight_tensors(train_df: pd.DataFrame):
    """Return (sent_weight_tensors, presence_weight_tensors, stage2_weight_tensors)."""
    cb_sent = compute_class_balanced_weights(
        train_df["sentiment"].values, [0, 1, 2], beta=CLASS_BALANCED_BETA, normalize="max")
    cb_sent_nc = compute_class_balanced_weights(
        train_df["sentiment"].values, [0, 1, 2], beta=CLASS_BALANCED_BETA, normalize="max", cap=1.0)
    cb_sent_mean_capped = compute_class_balanced_weights(
        train_df["sentiment"].values, [0, 1, 2], beta=CLASS_BALANCED_BETA, normalize="mean", cap=6.0)
    sent_tensors = {
        "class_balanced": torch.tensor(cb_sent,             dtype=torch.float),
        "neutral_capped": torch.tensor(cb_sent_nc,          dtype=torch.float),
        "mean_capped":    torch.tensor(cb_sent_mean_capped, dtype=torch.float),
    }

    pres_tensors, s2_tensors = {}, {}
    for col in ASPECT_COLS:
        vals  = pd.to_numeric(train_df[col], errors="coerce").fillna(ABSENT_ASPECT_CLASS).astype(int)
        pres_labels = (vals != ABSENT_ASPECT_CLASS).astype(int)
        pres_tensors[col] = torch.tensor(
            compute_class_balanced_weights(pres_labels, [0, 1], beta=0.9995, normalize="mean", cap=6.0),
            dtype=torch.float,
        )
        present_vals = vals[vals != ABSENT_ASPECT_CLASS]
        s2_w = (compute_class_balanced_weights(present_vals, [0, 1, 2], beta=0.9995, normalize="mean", cap=6.0)
                if len(present_vals) else np.ones(3, dtype=np.float32))
        s2_tensors[col] = torch.tensor(s2_w, dtype=torch.float)

    return sent_tensors, pres_tensors, s2_tensors

# ---------------------------------------------------------------------------
# Joint label sampler
# ---------------------------------------------------------------------------

def build_joint_label_sampler(frame: pd.DataFrame, temperature=SAMPLER_TEMPERATURE, include_neutral_bucket=False):
    aspect_values = frame[ASPECT_COLS].apply(pd.to_numeric, errors="coerce").fillna(3).astype(int)
    present_count = aspect_values.ne(3).sum(axis=1).clip(upper=N_ASPECTS).astype(int)

    group = "sent=" + frame["sentiment"].astype(str) + "|present=" + present_count.astype(str)
    if include_neutral_bucket:
        neutral_bucket = aspect_values.eq(1).sum(axis=1).clip(upper=2).astype(int)
        group = group + "|neutral=" + neutral_bucket.astype(str)

    counts         = group.value_counts()
    sample_weights = group.map(lambda g: counts[g] ** (-temperature)).astype(float)
    sample_weights = (sample_weights / sample_weights.mean()).clip(upper=SAMPLER_WEIGHT_CAP)
    sample_weights = sample_weights / sample_weights.mean()

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights.values, dtype=torch.double),
        num_samples=len(sample_weights), replacement=True,
        generator=GLOBAL_GENERATOR,
    )
    summary = (
        pd.DataFrame({"group": group, "weight": sample_weights})
        .groupby("group", as_index=False)
        .agg(count=("group", "size"), avg_weight=("weight", "mean"))
        .sort_values(["count", "group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return sampler, summary

# ---------------------------------------------------------------------------
# TrainingArguments factory
# ---------------------------------------------------------------------------

def build_training_args(output_dir: str, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE) -> TrainingArguments:
    use_fp16 = torch.cuda.is_available()
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.05,
        max_grad_norm=1.0,
        fp16=use_fp16,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_combined",
        greater_is_better=True,
        report_to="none",
    )
    if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)

# ---------------------------------------------------------------------------
# Multi-task Trainer
# ---------------------------------------------------------------------------

class ABSAMultiTaskTrainer(Trainer):
    def __init__(
        self, *args,
        sent_weights=None, loss_name="ce", use_class_weights=False,
        focal_gamma=BASE_FOCAL_GAMMA, focal_config=None,
        sentiment_loss_weight=DEFAULT_SENTIMENT_LOSS_WEIGHT,
        aspect_loss_weight=DEFAULT_ASPECT_LOSS_WEIGHT,
        scl_weight=0.0, train_sampler=None,
        presence_weights: dict | None = None,
        stage2_weights:   dict | None = None,
        aspect_loss_multipliers: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sent_weights      = sent_weights
        self.loss_name         = loss_name
        self.use_class_weights = use_class_weights
        self.focal_gamma       = focal_gamma
        self.focal_config      = focal_config or {}
        total = float(sentiment_loss_weight) + float(aspect_loss_weight)
        self.sentiment_loss_weight = float(sentiment_loss_weight) / total
        self.aspect_loss_weight    = float(aspect_loss_weight)    / total
        self.scl_weight       = scl_weight
        self.train_sampler    = train_sampler
        self.presence_weights = presence_weights or {}
        self.stage2_weights   = stage2_weights   or {}
        self.aspect_loss_multipliers = aspect_loss_multipliers or {}

    def get_train_dataloader(self):
        if self.train_sampler is None:
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self.train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
            generator=GLOBAL_GENERATOR,
        )

    def _sentiment_loss(self, logits, labels):
        sent_logits, _, _ = parse_logits(logits)
        w     = self.sent_weights.to(logits.device) if self.use_class_weights and self.sent_weights is not None else None
        alpha = w if self.loss_name == "improved_focal" else None
        fp    = dict(self.focal_config.get("sentiment") or {})
        return compute_label_loss(sent_logits, labels[:, 0],
                                  loss_name=self.loss_name, class_weights=w,
                                  gamma=self.focal_gamma, alpha=alpha, focal_params=fp)

    def _aspect_loss(self, logits, labels):
        _, pres_logits, asp_sent_logits = parse_logits(logits)
        aspect_labels = labels[:, 1:]
        s1_losses, s2_losses = [], []
        for i, col in enumerate(ASPECT_COLS):
            asp_label  = aspect_labels[:, i]
            pres_label = (asp_label != ABSENT_ASPECT_CLASS).long()
            mult       = self.aspect_loss_multipliers.get(col, 1.0)
            pres_w     = self.presence_weights.get(col)
            if pres_w is not None:
                pres_w = pres_w.to(logits.device)
            s1_losses.append(mult * F.cross_entropy(pres_logits[:, i, :], pres_label, weight=pres_w))

            present_mask = asp_label != ABSENT_ASPECT_CLASS
            if present_mask.any():
                fp = dict(self.focal_config.get("aspect", {}))
                if "gamma_by_class" in fp:
                    fp["gamma_by_class"] = fp["gamma_by_class"][:3]
                for key in ("ignore_easy_absent", "ignore_threshold", "absent_class"):
                    fp.pop(key, None)
                s2_w = self.stage2_weights.get(col)
                if s2_w is not None:
                    s2_w = s2_w.to(logits.device)
                s2_losses.append(mult * compute_label_loss(
                    asp_sent_logits[:, i, :][present_mask], asp_label[present_mask],
                    loss_name=self.loss_name, class_weights=s2_w,
                    gamma=self.focal_gamma, focal_params=fp,
                ))

        loss_s1 = sum(s1_losses) / len(s1_losses)
        loss_s2 = sum(s2_losses) / len(s2_losses) if s2_losses else logits.new_zeros(())
        return STAGE1_LOSS_WEIGHT * loss_s1 + STAGE2_LOSS_WEIGHT * loss_s2

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = (self.sentiment_loss_weight * self._sentiment_loss(logits, labels)
                   + self.aspect_loss_weight  * self._aspect_loss(logits, labels))
        if self.scl_weight > 0 and model.training:
            cls_vec = getattr(model, "_last_cls", None)
            if cls_vec is not None:
                loss = loss + self.scl_weight * scl_criterion(cls_vec, labels[:, 0])
        return (loss, outputs) if return_outputs else loss
