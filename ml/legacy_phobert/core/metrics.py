"""Evaluation metrics for ABSA multi-task model."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import ABSENT_ASPECT_CLASS, ASPECT_COLS, N_ASPECTS, PRES_DIM, SENT_DIM, ASP_SENT_DIM


def compute_metrics(eval_pred):
    """HuggingFace Trainer-compatible metrics function."""
    logits, labels = eval_pred
    s = SENT_DIM
    p = N_ASPECTS * PRES_DIM

    pred_sent     = np.argmax(logits[:, :s], axis=-1)
    pred_presence = np.argmax(logits[:, s : s + p].reshape(-1, N_ASPECTS, PRES_DIM), axis=-1)
    pred_asp_sent = np.argmax(logits[:, s + p :].reshape(-1, N_ASPECTS, ASP_SENT_DIM), axis=-1)
    pred_aspects  = np.where(pred_presence == 0, ABSENT_ASPECT_CLASS, pred_asp_sent)

    true_sent    = labels[:, 0]
    true_aspects = labels[:, 1:]

    _, _, f1_sent, _ = precision_recall_fscore_support(
        true_sent, pred_sent, labels=[0, 1, 2], average="macro", zero_division=0)

    present_mask           = true_aspects.flatten() != ABSENT_ASPECT_CLASS
    f1_asp_present         = 0.0
    f1_asp_neutral_present = 0.0
    if present_mask.any():
        tp = true_aspects.flatten()[present_mask]
        pp = pred_aspects.flatten()[present_mask]
        f1_asp_present         = precision_recall_fscore_support(tp, pp, labels=[0,1,2], average="macro", zero_division=0)[2]
        f1_asp_neutral_present = precision_recall_fscore_support(tp, pp, labels=[1],    average="macro", zero_division=0)[2]

    _, _, f1_asp_all, _ = precision_recall_fscore_support(
        true_aspects.flatten(), pred_aspects.flatten(), labels=[0,1,2,3], average="macro", zero_division=0)

    asp_f1s = {}
    for i, col in enumerate(ASPECT_COLS):
        mask = true_aspects[:, i] != ABSENT_ASPECT_CLASS
        if mask.any():
            _, _, f1_i, _ = precision_recall_fscore_support(
                true_aspects[:, i][mask], pred_aspects[:, i][mask],
                labels=[0, 1, 2], average="macro", zero_division=0)
        else:
            f1_i = 0.0
        asp_f1s[f"f1_{col}"] = round(float(f1_i), 4)

    acc = accuracy_score(
        np.concatenate([true_sent, true_aspects.flatten()]),
        np.concatenate([pred_sent, pred_aspects.flatten()]),
    )
    return {
        "f1_sentiment":               round(float(f1_sent), 4),
        "f1_aspect_all":              round(float(f1_asp_all), 4),
        "f1_aspect_present":          round(float(f1_asp_present), 4),
        "f1_aspect_neutral_present":  round(float(f1_asp_neutral_present), 4),
        "f1_combined":                round(0.5 * f1_sent + 0.5 * f1_asp_present, 4),
        "accuracy":                   round(float(acc), 4),
        **asp_f1s,
    }


def metrics_from_decoded(true_sent, pred_sent, true_asps, pred_asps) -> dict:
    """Compute metrics from already-decoded label arrays (post calibration)."""
    f1_s    = precision_recall_fscore_support(true_sent, pred_sent, labels=[0,1,2], average="macro", zero_division=0)[2]
    tf, pf  = true_asps.flatten(), pred_asps.flatten()
    f1_all  = precision_recall_fscore_support(tf, pf, labels=[0,1,2,3], average="macro", zero_division=0)[2]
    pm      = tf != ABSENT_ASPECT_CLASS
    f1_pres = f1_neu = 0.0
    if pm.any():
        f1_pres = precision_recall_fscore_support(tf[pm], pf[pm], labels=[0,1,2], average="macro", zero_division=0)[2]
        f1_neu  = precision_recall_fscore_support(tf[pm], pf[pm], labels=[1],    average="macro", zero_division=0)[2]
    return {
        "f1_sentiment":              round(float(f1_s),    4),
        "f1_aspect_all":             round(float(f1_all),  4),
        "f1_aspect_present":         round(float(f1_pres), 4),
        "f1_aspect_neutral_present": round(float(f1_neu),  4),
        "f1_combined":               round(float(0.5 * f1_s + 0.5 * f1_pres), 4),
    }
