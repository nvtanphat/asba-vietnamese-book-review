"""Threshold calibration, decoding, and single-review inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .config import (
    ABSENT_ASPECT_CLASS, ASPECT_COLS, ASPECT_LABELS,
    MAX_LENGTH, N_ASPECTS, PRES_DIM, SENT_DIM, ASP_SENT_DIM,
    SENTIMENT_LABELS, THRESHOLD_NEUTRAL_WEIGHT,
)
from .data import build_absa_input
from .model import parse_logits  # noqa: F401 (re-exported for convenience)


def softmax_np(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def calibrate_presence_thresholds(logits, labels, neutral_weight: float = THRESHOLD_NEUTRAL_WEIGHT):
    """Grid-search per-aspect presence thresholds on a validation set."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    s, p   = SENT_DIM, N_ASPECTS * PRES_DIM
    pres_probs    = softmax_np(logits[:, s : s + p].reshape(-1, N_ASPECTS, PRES_DIM))[:, :, 1]
    pred_asp_sent = np.argmax(logits[:, s + p :].reshape(-1, N_ASPECTS, ASP_SENT_DIM), axis=-1)

    thresholds, rows = {}, []
    for i, col in enumerate(ASPECT_COLS):
        y_true       = labels[:, 1 + i]
        present_mask = y_true != ABSENT_ASPECT_CLASS
        scores       = pres_probs[:, i]
        best_t, best_obj, best_f1, best_neu = 0.5, -1.0, 0.0, 0.0
        for t in np.linspace(0.05, 0.95, 19):
            y_pred = np.where(scores >= t, pred_asp_sent[:, i], ABSENT_ASPECT_CLASS)
            if present_mask.any():
                f1  = precision_recall_fscore_support(y_true[present_mask], y_pred[present_mask], labels=[0,1,2], average="macro", zero_division=0)[2]
                neu = precision_recall_fscore_support(y_true[present_mask], y_pred[present_mask], labels=[1],    average="macro", zero_division=0)[2]
            else:
                f1 = neu = 0.0
            obj = (1.0 - neutral_weight) * f1 + neutral_weight * neu
            if obj > best_obj:
                best_t, best_obj, best_f1, best_neu = float(t), float(obj), float(f1), float(neu)
        thresholds[col] = best_t
        rows.append({"aspect": col, "threshold": best_t, "macro_f1": best_f1, "neutral_f1": best_neu})

    return thresholds, pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def decode_with_thresholds(logits, thresholds: dict):
    """Apply calibrated presence thresholds to raw logits."""
    logits = np.asarray(logits)
    s, p   = SENT_DIM, N_ASPECTS * PRES_DIM
    pred_sent     = np.argmax(logits[:, :s], axis=-1)
    pres_probs    = softmax_np(logits[:, s : s + p].reshape(-1, N_ASPECTS, PRES_DIM))[:, :, 1]
    pred_presence = np.zeros((logits.shape[0], N_ASPECTS), dtype=int)
    for i, col in enumerate(ASPECT_COLS):
        pred_presence[:, i] = (pres_probs[:, i] >= thresholds.get(col, 0.5)).astype(int)
    pred_asp_sent = np.argmax(logits[:, s + p :].reshape(-1, N_ASPECTS, ASP_SENT_DIM), axis=-1)
    pred_asps     = np.where(pred_presence == 0, ABSENT_ASPECT_CLASS, pred_asp_sent)
    return pred_sent, pred_presence, pred_asp_sent, pred_asps


def predict_review(text: str, model, tokenizer, thresholds: dict, device) -> dict:
    """Single-review inference using calibrated presence thresholds."""
    import torch
    enc = tokenizer(
        build_absa_input(text),
        padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
    logits     = out.logits.cpu().numpy()
    pred_sent, _, _, pred_asps = decode_with_thresholds(logits, thresholds)
    result = {"overall_sentiment": SENTIMENT_LABELS.get(int(pred_sent[0]), "Unknown"), "aspects": {}}
    for i, col in enumerate(ASPECT_COLS):
        val = int(pred_asps[0, i])
        if val != ABSENT_ASPECT_CLASS:
            result["aspects"][col.replace("as_", "")] = ASPECT_LABELS.get(val, "Unknown")
    return result
