from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from ml.data.schema import ASPECT_COLS, ABSENT_CLASS


def calibrate_absent_thresholds(
    probabilities: list[np.ndarray],
    y_true: np.ndarray,
    grid: np.ndarray | list[float] | None = None,
    neutral_weight: float = 0.15,
) -> dict[str, float]:
    """Calibrate per-aspect present threshold from P(present)=1-P(absent) on validation data.

    Optimizes a balanced Present-Only Macro F1 over sentiment classes [0, 1, 2]
    across all validation samples, penalizing both false-positive hallucinations
    (true absent predicted present) and false negatives (true present predicted absent),
    with neutral protection, without being dominated by the overwhelming absent class support.
    """
    if grid is None:
        grid = np.linspace(0.10, 0.85, 31)
    grid = np.asarray(grid, dtype=float)
    thresholds: dict[str, float] = {}

    for i, col in enumerate(ASPECT_COLS, start=1):
        probs = np.asarray(probabilities[i])
        present_score = 1.0 - probs[:, ABSENT_CLASS]
        sentiment_pred = np.argmax(probs[:, :3], axis=1)
        yt = y_true[:, i]

        best_t, best_score = 0.5, -1.0
        for t in grid:
            pred = np.where(present_score >= t, sentiment_pred, ABSENT_CLASS)
            # 3-class present-only macro F1 evaluated across all samples [0, 1, 2]
            f1_pres = precision_recall_fscore_support(
                yt, pred, labels=[0, 1, 2], average="macro", zero_division=0
            )[2]
            f1_neu = precision_recall_fscore_support(
                yt, pred, labels=[1], average="macro", zero_division=0
            )[2]

            score = (1.0 - neutral_weight) * f1_pres + neutral_weight * f1_neu
            if score > best_score:
                best_score, best_t = float(score), float(t)

        if best_score <= 0.0:
            best_t = 0.5
        thresholds[col] = best_t
    return thresholds


def decode_probabilities(
    probabilities: list[np.ndarray], thresholds: dict[str, float] | None = None
) -> np.ndarray:
    probs = [np.asarray(x) for x in probabilities]
    n = probs[0].shape[0]
    pred = np.zeros((n, 7), dtype=int)
    pred[:, 0] = probs[0].argmax(axis=1)
    thresholds = thresholds or {}
    for i, col in enumerate(ASPECT_COLS, start=1):
        p = probs[i]
        if p.shape[1] != 4:
            raise ValueError(f"Aspect probabilities must have 4 classes, got {p.shape}")
        present_score = 1.0 - p[:, ABSENT_CLASS]
        sentiment = p[:, :3].argmax(axis=1)
        pred[:, i] = np.where(present_score >= thresholds.get(col, 0.5), sentiment, ABSENT_CLASS)
    return pred

