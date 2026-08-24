from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ml.data.schema import ASPECT_COLS, ABSENT_CLASS


def _macro_f1(y_true, y_pred, labels=None):
    kwargs = {"average": "macro", "zero_division": 0}
    if labels is not None:
        kwargs["labels"] = labels
    return float(precision_recall_fscore_support(y_true, y_pred, **kwargs)[2])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluate [overall sentiment + 6 four-class aspect targets]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 7:
        raise ValueError(f"Expected (N, 7) arrays, got {y_true.shape} and {y_pred.shape}")

    sent_true, sent_pred = y_true[:, 0], y_pred[:, 0]
    asp_true, asp_pred = y_true[:, 1:], y_pred[:, 1:]

    sent_f1 = _macro_f1(sent_true, sent_pred, [0, 1, 2])
    asp_all_f1 = _macro_f1(asp_true.ravel(), asp_pred.ravel(), [0, 1, 2, 3])

    true_presence = (asp_true != ABSENT_CLASS).astype(int)
    pred_presence = (asp_pred != ABSENT_CLASS).astype(int)
    presence_f1 = _macro_f1(true_presence.ravel(), pred_presence.ravel(), [0, 1])

    present_mask = asp_true != ABSENT_CLASS
    asp_present_f1 = 0.0
    asp_neutral_f1 = 0.0
    if present_mask.any():
        asp_present_f1 = _macro_f1(asp_true[present_mask], asp_pred[present_mask], [0, 1, 2])
        asp_neutral_f1 = _macro_f1(asp_true[present_mask], asp_pred[present_mask], [1])

    per_aspect_4class = []
    result = {
        "f1_sentiment": sent_f1,
        "f1_aspect_all": asp_all_f1,
        "f1_aspect_presence": presence_f1,
        "f1_aspect_present": asp_present_f1,
        "f1_aspect_neutral_present": asp_neutral_f1,
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "label_accuracy": float(accuracy_score(y_true.ravel(), y_pred.ravel())),
    }

    for i, col in enumerate(ASPECT_COLS):
        mask = asp_true[:, i] != ABSENT_CLASS
        result[f"f1_{col}"] = _macro_f1(asp_true[:, i][mask], asp_pred[:, i][mask], [0, 1, 2]) if mask.any() else 0.0
        result[f"presence_f1_{col}"] = _macro_f1(true_presence[:, i], pred_presence[:, i], [0, 1])
        f1_4 = _macro_f1(asp_true[:, i], asp_pred[:, i])
        result[f"f1_4class_{col}"] = f1_4
        per_aspect_4class.append(f1_4)

    f1_aspect_4class_mean = float(np.mean(per_aspect_4class))
    result["f1_aspect_4class_mean"] = f1_aspect_4class_mean
    # Primary research score: 4-class aspect F1 penalizes both missed and hallucinated aspects.
    result["f1_combined"] = 0.5 * sent_f1 + 0.5 * f1_aspect_4class_mean
    # Kept only to compare with the historical present-only SentenAI reporting convention.
    result["f1_combined_legacy_present_only"] = 0.5 * sent_f1 + 0.5 * asp_present_f1
    return {k: round(float(v), 6) for k, v in result.items()}
