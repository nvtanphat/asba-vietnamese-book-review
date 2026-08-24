from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .calibration import calibrate_absent_thresholds, decode_probabilities
from .metrics import evaluate_predictions


def evaluate_model(model, val_texts, val_y, test_texts, test_y, *, output_dir: str | Path | None = None):
    val_probs = model.predict_proba(list(val_texts))
    thresholds = calibrate_absent_thresholds(val_probs, np.asarray(val_y))
    val_pred = decode_probabilities(val_probs, thresholds)
    test_probs = model.predict_proba(list(test_texts))
    test_pred = decode_probabilities(test_probs, thresholds)
    result = {
        "thresholds": thresholds,
        "val": evaluate_predictions(np.asarray(val_y), val_pred),
        "test": evaluate_predictions(np.asarray(test_y), test_pred),
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(out / "test_predictions.npy", test_pred)
    return result
