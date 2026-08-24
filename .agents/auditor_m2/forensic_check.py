import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from ml.data.schema import ASPECT_COLS, ABSENT_CLASS
from ml.evaluation.calibration import calibrate_absent_thresholds, decode_probabilities
from ml.evaluation.metrics import evaluate_predictions

def test_independent_argmax_verification():
    """Verify that calibrate_absent_thresholds produces the EXACT mathematical argmax
    computed by an independent reference calculation."""
    np.random.seed(12345)
    n_samples = 300
    y_true = np.zeros((n_samples, 7), dtype=int)
    y_true[:, 0] = np.random.randint(0, 3, size=n_samples)
    for j in range(1, 7):
        y_true[:, j] = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.1, 0.05, 0.15, 0.7])

    probs = [np.random.dirichlet([1, 1, 1], size=n_samples)]
    for j in range(1, 7):
        probs.append(np.random.dirichlet([1, 1, 1, 3], size=n_samples))

    grid = np.linspace(0.1, 0.9, 17)
    neutral_weight = 0.25

    calibrated_th = calibrate_absent_thresholds(probs, y_true, grid=grid, neutral_weight=neutral_weight)

    # Independent reference calculation
    for i, col in enumerate(ASPECT_COLS, start=1):
        p = probs[i]
        present_score = 1.0 - p[:, ABSENT_CLASS]
        sentiment_pred = np.argmax(p[:, :3], axis=1)
        yt = y_true[:, i]

        best_score = -1.0
        best_t = 0.5
        for t in grid:
            pred = np.where(present_score >= t, sentiment_pred, ABSENT_CLASS)
            f1_pres = precision_recall_fscore_support(
                yt, pred, labels=[0, 1, 2], average="macro", zero_division=0
            )[2]
            f1_neu = precision_recall_fscore_support(
                yt, pred, labels=[1], average="macro", zero_division=0
            )[2]
            score = (1.0 - neutral_weight) * f1_pres + neutral_weight * f1_neu
            if score > best_score:
                best_score = float(score)
                best_t = float(t)
        if best_score <= 0.0:
            best_t = 0.5

        assert np.isclose(calibrated_th[col], best_t), (
            f"Aspect {col}: calibrated {calibrated_th[col]} != reference {best_t}"
        )
    print("[PASS] Independent argmax verification PASSED.")


def test_sensitivity_to_varied_probabilities():
    """Verify that thresholds are not static/hardcoded and shift logically with shifted probability inputs."""
    np.random.seed(999)
    n = 200
    y_true = np.zeros((n, 7), dtype=int)
    # aspect 1 (as_content): 100 pos (2), 100 absent (3)
    y_true[:100, 1] = 2
    y_true[100:, 1] = 3

    # Case A: Model outputs high P(absent) for all
    probs_a = [np.full((n, 3), 1/3)]
    for _ in range(6):
        p = np.zeros((n, 4))
        p[:100, 2] = 0.3
        p[:100, 3] = 0.7
        p[100:, 3] = 0.95
        p[100:, 2] = 0.05
        probs_a.append(p)

    # Case B: Model outputs high P(present) for all
    probs_b = [np.full((n, 3), 1/3)]
    for _ in range(6):
        p = np.zeros((n, 4))
        p[:100, 2] = 0.9
        p[:100, 3] = 0.1
        p[100:, 3] = 0.6
        p[100:, 2] = 0.4
        probs_b.append(p)

    th_a = calibrate_absent_thresholds(probs_a, y_true)
    th_b = calibrate_absent_thresholds(probs_b, y_true)

    print(f"Sensitivity test thresholds - Case A: {th_a['as_content']}, Case B: {th_b['as_content']}")
    assert th_a["as_content"] != th_b["as_content"], "Thresholds must be data-dependent, not static"
    print("[PASS] Sensitivity to varied probabilities PASSED.")


def test_real_validation_data_distribution():
    """Test on actual dataset split distribution."""
    import json
    with open("data/splits/val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    n = len(val_data)
    y_val = np.zeros((n, 7), dtype=int)
    for idx, item in enumerate(val_data):
        y_val[idx, 0] = item["sentiment"]
        for j, col in enumerate(ASPECT_COLS, start=1):
            y_val[idx, j] = item[col]

    print(f"Validation set loaded: {n} samples")
    for j, col in enumerate(ASPECT_COLS, start=1):
        counts = np.bincount(y_val[:, j], minlength=4)
        pct_absent = counts[3] / n * 100
        print(f"  {col:15s}: absent={counts[3]} ({pct_absent:.1f}%), neg={counts[0]}, neu={counts[1]}, pos={counts[2]}")

    # Generate synthetic realistic probabilities calibrated to val ground truth with slight noise
    probs = [np.full((n, 3), 0.1)]
    probs[0][np.arange(n), y_val[:, 0]] = 0.8
    probs[0] /= probs[0].sum(axis=1, keepdims=True)

    for j, col in enumerate(ASPECT_COLS, start=1):
        p = np.full((n, 4), 0.05)
        # Ground truth class gets higher probability, with realistic noise
        p[np.arange(n), y_val[:, j]] = 0.75 + np.random.uniform(-0.15, 0.15, size=n)
        p = np.clip(p, 0.01, 0.99)
        p /= p.sum(axis=1, keepdims=True)
        probs.append(p)

    th = calibrate_absent_thresholds(probs, y_val)
    print("Calibrated thresholds on realistic validation data:")
    for col, val in th.items():
        print(f"  {col}: {val}")
        assert 0.10 <= val <= 0.85, f"Threshold {val} out of grid bounds"

    pred = decode_probabilities(probs, th)
    metrics = evaluate_predictions(y_val, pred)
    print(f"Validation F1 Combined: {metrics['f1_combined']:.4f}")
    print(f"Validation F1 as_price: {metrics['f1_as_price']:.4f}")
    print(f"Validation F1 as_service: {metrics['f1_as_service']:.4f}")

    assert metrics["f1_as_price"] >= 0.40, "Minority aspect price F1 must meet threshold"
    assert metrics["f1_as_service"] >= 0.40, "Minority aspect service F1 must meet threshold"
    print("[PASS] Real validation data test PASSED.")


if __name__ == "__main__":
    test_independent_argmax_verification()
    test_sensitivity_to_varied_probabilities()
    test_real_validation_data_distribution()
    print("\nALL FORENSIC CHECKS PASSED.")
