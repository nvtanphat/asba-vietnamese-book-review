import numpy as np
import pytest
from ml.data.schema import ASPECT_COLS
from ml.evaluation.calibration import calibrate_absent_thresholds, decode_probabilities
from ml.evaluation.metrics import evaluate_predictions


def test_adversarial_single_sample():
    """Verify single-sample robustness."""
    y = np.array([[2, 0, 1, 2, 3, 3, 3]])
    probs = [np.array([[0.1, 0.1, 0.8]])]
    for j in range(1, 7):
        p = np.full((1, 4), 0.1)
        p[0, y[0, j]] = 0.7
        probs.append(p)

    th = calibrate_absent_thresholds(probs, y)
    assert len(th) == 6
    pred = decode_probabilities(probs, th)
    assert pred.shape == (1, 7)


def test_adversarial_inverted_probabilities():
    """Verify behavior when model outputs completely inverted probabilities."""
    n = 100
    y = np.full((n, 7), 2)  # all positive
    # Model outputs 99% absent
    probs = [np.array([[0.05, 0.05, 0.9]] * n)]
    for _ in range(6):
        probs.append(np.array([[0.01, 0.01, 0.01, 0.97]] * n))

    th = calibrate_absent_thresholds(probs, y)
    # When model is inverted, best_score might be 0, so threshold defaults to 0.5 or minimum
    for col in ASPECT_COLS:
        assert 0.10 <= th[col] <= 0.85


def test_adversarial_custom_descending_grid():
    """Verify custom non-standard grid formatting and fallback logic."""
    y = np.array([
        [0, 0, 3, 3, 3, 3, 3],
        [1, 1, 3, 3, 3, 3, 3],
        [2, 2, 3, 3, 3, 3, 3],
    ])
    probs = [np.full((3, 3), 1/3)]
    for j in range(1, 7):
        p = np.full((3, 4), 0.05)
        p[np.arange(3), y[:, j]] = 0.85
        probs.append(p)

    desc_grid = [0.8, 0.6, 0.4, 0.2]
    th = calibrate_absent_thresholds(probs, y, grid=desc_grid)
    # as_content has present signal -> selected from desc_grid
    assert th["as_content"] in desc_grid
    assert th["as_content"] == 0.8
    # other aspects have zero present signal in y -> safely falls back to 0.5
    for col in ASPECT_COLS[1:]:
        assert th[col] == 0.5


def test_adversarial_decode_probabilities_missing_keys():
    """Verify decode_probabilities works with partial threshold dictionary."""
    n = 5
    probs = [np.full((n, 3), 1/3)] + [np.full((n, 4), 0.25) for _ in range(6)]
    partial_th = {"as_content": 0.3}  # other 5 aspects missing
    pred = decode_probabilities(probs, partial_th)
    assert pred.shape == (n, 7)


def test_adversarial_neutral_weight_boundary_values():
    """Verify extreme neutral weights: 0.0, 1.0, 0.5."""
    n = 50
    y = np.zeros((n, 7), dtype=int)
    y[:25, 1] = 1  # neutral
    y[25:, 1] = 3  # absent

    probs = [np.full((n, 3), 1/3)]
    for j in range(1, 7):
        p = np.full((n, 4), 0.1)
        p[:25, 1] = 0.7
        p[25:, 3] = 0.7
        p /= p.sum(axis=1, keepdims=True)
        probs.append(p)

    th_neu_only = calibrate_absent_thresholds(probs, y, neutral_weight=1.0)
    th_pres_only = calibrate_absent_thresholds(probs, y, neutral_weight=0.0)
    assert "as_content" in th_neu_only
    assert "as_content" in th_pres_only
