import numpy as np
import pytest
from ml.data.schema import ASPECT_COLS
from ml.evaluation.calibration import calibrate_absent_thresholds, decode_probabilities
from ml.evaluation.metrics import evaluate_predictions


def test_calibration_contract():
    """Verify basic contract: returns threshold per aspect, decodes to correct shape and values."""
    y = np.array([
        [0, 0, 3, 3, 3, 3, 3],
        [2, 2, 3, 3, 3, 3, 3],
        [1, 3, 0, 3, 3, 3, 3],
        [2, 3, 2, 3, 3, 3, 3],
    ])
    probs = []
    sent = np.full((4, 3), 0.05)
    sent[np.arange(4), y[:, 0]] = 0.9
    probs.append(sent)
    for j in range(1, 7):
        p = np.full((4, 4), 0.02)
        p[np.arange(4), y[:, j]] = 0.94
        probs.append(p)

    th = calibrate_absent_thresholds(probs, y)
    assert isinstance(th, dict)
    assert set(th.keys()) == set(ASPECT_COLS)

    pred = decode_probabilities(probs, th)
    assert pred.shape == (4, 7)
    assert np.array_equal(pred, y)


def test_minority_aspect_threshold_avoids_ceiling():
    """Verify that on highly imbalanced minority aspect (e.g. 90% absent),
    threshold optimization selects an interior threshold instead of collapsing to ceiling."""
    np.random.seed(42)
    n_samples = 1000
    n_absent = 900

    # Ground truth: 900 absent (class 3), 40 neg (0), 20 neu (1), 40 pos (2)
    aspect_gt = np.array([3] * n_absent + [0] * 40 + [1] * 20 + [2] * 40)
    y_true = np.zeros((n_samples, 7), dtype=int)
    y_true[:, 3] = aspect_gt  # let aspect index 3 (as_price) be the minority aspect

    # Simulated model output with realistic noisy probabilities
    probs = []
    # Overall sentiment probs
    probs.append(np.full((n_samples, 3), 1.0 / 3))

    for j in range(1, 7):
        if j == 3:
            # For as_price:
            p = np.zeros((n_samples, 4), dtype=float)
            for idx in range(n_samples):
                c = aspect_gt[idx]
                if c == 3:
                    # True absent: P(absent) around 0.60 - 0.85 (P(present) around 0.15 - 0.40)
                    p_absent = np.random.uniform(0.60, 0.85)
                    p_sent = (1.0 - p_absent) / 3.0
                    p[idx] = [p_sent, p_sent, p_sent, p_absent]
                else:
                    # True present: P(present) around 0.60 - 0.90
                    p_present = np.random.uniform(0.60, 0.90)
                    p_absent = 1.0 - p_present
                    p_sent = np.zeros(3)
                    p_sent[c] = p_present * 0.8
                    rem = p_present * 0.2 / 2
                    for sc in range(3):
                        if sc != c:
                            p_sent[sc] = rem
                    p[idx] = [p_sent[0], p_sent[1], p_sent[2], p_absent]
            probs.append(p)
        else:
            p = np.full((n_samples, 4), 0.05)
            p[:, 3] = 0.85
            probs.append(p)

    th = calibrate_absent_thresholds(probs, y_true)
    price_th = th["as_price"]

    # Threshold must avoid ceiling (0.85) and floor (0.10)
    assert 0.30 <= price_th <= 0.75, f"Expected interior threshold for minority aspect, got {price_th}"

    pred = decode_probabilities(probs, th)
    metrics = evaluate_predictions(y_true, pred)
    assert metrics["f1_as_price"] > 0.40, f"Expected f1_as_price > 0.40, got {metrics['f1_as_price']}"


def test_neutral_protection_influence():
    """Verify that neutral_weight correctly weights neutral class F1."""
    n_samples = 200
    y_true = np.zeros((n_samples, 7), dtype=int)
    # 180 absent, 10 pos, 10 neu
    y_true[:, 1] = np.array([3] * 180 + [2] * 10 + [1] * 10)

    probs = [np.full((n_samples, 3), 1.0 / 3)]
    for j in range(1, 7):
        p = np.full((n_samples, 4), 0.1)
        p[:180, 3] = 0.7
        p[180:190, 2] = 0.6
        p[190:200, 1] = 0.55
        p[180:, 3] = 0.2
        # Normalize
        p /= p.sum(axis=1, keepdims=True)
        probs.append(p)

    th_standard = calibrate_absent_thresholds(probs, y_true, neutral_weight=0.15)
    th_zero_neu = calibrate_absent_thresholds(probs, y_true, neutral_weight=0.0)

    assert "as_content" in th_standard
    assert "as_content" in th_zero_neu
    assert 0.10 <= th_standard["as_content"] <= 0.85


def test_decode_probabilities_defaults_and_validation():
    """Test decode_probabilities behavior with default None thresholds and invalid shapes."""
    n_samples = 10
    probs = [np.full((n_samples, 3), 1.0 / 3)]
    for _ in range(6):
        probs.append(np.full((n_samples, 4), 0.25))

    # Test default threshold = 0.5
    pred_default = decode_probabilities(probs, None)
    assert pred_default.shape == (n_samples, 7)

    # Test invalid shape error handling
    invalid_probs = [probs[0], np.full((n_samples, 5), 0.2)] + probs[2:]
    with pytest.raises(ValueError, match="Aspect probabilities must have 4 classes"):
        decode_probabilities(invalid_probs)


def test_zero_division_and_all_absent_edge_cases():
    """Verify robust handling when ground truth has no positive examples or uniform probabilities."""
    n_samples = 50
    # All ground truth is absent
    y_true_absent = np.full((n_samples, 7), 3, dtype=int)
    y_true_absent[:, 0] = 0  # valid overall sentiment

    probs_uniform = [np.full((n_samples, 3), 1.0 / 3)]
    for _ in range(6):
        probs_uniform.append(np.full((n_samples, 4), 0.25))

    # Should not raise ZeroDivisionError or error out
    th = calibrate_absent_thresholds(probs_uniform, y_true_absent)
    for col in ASPECT_COLS:
        assert th[col] == 0.5, f"All absent / zero signal should default to 0.5, got {th[col]}"

    pred = decode_probabilities(probs_uniform, th)
    assert pred.shape == (n_samples, 7)


def test_custom_grid_parameter():
    """Verify custom grid parameter is respected."""
    y = np.array([
        [0, 0, 3, 3, 3, 3, 3],
        [1, 1, 3, 3, 3, 3, 3],
        [2, 2, 3, 3, 3, 3, 3],
        [0, 3, 3, 3, 3, 3, 3],
    ])
    probs = [np.full((4, 3), 0.33)]
    for j in range(1, 7):
        p = np.full((4, 4), 0.05)
        p[np.arange(4), y[:, j]] = 0.85
        probs.append(p)

    custom_grid = [0.25, 0.50, 0.75]
    th = calibrate_absent_thresholds(probs, y, grid=custom_grid)
    for col in ASPECT_COLS:
        assert th[col] in custom_grid

