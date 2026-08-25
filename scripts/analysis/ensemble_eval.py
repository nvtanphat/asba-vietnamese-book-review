"""Evaluate a probability-level ensemble of two trained models on the frozen val/test
splits, using the exact same calibration and metric functions ml/train.py uses for every
individual model — so the ensemble's f1_combined is directly comparable to the numbers
already in the MLOps registry.

Averages per-task softmax probabilities (saved by scripts/analysis/ensemble_probs.py),
re-calibrates per-aspect presence thresholds on validation (never touching test for
calibration — same test-seal policy as every other model in this benchmark), then reports
test metrics.

Usage (from repo root):
    python scripts/analysis/ensemble_eval.py --model-dirs artifacts/final/model experiments/xlmr --weights 1 1
"""

from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np

from ml.data import TARGET_COLS, load_splits
from ml.evaluation.calibration import calibrate_absent_thresholds, decode_probabilities
from ml.evaluation.metrics import evaluate_predictions

N_TASKS = 7


def _load_probs(model_dir: Path, split: str) -> list[np.ndarray]:
    path = model_dir / f"probs_{split}.npz"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/analysis/ensemble_probs.py --model-dir {model_dir} first.")
    npz = np.load(path)
    return [npz[f"arr_{i}"] for i in range(N_TASKS)]


def _weighted_average(all_probs: list[list[np.ndarray]], weights: list[float]) -> list[np.ndarray]:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return [
        sum(w[m] * all_probs[m][t] for m in range(len(all_probs)))
        for t in range(N_TASKS)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True, help="Directories each holding probs_val.npz/probs_test.npz")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="Default: equal weight for every model")
    ap.add_argument("--data-root", default="data/splits")
    args = ap.parse_args()

    model_dirs = [Path(d) for d in args.model_dirs]
    weights = args.weights or [1.0] * len(model_dirs)
    if len(weights) != len(model_dirs):
        raise SystemExit("--weights must have one value per --model-dirs entry")

    _, val_df, test_df = load_splits(args.data_root)
    val_y = val_df[TARGET_COLS].to_numpy(dtype=int)
    test_y = test_df[TARGET_COLS].to_numpy(dtype=int)

    val_probs_per_model = [_load_probs(d, "val") for d in model_dirs]
    test_probs_per_model = [_load_probs(d, "test") for d in model_dirs]

    ens_val_probs = _weighted_average(val_probs_per_model, weights)
    ens_test_probs = _weighted_average(test_probs_per_model, weights)

    print(f"Ensembling {len(model_dirs)} models: {[str(d) for d in model_dirs]} (weights={weights})")
    print("Calibrating per-aspect thresholds on validation (test stays sealed)...")
    thresholds = calibrate_absent_thresholds(ens_val_probs, val_y)
    print("Ensemble thresholds:", json.dumps(thresholds, indent=2))

    val_pred = decode_probabilities(ens_val_probs, thresholds)
    test_pred = decode_probabilities(ens_test_probs, thresholds)

    val_metrics = evaluate_predictions(val_y, val_pred)
    test_metrics = evaluate_predictions(test_y, test_pred)

    print("\n=== Ensemble val metrics ===")
    print(json.dumps({k: val_metrics[k] for k in ("f1_combined", "f1_sentiment", "f1_aspect_4class_mean")}, indent=2))
    print("\n=== Ensemble test metrics ===")
    print(json.dumps(test_metrics, indent=2))

    out = {"model_dirs": [str(d) for d in model_dirs], "weights": weights, "thresholds": thresholds, "val": val_metrics, "test": test_metrics}
    out_path = Path("experiments/ensemble_phobert_xlmr_metrics.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
