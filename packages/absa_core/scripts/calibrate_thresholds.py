"""Calibrate per-aspect presence thresholds for the deployed 'phobert' checkpoint.

The model's presence heads output a raw probability per aspect; there is no single
threshold that works well for all six aspects (some aspects are rare/subtle, others
common/obvious). `ABSAPredictor` used to fall back to a single hardcoded 0.65 for
every aspect. This script calibrates a per-aspect threshold against the *actual*
deployed checkpoint and writes the result to `absa_core/models/thresholds.json`,
which `ABSAPredictor` loads at runtime.

The objective is F-beta on the *binary presence decision itself* (present vs.
absent), not sentiment-conditional-on-presence — an earlier version of this script
copied ml/core/evaluate.py's objective, which only scores sentiment F1 on rows
where the aspect is truly present and never penalizes false positives on absent
rows. That pushed every threshold toward ~0.05 and traded missed complaints for
every review being tagged with 3-4 spurious aspects. beta>1 still favours recall
(a missed complaint is worse than an extra chip a human can dismiss) without
collapsing precision entirely.

Usage (from repo root):
    uv run python packages/absa_core/scripts/calibrate_thresholds.py
    uv run python packages/absa_core/scripts/calibrate_thresholds.py --val data/processed/val_clean.json --beta 1.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import fbeta_score, precision_score, recall_score

from absa_core.models.architectures import ASPECT_COLS, PRES_DIM, SENT_DIM
from absa_core.models.predictor import ABSAPredictor
from absa_core.preprocessing.pipeline import clean_text_series

ABSENT_ASPECT_CLASS = 3
BATCH_SIZE = 32
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "absa_core" / "models" / "thresholds.json"


def load_val_frame(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["content", "sentiment"]).copy()
    for col in ASPECT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(ABSENT_ASPECT_CLASS).astype(int)
    return df.reset_index(drop=True)


@torch.no_grad()
def collect_logits(predictor: ABSAPredictor, texts: list[str]) -> np.ndarray:
    """Run the real deployed model over `texts` using the exact production preprocessing."""
    cleaned = clean_text_series(pd.Series(texts), lowercase=True).tolist()
    cleaned = [str(t) for t in cleaned]

    all_logits = []
    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i : i + BATCH_SIZE]
        inputs = predictor.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=predictor.max_length
        ).to(predictor.device)
        logits = predictor.model(**inputs).logits
        all_logits.append(logits.cpu().numpy())
        print(f"  {min(i + BATCH_SIZE, len(cleaned))}/{len(cleaned)}", end="\r")
    print()
    return np.concatenate(all_logits, axis=0)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def calibrate(logits: np.ndarray, labels: np.ndarray, beta: float = 1.5) -> tuple[dict[str, float], pd.DataFrame]:
    """Per-aspect threshold that maximizes F-beta of the *binary presence decision*.

    beta > 1 weights recall over precision: a missed complaint (false negative) is
    worse than an extra aspect chip a human reviewer can dismiss (false positive).
    """
    s, p = SENT_DIM, len(ASPECT_COLS) * PRES_DIM
    pres_probs = softmax_np(logits[:, s : s + p].reshape(-1, len(ASPECT_COLS), PRES_DIM))[:, :, 1]

    thresholds, rows = {}, []
    for i, col in enumerate(ASPECT_COLS):
        y_present = (labels[:, i] != ABSENT_ASPECT_CLASS).astype(int)
        scores = pres_probs[:, i]
        best_t, best_fbeta, best_precision, best_recall = 0.5, -1.0, 0.0, 0.0
        for t in np.linspace(0.05, 0.95, 19):
            y_pred = (scores >= t).astype(int)
            fbeta = fbeta_score(y_present, y_pred, beta=beta, zero_division=0)
            if fbeta > best_fbeta:
                precision = precision_score(y_present, y_pred, zero_division=0)
                recall = recall_score(y_present, y_pred, zero_division=0)
                best_t, best_fbeta, best_precision, best_recall = float(t), float(fbeta), precision, recall
        thresholds[col] = round(best_t, 3)
        rows.append(
            {
                "aspect": col,
                "threshold": best_t,
                "fbeta": best_fbeta,
                "precision": best_precision,
                "recall": best_recall,
                "n_present": int(y_present.sum()),
            }
        )

    return thresholds, pd.DataFrame(rows).sort_values("fbeta", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", default="data/processed/val_clean.json")
    parser.add_argument("--variant", default="phobert")
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument("--beta", type=float, default=1.5, help="F-beta; >1 favors recall over precision")
    args = parser.parse_args()

    print(f"Loading validation set: {args.val}")
    df = load_val_frame(Path(args.val))
    print(f"  {len(df)} rows")

    print(f"Loading predictor (variant={args.variant}) — this loads the real deployed checkpoint...")
    predictor = ABSAPredictor(model_variant=args.variant)
    print(f"  model_source={predictor.model_source}")

    print("Running inference over validation set...")
    logits = collect_logits(predictor, df["content"].tolist())
    labels = df[ASPECT_COLS].to_numpy()

    print(f"Calibrating per-aspect thresholds (beta={args.beta})...")
    thresholds, report = calibrate(logits, labels, beta=args.beta)
    print(report.round(4).to_string(index=False))

    payload = {
        "model_id": predictor.model_id,
        "model_source": predictor.model_source,
        "variant": args.variant,
        "n_val_rows": len(df),
        "beta": args.beta,
        "thresholds": thresholds,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved calibrated thresholds -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
