"""Print test-set predictions that disagree with ground truth for a trained model.

Reads experiments/<model>/test_predictions.npy, which evaluate_model() saves whenever a
model is run with --run-test (python -m ml.train --model <name> --run-test), and compares
it row-by-row against the frozen test split's true labels.
"""
from __future__ import annotations
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which can't print Vietnamese text

from ml.data import TARGET_COLS, load_splits
from ml.data.schema import ASPECT_SENTIMENT_NAMES, SENTIMENT_NAMES


def label_name(task_idx: int, value: int) -> str:
    names = SENTIMENT_NAMES if task_idx == 0 else ASPECT_SENTIMENT_NAMES
    return names.get(int(value), str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name (e.g. phobert, linear_svm)")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--data-root", default="data/splits")
    parser.add_argument("--task", default="all", help="'all', 'sentiment', or an aspect column (e.g. as_price)")
    parser.add_argument("--present-only", action="store_true", help="For aspect tasks, only show mismatches where the aspect is truly present (skip pure absent-vs-present confusion)")
    parser.add_argument("--limit", type=int, default=50, help="Max mismatches printed per task (0 = no limit)")
    parser.add_argument("--text-chars", type=int, default=160, help="Truncate review text to this many characters")
    args = parser.parse_args()

    pred_path = Path(args.experiments_root) / args.model / "test_predictions.npy"
    if not pred_path.exists():
        raise SystemExit(
            f"No saved test predictions at {pred_path}. Run with --run-test "
            f"(python -m ml.train --model {args.model} --run-test, or for a Kaggle run, "
            f"download the output and copy test_predictions.npy into that folder) so "
            f"evaluate_model() has produced it."
        )
    pred = np.load(pred_path)

    _, _, test_df = load_splits(args.data_root)
    true = test_df[TARGET_COLS].to_numpy(dtype=int)
    texts = test_df["text"].tolist()
    if true.shape != pred.shape:
        raise SystemExit(f"Shape mismatch: true={true.shape} vs pred={pred.shape} — predictions may be from a different data split/version.")

    tasks = TARGET_COLS if args.task == "all" else [args.task]
    for task in tasks:
        if task not in TARGET_COLS:
            raise SystemExit(f"Unknown task {task!r}. Choose from {TARGET_COLS} or 'all'.")
        idx = TARGET_COLS.index(task)
        mismatch = true[:, idx] != pred[:, idx]
        if args.present_only and idx > 0:
            mismatch &= true[:, idx] != 3
        rows = np.where(mismatch)[0]

        print(f"\n=== {task}: {len(rows)}/{len(true)} mismatches ({len(rows) / len(true):.1%}) ===")
        shown = rows if args.limit <= 0 else rows[: args.limit]
        for row_i in shown:
            text = texts[row_i]
            if len(text) > args.text_chars:
                text = text[: args.text_chars] + "..."
            print(f"[{row_i}] true={label_name(idx, true[row_i, idx])} pred={label_name(idx, pred[row_i, idx])} | {text}")
        if args.limit > 0 and len(rows) > args.limit:
            print(f"... {len(rows) - args.limit} more mismatches not shown (use --limit 0 for all)")


if __name__ == "__main__":
    main()
