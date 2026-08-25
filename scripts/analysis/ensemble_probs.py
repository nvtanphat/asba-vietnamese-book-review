"""Compute per-task softmax probabilities for a trained pretrained_encoder model on the
frozen val/test splits and save them to <experiments-root>/<model>/probs_val.npz /
probs_test.npz.

Needed for probability-level ensembling: experiments/<model>/test_predictions.npy only
stores hard class labels, not the probabilities an ensemble needs to average before the
final argmax/threshold decision.

Usage (from repo root):
    python scripts/analysis/ensemble_probs.py --model phobert
    python scripts/analysis/ensemble_probs.py --model xlmr
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
import torch

from ml.data import load_splits

TASK_DIMS = [3, 4, 4, 4, 4, 4, 4]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. phobert, xlmr")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--model-dir", default=None, help="Override: directory with model.pt/encoder/tokenizer/metadata.json directly inside (default: <experiments-root>/<model>)")
    ap.add_argument("--data-root", default="data/splits")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else Path(args.experiments_root) / args.model
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = meta.get("config", {})

    from transformers import AutoTokenizer

    from absa_core.models.unified_architectures import EncoderMultiTaskNetwork
    from absa_core.preprocessing.pipeline import clean_text_series
    import pandas as pd

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    net = EncoderMultiTaskNetwork(
        str(model_dir / "encoder"),
        float(cfg.get("dropout", 0.15)),
        pooling_type=str(cfg.get("pooling_type", "masked_mean")),
        head_type=str(cfg.get("head_type", "hierarchical")),
    )
    net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    net.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    max_length = int(cfg.get("max_length", 160))
    segmenter = cfg.get("word_segmenter", "none")

    _, val_df, test_df = load_splits(args.data_root)

    def compute(df) -> list[np.ndarray]:
        cleaned = [str(t) for t in clean_text_series(pd.Series(df["text"].tolist()), lowercase=True).tolist()]
        if segmenter == "pyvi":
            from pyvi import ViTokenizer

            cleaned = [ViTokenizer.tokenize(t) for t in cleaned]

        n = len(cleaned)
        probs_per_task = [np.zeros((n, d), dtype=np.float32) for d in TASK_DIMS]
        with torch.no_grad():
            for start in range(0, n, args.batch_size):
                batch = cleaned[start : start + args.batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                logits = net(enc["input_ids"], enc["attention_mask"])
                for t, task_logits in enumerate(logits):
                    probs_per_task[t][start : start + len(batch)] = _softmax(task_logits.numpy())
                if start % (args.batch_size * 20) == 0:
                    print(f"  {start}/{n}", flush=True)
        return probs_per_task

    print(f"Computing val probabilities for {args.model} ({len(val_df)} rows)...")
    val_probs = compute(val_df)
    np.savez(model_dir / "probs_val.npz", *val_probs)

    print(f"Computing test probabilities for {args.model} ({len(test_df)} rows)...")
    test_probs = compute(test_df)
    np.savez(model_dir / "probs_test.npz", *test_probs)

    print(f"Wrote {model_dir / 'probs_val.npz'} and {model_dir / 'probs_test.npz'}")


if __name__ == "__main__":
    main()
