"""ONNX Runtime-backed probability ensemble of two or more pretrained_encoder members
(currently phobert + xlmr), produced by scripts/analysis/ensemble_probs.py +
scripts/analysis/ensemble_eval.py (weight/threshold calibration on validation) and
packages/absa_core/scripts/export_onnx_unified.py (per-member ONNX export).

Averages each member's per-task softmax probabilities (not raw logits, and not hard
votes — matching what was actually measured to beat every solo model on the sealed test
split), then decodes with the ensemble-calibrated thresholds. Mirrors
UnifiedArtifactPredictor.predict()'s output contract exactly so absa_service needs no
changes to call this instead.

Trade-off (accept knowingly): every request runs every member sequentially, so latency is
roughly the sum of each member's latency, not the max — chosen deliberately over a single
faster model because the accuracy gain from ensembling was measured to be real (test
f1_combined 0.791 vs 0.774/0.782 solo), not run in parallel threads to keep the API's
one-thread-per-worker CPU budget predictable under concurrent requests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

from absa_core.preprocessing.pipeline import clean_text_series

from .architectures import ASPECT_COLS

TASK_DIMS = [3, 4, 4, 4, 4, 4, 4]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _split(flat: np.ndarray) -> list[np.ndarray]:
    out, offset = [], 0
    for d in TASK_DIMS:
        out.append(flat[:, offset : offset + d])
        offset += d
    return out


class _Member:
    """One ensemble member: its own ONNX session, tokenizer, and preprocessing config —
    phobert needs pyvi word segmentation before tokenizing, xlmr's SentencePiece tokenizer
    does not, so this can't be shared across members."""

    def __init__(self, member_dir: Path, num_threads: int = 1):
        meta_path = member_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Ensemble member metadata not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = meta.get("config", {})
        self.max_length = int(cfg.get("max_length", 160))
        self.word_segmenter = cfg.get("word_segmenter", "none")

        onnx_path = member_dir / "model.int8.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. Run "
                "packages/absa_core/scripts/export_onnx_unified.py for this member first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(member_dir / "tokenizer")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def probs(self, texts: list[str]) -> list[np.ndarray]:
        cleaned = [str(t) for t in clean_text_series(pd.Series(texts), lowercase=True).tolist()]
        if self.word_segmenter == "pyvi":
            from pyvi import ViTokenizer

            cleaned = [ViTokenizer.tokenize(t) for t in cleaned]
        enc = self.tokenizer(cleaned, return_tensors="np", padding=True, truncation=True, max_length=self.max_length)
        (flat,) = self.session.run(
            ["logits"],
            {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)},
        )
        return [_softmax(part) for part in _split(flat)]


class EnsembleOnnxPredictor:
    """Drop-in replacement for UnifiedArtifactPredictor backed by an ONNX ensemble."""

    def __init__(self, artifact_dir: str | Path, num_threads: int = 1):
        self.root = Path(artifact_dir)
        meta = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        member_names = meta["members"]
        self.members = [_Member(self.root / name, num_threads=num_threads) for name in member_names]
        weights = np.asarray(meta.get("weights", [1.0] * len(self.members)), dtype=float)
        self.weights = weights / weights.sum()
        th_path = self.root / "thresholds.json"
        self.thresholds = json.loads(th_path.read_text(encoding="utf-8")) if th_path.exists() else {}

    def _ensemble_probs(self, texts: list[str]) -> list[np.ndarray]:
        per_member = [m.probs(texts) for m in self.members]
        return [
            sum(self.weights[i] * per_member[i][t] for i in range(len(self.members)))
            for t in range(len(TASK_DIMS))
        ]

    def predict(self, texts: list[str] | str) -> list[dict[str, Any]]:
        if isinstance(texts, str):
            texts = [texts]
        probs = self._ensemble_probs(texts)
        results = []
        for r in range(len(texts)):
            overall_probs = probs[0][r].tolist()
            aspects: dict[str, int] = {}
            aspect_probs: dict[str, dict[str, Any]] = {}
            for i, col in enumerate(ASPECT_COLS, 1):
                p = probs[i][r]
                presence = float(1.0 - p[3])
                present = presence >= float(self.thresholds.get(col, 0.5))
                sent = int(np.argmax(p[:3])) if present else -1
                aspects[col] = sent
                aspect_probs[col] = {"presence": round(presence, 3), "sentiment": p[:3].tolist() if present else [0.0, 0.0, 0.0]}
            results.append(
                {
                    "overall": int(np.argmax(probs[0][r])),
                    "overall_probs": overall_probs,
                    "aspects": aspects,
                    "aspect_probs": aspect_probs,
                }
            )
        return results
