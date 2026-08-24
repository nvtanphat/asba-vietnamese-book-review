"""ONNX Runtime-backed predictor for the 'phobert' variant.

A CPU-optimized drop-in alternative to ABSAPredictor, produced by
scripts/export_onnx.py (fp32 export + optional INT8 dynamic quantization). Mirrors
ABSAPredictor's public `.predict(texts)` contract exactly so absa_service can switch
between the two via a config flag without touching any call site.

Only the tokenizer still comes from `transformers` — ONNX only captures the neural
net graph + weights, not the (small, separately-cached) tokenizer files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

from absa_core.preprocessing.pipeline import clean_text_series

from .architectures import ASPECT_COLS
from .predictor import _FALLBACK_PRESENCE_THRESHOLD, _load_calibrated_thresholds

_SENT_DIM = 3
_MAX_LENGTH = 256  # matches MODEL_VARIANTS["phobert"]["max_length"] in predictor.py


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


_ABSA_PROMPT_PREFIX = "ABSA review aspect content physical price packaging delivery service: "


class ONNXABSAPredictor:
    """ONNX Runtime equivalent of `ABSAPredictor(model_variant="phobert")`."""

    def __init__(
        self,
        onnx_model_path: str,
        tokenizer_source: str,
        tokenizer_subfolder: str | None = "phobert",
        local_files_only: bool = False,
        num_threads: int = 1,
    ):
        if not Path(onnx_model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at '{onnx_model_path}'. Run "
                "packages/absa_core/scripts/export_onnx.py first (requires the "
                "absa-core[onnx] extra)."
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            onnx_model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        tokenizer_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if tokenizer_subfolder:
            tokenizer_kwargs["subfolder"] = tokenizer_subfolder
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        self.phobert_thresholds = _load_calibrated_thresholds()

    def _preprocess(self, texts: list[str]) -> list[str]:
        cleaned = clean_text_series(pd.Series(texts), lowercase=True).tolist()
        return [str(t) for t in cleaned]

    def predict(self, texts: list[str] | str) -> list[dict[str, Any]]:
        if isinstance(texts, str):
            texts = [texts]

        cleaned = self._preprocess(texts)
        cleaned = [f"{_ABSA_PROMPT_PREFIX}{t}" for t in cleaned]
        enc = self.tokenizer(
            cleaned, return_tensors="np", padding=True, truncation=True, max_length=_MAX_LENGTH
        )
        (logits,) = self.session.run(
            ["logits"],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )
        return self._decode(logits)

    def _decode(self, logits: np.ndarray) -> list[dict[str, Any]]:
        n = len(ASPECT_COLS)
        sent = logits[:, :_SENT_DIM]
        pres = logits[:, _SENT_DIM : _SENT_DIM + n * 2].reshape(-1, n, 2)
        asp_sent = logits[:, _SENT_DIM + n * 2 :].reshape(-1, n, 3)

        overall_probs = _softmax(sent)
        pres_probs = _softmax(pres)
        asp_probs = _softmax(asp_sent)

        results = []
        for i in range(logits.shape[0]):
            aspects: dict[str, int] = {}
            aspect_probs: dict[str, dict[str, Any]] = {}
            for j, col in enumerate(ASPECT_COLS):
                presence_conf = float(pres_probs[i, j, 1])
                threshold = self.phobert_thresholds.get(col, _FALLBACK_PRESENCE_THRESHOLD)
                is_present = presence_conf > threshold
                aspects[col] = int(np.argmax(asp_probs[i, j])) if is_present else -1
                aspect_probs[col] = {
                    "presence": round(presence_conf, 3),
                    "sentiment": asp_probs[i, j].tolist(),
                }
            results.append(
                {
                    "overall": int(np.argmax(overall_probs[i])),
                    "overall_probs": overall_probs[i].tolist(),
                    "aspects": aspects,
                    "aspect_probs": aspect_probs,
                }
            )
        return results
