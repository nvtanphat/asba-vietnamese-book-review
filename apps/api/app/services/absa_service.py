"""Thin service wrapping `absa_core.ABSAPredictor` with lazy, singleton loading.

The predictor (and its ~513 MB PhoBERT weights) is loaded on first call, not at import,
so the API process starts instantly and only pays the cost when ABSA is actually used.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from app.core.config import get_settings

_SENTIMENT_NAMES = ["negative", "neutral", "positive"]
_ASPECT_LABELS = {
    "as_content": "Nội dung",
    "as_physical": "Hình thức",
    "as_price": "Giá cả",
    "as_packaging": "Đóng gói",
    "as_delivery": "Giao hàng",
    "as_service": "Dịch vụ",
}

_predictor = None
_lock = threading.Lock()


def _get_predictor():
    global _predictor
    if _predictor is None:
        with _lock:
            if _predictor is None:
                settings = get_settings()
                from pathlib import Path
                unified_meta = Path(settings.absa_artifact_dir) / "metadata.json"
                if settings.absa_prefer_unified_artifact and unified_meta.exists():
                    from absa_core.models import UnifiedArtifactPredictor
                    _predictor = UnifiedArtifactPredictor(settings.absa_artifact_dir)
                elif settings.absa_use_onnx:
                    from absa_core.models.onnx_predictor import ONNXABSAPredictor

                    _predictor = ONNXABSAPredictor(
                        onnx_model_path=settings.absa_onnx_model_path,
                        tokenizer_source=settings.absa_model_id,
                        num_threads=settings.absa_torch_threads,
                    )
                else:
                    import torch
                    from absa_core.models import ABSAPredictor

                    # Must be set before the model runs its first forward pass; safe to
                    # call repeatedly (idempotent) if some other import already touched torch.
                    torch.set_num_threads(settings.absa_torch_threads)
                    _predictor = ABSAPredictor(
                        model_id=settings.absa_model_id,
                        model_variant=settings.absa_model_variant,
                    )
    return _predictor


def analyze(text: str) -> dict[str, Any]:
    """Return a UI-friendly ABSA result for one review.

    Shape: {text, overall, overall_probs, aspects:[{aspect,label,sentiment,presence}]}
    Only aspects detected as *present* are included.
    """
    predictor = _get_predictor()
    started = time.perf_counter()
    raw = predictor.predict(text)[0]
    latency_ms = (time.perf_counter() - started) * 1000.0

    aspects = []
    for col, sent_idx in raw["aspects"].items():
        if sent_idx < 0:  # aspect not present
            continue
        probs = raw["aspect_probs"][col]
        aspects.append(
            {
                "aspect": col,
                "label": _ASPECT_LABELS.get(col, col),
                "sentiment": _SENTIMENT_NAMES[sent_idx],
                "presence": probs.get("presence", 0.0),
            }
        )

    response = {
        "text": text,
        "overall": _SENTIMENT_NAMES[raw["overall"]],
        "overall_probs": raw["overall_probs"],
        "aspects": aspects,
    }
    settings = get_settings()
    if settings.absa_telemetry_enabled:
        from app.services.model_telemetry import record_prediction
        record_prediction(
            text=text,
            response=response,
            predictor=predictor,
            path=settings.absa_telemetry_path,
            latency_ms=latency_ms,
            sample_rate=max(0.0, min(1.0, settings.absa_telemetry_sample_rate)),
        )
    return response
