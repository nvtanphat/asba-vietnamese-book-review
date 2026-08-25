from __future__ import annotations

import hashlib
import json
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def _features(text: str) -> dict[str, float]:
    text = str(text or "")
    n = max(len(text), 1)
    return {
        "text_length": float(len(text)),
        "word_count": float(len(text.split())),
        "ascii_ratio": float(sum(ord(ch) < 128 for ch in text) / n),
        "digit_ratio": float(sum(ch.isdigit() for ch in text) / n),
        "punct_ratio": float(sum((not ch.isalnum()) and (not ch.isspace()) for ch in text) / n),
    }


def record_prediction(
    *,
    text: str,
    response: dict[str, Any],
    predictor: Any,
    path: str | Path,
    latency_ms: float,
    sample_rate: float = 1.0,
) -> None:
    """Append privacy-preserving inference telemetry.

    Raw review text is never written. We store a short one-way hash and coarse text
    features sufficient for drift checks, plus predicted labels/model identity.
    """
    if sample_rate <= 0 or (sample_rate < 1 and random.random() > sample_rate):
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_hash": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16],
        **_features(text),
        "overall": response.get("overall"),
        "aspect_count": len(response.get("aspects", [])),
        "aspects": [a.get("aspect") for a in response.get("aspects", [])],
        "model": getattr(predictor, "model_name", getattr(predictor, "model_variant", "unknown")),
        "family": getattr(predictor, "family", None),
        "latency_ms": round(float(latency_ms), 3),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with _lock, p.open("a", encoding="utf-8") as f:
        f.write(line)
