from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import atomic_write_json, utc_now

FEATURES = ["text_length", "word_count", "ascii_ratio", "digit_ratio", "punct_ratio"]


def text_features(text: str) -> dict[str, float]:
    text = str(text or "")
    n = max(len(text), 1)
    return {
        "text_length": float(len(text)),
        "word_count": float(len(text.split())),
        "ascii_ratio": float(sum(ord(ch) < 128 for ch in text) / n),
        "digit_ratio": float(sum(ch.isdigit() for ch in text) / n),
        "punct_ratio": float(sum((not ch.isalnum()) and (not ch.isspace()) for ch in text) / n),
    }


def _load_texts(path: str | Path) -> list[str]:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        # API telemetry intentionally does not store raw review text; features are present directly.
        if rows and "text_length" in rows[0]:
            return []
        return [str(r.get("text") or r.get("content") or "") for r in rows]
    df = pd.read_json(p)
    col = "text" if "text" in df.columns else "content"
    return df[col].fillna("").astype(str).tolist()


def _load_feature_rows(path: str | Path) -> list[dict[str, float]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows and all(k in rows[0] for k in FEATURES):
            return [{k: float(r.get(k, 0.0)) for k in FEATURES} for r in rows]
    return [text_features(t) for t in _load_texts(path)]


def _bins(values: np.ndarray, n_bins: int = 10) -> list[float]:
    if len(values) == 0:
        return [-math.inf, math.inf]
    q = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    q = np.unique(q).astype(float)
    if len(q) < 2:
        center = float(values[0])
        # Keep two bins even for a constant reference distribution so a shifted
        # production distribution cannot collapse into the same [-inf, +inf] bucket.
        q = np.array([-math.inf, center + 1e-9, math.inf], dtype=float)
        return q.tolist()
    q[0] = -math.inf
    q[-1] = math.inf
    return q.tolist()


def _hist(values: np.ndarray, bins: list[float]) -> list[float]:
    counts, _ = np.histogram(values, bins=np.asarray(bins, dtype=float))
    probs = counts.astype(float) + 1e-8
    probs /= probs.sum()
    return probs.tolist()


def build_reference_profile(path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    rows = _load_feature_rows(path)
    profile: dict[str, Any] = {"created_at": utc_now(), "source": str(path), "count": len(rows), "features": {}}
    for feature in FEATURES:
        values = np.asarray([r[feature] for r in rows], dtype=float)
        bins = _bins(values)
        profile["features"][feature] = {
            "bins": bins,
            "probs": _hist(values, bins),
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std()) if len(values) else None,
        }
    if output:
        atomic_write_json(output, profile)
    return profile


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(np.clip(p / m, 1e-12, None)))
    kl_qm = np.sum(q * np.log(np.clip(q / m, 1e-12, None)))
    return float(0.5 * (kl_pm + kl_qm))


def check_drift(reference_profile: str | Path | dict[str, Any], current_path: str | Path, *, warning_js: float = 0.10, critical_js: float = 0.20) -> dict[str, Any]:
    ref = reference_profile if isinstance(reference_profile, dict) else json.loads(Path(reference_profile).read_text(encoding="utf-8"))
    rows = _load_feature_rows(current_path)
    report: dict[str, Any] = {"created_at": utc_now(), "reference_count": ref.get("count", 0), "current_count": len(rows), "features": {}}
    worst = "OK"
    for feature in FEATURES:
        bins = ref["features"][feature]["bins"]
        current = np.asarray([r[feature] for r in rows], dtype=float)
        p = np.asarray(ref["features"][feature]["probs"], dtype=float)
        q = np.asarray(_hist(current, bins), dtype=float)
        js = _js(p, q)
        status = "CRITICAL" if js >= critical_js else "WARNING" if js >= warning_js else "OK"
        if status == "CRITICAL": worst = "CRITICAL"
        elif status == "WARNING" and worst == "OK": worst = "WARNING"
        report["features"][feature] = {"js_divergence": js, "status": status}
    report["status"] = worst
    return report
