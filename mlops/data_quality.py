from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ASPECT_COLS = ["as_content", "as_physical", "as_price", "as_packaging", "as_delivery", "as_service"]
REQUIRED = ["review_id", "content", "sentiment", *ASPECT_COLS]


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_json(p)


def validate_dataset(path: str | Path) -> dict[str, Any]:
    df = _read(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    checks: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing,
        "duplicate_review_ids": int(df["review_id"].duplicated().sum()) if "review_id" in df else None,
        "empty_content": int(df["content"].isna().sum() + df["content"].fillna("").astype(str).str.strip().eq("").sum()) if "content" in df else None,
        "invalid_overall_labels": None,
        "invalid_aspect_labels": {},
    }
    if "sentiment" in df:
        overall = pd.to_numeric(df["sentiment"], errors="coerce")
        checks["invalid_overall_labels"] = int((overall.notna() & ~overall.isin([0, 1, 2])).sum())
    for col in ASPECT_COLS:
        if col in df:
            s = pd.to_numeric(df[col], errors="coerce")
            checks["invalid_aspect_labels"][col] = int((s.notna() & ~s.isin([0, 1, 2])).sum())

    failures = []
    if missing:
        failures.append("missing_required_columns")
    if checks["duplicate_review_ids"] not in (None, 0):
        failures.append("duplicate_review_ids")
    if checks["invalid_overall_labels"] not in (None, 0):
        failures.append("invalid_overall_labels")
    if any(v for v in checks["invalid_aspect_labels"].values()):
        failures.append("invalid_aspect_labels")
    checks["status"] = "PASS" if not failures else "FAIL"
    checks["failures"] = failures
    return checks


def dumps(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)
