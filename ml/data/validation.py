from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

from .schema import ASPECT_COLS


def normalized_signature(text: str) -> str:
    return " ".join(str(text).lower().split())


def validate_no_text_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    sets = [set(df["text"].map(normalized_signature)) for df in (train, val, test)]
    return {
        "train_val": len(sets[0] & sets[1]),
        "train_test": len(sets[0] & sets[2]),
        "val_test": len(sets[1] & sets[2]),
    }


def frame_fingerprint(frame: pd.DataFrame) -> str:
    cols = [c for c in ["review_id", "text", "sentiment", *ASPECT_COLS] if c in frame.columns]
    payload = frame[cols].sort_values(cols[0]).to_json(orient="records", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(path: str | Path, *, seed: int, strategy: str, splits: dict[str, pd.DataFrame]) -> dict:
    manifest = {
        "seed": seed,
        "strategy": strategy,
        "splits": {
            name: {
                "rows": len(df),
                "fingerprint": frame_fingerprint(df),
                "sentiment_distribution": df["sentiment"].value_counts(normalize=True).sort_index().round(6).to_dict(),
                "aspect_presence": {c: round(float((df[c] != 3).mean()), 6) for c in ASPECT_COLS},
            }
            for name, df in splits.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
