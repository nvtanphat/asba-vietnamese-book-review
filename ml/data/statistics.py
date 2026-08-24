from __future__ import annotations
import pandas as pd
from .schema import ASPECT_COLS


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "sentiment": frame["sentiment"].value_counts(dropna=False).sort_index().to_dict(),
        "aspect_presence": {c: int((frame[c] != 3).sum()) for c in ASPECT_COLS},
        "aspect_presence_rate": {c: float((frame[c] != 3).mean()) for c in ASPECT_COLS},
    }
