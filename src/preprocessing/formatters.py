from __future__ import annotations

import re
from typing import Any

import pandas as pd


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def _to_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def normalize_format(value: Any) -> str | None:
    text = _to_text(value)
    if text is None:
        return None
    text = PUNCT_RE.sub(r"\1\1", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_format)
