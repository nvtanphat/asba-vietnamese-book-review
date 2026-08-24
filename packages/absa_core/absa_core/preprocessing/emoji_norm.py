from __future__ import annotations

import re
from typing import Any

import emoji
import pandas as pd

from .map_utils import load_json_map

SKIN_TONE_SUFFIX_RE = re.compile(r"_(?:light|medium_light|medium|medium_dark|dark)_skin_tone$")
EMOJI_MAP = load_json_map("emoji_map.json")


def _to_text(value: Any) -> str | None:
    """Convert the input to text and preserve None/NaN as None."""
    if value is None or pd.isna(value):
        return None
    return str(value)


def _normalize_alias(alias: str) -> str:
    """Normalize an emoji alias to lowercase snake_case."""
    alias = alias.lower()
    alias = re.sub(r"[^a-z0-9]+", "_", alias)
    return alias.strip("_")


def demojize_text(value: Any, separator: str = " ") -> str | None:
    """Replace emoji characters with mapped Vietnamese aliases."""
    text = _to_text(value)
    if text is None:
        return None

    text = emoji.demojize(text, language="en")

    def replace_token(match: re.Match[str]) -> str:
        key = _normalize_alias(match.group(1))
        base_key = _normalize_alias(SKIN_TONE_SUFFIX_RE.sub("", key))
        return EMOJI_MAP.get(key, EMOJI_MAP.get(base_key, f"emoji_{base_key}"))

    # Only match valid emoji aliases so normal text between ':' characters stays intact.
    text = re.sub(r":([A-Za-z0-9_+\-]+):", replace_token, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_series(series: pd.Series) -> pd.Series:
    """Apply emoji normalization to an entire pandas Series."""
    return series.map(demojize_text)
