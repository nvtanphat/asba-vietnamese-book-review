from __future__ import annotations

import re
from typing import Any

import emoji
import pandas as pd

from .map_utils import load_json_map


SKIN_TONE_SUFFIX_RE = re.compile(r"_(?:light|medium_light|medium|medium_dark|dark)_skin_tone$")
EMOJI_MAP = load_json_map("emoji_map.json")


def _to_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _normalize_alias(alias: str) -> str:
    alias = alias.lower()
    alias = re.sub(r"[^a-z0-9]+", "_", alias)
    return alias.strip("_")


def demojize_text(value: Any, separator: str = " ") -> str | None:
    text = _to_text(value)
    if text is None:
        return None

    text = emoji.demojize(text, language="en")

    def replace_token(match: re.Match[str]) -> str:
        key = _normalize_alias(match.group(1))
        base_key = _normalize_alias(SKIN_TONE_SUFFIX_RE.sub("", key))
        return EMOJI_MAP.get(key, EMOJI_MAP.get(base_key, f"emoji_{base_key}"))

    text = re.sub(r":([^:]+):", replace_token, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(demojize_text)
