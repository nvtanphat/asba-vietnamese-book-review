from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


SHORT_TEXT_MIN_CHARS = 10
BLANK_MARKERS = {"", "null", "none", "nan"}
NAME_ERROR_MARKERS = {"#name?"}
WHITESPACE_RE = re.compile(r"\s+")


def _to_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def normalize_for_duplicate(text: Any) -> str:
    value = unicodedata.normalize("NFKC", _to_text(text)).casefold()
    return WHITESPACE_RE.sub(" ", value).strip()


def is_symbol_only(text: Any) -> bool:
    value = normalize_for_duplicate(text)
    if not value:
        return False
    return all(not char.isalnum() for char in value)


def is_digit_only(text: Any) -> bool:
    value = normalize_for_duplicate(text)
    return value.isdigit() if value else False


def is_meaningful_text(text: Any, min_chars: int = SHORT_TEXT_MIN_CHARS) -> bool:
    value = normalize_for_duplicate(text)
    if not value:
        return False
    if value in BLANK_MARKERS or value in NAME_ERROR_MARKERS:
        return False
    if is_digit_only(value) or is_symbol_only(value):
        return False
    return len(value) >= min_chars


def drop_noise_rows(
    frame: pd.DataFrame,
    text_column: str = "content",
    min_chars: int = SHORT_TEXT_MIN_CHARS,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    target = frame.copy()
    text_series = target[text_column]
    keep_mask = text_series.map(lambda value: is_meaningful_text(value, min_chars=min_chars))
    target = target.loc[keep_mask].copy()

    if drop_duplicates and not target.empty:
        normalized = target[text_column].map(normalize_for_duplicate)
        target = target.loc[~normalized.duplicated(keep="first")].copy()

    target.reset_index(drop=True, inplace=True)
    return target
