from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .map_utils import load_json_map


ELONGATED_RE = re.compile(r"(.)\1{2,}")
TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)
VOCAB_MAP = load_json_map("vocab_map.json")
VIETNAMESE_VOWELS = set("aeiouyăâêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")


def _to_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _collapse_elongation(token: str) -> str:
    def replace(match: re.Match[str]) -> str:
        char = match.group(1)
        if match.end() == len(token) and char.lower() not in VIETNAMESE_VOWELS:
            return char
        return char * 2

    return ELONGATED_RE.sub(replace, token)


def normalize_vocab(value: Any) -> str | None:
    text = _to_text(value)
    if text is None:
        return None

    parts: list[str] = []
    for chunk in TOKEN_RE.findall(text):
        if chunk.isspace():
            parts.append(" ")
            continue
        if chunk.isalnum() or "_" in chunk:
            lower = chunk.lower()
            if lower in VOCAB_MAP:
                parts.append(VOCAB_MAP[lower])
            else:
                parts.append(_collapse_elongation(chunk))
            continue
        parts.append(chunk)

    return re.sub(r"\s+", " ", "".join(parts)).strip()


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_vocab)
