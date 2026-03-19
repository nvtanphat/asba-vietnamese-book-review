from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .scan_constants import LABEL_COLUMN_PRIORITY, TEXT_COLUMN_PRIORITY

try:
    import emoji
except Exception:  # pragma: no cover - optional dependency fallback
    emoji = None

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?84|0)\d{8,10}\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MD_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MOJIBAKE_HINT_RE = re.compile(r"(Ã.|Â.|Ä.|á».|ðŸ|â€|â€™)")
PUNCT_REPEAT_RE = re.compile(r"([!?.,])\1{1,}")
ELONGATED_RE = re.compile(r"([A-Za-zÀ-ỹ])\1{2,}", re.UNICODE)
EMOJI_RE = re.compile(
    "["  # broad emoji range, good enough for scanner use
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]",
    re.UNICODE,
)

REPLACEMENT_CHAR = "\uFFFD"


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return stripped == "" or stripped.lower() in {"null", "none", "nan"}
    return False


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", to_text(text))
    return value.strip()


def normalize_for_duplicate(text: Any) -> str:
    value = normalize_text(text).casefold()
    value = re.sub(r"\s+", " ", value)
    return value


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_float(value: Any) -> float | None:
    if is_blank(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
            "p25": None,
            "p75": None,
        }

    ordered = sorted(values)
    count = len(ordered)
    mean = statistics.fmean(ordered)
    median = statistics.median(ordered)
    stdev = statistics.pstdev(ordered) if count > 1 else 0.0
    p25 = ordered[min(count - 1, max(0, round(0.25 * (count - 1))))]
    p75 = ordered[min(count - 1, max(0, round(0.75 * (count - 1))))]
    return {
        "count": count,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "median": median,
        "stdev": stdev,
        "p25": p25,
        "p75": p75,
    }


def percentage(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(part * 100.0 / whole, 2)


def detect_text_column(records: Sequence[Mapping[str, Any]]) -> str | None:
    if not records:
        return None
    columns = list(records[0].keys())
    for preferred in TEXT_COLUMN_PRIORITY:
        if preferred in columns:
            return preferred

    best_column = None
    best_score = -1
    for column in columns:
        score = 0
        for row in records[:100]:
            if not is_blank(row.get(column)):
                score += 1
        if score > best_score:
            best_score = score
            best_column = column
    return best_column


def detect_label_columns(records: Sequence[Mapping[str, Any]]) -> list[str]:
    if not records:
        return []
    columns = list(records[0].keys())
    ordered = [column for column in LABEL_COLUMN_PRIORITY if column in columns]
    if ordered:
        return ordered
    text_column = detect_text_column(records)
    return [column for column in columns if column != text_column]


def count_emojis(text: str) -> list[str]:
    if emoji is not None:
        try:
            return [item["emoji"] for item in emoji.emoji_list(text)]
        except Exception:
            pass
    return EMOJI_RE.findall(text)


def emoji_name(value: str) -> str:
    if emoji is not None:
        try:
            return emoji.demojize(value, language="en").strip(":")
        except Exception:
            return value
    return value


def is_symbol_only(text: str) -> bool:
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return False
    return all(not ch.isalnum() for ch in stripped)


def is_digit_only(text: str) -> bool:
    stripped = re.sub(r"\s+", "", text)
    return stripped.isdigit() if stripped else False


def top_items(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def to_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)
