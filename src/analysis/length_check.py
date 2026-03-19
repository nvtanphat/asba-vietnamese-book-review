from __future__ import annotations

from typing import Any, Mapping, Sequence

from .helpers import numeric_summary, percentage, to_dataframe
from .scan_constants import MIN_TEXT_LENGTH


def scan(records: Sequence[Mapping[str, Any]], text_column: str | None = None) -> dict[str, Any]:
    if not records or not text_column:
        return {
            "text_column": text_column,
            "length_summary": numeric_summary([]),
            "word_summary": numeric_summary([]),
            "length_buckets": {},
            "shorter_than_min_length": 0,
        }

    df = to_dataframe(records)
    series = df[text_column].astype("string").fillna("")
    lengths = series.str.len().astype(int).tolist()
    words = series.str.split().map(len).tolist()

    shorter_than_min_length = sum(length < MIN_TEXT_LENGTH for length in lengths)
    buckets = {
        f"<{MIN_TEXT_LENGTH}": 0,
        f"{MIN_TEXT_LENGTH}-{MIN_TEXT_LENGTH * 2 - 1}": 0,
        f"{MIN_TEXT_LENGTH * 2}-{MIN_TEXT_LENGTH * 5 - 1}": 0,
        f"{MIN_TEXT_LENGTH * 5}-{MIN_TEXT_LENGTH * 10 - 1}": 0,
        f">={MIN_TEXT_LENGTH * 10}": 0,
    }

    for length in lengths:
        if length < MIN_TEXT_LENGTH:
            buckets[f"<{MIN_TEXT_LENGTH}"] += 1
        elif length < MIN_TEXT_LENGTH * 2:
            buckets[f"{MIN_TEXT_LENGTH}-{MIN_TEXT_LENGTH * 2 - 1}"] += 1
        elif length < MIN_TEXT_LENGTH * 5:
            buckets[f"{MIN_TEXT_LENGTH * 2}-{MIN_TEXT_LENGTH * 5 - 1}"] += 1
        elif length < MIN_TEXT_LENGTH * 10:
            buckets[f"{MIN_TEXT_LENGTH * 5}-{MIN_TEXT_LENGTH * 10 - 1}"] += 1
        else:
            buckets[f">={MIN_TEXT_LENGTH * 10}"] += 1

    return {
        "text_column": text_column,
        "length_summary": numeric_summary(lengths),
        "word_summary": numeric_summary(words),
        "length_buckets": buckets,
        "shorter_than_min_length": shorter_than_min_length,
        "shorter_than_min_length_ratio": percentage(shorter_than_min_length, len(records)),
    }
