from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .helpers import normalize_for_duplicate, percentage, to_dataframe


def scan(records: Sequence[Mapping[str, Any]], text_column: str | None = None) -> dict[str, Any]:
    if not records:
        return {
            "exact_duplicate_rows": 0,
            "normalized_duplicate_texts": 0,
            "exact_duplicate_rows_ratio": 0.0,
            "normalized_duplicate_texts_ratio": 0.0,
            "top_duplicate_texts": [],
        }

    df = to_dataframe(records)
    exact_duplicate_rows = int(df.duplicated().sum())

    text_counter = Counter()
    normalized_text_counter = Counter()
    if text_column and text_column in df.columns:
        text_series = df[text_column].fillna("").astype(str)
        text_counter.update(text_series.tolist())
        normalized_text_counter.update(text_series.map(normalize_for_duplicate).tolist())

    normalized_duplicate_texts = sum(count - 1 for count in normalized_text_counter.values() if count > 1)

    top_duplicate_texts = [
        {"text": text, "count": count}
        for text, count in text_counter.most_common(10)
        if count > 1
    ]

    return {
        "exact_duplicate_rows": exact_duplicate_rows,
        "normalized_duplicate_texts": normalized_duplicate_texts,
        "exact_duplicate_rows_ratio": percentage(exact_duplicate_rows, len(records)),
        "normalized_duplicate_texts_ratio": percentage(normalized_duplicate_texts, len(records)),
        "top_duplicate_texts": top_duplicate_texts,
    }
