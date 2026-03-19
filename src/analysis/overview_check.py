from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from .helpers import safe_float, to_dataframe


def scan(records: Sequence[Mapping[str, Any]], text_column: str | None = None) -> dict[str, Any]:
    df = to_dataframe(records)
    rows = int(df.shape[0])
    columns = list(df.columns)

    column_type_guess: dict[str, str] = {}
    non_empty_counts: dict[str, int] = {}
    numeric_like_counts: dict[str, int] = {}

    for column in columns:
        series = df[column]
        string_series = series.astype("string")
        blank_mask = string_series.str.strip().isin(["", "null", "none", "nan"])
        non_empty = int((~series.isna() & ~blank_mask).sum())
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_like = int(numeric_series.notna().sum())
        non_empty_counts[column] = non_empty
        numeric_like_counts[column] = numeric_like
        if rows and numeric_like / rows >= 0.8:
            column_type_guess[column] = "numeric_like"
        elif rows and non_empty / rows >= 0.8:
            column_type_guess[column] = "text_like"
        else:
            column_type_guess[column] = "mixed"

    return {
        "row_count": rows,
        "column_count": len(columns),
        "columns": columns,
        "text_column": text_column,
        "non_empty_counts": non_empty_counts,
        "numeric_like_counts": numeric_like_counts,
        "column_type_guess": column_type_guess,
    }
