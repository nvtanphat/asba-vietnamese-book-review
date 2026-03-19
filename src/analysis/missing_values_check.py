from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from .helpers import percentage, to_dataframe


def scan(records: Sequence[Mapping[str, Any]], text_column: str | None = None) -> dict[str, Any]:
    df = to_dataframe(records)
    rows = int(df.shape[0])
    columns = list(df.columns)

    per_column: dict[str, dict[str, Any]] = {}
    missing_rows = 0

    blank_like = {"", "null", "none", "nan"}
    normalized = df.astype("string")
    missing_mask = normalized.isna() | normalized.apply(
        lambda series: series.str.strip().str.lower().isin(blank_like)
    )
    missing_series = missing_mask.sum(axis=0)

    for column in columns:
        missing = int(missing_series[column])
        per_column[column] = {
            "missing_count": missing,
            "missing_ratio": percentage(missing, rows),
        }

    missing_rows = int(missing_mask.any(axis=1).sum())

    sorted_columns = sorted(
        per_column.items(),
        key=lambda item: (item[1]["missing_count"], item[0]),
        reverse=True,
    )

    return {
        "rows_with_at_least_one_missing": missing_rows,
        "rows_with_at_least_one_missing_ratio": percentage(missing_rows, rows),
        "per_column": per_column,
        "top_missing_columns": [
            {"column": column, **stats} for column, stats in sorted_columns[:10]
        ],
    }
