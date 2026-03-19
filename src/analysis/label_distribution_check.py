from __future__ import annotations

from typing import Any, Mapping, Sequence

from .helpers import detect_label_columns, percentage, to_dataframe


def scan(records: Sequence[Mapping[str, Any]], text_column: str | None = None) -> dict[str, Any]:
    if not records:
        return {
            "label_columns": [],
            "columns": {},
        }

    df = to_dataframe(records)
    label_columns = detect_label_columns(records)
    columns: dict[str, Any] = {}

    for column in label_columns:
        series = df[column]
        normalized = series.astype("string").str.strip()
        missing_mask = series.isna() | normalized.isna() | normalized.isin(["", "null", "none", "nan"])
        value_counts = normalized[~missing_mask].value_counts(dropna=False)
        total = int(len(df))
        missing = int(missing_mask.sum())
        non_missing = total - missing
        most_common = value_counts.head(10)
        min_non_zero = int(value_counts[value_counts > 0].min()) if not value_counts.empty else 0
        max_non_zero = int(value_counts.max()) if not value_counts.empty else 0

        columns[column] = {
            "missing_count": missing,
            "missing_ratio": percentage(missing, total),
            "value_counts": [
                {"value": str(value), "count": int(count), "ratio": percentage(int(count), total)}
                for value, count in most_common.items()
            ],
            "non_missing_count": non_missing,
            "imbalance_ratio": round(max_non_zero / min_non_zero, 4) if min_non_zero else None,
        }

    return {
        "label_columns": label_columns,
        "columns": columns,
    }
