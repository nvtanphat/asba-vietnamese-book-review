from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import emoji_norm, formatters, noise_cleaner, quality_filter, unicode_norm, vocab_norm

CSV_ENCODING = "utf-8-sig"
JSON_ENCODING = "utf-8"


def _normalize_without_lowercase(series: pd.Series) -> pd.Series:
    """Run the text-cleaning pipeline without case folding."""
    cleaned = unicode_norm.normalize_series(series)
    cleaned = noise_cleaner.normalize_series(cleaned)
    cleaned = emoji_norm.normalize_series(cleaned)
    cleaned = vocab_norm.normalize_series(cleaned)
    cleaned = formatters.normalize_series(cleaned)
    return cleaned


def lowercase_series(series: pd.Series) -> pd.Series:
    """Lowercase string values while preserving missing values."""
    return series.map(lambda value: value.lower() if isinstance(value, str) else value)


def clean_text_series(series: pd.Series, lowercase: bool = True) -> pd.Series:
    """Clean a text series and optionally convert the final output to lowercase."""
    cleaned = _normalize_without_lowercase(series)
    if lowercase:
        cleaned = lowercase_series(cleaned)
    return cleaned


def preprocess_dataframe(
    frame: pd.DataFrame,
    text_column: str = "content",
    output_column: str | None = None,
    keep_raw: bool = True,
    min_chars: int = quality_filter.SHORT_TEXT_MIN_CHARS,
    drop_duplicates: bool = True,
    keep_columns: list[str] | None = None,
    lowercase: bool = True,
) -> pd.DataFrame:
    """Preprocess a dataframe and keep only the requested columns."""
    target = frame.copy()
    source_column = text_column
    output_column = output_column or text_column

    if keep_raw:
        raw_column = f"{source_column}_raw"
        if raw_column not in target.columns:
            target[raw_column] = target[source_column]

    if lowercase:
        target[output_column] = clean_text_series(target[source_column])
    else:
        target[output_column] = _normalize_without_lowercase(target[source_column])

    if output_column != source_column:
        target.drop(columns=[source_column], inplace=True, errors="ignore")

    target = quality_filter.drop_noise_rows(
        target,
        text_column=output_column,
        min_chars=min_chars,
        drop_duplicates=drop_duplicates,
    )

    if keep_columns is not None:
        ordered_columns = []
        for column in keep_columns:
            if column in target.columns and column not in ordered_columns:
                ordered_columns.append(column)
        target = target.loc[:, ordered_columns]

    return target


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str = "content",
    keep_raw: bool = True,
    min_chars: int = quality_filter.SHORT_TEXT_MIN_CHARS,
    drop_duplicates: bool = True,
    keep_columns: list[str] | None = None,
    lowercase: bool = True,
) -> pd.DataFrame:
    """Load a CSV/JSON file, preprocess it, and optionally write the result."""
    path = Path(input_path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding=JSON_ENCODING) as handle:
            frame = pd.read_json(handle)
    else:
        frame = pd.read_csv(path, encoding=CSV_ENCODING)

    cleaned = preprocess_dataframe(
        frame,
        text_column=text_column,
        keep_raw=keep_raw,
        min_chars=min_chars,
        drop_duplicates=drop_duplicates,
        keep_columns=keep_columns,
        lowercase=lowercase,
    )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            with output.open("w", encoding=JSON_ENCODING, newline="") as handle:
                cleaned.to_json(handle, orient="records", force_ascii=False, indent=2)
        else:
            cleaned.to_csv(output, index=False, encoding=CSV_ENCODING)

    return cleaned
