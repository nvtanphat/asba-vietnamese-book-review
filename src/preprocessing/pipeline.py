from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import emoji_norm, formatters, noise_cleaner, quality_filter, unicode_norm, vocab_norm


def clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = unicode_norm.normalize_series(series)
    cleaned = noise_cleaner.normalize_series(cleaned)
    cleaned = emoji_norm.normalize_series(cleaned)
    cleaned = vocab_norm.normalize_series(cleaned)
    cleaned = formatters.normalize_series(cleaned)
    return cleaned


def preprocess_dataframe(
    frame: pd.DataFrame,
    text_column: str = "content",
    output_column: str | None = None,
    keep_raw: bool = True,
    min_chars: int = quality_filter.SHORT_TEXT_MIN_CHARS,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    target = frame.copy()
    source_column = text_column
    output_column = output_column or text_column

    if keep_raw:
        raw_column = f"{source_column}_raw"
        if raw_column not in target.columns:
            target[raw_column] = target[source_column]

    target[output_column] = clean_text_series(target[source_column])

    if output_column != source_column:
        target.drop(columns=[source_column], inplace=True, errors="ignore")

    target = quality_filter.drop_noise_rows(
        target,
        text_column=output_column,
        min_chars=min_chars,
        drop_duplicates=drop_duplicates,
    )
    return target


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    text_column: str = "content",
    keep_raw: bool = True,
    min_chars: int = quality_filter.SHORT_TEXT_MIN_CHARS,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    path = Path(input_path)
    frame = pd.read_json(path) if path.suffix.lower() == ".json" else pd.read_csv(path)
    cleaned = preprocess_dataframe(
        frame,
        text_column=text_column,
        keep_raw=keep_raw,
        min_chars=min_chars,
        drop_duplicates=drop_duplicates,
    )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            cleaned.to_json(output, orient="records", force_ascii=False, indent=2)
        else:
            cleaned.to_csv(output, index=False)

    return cleaned
