from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import regex as re

try:
    from ftfy import fix_text
except ImportError:  # pragma: no cover
    def fix_text(text: str) -> str:
        return text


_CTRL_OR_FORMAT = re.compile(r"[\p{Cc}\p{Cf}]")
_KEEP = {"\n", "\r", "\t", "\u200c", "\u200d"}


def normalize_unicode(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = fix_text(str(value))
    text = unicodedata.normalize("NFC", text)

    cleaned: list[str] = []
    for char in text:
        if char in _KEEP:
            cleaned.append(char)
            continue
        if _CTRL_OR_FORMAT.fullmatch(char):
            continue
        cleaned.append(char)

    return unicodedata.normalize("NFC", "".join(cleaned))


def normalize_text(value: Any) -> str | None:
    return normalize_unicode(value)


def repair_mojibake(value: Any) -> str | None:
    return normalize_unicode(value)


def normalize_nfc(value: Any) -> str | None:
    return normalize_unicode(value)


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_unicode)


def normalize_dataframe(
    frame: pd.DataFrame,
    text_column: str,
    output_column: str | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    target = frame if inplace else frame.copy()
    column = output_column or text_column
    target[column] = normalize_series(target[text_column])
    return target


def normalize_file(input_path: str | Path, output_path: str | Path | None = None, text_column: str = "content") -> pd.DataFrame:
    path = Path(input_path)
    frame = pd.read_json(path) if path.suffix.lower() == ".json" else pd.read_csv(path)
    cleaned = normalize_dataframe(frame, text_column=text_column)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            cleaned.to_json(output, orient="records", force_ascii=False, indent=2)
        else:
            cleaned.to_csv(output, index=False)

    return cleaned
