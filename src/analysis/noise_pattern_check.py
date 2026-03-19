from __future__ import annotations

from collections import Counter
import warnings
from typing import Any, Mapping, Sequence

from .helpers import (
    EMAIL_RE,
    ELONGATED_RE,
    MD_LINK_RE,
    PHONE_RE,
    PUNCT_REPEAT_RE,
    URL_RE,
    is_digit_only,
    is_symbol_only,
    percentage,
    to_text,
)

try:
    from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
except Exception:  # pragma: no cover - optional dependency fallback
    BeautifulSoup = None
    MarkupResemblesLocatorWarning = None


def scan(records: Sequence[Mapping[str, Any]], text_column: str | None = None) -> dict[str, Any]:
    if not records or not text_column:
        return {
            "text_column": text_column,
            "pattern_counts": {},
            "rows_with_any_noise": 0,
            "rows_with_any_noise_ratio": 0.0,
            "examples": {},
        }

    pattern_counts = Counter()
    examples: dict[str, list[str]] = {
        "url": [],
        "email": [],
        "phone": [],
        "html": [],
        "markdown_link": [],
        "punct_repeat": [],
        "elongated": [],
        "digit_only": [],
        "symbol_only": [],
    }
    rows_with_any_noise = 0

    for row in records:
        text = to_text(row.get(text_column))
        row_has_noise = False

        if URL_RE.search(text):
            pattern_counts["url"] += 1
            row_has_noise = True
            if len(examples["url"]) < 3:
                examples["url"].append(text)
        if EMAIL_RE.search(text):
            pattern_counts["email"] += 1
            row_has_noise = True
            if len(examples["email"]) < 3:
                examples["email"].append(text)
        if PHONE_RE.search(text):
            pattern_counts["phone"] += 1
            row_has_noise = True
            if len(examples["phone"]) < 3:
                examples["phone"].append(text)
        if BeautifulSoup is not None and "<" in text and ">" in text:
            with warnings.catch_warnings():
                if MarkupResemblesLocatorWarning is not None:
                    warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
                soup = BeautifulSoup(text, "html.parser")
            if soup.find():
                pattern_counts["html"] += 1
                row_has_noise = True
                if len(examples["html"]) < 3:
                    examples["html"].append(text)
                if row_has_noise:
                    rows_with_any_noise += 1
                continue
        elif "<" in text and ">" in text:
            pattern_counts["html"] += 1
            row_has_noise = True
            if len(examples["html"]) < 3:
                examples["html"].append(text)
        if MD_LINK_RE.search(text):
            pattern_counts["markdown_link"] += 1
            row_has_noise = True
            if len(examples["markdown_link"]) < 3:
                examples["markdown_link"].append(text)
        if PUNCT_REPEAT_RE.search(text):
            pattern_counts["punct_repeat"] += 1
            row_has_noise = True
            if len(examples["punct_repeat"]) < 3:
                examples["punct_repeat"].append(text)
        if ELONGATED_RE.search(text):
            pattern_counts["elongated"] += 1
            row_has_noise = True
            if len(examples["elongated"]) < 3:
                examples["elongated"].append(text)
        if is_digit_only(text):
            pattern_counts["digit_only"] += 1
            row_has_noise = True
            if len(examples["digit_only"]) < 3:
                examples["digit_only"].append(text)
        if is_symbol_only(text):
            pattern_counts["symbol_only"] += 1
            row_has_noise = True
            if len(examples["symbol_only"]) < 3:
                examples["symbol_only"].append(text)

        if row_has_noise:
            rows_with_any_noise += 1

    return {
        "text_column": text_column,
        "pattern_counts": dict(pattern_counts),
        "rows_with_any_noise": rows_with_any_noise,
        "rows_with_any_noise_ratio": percentage(rows_with_any_noise, len(records)),
        "examples": examples,
    }
