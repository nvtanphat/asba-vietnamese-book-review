from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .helpers import CONTROL_RE, MOJIBAKE_HINT_RE, REPLACEMENT_CHAR, ZERO_WIDTH_RE, percentage, to_text

try:
    from ftfy import fix_text
except Exception:  # pragma: no cover - optional dependency fallback
    fix_text = None


def scan(records: Sequence[Mapping[str, Any]], text_column: str | None = None) -> dict[str, Any]:
    if not records or not text_column:
        return {
            "text_column": text_column,
            "issue_counts": {},
            "rows_with_any_issue": 0,
            "rows_with_any_issue_ratio": 0.0,
            "examples": {},
        }

    issue_counts = Counter()
    examples: dict[str, list[str]] = {
        "replacement_char": [],
        "zero_width": [],
        "control_char": [],
        "mojibake_hint": [],
    }
    rows_with_any_issue = 0

    for row in records:
        text = to_text(row.get(text_column))
        row_has_issue = False

        if REPLACEMENT_CHAR in text:
            issue_counts["replacement_char"] += 1
            row_has_issue = True
            if len(examples["replacement_char"]) < 3:
                examples["replacement_char"].append(text)

        if ZERO_WIDTH_RE.search(text):
            issue_counts["zero_width"] += 1
            row_has_issue = True
            if len(examples["zero_width"]) < 3:
                examples["zero_width"].append(text)

        if CONTROL_RE.search(text):
            issue_counts["control_char"] += 1
            row_has_issue = True
            if len(examples["control_char"]) < 3:
                examples["control_char"].append(text)

        if MOJIBAKE_HINT_RE.search(text):
            issue_counts["mojibake_hint"] += 1
            row_has_issue = True
            if len(examples["mojibake_hint"]) < 3:
                examples["mojibake_hint"].append(text)

        if fix_text is not None and fix_text(text) != text:
            issue_counts["fixable_encoding"] += 1
            row_has_issue = True
            if "fixable_encoding" not in examples:
                examples["fixable_encoding"] = []
            if len(examples["fixable_encoding"]) < 3:
                examples["fixable_encoding"].append(text)

        if row_has_issue:
            rows_with_any_issue += 1

    return {
        "text_column": text_column,
        "issue_counts": dict(issue_counts),
        "rows_with_any_issue": rows_with_any_issue,
        "rows_with_any_issue_ratio": percentage(rows_with_any_issue, len(records)),
        "examples": examples,
    }
