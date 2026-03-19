from __future__ import annotations

from pathlib import Path

DEFAULT_INPUT_CANDIDATES = (
    Path("data/raw/tiki-book-review.csv"),
    Path("data/raw/tiki-book-review.json"),
)

DEFAULT_OUTPUT_PATH = Path("experiments/reports/data_scan_report.json")

TEXT_COLUMN_PRIORITY = (
    "content",
    "review_content",
    "review_text",
    "text",
    "review",
    "comment",
)

LABEL_COLUMN_PRIORITY = (
    "sentiment_llm",
    "rating",
    "as_content",
    "as_physical",
    "as_price",
    "as_packaging",
    "as_delivery",
    "as_service",
)

MIN_TEXT_LENGTH = 10

TOP_N = 10
