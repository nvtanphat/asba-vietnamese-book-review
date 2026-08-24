from __future__ import annotations

from dataclasses import dataclass

ASPECT_COLS = [
    "as_content",
    "as_physical",
    "as_price",
    "as_packaging",
    "as_delivery",
    "as_service",
]
TARGET_COLS = ["sentiment", *ASPECT_COLS]
SENTIMENT_CLASSES = (0, 1, 2)
ASPECT_CLASSES = (0, 1, 2, 3)
ABSENT_CLASS = 3
SENTIMENT_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
ASPECT_SENTIMENT_NAMES = {0: "negative", 1: "neutral", 2: "positive", 3: "absent"}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    num_classes: int


TASK_SPECS = [TaskSpec("sentiment", 3), *[TaskSpec(c, 4) for c in ASPECT_COLS]]
