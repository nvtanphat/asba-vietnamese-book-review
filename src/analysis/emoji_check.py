from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .helpers import count_emojis, percentage, to_text


def scan(records: Sequence[Mapping[str, Any]], text_column: str | None = None) -> dict[str, Any]:
    if not records or not text_column:
        return {
            "text_column": text_column,
            "rows_with_emoji": 0,
            "rows_with_emoji_ratio": 0.0,
            "emoji_total": 0,
            "top_emojis": [],
            "emoji_samples": [],
        }

    emoji_counter: Counter[str] = Counter()
    samples: list[str] = []
    rows_with_emoji = 0

    for row in records:
        text = to_text(row.get(text_column))
        emojis = count_emojis(text)
        if emojis:
            rows_with_emoji += 1
            if len(samples) < 10:
                samples.extend(emojis[: max(0, 10 - len(samples))])
        emoji_counter.update(emojis)

    return {
        "text_column": text_column,
        "rows_with_emoji": rows_with_emoji,
        "rows_with_emoji_ratio": percentage(rows_with_emoji, len(records)),
        "emoji_total": sum(emoji_counter.values()),
        "all_emojis": [
            {"emoji": emoji, "count": count}
            for emoji, count in emoji_counter.most_common()
        ],
        "emoji_samples": samples,
    }
