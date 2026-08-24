from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import utc_now


def generate_model_card(*, model: str, metrics: dict[str, Any], lineage: dict[str, Any], output: str | Path, notes: str = "") -> Path:
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Model Card — {model}",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Intended use",
        "Vietnamese Tiki book-review aspect-based sentiment analysis (ABSA). The production task predicts overall sentiment and six aspect sentiments.",
        "",
        "## Evaluation",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Data lineage",
        "",
        "```json",
        json.dumps(lineage, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Limitations",
        "The dataset is domain-specific, label/aspect distributions are imbalanced, and production drift must be monitored before reusing the model for other commerce domains.",
    ]
    if notes:
        lines += ["", "## Notes", notes]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
