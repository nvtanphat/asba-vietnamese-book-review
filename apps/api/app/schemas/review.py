from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class TikiSampleOut(BaseModel):
    """A live review fetched from Tiki, with a demo order code appended."""

    original_text: str
    text: str
    order_code: str
    product_id: int
    stars: int


class AnalyzedReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    overall: str
    overall_probs: list[float]
    aspects: list[dict[str, Any]]
    created_at: datetime
