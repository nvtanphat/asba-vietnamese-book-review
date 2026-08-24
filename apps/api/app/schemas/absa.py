from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeIn(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=4000, examples=["Giao hàng quá chậm, sách lại bị rách bìa"]
    )


class AbsaAspect(BaseModel):
    aspect: str          # as_delivery
    label: str           # Giao hàng
    sentiment: str       # negative | neutral | positive
    presence: float


class AbsaResult(BaseModel):
    text: str
    overall: str
    overall_probs: list[float]
    aspects: list[AbsaAspect]
