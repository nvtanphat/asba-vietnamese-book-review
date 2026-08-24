"""Pydantic schemas for reply-template CRUD."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TemplateIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    category: str = "Lỗi Vận chuyển"
    trigger_condition: str = "Kích hoạt thủ công"
    body: str = Field(..., min_length=1)
    sentiment_type: str = "Cảm xúc tiêu cực"


class TemplateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: str
    trigger_condition: str
    body: str
    sentiment_type: str
    created_at: datetime
