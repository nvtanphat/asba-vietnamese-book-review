"""Pydantic schemas for shop-level settings and team management."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SettingsOut(BaseModel):
    settings: dict[str, Any] = {}


class SettingsIn(BaseModel):
    settings: dict[str, Any]


class UserUpdateIn(BaseModel):
    role: str | None = None
    is_active: bool | None = None
