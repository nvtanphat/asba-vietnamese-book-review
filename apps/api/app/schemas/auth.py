"""Pydantic schemas for authentication endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ── Request schemas ──────────────────────────────────────────────────────


class RegisterIn(BaseModel):
    shop_name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=6)
    name: str = Field(..., min_length=1, max_length=120)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class InviteIn(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=120)
    password: str = Field(..., min_length=6)
    role: str = Field(default="agent", pattern=r"^(admin|agent)$")


# ── Response schemas ─────────────────────────────────────────────────────


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    role: str
    is_active: bool
    shop_id: int
    created_at: datetime


class ShopOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    plan: str
    created_at: datetime


class MeOut(BaseModel):
    user: UserOut
    shop: ShopOut
