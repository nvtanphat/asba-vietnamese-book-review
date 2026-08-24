"""ORM models — shop/user auth tables for the ABSA demo API."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Shop(Base):
    __tablename__ = "shops"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    slug: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    plan: Mapped[str] = mapped_column(String(20), default="free")  # free | pro
    settings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    users: Mapped[list["User"]] = relationship(back_populates="shop")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    shop_id: Mapped[int] = mapped_column(ForeignKey("shops.id"), index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="agent")  # admin | agent
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    shop: Mapped[Shop] = relationship(back_populates="users")


class Template(Base):
    """A saved reply-template script (predefined response), scoped per shop."""

    __tablename__ = "templates"

    id: Mapped[int] = mapped_column(primary_key=True)
    shop_id: Mapped[int] = mapped_column(ForeignKey("shops.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(60), default="Lỗi Vận chuyển")
    trigger_condition: Mapped[str] = mapped_column(String(120), default="Kích hoạt thủ công")
    body: Mapped[str] = mapped_column(Text)
    sentiment_type: Mapped[str] = mapped_column(String(40), default="Cảm xúc tiêu cực")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class AnalyzedReview(Base):
    """Durable record of one /absa/analyze call, scoped per shop.

    The client-side session history (localStorage) is ephemeral by design — this is
    its server-side, cross-device, cross-session twin, so long-run week/month trend
    charts survive a browser refresh or a different device logging in.
    """

    __tablename__ = "analyzed_reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    shop_id: Mapped[int] = mapped_column(ForeignKey("shops.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    overall: Mapped[str] = mapped_column(String(12))
    overall_probs: Mapped[list] = mapped_column(JSON)
    aspects: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
