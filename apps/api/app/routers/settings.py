"""Shop-level configuration (SLA targets, connector config, etc.)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.auth import require_role
from app.db.models import User
from app.db.session import get_db
from app.schemas.settings import SettingsIn, SettingsOut

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsOut)
def get_settings_(
    user: User = Depends(require_role("admin", "agent")),
) -> SettingsOut:
    return SettingsOut(settings=user.shop.settings or {})


@router.patch("", response_model=SettingsOut)
def update_settings(
    payload: SettingsIn,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
) -> SettingsOut:
    shop = admin.shop
    shop.settings = {**(shop.settings or {}), **payload.settings}
    db.commit()
    db.refresh(shop)
    return SettingsOut(settings=shop.settings or {})
