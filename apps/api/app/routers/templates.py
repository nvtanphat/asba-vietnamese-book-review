"""Reply-template CRUD endpoints (predefined response scripts)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.db.models import Template, User
from app.db.session import get_db
from app.schemas import TemplateIn, TemplateOut

router = APIRouter(prefix="/templates", tags=["templates"])


@router.get("", response_model=list[TemplateOut])
def list_templates(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Template]:
    stmt = (
        select(Template)
        .where(Template.shop_id == user.shop_id)
        .order_by(Template.created_at.desc())
    )
    return list(db.scalars(stmt).all())


@router.post("", response_model=TemplateOut, status_code=201)
def create_template(
    payload: TemplateIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Template:
    template = Template(shop_id=user.shop_id, **payload.model_dump())
    db.add(template)
    db.commit()
    db.refresh(template)
    return template


@router.patch("/{template_id}", response_model=TemplateOut)
def update_template(
    template_id: int,
    payload: TemplateIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Template:
    template = db.get(Template, template_id)
    if not template or template.shop_id != user.shop_id:
        raise HTTPException(404, "Template not found")
    for field, value in payload.model_dump().items():
        setattr(template, field, value)
    db.commit()
    db.refresh(template)
    return template


@router.delete("/{template_id}", status_code=204)
def delete_template(
    template_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    template = db.get(Template, template_id)
    if not template or template.shop_id != user.shop_id:
        raise HTTPException(404, "Template not found")
    db.delete(template)
    db.commit()
