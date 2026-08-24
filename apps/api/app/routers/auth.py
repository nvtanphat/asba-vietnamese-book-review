"""Authentication endpoints: register, login, invite, me."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    require_role,
    verify_password,
)
from app.core.limiter import limiter
from app.db.models import Shop, User
from app.db.session import get_db
from app.schemas.auth import InviteIn, LoginIn, MeOut, RegisterIn, ShopOut, TokenOut, UserOut
from app.schemas.settings import UserUpdateIn

router = APIRouter(prefix="/auth", tags=["auth"])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    return re.sub(r"[-\s]+", "-", slug).strip("-")[:60]


@router.post("/register", response_model=TokenOut, status_code=201)
@limiter.limit("10/minute")
def register(request: Request, payload: RegisterIn, db: Session = Depends(get_db)) -> TokenOut:
    """Create a new shop + admin user."""
    if db.scalar(select(User).where(User.email == payload.email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    slug = _slugify(payload.shop_name)
    # ensure unique slug
    base_slug = slug
    counter = 1
    while db.scalar(select(Shop).where(Shop.slug == slug)):
        slug = f"{base_slug}-{counter}"
        counter += 1

    shop = Shop(name=payload.shop_name, slug=slug)
    db.add(shop)
    db.flush()

    user = User(
        shop_id=shop.id,
        email=payload.email,
        name=payload.name,
        hashed_password=hash_password(payload.password),
        role="admin",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, shop.id, user.role)
    return TokenOut(access_token=token)


@router.post("/login", response_model=TokenOut)
@limiter.limit("5/minute")
def login(request: Request, payload: LoginIn, db: Session = Depends(get_db)) -> TokenOut:
    """Authenticate with email/password and receive a JWT."""
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account is deactivated")

    token = create_access_token(user.id, user.shop_id, user.role)
    return TokenOut(access_token=token)


@router.post("/invite", response_model=UserOut, status_code=201)
def invite(
    payload: InviteIn,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
) -> User:
    """Admin invites a new team member to their shop."""
    if db.scalar(select(User).where(User.email == payload.email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    user = User(
        shop_id=admin.shop_id,
        email=payload.email,
        name=payload.name,
        hashed_password=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/me", response_model=MeOut)
def me(user: User = Depends(get_current_user)) -> MeOut:
    """Return the current user's profile and shop info."""
    return MeOut(user=UserOut.model_validate(user), shop=ShopOut.model_validate(user.shop))


@router.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[User]:
    """List all team members in the current shop."""
    stmt = select(User).where(User.shop_id == user.shop_id).order_by(User.created_at)
    return list(db.scalars(stmt).all())


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    payload: UserUpdateIn,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
) -> User:
    """Admin updates a team member's role or active status."""
    target = db.get(User, user_id)
    if not target or target.shop_id != admin.shop_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if payload.role is not None:
        target.role = payload.role
    if payload.is_active is not None:
        target.is_active = payload.is_active
    db.commit()
    db.refresh(target)
    return target
