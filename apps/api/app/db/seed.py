"""Create tables and seed a demo shop + users.

Run:  uv run python -m app.db.seed     (from apps/api, or via `make seed`)

WARNING: seed() drops all tables before reseeding. Only run this by hand when you want to
reset the demo database — it is never called automatically by the API on startup.
"""

from __future__ import annotations

from app.core.auth import hash_password
from app.db.base import Base
from app.db.models import Shop, User
from app.db.session import SessionLocal, engine


def seed() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        shop = Shop(name="Demo Bookstore", slug="demo-bookstore", plan="pro")
        db.add(shop)
        db.flush()

        admin = User(
            shop_id=shop.id,
            email="admin@demo.com",
            name="Admin Demo",
            hashed_password=hash_password("admin123"),
            role="admin",
        )
        agent = User(
            shop_id=shop.id,
            email="agent@demo.com",
            name="Agent CSKH",
            hashed_password=hash_password("agent123"),
            role="agent",
        )
        db.add_all([admin, agent])
        db.commit()
        print(f"Seeded shop '{shop.name}' with:")
        print("  - 2 users (admin@demo.com / admin123, agent@demo.com / agent123)")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
