"""Shared pytest fixtures: isolated in-memory DB, mocked ABSA, authenticated client."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.limiter import limiter
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.services import absa_service


@pytest.fixture()
def test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(test_engine, monkeypatch):
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    def fake_analyze(text: str) -> dict:
        lowered = text.lower()
        # General negative complaint with no identifiable aspect (e.g. vague insult).
        general_negative_words = ["cức", "tởm", "chán ghê"]
        if any(w in lowered for w in general_negative_words):
            return {
                "text": text,
                "overall": "negative",
                "overall_probs": [0.76, 0.14, 0.10],
                "aspects": [],
            }
        negative_words = ["trễ", "lỗi", "vỡ", "hỏng", "chậm", "tệ"]
        is_negative = any(w in lowered for w in negative_words)
        if is_negative:
            return {
                "text": text,
                "overall": "negative",
                "overall_probs": [0.85, 0.1, 0.05],
                "aspects": [
                    {"aspect": "as_delivery", "label": "Giao hàng", "sentiment": "negative", "presence": 0.9},
                ],
            }
        return {
            "text": text,
            "overall": "positive",
            "overall_probs": [0.05, 0.1, 0.85],
            "aspects": [],
        }

    monkeypatch.setattr(absa_service, "analyze", fake_analyze)
    app.dependency_overrides[get_db] = override_get_db
    # The rate limiter's storage is process-global, not per-TestClient — without a
    # reset, whichever test happens to run first exhausts /auth/register's quota for
    # every later test too (they all share the TestClient's fixed fake IP).
    limiter.reset()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def auth_headers(client: TestClient) -> dict[str, str]:
    """Register a fresh demo shop/user and return an Authorization header for it."""
    resp = client.post(
        "/auth/register",
        json={
            "shop_name": "Test Shop",
            "email": "owner@test.com",
            "password": "testpass123",
            "name": "Owner",
        },
    )
    assert resp.status_code == 201, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
