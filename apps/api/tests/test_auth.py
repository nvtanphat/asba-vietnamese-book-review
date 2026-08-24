"""Auth flow: register, login, /me, role-gated failures, password hashing, and access token validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from jose import jwt
from fastapi import HTTPException

from app.core.auth import (
    _decode_token,
    create_access_token,
    hash_password,
    verify_password,
)
from app.core.config import get_settings
from app.main import _check_jwt_secret


class TestPasswordHashing:
    def test_hash_is_not_the_plain_password(self):
        hashed = hash_password("hunter2")
        assert hashed != "hunter2"

    def test_verify_password_accepts_correct_password(self):
        hashed = hash_password("hunter2")
        assert verify_password("hunter2", hashed) is True

    def test_verify_password_rejects_wrong_password(self):
        hashed = hash_password("hunter2")
        assert verify_password("wrong-password", hashed) is False

    def test_same_password_hashes_differently_each_time(self):
        # bcrypt uses a random salt per call, so hashes must not collide,
        # but both must still verify against the original password.
        first = hash_password("hunter2")
        second = hash_password("hunter2")
        assert first != second
        assert verify_password("hunter2", first) is True
        assert verify_password("hunter2", second) is True


class TestAccessToken:
    def test_create_and_decode_round_trip(self):
        token = create_access_token(user_id=42, shop_id=7, role="owner")
        payload = _decode_token(token)
        assert payload["sub"] == "42"
        assert payload["shop_id"] == 7
        assert payload["role"] == "owner"

    def test_decode_rejects_garbage_token(self):
        with pytest.raises(HTTPException) as exc_info:
            _decode_token("not-a-real-jwt-token")
        assert exc_info.value.status_code == 401

    def test_decode_rejects_expired_token(self):
        settings = get_settings()
        expired_payload = {
            "sub": "42",
            "shop_id": 7,
            "role": "owner",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        }
        expired_token = jwt.encode(
            expired_payload, settings.jwt_secret, algorithm=settings.jwt_algorithm
        )
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(expired_token)
        assert exc_info.value.status_code == 401

    def test_decode_rejects_token_signed_with_wrong_secret(self):
        forged_payload = {
            "sub": "1",
            "shop_id": 1,
            "role": "owner",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        forged_token = jwt.encode(forged_payload, "not-the-real-secret", algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(forged_token)
        assert exc_info.value.status_code == 401


def test_register_returns_token(client):
    resp = client.post(
        "/auth/register",
        json={
            "shop_name": "My Shop",
            "email": "admin@myshop.com",
            "password": "secret123",
            "name": "Admin",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"


def test_register_duplicate_email_conflicts(client):
    payload = {
        "shop_name": "Shop A",
        "email": "dup@test.com",
        "password": "secret123",
        "name": "A",
    }
    assert client.post("/auth/register", json=payload).status_code == 201
    payload["shop_name"] = "Shop B"
    resp = client.post("/auth/register", json=payload)
    assert resp.status_code == 409


def test_login_success(client, auth_headers):
    resp = client.post("/auth/login", json={"email": "owner@test.com", "password": "testpass123"})
    assert resp.status_code == 200
    assert resp.json()["access_token"]


def test_login_wrong_password_rejected(client, auth_headers):
    resp = client.post("/auth/login", json={"email": "owner@test.com", "password": "wrong"})
    assert resp.status_code == 401


def test_me_requires_auth(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_me_returns_profile(client, auth_headers):
    resp = client.get("/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["user"]["email"] == "owner@test.com"
    assert body["shop"]["name"] == "Test Shop"


def test_list_users_includes_registered_admin(client, auth_headers):
    resp = client.get("/auth/users", headers=auth_headers)
    assert resp.status_code == 200
    emails = [u["email"] for u in resp.json()]
    assert "owner@test.com" in emails


def test_invite_and_update_user(client, auth_headers):
    resp = client.post(
        "/auth/invite",
        json={"email": "agent@test.com", "name": "Agent", "password": "secret123", "role": "agent"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    user_id = resp.json()["id"]

    resp = client.patch(f"/auth/users/{user_id}", json={"is_active": False}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False


class TestJwtSecretGuard:
    def test_refuses_to_start_with_default_secret_in_production(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "environment", "production", raising=False)
        with pytest.raises(RuntimeError, match="default JWT_SECRET"):
            _check_jwt_secret()

    def test_warns_but_allows_default_secret_outside_production(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "environment", "development", raising=False)
        with pytest.warns(UserWarning, match="default JWT_SECRET"):
            _check_jwt_secret()

    def test_allows_production_with_a_real_secret(self, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "environment", "production", raising=False)
        monkeypatch.setattr(settings, "jwt_secret", "a-real-random-secret", raising=False)
        _check_jwt_secret()  # must not raise


class TestRateLimiting:
    def test_login_is_rate_limited_after_five_attempts_per_minute(self, client, auth_headers):
        # auth_headers already spent one /auth/register call, but /auth/login has its
        # own independent quota — 5 wrong-password attempts should all still go through...
        for _ in range(5):
            resp = client.post("/auth/login", json={"email": "owner@test.com", "password": "wrong"})
            assert resp.status_code == 401

        # ...and the 6th, within the same minute, must be throttled rather than evaluated.
        resp = client.post("/auth/login", json={"email": "owner@test.com", "password": "wrong"})
        assert resp.status_code == 429

    def test_register_is_rate_limited_after_ten_attempts_per_minute(self, client):
        for i in range(10):
            resp = client.post(
                "/auth/register",
                json={
                    "shop_name": f"Spam Shop {i}",
                    "email": f"spam{i}@test.com",
                    "password": "secret123",
                    "name": "Spammer",
                },
            )
            assert resp.status_code == 201

        resp = client.post(
            "/auth/register",
            json={
                "shop_name": "Spam Shop 11",
                "email": "spam11@test.com",
                "password": "secret123",
                "name": "Spammer",
            },
        )
        assert resp.status_code == 429
