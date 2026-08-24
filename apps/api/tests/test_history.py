"""Persisted analysis history: /absa/analyze writes a row, /reviews/history reads it back."""

from __future__ import annotations

from datetime import datetime, timezone


def test_analyze_persists_to_history(client, auth_headers):
    resp = client.post("/absa/analyze", json={"text": "Giao hàng trễ quá"}, headers=auth_headers)
    assert resp.status_code == 200

    history = client.get("/reviews/history", headers=auth_headers).json()
    assert len(history) == 1
    assert history[0]["text"] == "Giao hàng trễ quá"
    assert history[0]["overall"] == "negative"
    assert history[0]["aspects"][0]["aspect"] == "as_delivery"


def test_history_requires_authentication(client):
    assert client.get("/reviews/history").status_code == 401
    assert client.get("/reviews/history/summary").status_code == 401


def test_history_scoped_per_shop(client, auth_headers):
    client.post("/absa/analyze", json={"text": "Giao hàng trễ quá"}, headers=auth_headers)

    other_resp = client.post(
        "/auth/register",
        json={"shop_name": "Other Shop", "email": "other-history@test.com", "password": "secret123", "name": "Other"},
    )
    other_headers = {"Authorization": f"Bearer {other_resp.json()['access_token']}"}

    assert client.get("/reviews/history", headers=other_headers).json() == []
    assert client.get("/reviews/history", headers=auth_headers).json() != []


def test_history_summary_aggregates_by_period(client, auth_headers):
    client.post("/absa/analyze", json={"text": "Giao hàng trễ quá"}, headers=auth_headers)
    client.post("/absa/analyze", json={"text": "Sản phẩm tuyệt vời"}, headers=auth_headers)
    client.post("/absa/analyze", json={"text": "Giao hàng bị lỗi nữa rồi"}, headers=auth_headers)

    resp = client.get("/reviews/history/summary", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["groupby"] == "week"
    assert len(body["buckets"]) == 1  # all 3 calls happen in the same test run / same week

    bucket = body["buckets"][0]
    current_period = f"{datetime.now(timezone.utc).isocalendar()[0]}-W{datetime.now(timezone.utc).isocalendar()[1]:02d}"
    assert bucket["period"] == current_period
    assert bucket["total"] == 3
    assert bucket["positive"] == 1
    assert bucket["negative"] == 2
    assert bucket["aspect_negative"]["as_delivery"] == 2


def test_history_summary_groupby_month(client, auth_headers):
    client.post("/absa/analyze", json={"text": "Giao hàng trễ quá"}, headers=auth_headers)

    resp = client.get("/reviews/history/summary?groupby=month", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["groupby"] == "month"
    assert body["buckets"][0]["period"] == datetime.now(timezone.utc).strftime("%Y-%m")


def test_history_summary_rejects_bad_groupby(client, auth_headers):
    resp = client.get("/reviews/history/summary?groupby=year", headers=auth_headers)
    assert resp.status_code == 422
