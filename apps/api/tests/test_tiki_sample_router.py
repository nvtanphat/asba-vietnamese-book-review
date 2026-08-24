from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.services.tiki_fetch import TikiFetchError


class TestTikiSample:
    def test_requires_authentication(self, client: TestClient):
        resp = client.get("/reviews/tiki-sample")
        assert resp.status_code == 401

    def test_returns_fetched_review(self, client: TestClient, auth_headers: dict[str, str]):
        fake_result = {
            "original_text": "Giao hàng chậm quá, đợi cả tuần",
            "text": "Giao hàng chậm quá, đợi cả tuần (Đơn hàng DH10231)",
            "order_code": "DH10231",
            "product_id": 480040,
            "stars": 1,
        }
        with patch("app.routers.reviews.fetch_live_negative_review", return_value=fake_result):
            resp = client.get("/reviews/tiki-sample", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == fake_result

    def test_returns_502_when_tiki_fetch_fails(
        self, client: TestClient, auth_headers: dict[str, str]
    ):
        with patch(
            "app.routers.reviews.fetch_live_negative_review",
            side_effect=TikiFetchError("Không kết nối được tới Tiki"),
        ):
            resp = client.get("/reviews/tiki-sample", headers=auth_headers)
        assert resp.status_code == 502
