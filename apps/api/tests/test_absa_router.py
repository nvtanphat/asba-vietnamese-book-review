from __future__ import annotations

from fastapi.testclient import TestClient


class TestAnalyzeAuth:
    def test_requires_authentication(self, client: TestClient):
        resp = client.post("/absa/analyze", json={"text": "Giao hàng chậm quá"})
        assert resp.status_code == 401

    def test_rejects_oversized_text_before_touching_the_model(
        self, client: TestClient, auth_headers: dict[str, str]
    ):
        # Validation happens before the (expensive) model is ever invoked, so this must
        # fail fast with 422 regardless of whether the PhoBERT model is loaded.
        too_long = "a" * 4001
        resp = client.post("/absa/analyze", json={"text": too_long}, headers=auth_headers)
        assert resp.status_code == 422

    def test_rejects_empty_text(self, client: TestClient, auth_headers: dict[str, str]):
        resp = client.post("/absa/analyze", json={"text": ""}, headers=auth_headers)
        assert resp.status_code == 422
