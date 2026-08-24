"""Shop-level settings persistence."""

from __future__ import annotations


def test_settings_default_empty(client, auth_headers):
    resp = client.get("/settings", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == {"settings": {}}


def test_settings_patch_merges_and_persists(client, auth_headers):
    resp = client.patch("/settings", json={"settings": {"sla_warning": "2h"}}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["settings"]["sla_warning"] == "2h"

    resp = client.patch("/settings", json={"settings": {"shopee_configured": True}}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()["settings"]
    assert body["sla_warning"] == "2h"
    assert body["shopee_configured"] is True

    resp = client.get("/settings", headers=auth_headers)
    assert resp.json()["settings"]["sla_warning"] == "2h"
