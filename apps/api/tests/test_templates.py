"""Reply-template CRUD, scoped per shop."""

from __future__ import annotations


def test_create_list_update_delete_template(client, auth_headers):
    resp = client.post(
        "/templates",
        json={
            "name": "Xin lỗi giao hàng trễ",
            "category": "Lỗi Vận chuyển",
            "trigger_condition": "Khi Vận chuyển Tiêu cực",
            "body": "Xin lỗi {customer_name} vì đơn {order_number} bị trễ.",
            "sentiment_type": "Cảm xúc tiêu cực",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    template_id = resp.json()["id"]

    listed = client.get("/templates", headers=auth_headers).json()
    assert any(t["id"] == template_id for t in listed)

    resp = client.patch(
        f"/templates/{template_id}",
        json={
            "name": "Xin lỗi giao hàng trễ (v2)",
            "category": "Lỗi Vận chuyển",
            "trigger_condition": "Khi Vận chuyển Tiêu cực",
            "body": "Bản cập nhật nội dung mẫu.",
            "sentiment_type": "Cảm xúc tiêu cực",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Xin lỗi giao hàng trễ (v2)"

    resp = client.delete(f"/templates/{template_id}", headers=auth_headers)
    assert resp.status_code == 204

    listed = client.get("/templates", headers=auth_headers).json()
    assert all(t["id"] != template_id for t in listed)


def test_template_not_found_for_other_shop(client, auth_headers):
    resp = client.post(
        "/templates",
        json={"name": "T1", "body": "Nội dung"},
        headers=auth_headers,
    )
    template_id = resp.json()["id"]

    other_resp = client.post(
        "/auth/register",
        json={"shop_name": "Other Shop", "email": "other2@test.com", "password": "secret123", "name": "Other"},
    )
    other_headers = {"Authorization": f"Bearer {other_resp.json()['access_token']}"}

    resp = client.patch(
        f"/templates/{template_id}",
        json={"name": "Hijacked", "body": "x"},
        headers=other_headers,
    )
    assert resp.status_code == 404
