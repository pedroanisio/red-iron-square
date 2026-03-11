"""Tests for campaign API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
from src.api.app import create_app


def _client(tmp_path: object) -> TestClient:
    """Build a test client with an isolated database."""
    app = create_app(database_path=str(tmp_path) + "/test.db")
    return TestClient(app)


def _default_config() -> dict:
    """Return a minimal valid run configuration."""
    return {
        "personality": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
        "actions": [
            {"name": "bold", "modifiers": {"O": 1.0}},
            {"name": "safe", "modifiers": {"C": 0.9}},
        ],
        "temperature": 1.0,
        "seed": 42,
    }


def test_create_campaign(tmp_path: object) -> None:
    """POST /campaigns creates a campaign and its primary run."""
    client = _client(tmp_path)
    resp = client.post(
        "/campaigns",
        json={
            "name": "Test Campaign",
            "goals": ["explore"],
            "config_template": _default_config(),
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["name"] == "Test Campaign"
    assert data["status"] == "active"


def test_list_campaigns(tmp_path: object) -> None:
    """GET /campaigns returns all created campaigns."""
    client = _client(tmp_path)
    client.post(
        "/campaigns",
        json={
            "name": "C1",
            "config_template": _default_config(),
        },
    )
    client.post(
        "/campaigns",
        json={
            "name": "C2",
            "config_template": _default_config(),
        },
    )
    resp = client.get("/campaigns")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 2


def test_get_campaign(tmp_path: object) -> None:
    """GET /campaigns/{id} includes run links."""
    client = _client(tmp_path)
    created = client.post(
        "/campaigns",
        json={
            "name": "Test",
            "config_template": _default_config(),
        },
    ).json()["data"]
    resp = client.get(f"/campaigns/{created['campaign_id']}")
    assert resp.status_code == 200
    assert "runs" in resp.json()["data"]


def test_get_campaign_not_found(tmp_path: object) -> None:
    """GET /campaigns/{id} returns 404 for missing campaign."""
    client = _client(tmp_path)
    resp = client.get("/campaigns/nonexistent")
    assert resp.status_code == 404


def test_campaign_summary(tmp_path: object) -> None:
    """GET /campaigns/{id}/summary includes tick aggregation."""
    client = _client(tmp_path)
    created = client.post(
        "/campaigns",
        json={
            "name": "Test",
            "config_template": _default_config(),
        },
    ).json()["data"]
    resp = client.get(f"/campaigns/{created['campaign_id']}/summary")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "total_ticks" in data
    assert "run_count" in data


def test_add_checkpoint_rule(tmp_path: object) -> None:
    """POST /campaigns/{id}/rules persists a trigger rule."""
    client = _client(tmp_path)
    created = client.post(
        "/campaigns",
        json={
            "name": "Test",
            "config_template": _default_config(),
        },
    ).json()["data"]
    resp = client.post(
        f"/campaigns/{created['campaign_id']}/rules",
        json={"trigger_type": "every_n_ticks", "trigger_config": {"n": 5}},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "created"


def test_trigger_checkpoint(tmp_path: object) -> None:
    """POST /campaigns/{id}/checkpoint evaluates rules."""
    client = _client(tmp_path)
    created = client.post(
        "/campaigns",
        json={
            "name": "Test",
            "config_template": _default_config(),
        },
    ).json()["data"]
    cid = created["campaign_id"]
    client.post(
        f"/campaigns/{cid}/rules",
        json={"trigger_type": "every_n_ticks", "trigger_config": {"n": 5}},
    )
    resp = client.post(f"/campaigns/{cid}/checkpoint")
    assert resp.status_code == 200
    assert "fired" in resp.json()["data"]


def test_checkpoint_rule_not_found(tmp_path: object) -> None:
    """POST /campaigns/{id}/rules returns 404 for missing campaign."""
    client = _client(tmp_path)
    resp = client.post(
        "/campaigns/nonexistent/rules",
        json={"trigger_type": "manual", "trigger_config": {}},
    )
    assert resp.status_code == 404
