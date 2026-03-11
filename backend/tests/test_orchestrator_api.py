"""Tests for orchestration API endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from src.api.app import create_app


def _client(tmp_path: Path) -> TestClient:
    """Create a test client with an isolated database."""
    app = create_app(database_path=str(tmp_path / "test.db"))
    return TestClient(app)


def _create_run(client: TestClient) -> str:
    """Create a minimal run and return its ID."""
    resp = client.post(
        "/runs",
        json={
            "personality": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
            "actions": [
                {"name": "bold", "modifiers": {"O": 1.0}},
                {"name": "safe", "modifiers": {"C": 0.9}},
            ],
            "temperature": 1.0,
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    result: str = resp.json()["data"]["run_id"]
    return result


def test_orchestrate_single_cycle(tmp_path: Path) -> None:
    """Single orchestration cycle returns a decision dict."""
    client = _client(tmp_path)
    run_id = _create_run(client)
    resp = client.post(f"/runs/{run_id}/orchestrate", json={"cycles": 1})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["cycle"] == 0
    assert data["action_type"] == "scenario"


def test_orchestrate_multiple_cycles(tmp_path: Path) -> None:
    """Multiple cycles returns a list of decisions."""
    client = _client(tmp_path)
    run_id = _create_run(client)
    resp = client.post(f"/runs/{run_id}/orchestrate", json={"cycles": 3})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 3


def test_orchestrator_log(tmp_path: Path) -> None:
    """Log endpoint returns persisted decisions."""
    client = _client(tmp_path)
    run_id = _create_run(client)
    client.post(f"/runs/{run_id}/orchestrate", json={"cycles": 2})
    resp = client.get(f"/runs/{run_id}/orchestrator-log")
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 2


def test_orchestrator_log_empty(tmp_path: Path) -> None:
    """Log endpoint returns empty list when no decisions exist."""
    client = _client(tmp_path)
    run_id = _create_run(client)
    resp = client.get(f"/runs/{run_id}/orchestrator-log")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_resume_not_paused(tmp_path: Path) -> None:
    """Resume raises 400 when run is not paused."""
    client = _client(tmp_path)
    run_id = _create_run(client)
    resp = client.post(f"/runs/{run_id}/resume", json={})
    assert resp.status_code == 400


def test_orchestrate_with_goals(tmp_path: Path) -> None:
    """Goals are accepted and cycle completes successfully."""
    client = _client(tmp_path)
    run_id = _create_run(client)
    resp = client.post(
        f"/runs/{run_id}/orchestrate",
        json={"cycles": 1, "goals": ["maximize entropy"]},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["action_type"] == "scenario"
    # Verify goals appear in the persisted log
    log_resp = client.get(f"/runs/{run_id}/orchestrator-log")
    log_data = log_resp.json()["data"]
    assert log_data[0]["input"]["goals"] == ["maximize entropy"]


def test_orchestrate_with_campaign_id(tmp_path: Path) -> None:
    """Campaign ID is recorded in the persisted decision."""
    client = _client(tmp_path)
    run_id = _create_run(client)
    resp = client.post(
        f"/runs/{run_id}/orchestrate",
        json={"cycles": 1, "campaign_id": "camp-1"},
    )
    assert resp.status_code == 200
    # Verify campaign_id in the log
    log_resp = client.get(f"/runs/{run_id}/orchestrator-log")
    log_data = log_resp.json()["data"]
    assert log_data[0]["campaign_id"] == "camp-1"
