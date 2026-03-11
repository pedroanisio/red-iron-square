"""Tests for the Flask UI.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import pytest

pytest.importorskip("flask")

from src.ui.app import create_ui_app


class FakeUiClient:
    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def create_run(self, payload: dict[str, object]) -> dict[str, object]:
        return {"run_id": "run-123"}

    def get_run(self, run_id: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "mode": "temporal",
            "status": "active",
            "tick_count": 1,
            "config": {},
            "parent_run_id": None,
            "parent_tick": None,
            "latest_tick": {"tick": 0, "action": "safe", "outcome": 0.6},
            "phases": [],
            "agent_invocation_count": 2,
            "intervention_count": 1,
            "created_at": "now",
            "updated_at": "now",
        }

    def get_trajectory(self, run_id: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "tick_count": 1,
            "ticks": [{"tick": 0, "action": "safe", "outcome": 0.6, "emotions": []}],
            "phases": [],
            "agent_invocations": [
                {
                    "agent_name": "observer_agent",
                    "purpose": "summarize_window",
                    "metadata": {"model": "fake"},
                    "output": {"summary": "stable"},
                }
            ],
            "interventions": [
                {
                    "action": "patch_params",
                    "reason": "reduce randomness",
                    "applied": True,
                    "payload": {"temperature": 0.5},
                }
            ],
        }

    def assist_step(self, run_id: str, payload: dict[str, object]) -> dict[str, object]:
        return {}

    def intervention(
        self, run_id: str, payload: dict[str, object]
    ) -> dict[str, object]:
        return {}

    def tick(self, run_id: str, payload: dict[str, object]) -> dict[str, object]:
        return {}


def test_index_renders() -> None:
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Red Iron Square" in response.data


def test_index_loads_run_view() -> None:
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/?run_id=run-123")

    assert response.status_code == 200
    assert b"run-123" in response.data
    assert b"AI Calls" in response.data
