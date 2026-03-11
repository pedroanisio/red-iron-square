"""API tests for the Two Minds demo routes.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from src.api import create_app  # noqa: E402

from tests.api_support import make_database_path  # noqa: E402


def _client(test_name: str) -> TestClient:
    return TestClient(create_app(database_path=make_database_path(test_name)))


def test_demo_session_lifecycle_routes() -> None:
    client = _client("test_demo_session_lifecycle_routes")

    created = client.post("/demo/sessions", json={"act_number": 1})
    session_id = created.json()["data"]["session_id"]
    fetched = client.get(f"/demo/sessions/{session_id}")
    scripted = client.post(f"/demo/sessions/{session_id}/scripted/promotion")
    swapped = client.post(f"/demo/sessions/{session_id}/swap")

    assert created.status_code == fastapi.status.HTTP_200_OK
    assert fetched.status_code == fastapi.status.HTTP_200_OK
    assert scripted.json()["data"]["turn_count"] == 1
    assert swapped.json()["data"]["swapped"] is True


def test_demo_websocket_stream_emits_turn_events_in_order() -> None:
    client = _client("test_demo_websocket_stream_emits_turn_events_in_order")
    session_id = client.post("/demo/sessions", json={"act_number": 1}).json()["data"][
        "session_id"
    ]

    with client.websocket_connect(f"/demo/sessions/{session_id}/stream") as websocket:
        response = client.post(f"/demo/sessions/{session_id}/scripted/promotion")
        assert response.status_code == fastapi.status.HTTP_200_OK
        event_types: list[str] = []
        while True:
            message = websocket.receive_json()
            event_types.append(message["event_type"])
            if message["event_type"] == "turn_completed":
                break

    assert event_types == [
        "scenario_received",
        "agent_state_updated",
        "agent_text_started",
        "agent_text_completed",
        "audio_unavailable",
        "agent_state_updated",
        "agent_text_started",
        "agent_text_completed",
        "audio_unavailable",
        "turn_completed",
    ]


def test_demo_custom_scenario_route_returns_session_payload() -> None:
    client = _client("test_demo_custom_scenario_route_returns_session_payload")
    session_id = client.post("/demo/sessions", json={"act_number": 3}).json()["data"][
        "session_id"
    ]

    response = client.post(
        f"/demo/sessions/{session_id}/scenarios",
        json={"text": "What if Marco gets fired?"},
    )

    assert response.status_code == fastapi.status.HTTP_200_OK
    assert response.json()["data"]["turn_count"] == 1
