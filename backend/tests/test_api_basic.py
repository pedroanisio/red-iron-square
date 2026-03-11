"""Basic API tests.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from tests.api_support import make_client  # noqa: E402


def test_health() -> None:
    client = make_client("test_health")
    response = client.get("/health")
    assert response.status_code == fastapi.status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


def test_decide() -> None:
    client = make_client("test_decide")
    response = client.post(
        "/decide",
        json={
            "personality": {
                "O": 0.8,
                "C": 0.5,
                "E": 0.3,
                "A": 0.7,
                "N": 0.4,
                "R": 0.9,
                "I": 0.6,
                "T": 0.2,
            },
            "scenario": {"name": "pitch_meeting", "values": {"O": 0.9, "N": 0.7}},
            "actions": [
                {"name": "bold", "modifiers": {"O": 1.0, "R": 0.8, "N": -0.3}},
                {"name": "safe", "modifiers": {"C": 0.9, "T": 0.8}},
            ],
        },
    )
    payload = response.json()["data"]
    assert response.status_code == fastapi.status.HTTP_200_OK
    assert payload["chosen_action"] in {"bold", "safe"}
    assert payload["probabilities"]["bold"] + payload["probabilities"][
        "safe"
    ] == pytest.approx(1.0)


def test_self_aware_simulate() -> None:
    client = make_client("test_self_aware_simulate")
    response = client.post(
        "/simulate",
        json={
            "personality": {
                "O": 0.8,
                "C": 0.5,
                "E": 0.3,
                "A": 0.7,
                "N": 0.4,
                "R": 0.9,
                "I": 0.6,
                "T": 0.2,
            },
            "actions": [
                {"name": "bold", "modifiers": {"O": 1.0, "R": 0.8, "N": -0.3}},
                {"name": "safe", "modifiers": {"C": 0.9, "T": 0.8}},
            ],
            "scenarios": [{"name": "pitch_meeting", "values": {"O": 0.9, "N": 0.7}}],
            "outcomes": [0.6],
            "self_model": {
                "O": 0.7,
                "C": 0.5,
                "E": 0.4,
                "A": 0.6,
                "N": 0.4,
                "R": 0.8,
                "I": 0.6,
                "T": 0.3,
            },
        },
    )
    payload = response.json()["data"]
    assert response.status_code == fastapi.status.HTTP_200_OK
    assert len(payload["ticks"]) == 1
    assert "psi_hat" in payload["ticks"][0]
