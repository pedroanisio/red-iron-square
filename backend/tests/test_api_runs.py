"""Stateful run API tests.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from tests.api_support import create_base_run, make_client  # noqa: E402


def test_run_lifecycle() -> None:
    client = make_client("test_run_lifecycle")
    create_response = client.post(
        "/runs",
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
            "temperature": 1.0,
            "seed": 42,
        },
    )
    run = create_response.json()["data"]
    run_id = run["run_id"]
    assert create_response.status_code == fastapi.status.HTTP_200_OK
    assert run["tick_count"] == 0
    assert run["mode"] == "self_aware"

    tick_response = client.post(
        f"/runs/{run_id}/tick",
        json={
            "scenario": {"name": "pitch_meeting", "values": {"O": 0.9, "N": 0.7}},
            "outcome": 0.6,
        },
    )
    tick_payload = tick_response.json()["data"]
    assert tick_response.status_code == fastapi.status.HTTP_200_OK
    assert tick_payload["tick"] == 0
    assert "psi_hat" in tick_payload
    assert "precision" in tick_payload
    assert "prediction_errors" in tick_payload

    run_after_tick = client.get(f"/runs/{run_id}").json()["data"]
    assert run_after_tick["tick_count"] == 1
    assert run_after_tick["latest_tick"]["tick"] == 0
    assert "precision" in run_after_tick["latest_tick"]
    assert "prediction_errors" in run_after_tick["latest_tick"]

    phase_response = client.post(
        f"/runs/{run_id}/phases",
        json={"start_tick": 0, "label": "probe", "notes": "First high-O scenario."},
    )
    assert phase_response.status_code == fastapi.status.HTTP_200_OK
    assert phase_response.json()["data"]["label"] == "probe"

    patch_response = client.patch(f"/runs/{run_id}/params", json={"temperature": 0.7})
    assert patch_response.json()["data"]["config"]["temperature"] == pytest.approx(0.7)

    trajectory = client.get(f"/runs/{run_id}/trajectory").json()["data"]
    assert trajectory["tick_count"] == 1
    assert len(trajectory["ticks"]) == 1
    assert trajectory["phases"][0]["label"] == "probe"
    assert "precision" in trajectory["ticks"][0]
    assert "prediction_errors" in trajectory["ticks"][0]


def test_replay_and_branch() -> None:
    client = make_client("test_replay_and_branch")
    run_id = create_base_run(client)
    for name, values, outcome in [
        ("s0", {"O": 0.9, "N": 0.7}, 0.6),
        ("s1", {"C": 0.8, "N": 0.3}, 0.2),
    ]:
        response = client.post(
            f"/runs/{run_id}/tick",
            json={"scenario": {"name": name, "values": values}, "outcome": outcome},
        )
        assert response.status_code == fastapi.status.HTTP_200_OK

    replay_run = client.post(f"/runs/{run_id}/replay").json()["data"]["run"]
    assert replay_run["parent_run_id"] == run_id
    assert replay_run["parent_tick"] == 1
    assert replay_run["tick_count"] == 2

    original_trajectory = client.get(f"/runs/{run_id}/trajectory").json()["data"]
    replay_trajectory = client.get(f"/runs/{replay_run['run_id']}/trajectory").json()[
        "data"
    ]
    assert replay_trajectory["ticks"] == original_trajectory["ticks"]

    branch_run = client.post(
        f"/runs/{run_id}/branches",
        json={"parent_tick": 0, "temperature": 0.5},
    ).json()["data"]["run"]
    assert branch_run["parent_run_id"] == run_id
    assert branch_run["parent_tick"] == 0
    assert branch_run["tick_count"] == 1
    assert branch_run["config"]["temperature"] == pytest.approx(0.5)
