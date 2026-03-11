"""Tests for listing runs.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from tests.api_support import create_base_run, make_client  # noqa: E402


def test_list_runs_empty() -> None:
    client = make_client("test_list_runs_empty")
    response = client.get("/runs")
    assert response.status_code == fastapi.status.HTTP_200_OK
    assert response.json()["data"] == []


def test_list_runs_returns_created_runs() -> None:
    client = make_client("test_list_runs_returns")
    run_id_a = create_base_run(client)
    run_id_b = create_base_run(client)

    response = client.get("/runs")
    runs = response.json()["data"]

    assert len(runs) == 2
    returned_ids = {r["run_id"] for r in runs}
    assert run_id_a in returned_ids
    assert run_id_b in returned_ids


def test_list_runs_most_recent_first() -> None:
    client = make_client("test_list_runs_order")
    run_id_a = create_base_run(client)
    run_id_b = create_base_run(client)

    # Tick run_a so it gets updated_at refreshed (most recent)
    client.post(
        f"/runs/{run_id_a}/tick",
        json={"scenario": {"name": "s0", "values": {"O": 0.9}}, "outcome": 0.5},
    )

    runs = client.get("/runs").json()["data"]
    assert runs[0]["run_id"] == run_id_a
    assert runs[1]["run_id"] == run_id_b


def test_list_runs_includes_tick_count() -> None:
    client = make_client("test_list_runs_tick_count")
    run_id = create_base_run(client)
    client.post(
        f"/runs/{run_id}/tick",
        json={"scenario": {"name": "s0", "values": {"O": 0.9}}, "outcome": 0.5},
    )

    runs = client.get("/runs").json()["data"]
    match = [r for r in runs if r["run_id"] == run_id][0]
    assert match["tick_count"] == 1
