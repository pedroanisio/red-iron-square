"""End-to-end API flow tests.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from src.api import create_app  # noqa: E402

from tests.api_support import (  # noqa: E402
    create_base_run,
    make_client,
    make_database_path,
)


def test_stateful_agent_flow_end_to_end() -> None:
    """Exercise the full stateful API flow through one persisted run."""
    client = make_client("test_stateful_agent_flow_end_to_end")

    health_response = client.get("/health")
    assert health_response.status_code == fastapi.status.HTTP_200_OK
    assert health_response.json() == {"status": "ok"}

    run_id = create_base_run(client)

    first_tick = client.post(
        f"/runs/{run_id}/tick",
        json={
            "scenario": {"name": "opening_probe", "values": {"O": 0.85, "N": 0.45}},
            "outcome": 0.4,
        },
    )
    second_tick = client.post(
        f"/runs/{run_id}/tick",
        json={
            "scenario": {"name": "stabilizer", "values": {"C": 0.75, "T": 0.8}},
            "outcome": 0.2,
        },
    )

    assert first_tick.status_code == fastapi.status.HTTP_200_OK
    assert second_tick.status_code == fastapi.status.HTTP_200_OK
    assert first_tick.json()["data"]["tick"] == 0
    assert second_tick.json()["data"]["tick"] == 1

    assist_response = client.post(
        f"/runs/{run_id}/assist/step",
        json={"goals": ["probe novelty", "preserve stability"], "window": 3},
    )
    assert assist_response.status_code == fastapi.status.HTTP_200_OK
    assist_payload = assist_response.json()["data"]
    assert assist_payload["scenario"]["name"] == "llm_probe"
    assert assist_payload["tick"]["tick"] == 2
    assert assist_payload["narrative"]["summary"] == (
        "The run remains exploratory but stable."
    )
    assert len(assist_payload["invocations"]) == 2

    intervention_response = client.post(
        f"/runs/{run_id}/intervention",
        json={"goals": ["stabilize behavior"], "window": 3, "apply_patch": True},
    )
    assert intervention_response.status_code == fastapi.status.HTTP_200_OK
    intervention_payload = intervention_response.json()["data"]
    assert intervention_payload["recommendation"]["action"] == "patch_params"
    assert intervention_payload["decision"]["applied"] is True
    assert intervention_payload["updated_run"]["config"][
        "temperature"
    ] == pytest.approx(0.55)

    phase_response = client.post(
        f"/runs/{run_id}/phases",
        json={
            "start_tick": 0,
            "end_tick": 2,
            "label": "stabilized_exploration",
            "notes": "Manual probes followed by agent-guided stabilization.",
        },
    )
    assert phase_response.status_code == fastapi.status.HTTP_200_OK
    assert phase_response.json()["data"]["label"] == "stabilized_exploration"

    run_response = client.get(f"/runs/{run_id}")
    assert run_response.status_code == fastapi.status.HTTP_200_OK
    run_payload = run_response.json()["data"]
    assert run_payload["tick_count"] == 3
    assert run_payload["latest_tick"]["tick"] == 2
    assert run_payload["config"]["temperature"] == pytest.approx(0.55)
    assert run_payload["agent_invocation_count"] == 3
    assert run_payload["intervention_count"] == 1

    trajectory_response = client.get(f"/runs/{run_id}/trajectory")
    assert trajectory_response.status_code == fastapi.status.HTTP_200_OK
    trajectory_payload = trajectory_response.json()["data"]
    assert trajectory_payload["tick_count"] == 3
    assert [tick["tick"] for tick in trajectory_payload["ticks"]] == [0, 1, 2]
    assert [tick["scenario"]["name"] for tick in trajectory_payload["ticks"]] == [
        "opening_probe",
        "stabilizer",
        "llm_probe",
    ]
    assert len(trajectory_payload["agent_invocations"]) == 3
    assert trajectory_payload["agent_invocations"][0]["agent_name"] == "scenario_agent"
    assert trajectory_payload["agent_invocations"][1]["agent_name"] == "observer_agent"
    assert trajectory_payload["agent_invocations"][2]["agent_name"] == (
        "intervention_agent"
    )
    assert len(trajectory_payload["interventions"]) == 1
    assert trajectory_payload["phases"][0]["label"] == "stabilized_exploration"

    replay_response = client.post(f"/runs/{run_id}/replay")
    assert replay_response.status_code == fastapi.status.HTTP_200_OK
    replay_payload = replay_response.json()["data"]["run"]
    assert replay_payload["parent_run_id"] == run_id
    assert replay_payload["parent_tick"] == 2
    assert replay_payload["tick_count"] == 3

    branch_response = client.post(
        f"/runs/{run_id}/branches",
        json={"parent_tick": 1, "temperature": 0.35},
    )
    assert branch_response.status_code == fastapi.status.HTTP_200_OK
    branch_payload = branch_response.json()["data"]["run"]
    assert branch_payload["parent_run_id"] == run_id
    assert branch_payload["parent_tick"] == 1
    assert branch_payload["tick_count"] == 2
    assert branch_payload["config"]["temperature"] == pytest.approx(0.35)


@pytest.mark.real_llm
def test_stateful_agent_flow_end_to_end_with_real_anthropic() -> None:
    """Exercise assisted and intervention flows against the real Anthropic client."""
    backend_root = Path(__file__).resolve().parents[1]
    project_root = backend_root.parent
    load_dotenv(project_root / ".env")
    load_dotenv(backend_root / ".env")
    if not (
        os.getenv("ANTROPHIC_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_AUTH_TOKEN")
    ):
        pytest.skip(
            "Set ANTHROPIC_API_KEY, legacy ANTROPHIC_API_KEY, "
            "or ANTHROPIC_AUTH_TOKEN to run the real "
            "Anthropic e2e test."
        )

    database_path = make_database_path(
        "test_stateful_agent_flow_end_to_end_with_real_anthropic"
    )
    client = TestClient(create_app(database_path))

    health_response = client.get("/health")
    assert health_response.status_code == fastapi.status.HTTP_200_OK

    run_id = create_base_run(client)

    manual_tick = client.post(
        f"/runs/{run_id}/tick",
        json={
            "scenario": {
                "name": "baseline_context",
                "values": {"O": 0.65, "C": 0.6, "N": 0.35},
            },
            "outcome": 0.3,
        },
    )
    assert manual_tick.status_code == fastapi.status.HTTP_200_OK
    assert manual_tick.json()["data"]["tick"] == 0

    assist_response = client.post(
        f"/runs/{run_id}/assist/step",
        json={"goals": ["increase insight", "avoid instability"], "window": 3},
    )
    assert assist_response.status_code == fastapi.status.HTTP_200_OK
    assist_payload = assist_response.json()["data"]
    assert isinstance(assist_payload["scenario"]["name"], str)
    assert assist_payload["scenario"]["name"]
    assert isinstance(assist_payload["scenario"]["values"], dict)
    assert assist_payload["scenario"]["values"]
    assert all(
        isinstance(value, float | int) and 0.0 <= float(value) <= 1.0
        for value in assist_payload["scenario"]["values"].values()
    )
    assert assist_payload["tick"]["tick"] == 1
    assert isinstance(assist_payload["narrative"]["summary"], str)
    assert assist_payload["narrative"]["summary"]
    assert len(assist_payload["invocations"]) == 2
    assert {invocation["purpose"] for invocation in assist_payload["invocations"]} == {
        "propose_scenario",
        "summarize_window",
    }
    assert all(
        invocation["metadata"]["provider"] == "anthropic"
        for invocation in assist_payload["invocations"]
    )

    intervention_response = client.post(
        f"/runs/{run_id}/intervention",
        json={"goals": ["stabilize behavior"], "window": 3, "apply_patch": True},
    )
    assert intervention_response.status_code == fastapi.status.HTTP_200_OK
    intervention_payload = intervention_response.json()["data"]
    assert intervention_payload["recommendation"]["action"] in {
        "continue",
        "probe",
        "narrate",
        "analyze",
        "patch_params",
        "pause",
        "terminate",
    }
    assert isinstance(intervention_payload["recommendation"]["reason"], str)
    assert intervention_payload["recommendation"]["reason"]
    assert intervention_payload["invocation"]["metadata"]["provider"] == "anthropic"

    if intervention_payload["recommendation"]["action"] == "patch_params":
        assert intervention_payload["decision"]["applied"] is True
        assert intervention_payload["updated_run"] is not None
        assert (
            0.0 <= intervention_payload["updated_run"]["config"]["temperature"] <= 1.0
        )
    else:
        assert intervention_payload["decision"]["applied"] is False

    trajectory_response = client.get(f"/runs/{run_id}/trajectory")
    assert trajectory_response.status_code == fastapi.status.HTTP_200_OK
    trajectory_payload = trajectory_response.json()["data"]
    assert trajectory_payload["tick_count"] == 2
    assert len(trajectory_payload["ticks"]) == 2
    assert len(trajectory_payload["agent_invocations"]) == 3
    assert (
        trajectory_payload["agent_invocations"][0]["metadata"]["provider"]
        == "anthropic"
    )
    assert (
        trajectory_payload["agent_invocations"][1]["metadata"]["provider"]
        == "anthropic"
    )
    assert (
        trajectory_payload["agent_invocations"][2]["metadata"]["provider"]
        == "anthropic"
    )
