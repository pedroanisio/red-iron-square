"""Agent-assisted API tests.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from src.api import create_app  # noqa: E402

from tests.api_support import (  # noqa: E402
    create_base_run,
    make_client,
    make_database_path,
)


def test_assisted_step_persists_agent_invocations() -> None:
    client = make_client("test_assisted_step_persists_agent_invocations")
    run_id = create_base_run(client)

    assist_response = client.post(
        f"/runs/{run_id}/assist/step",
        json={"goals": ["probe novelty"], "window": 5},
    )
    payload = assist_response.json()["data"]
    assert assist_response.status_code == fastapi.status.HTTP_200_OK
    assert payload["scenario"]["name"] == "llm_probe"
    assert payload["narrative"]["summary"] == "The run remains exploratory but stable."
    assert len(payload["invocations"]) == 2

    run_response = client.get(f"/runs/{run_id}").json()["data"]
    assert run_response["agent_invocation_count"] == 2

    trajectory = client.get(f"/runs/{run_id}/trajectory").json()["data"]
    assert len(trajectory["agent_invocations"]) == 2
    assert trajectory["agent_invocations"][0]["agent_name"] == "scenario_agent"


def test_intervention_endpoint_persists_and_applies_patch() -> None:
    client = make_client("test_intervention_endpoint_persists_and_applies_patch")
    run_id = create_base_run(client)

    intervention_response = client.post(
        f"/runs/{run_id}/intervention",
        json={"goals": ["stabilize behavior"], "window": 5, "apply_patch": True},
    )
    payload = intervention_response.json()["data"]
    assert intervention_response.status_code == fastapi.status.HTTP_200_OK
    assert payload["recommendation"]["action"] == "patch_params"
    assert payload["decision"]["applied"] is True
    assert payload["updated_run"]["config"]["temperature"] == pytest.approx(0.55)

    run_response = client.get(f"/runs/{run_id}").json()["data"]
    assert run_response["intervention_count"] == 1
    assert run_response["agent_invocation_count"] == 1

    trajectory = client.get(f"/runs/{run_id}/trajectory").json()["data"]
    assert len(trajectory["interventions"]) == 1
    assert trajectory["interventions"][0]["action"] == "patch_params"


def test_assisted_step_returns_503_when_anthropic_credentials_are_missing() -> None:
    database_path = make_database_path(
        "test_assisted_step_missing_anthropic_credentials"
    )
    with patch.dict(
        "os.environ",
        {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""},
        clear=False,
    ):
        client = TestClient(create_app(database_path))
        run_id = create_base_run(client)

        assist_response = client.post(
            f"/runs/{run_id}/assist/step",
            json={"goals": ["probe novelty"], "window": 5},
        )

    assert assist_response.status_code == fastapi.status.HTTP_503_SERVICE_UNAVAILABLE
    assert "ANTHROPIC_API_KEY" in assist_response.json()["detail"]


def test_assisted_step_returns_503_when_openai_credentials_are_missing() -> None:
    database_path = make_database_path("test_assisted_step_missing_openai_credentials")
    with patch.dict(
        "os.environ",
        {
            "RED_IRON_SQUARE_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "",
        },
        clear=False,
    ):
        client = TestClient(create_app(database_path))
        run_id = create_base_run(client)

        assist_response = client.post(
            f"/runs/{run_id}/assist/step",
            json={"goals": ["probe novelty"], "window": 5},
        )

    assert assist_response.status_code == fastapi.status.HTTP_503_SERVICE_UNAVAILABLE
    assert "OPENAI_API_KEY" in assist_response.json()["detail"]
