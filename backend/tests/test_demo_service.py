"""Service tests for the Two Minds demo flow.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from src.api.run_service import RunService
from src.api.run_store import RunStore
from src.demo.service import DemoSessionService

from tests.api_support import make_database_path


def _service(test_name: str) -> DemoSessionService:
    return DemoSessionService(RunService(RunStore(make_database_path(test_name))))


def test_create_session_builds_two_agents() -> None:
    service = _service("test_create_session_builds_two_agents")

    payload = service.create_session(act_number=1)

    assert payload["act_number"] == 1
    assert {agent["key"] for agent in payload["agents"]} == {"luna", "marco"}


def test_scripted_turn_persists_state_between_calls() -> None:
    service = _service("test_scripted_turn_persists_state_between_calls")
    session_id = service.create_session(act_number=1)["session_id"]

    first = service.run_scripted(session_id, "promotion")
    first_state = service.get_session(session_id)
    second = service.run_scripted(session_id, "phone_call")
    second_state = service.get_session(session_id)

    assert first["turn_count"] == 1
    assert second["turn_count"] == 2
    luna_before = next(
        agent for agent in first_state["agents"] if agent["key"] == "luna"
    )
    luna_after = next(
        agent for agent in second_state["agents"] if agent["key"] == "luna"
    )
    assert luna_before["mood"] != luna_after["mood"]


def test_swap_personalities_resets_turns_and_traits() -> None:
    service = _service("test_swap_personalities_resets_turns_and_traits")
    session_id = service.create_session(act_number=2)["session_id"]
    before_swap = service.get_session(session_id)
    luna_traits_before = service._store.get(session_id).session.agents["luna"].traits

    swap = service.swap_personalities(session_id)
    after_swap = service.get_session(session_id)
    luna_traits_after = service._store.get(session_id).session.agents["luna"].traits

    assert swap["swapped"] is True
    assert after_swap["turn_count"] == 0
    assert before_swap["agents"][0]["name"] == after_swap["agents"][0]["name"]
    assert luna_traits_before != luna_traits_after


def test_custom_turn_returns_neutral_fallback_scenario() -> None:
    service = _service("test_custom_turn_returns_neutral_fallback_scenario")
    session_id = service.create_session(act_number=3)["session_id"]

    payload = service.run_custom(session_id, "What if Luna meets an old friend?")

    assert payload["turn_count"] == 1
    assert len(payload["agents"]) == 2
