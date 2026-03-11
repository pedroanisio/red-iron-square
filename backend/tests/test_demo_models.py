"""Tests for Two Minds demo contracts.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from src.demo.models import DemoAgentSnapshot, DemoSessionState
from src.demo.personas import (
    DEFAULT_PERSONAS,
    DISPLAY_EMOTION_LABELS,
    SCRIPTED_SCENARIOS,
)
from src.demo.schemas import (
    DemoAgentResponse,
    DemoCustomScenarioRequest,
    DemoScriptedScenarioResponse,
    DemoSessionCreateRequest,
    DemoSessionResponse,
    DemoSwapResponse,
)


def test_default_personas_match_guideline_profiles() -> None:
    """Luna and Marco should expose the expected contrasting traits."""
    luna = DEFAULT_PERSONAS["luna"]
    marco = DEFAULT_PERSONAS["marco"]

    assert luna.name == "Luna"
    assert luna.traits["N"] == 0.8
    assert luna.traits["T"] == 0.7
    assert marco.name == "Marco"
    assert marco.traits["O"] == 0.9
    assert marco.traits["R"] == 0.9


def test_scripted_scenarios_include_forced_outcomes() -> None:
    """The three scripted acts should be codified with deterministic outcomes."""
    assert set(SCRIPTED_SCENARIOS) == {"promotion", "phone_call", "three_months"}
    assert SCRIPTED_SCENARIOS["promotion"].forced_outcome == 0.3
    assert SCRIPTED_SCENARIOS["phone_call"].forced_outcome == -0.4
    assert SCRIPTED_SCENARIOS["three_months"].forced_outcome == -0.2


def test_swap_reset_rebuilds_session_state() -> None:
    """Swapping personalities should reset counters and agent snapshots."""
    session = DemoSessionState(
        session_id="demo-1",
        act_number=2,
        turn_count=4,
        scripted_index=2,
        agents={
            "luna": DemoAgentSnapshot(
                key="luna",
                name="Luna",
                summary="Thoughtful",
                traits=DEFAULT_PERSONAS["marco"].traits,
                mood=-0.8,
            ),
        },
    )

    session.reset_for_swap(DEFAULT_PERSONAS)

    assert session.turn_count == 0
    assert session.scripted_index == 0
    assert session.agents["luna"].traits == DEFAULT_PERSONAS["luna"].traits
    assert session.agents["luna"].mood == 0.0


def test_session_schemas_validate_expected_shapes() -> None:
    """Session request and response payloads should stay frontend-safe."""
    create_request = DemoSessionCreateRequest(act_number=2)
    custom_request = DemoCustomScenarioRequest(text="What if Luna sees an old friend?")
    scripted_response = DemoScriptedScenarioResponse(
        session_id="demo-1",
        scenario_key="promotion",
        turn_count=1,
    )
    session_response = DemoSessionResponse(
        session_id="demo-1",
        act_number=1,
        turn_count=0,
        agents=[
            DemoAgentResponse(
                key="luna",
                name="Luna",
                summary="Thoughtful",
                mood=0.0,
                energy=0.5,
                calm=0.5,
                emotion_label="Neutral",
            ),
        ],
    )
    swap_response = DemoSwapResponse(session_id="demo-1", act_number=2)

    assert create_request.act_number == 2
    assert custom_request.text.startswith("What if")
    assert scripted_response.scenario_key == "promotion"
    assert session_response.agents[0].emotion_label == "Neutral"
    assert swap_response.swapped is True


def test_display_emotion_labels_include_family_safe_copy() -> None:
    """Display labels should avoid internal jargon on the frontend."""
    assert DISPLAY_EMOTION_LABELS["IDENTITY_THREAT"] == "Inner conflict"
    assert DISPLAY_EMOTION_LABELS["AUTHENTICITY"] == "Feeling like myself"
