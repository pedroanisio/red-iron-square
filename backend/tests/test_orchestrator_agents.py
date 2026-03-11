"""Tests for orchestrator agent registry."""

from __future__ import annotations

from typing import Any

from src.llm.schemas import (
    AnalysisReport,
    InterventionRecommendation,
    LLMInvocationMetadata,
    LLMInvocationResult,
    NarrativeChunk,
    ScenarioProposal,
)
from src.orchestrator.agents import (
    AGENT_REGISTRY,
    run_analyst_agent,
    run_intervention_agent,
    run_observer_agent,
    run_scenario_agent,
)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _invocation_result() -> LLMInvocationResult:
    """Build a minimal LLMInvocationResult for testing."""
    return LLMInvocationResult(
        metadata=LLMInvocationMetadata(
            model="test-model",
            prompt_tokens=10,
            completion_tokens=5,
        ),
        raw_text="{}",
    )


def _fake_proposal() -> ScenarioProposal:
    return ScenarioProposal(
        name="bold_probe",
        description="Test scenario",
        values={"curiosity": 0.9},
        rationale="testing",
    )


def _fake_narrative() -> NarrativeChunk:
    return NarrativeChunk(
        summary="Stable trajectory observed",
        tick_start=0,
        tick_end=5,
        evidence=["mood stayed flat"],
    )


def _fake_report() -> AnalysisReport:
    return AnalysisReport(
        dominant_regime="exploratory",
        notable_emotions=["curiosity"],
        anomalies=[],
        recommendations=["continue"],
    )


def _fake_intervention() -> InterventionRecommendation:
    return InterventionRecommendation(
        action="continue",
        reason="All stable",
        temperature=None,
    )


# ---------------------------------------------------------------------------
# Fake runtime that returns real Pydantic models
# ---------------------------------------------------------------------------


class FakeRuntime:
    """Stand-in for AgentRuntime; returns real schema objects."""

    def propose_scenario(
        self,
        *,
        current_state: dict[str, Any],
        trajectory_window: list[dict[str, Any]],
        goals: list[str],
    ) -> tuple[ScenarioProposal, LLMInvocationResult]:
        return _fake_proposal(), _invocation_result()

    def summarize_window(
        self,
        *,
        ticks: list[dict[str, Any]],
    ) -> tuple[NarrativeChunk, LLMInvocationResult]:
        return _fake_narrative(), _invocation_result()

    def analyze_window(
        self,
        *,
        ticks: list[dict[str, Any]],
    ) -> tuple[AnalysisReport, LLMInvocationResult]:
        return _fake_report(), _invocation_result()

    def recommend_intervention(
        self,
        *,
        current_state: dict[str, Any],
        ticks: list[dict[str, Any]],
        goals: list[str],
    ) -> tuple[InterventionRecommendation, LLMInvocationResult]:
        return _fake_intervention(), _invocation_result()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scenario_agent_returns_expected_shape() -> None:
    """run_scenario_agent wraps propose_scenario correctly."""
    result = run_scenario_agent(
        FakeRuntime(),
        current_state={"mood": 0.5},
        trajectory_window=[{"tick": 1}],
        goals=["explore"],
    )
    assert result["action_type"] == "scenario"
    assert result["output"]["name"] == "bold_probe"
    assert "bold_probe" in result["rationale"]


def test_observer_agent_returns_expected_shape() -> None:
    """run_observer_agent wraps summarize_window correctly."""
    result = run_observer_agent(
        FakeRuntime(),
        ticks=[{"tick": 0}],
    )
    assert result["action_type"] == "observe"
    assert "Stable" in result["rationale"]
    assert result["output"]["tick_start"] == 0


def test_analyst_agent_returns_expected_shape() -> None:
    """run_analyst_agent wraps analyze_window correctly."""
    result = run_analyst_agent(
        FakeRuntime(),
        ticks=[{"tick": 0}],
    )
    assert result["action_type"] == "analyze"
    assert "exploratory" in result["rationale"]


def test_intervention_agent_returns_expected_shape() -> None:
    """run_intervention_agent wraps recommend_intervention correctly."""
    result = run_intervention_agent(
        FakeRuntime(),
        current_state={"mood": 0.0},
        ticks=[{"tick": 0}],
        goals=["stabilize"],
    )
    assert result["action_type"] == "intervene"
    assert result["output"]["action"] == "continue"
    assert "All stable" in result["rationale"]


def test_registry_has_all_agents() -> None:
    """AGENT_REGISTRY maps the four canonical agent names."""
    assert set(AGENT_REGISTRY.keys()) == {"scenario", "observe", "analyze", "intervene"}


def test_registry_callables_match_functions() -> None:
    """Each registry entry points to the correct function."""
    assert AGENT_REGISTRY["scenario"] is run_scenario_agent
    assert AGENT_REGISTRY["observe"] is run_observer_agent
    assert AGENT_REGISTRY["analyze"] is run_analyst_agent
    assert AGENT_REGISTRY["intervene"] is run_intervention_agent
