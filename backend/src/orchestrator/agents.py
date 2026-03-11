"""Agent registry for orchestration.

Each public function wraps one ``AgentRuntime`` method, normalising
the return value into a lightweight ``dict`` suitable for the
orchestration loop.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.llm.schemas import (
    AnalysisReport,
    InterventionRecommendation,
    LLMInvocationResult,
    NarrativeChunk,
    ScenarioProposal,
)

# ---------------------------------------------------------------------------
# Protocol — structural subtype of AgentRuntime
# ---------------------------------------------------------------------------


class AgentRuntimeProtocol(Protocol):
    """Structural interface matching ``AgentRuntime``."""

    def propose_scenario(
        self,
        *,
        current_state: dict[str, Any],
        trajectory_window: list[dict[str, Any]],
        goals: list[str],
    ) -> tuple[ScenarioProposal, LLMInvocationResult]:
        """Produce a validated scenario proposal."""

    def summarize_window(
        self,
        *,
        ticks: list[dict[str, Any]],
    ) -> tuple[NarrativeChunk, LLMInvocationResult]:
        """Produce a narrative summary grounded in recent ticks."""

    def analyze_window(
        self,
        *,
        ticks: list[dict[str, Any]],
    ) -> tuple[AnalysisReport, LLMInvocationResult]:
        """Produce a structured analysis report."""

    def recommend_intervention(
        self,
        *,
        current_state: dict[str, Any],
        ticks: list[dict[str, Any]],
        goals: list[str],
    ) -> tuple[InterventionRecommendation, LLMInvocationResult]:
        """Produce a validated intervention recommendation."""


# ---------------------------------------------------------------------------
# Agent wrapper functions
# ---------------------------------------------------------------------------


def run_scenario_agent(
    runtime: AgentRuntimeProtocol,
    *,
    current_state: dict[str, Any],
    trajectory_window: list[dict[str, Any]],
    goals: list[str],
) -> dict[str, Any]:
    """Propose a scenario via the ScenarioAgent."""
    proposal, _meta = runtime.propose_scenario(
        current_state=current_state,
        trajectory_window=trajectory_window,
        goals=goals,
    )
    return {
        "action_type": "scenario",
        "output": proposal.model_dump(),
        "rationale": f"Proposed scenario: {proposal.name}",
    }


def run_observer_agent(
    runtime: AgentRuntimeProtocol,
    *,
    ticks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize recent trajectory via the ObserverAgent."""
    narrative, _meta = runtime.summarize_window(ticks=ticks)
    return {
        "action_type": "observe",
        "output": narrative.model_dump(),
        "rationale": f"Observation: {narrative.summary[:100]}",
    }


def run_analyst_agent(
    runtime: AgentRuntimeProtocol,
    *,
    ticks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze trajectory window via the AffectAnalystAgent."""
    report, _meta = runtime.analyze_window(ticks=ticks)
    return {
        "action_type": "analyze",
        "output": report.model_dump(),
        "rationale": f"Analysis: regime={report.dominant_regime}",
    }


def run_intervention_agent(
    runtime: AgentRuntimeProtocol,
    *,
    current_state: dict[str, Any],
    ticks: list[dict[str, Any]],
    goals: list[str],
) -> dict[str, Any]:
    """Recommend intervention via the InterventionAgent."""
    rec, _meta = runtime.recommend_intervention(
        current_state=current_state,
        ticks=ticks,
        goals=goals,
    )
    return {
        "action_type": "intervene",
        "output": rec.model_dump(),
        "rationale": f"Intervention: {rec.action} — {rec.reason}",
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, Any] = {
    "scenario": run_scenario_agent,
    "observe": run_observer_agent,
    "analyze": run_analyst_agent,
    "intervene": run_intervention_agent,
}
