"""Domain models for orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.api.run_models import utc_now


@dataclass(frozen=True)
class OrchestrationContext:
    """Input context for one orchestration cycle."""

    run_id: str
    campaign_id: str | None
    cycle: int
    goals: list[str]
    recent_ticks: list[dict[str, Any]]
    latest_state: dict[str, Any] | None
    run_status: str


@dataclass(frozen=True)
class OrchestratorDecision:
    """One persisted orchestrator decision."""

    run_id: str
    campaign_id: str | None
    cycle: int
    action_type: str  # scenario, observe, analyze, intervene, pause, terminate
    input_json: dict[str, Any]
    output_json: dict[str, Any]
    rationale: str
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class AgentResult:
    """Output from one agent invocation within orchestration."""

    action_type: str
    output: dict[str, Any]
    rationale: str
