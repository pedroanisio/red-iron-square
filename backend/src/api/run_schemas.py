"""Schemas for stateful simulation runs.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.api.schemas import ActionInput, ScenarioInput


class RunConfig(BaseModel):
    """Configuration used to initialize a simulation run."""

    personality: dict[str, float]
    actions: list[ActionInput]
    temperature: float = 1.0
    self_model: dict[str, float] | None = None
    seed: int | None = None


class RunCreateRequest(RunConfig):
    """Request body for creating a run."""


class RunTickRequest(BaseModel):
    """Request body for advancing a run by one tick."""

    scenario: ScenarioInput
    outcome: float | None = None


class RunPatchRequest(BaseModel):
    """Mutable run parameters."""

    temperature: float | None = None


class RunBranchRequest(BaseModel):
    """Request body for creating a branch from an existing run."""

    parent_tick: int | None = None
    temperature: float | None = None


class RunReplayResponse(BaseModel):
    """Response body for replay or branch creation."""

    run: dict[str, Any]


class AssistedStepRequest(BaseModel):
    """Request body for an assisted run step."""

    goals: list[str]
    window: int = 5


class AssistedStepResponse(BaseModel):
    """Response body for one assisted step."""

    scenario: dict[str, Any]
    tick: dict[str, Any]
    narrative: dict[str, Any]
    invocations: list[dict[str, Any]]


class InterventionRequest(BaseModel):
    """Request body for an intervention recommendation."""

    goals: list[str]
    window: int = 10
    apply_patch: bool = False


class InterventionResponse(BaseModel):
    """Response body for an intervention recommendation."""

    recommendation: dict[str, Any]
    invocation: dict[str, Any]
    decision: dict[str, Any]
    updated_run: dict[str, Any] | None = None


class PhaseCreateRequest(BaseModel):
    """Request body for phase annotation creation."""

    start_tick: int
    label: str
    end_tick: int | None = None
    notes: str = ""


class RunSummary(BaseModel):
    """API-facing summary of a persisted run."""

    run_id: str
    mode: str
    status: str
    tick_count: int
    config: dict[str, Any]
    parent_run_id: str | None = None
    parent_tick: int | None = None
    latest_tick: dict[str, Any] | None = None
    phases: list[dict[str, Any]]
    agent_invocation_count: int = 0
    intervention_count: int = 0
    created_at: str
    updated_at: str


class TrajectoryResponse(BaseModel):
    """API-facing trajectory payload."""

    run_id: str
    tick_count: int
    ticks: list[dict[str, Any]]
    phases: list[dict[str, Any]]
    agent_invocations: list[dict[str, Any]]
    interventions: list[dict[str, Any]]
