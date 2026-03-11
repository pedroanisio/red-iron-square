"""Typed models for the Flask UI layer.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import TypedDict


class RunListItem(TypedDict):
    """A run as returned by list_runs."""

    run_id: str
    mode: str
    status: str
    tick_count: int
    updated_at: str


class StateSnapshot(TypedDict, total=False):
    """Internal agent state at a point in time."""

    mood: float
    arousal: float
    energy: float
    satisfaction: float
    frustration: float


class EmotionReading(TypedDict):
    """One emotion measurement."""

    label: str
    intensity: float
    description: str


class TickData(TypedDict, total=False):
    """One tick event from the trajectory."""

    tick: int
    action: str
    outcome: float
    state_before: StateSnapshot
    state_after: StateSnapshot
    emotions: list[EmotionReading]
    scenario: dict[str, object]
    probabilities: list[float]
    activations: dict[str, float]
    # Self-aware fields
    identity_drift: float
    self_coherence: float
    self_accuracy: float


class AgentInvocation(TypedDict, total=False):
    """One agent call record."""

    agent_name: str
    purpose: str
    metadata: dict[str, object]
    output: dict[str, object]
    created_at: str


class InterventionDecision(TypedDict, total=False):
    """One intervention decision record."""

    action: str
    reason: str
    payload: dict[str, object]
    applied: bool
    created_at: str


class PhaseAnnotation(TypedDict, total=False):
    """One phase annotation."""

    start_tick: int
    end_tick: int | None
    label: str
    notes: str


class LatestTick(TypedDict, total=False):
    """The most recent tick summary in RunSummary."""

    tick: int
    action: str
    outcome: float
    state_after: StateSnapshot
    emotions: list[EmotionReading]


class RunConfig(TypedDict, total=False):
    """Run configuration."""

    personality: dict[str, float]
    actions: list[dict[str, object]]
    temperature: float
    seed: int | None
    self_model: dict[str, float] | None


class RunSummary(TypedDict, total=False):
    """Full run summary as returned by get_run."""

    run_id: str
    mode: str
    status: str
    tick_count: int
    config: RunConfig
    parent_run_id: str | None
    parent_tick: int | None
    latest_tick: LatestTick | None
    phases: list[PhaseAnnotation]
    agent_invocation_count: int
    intervention_count: int
    created_at: str
    updated_at: str


class TrajectoryData(TypedDict):
    """Full trajectory as returned by get_trajectory."""

    run_id: str
    tick_count: int
    ticks: list[TickData]
    phases: list[PhaseAnnotation]
    agent_invocations: list[AgentInvocation]
    interventions: list[InterventionDecision]


class ReplayResult(TypedDict):
    """Result of replay_run."""

    run: RunListItem


class BranchResult(TypedDict):
    """Result of branch_run."""

    run: RunListItem
