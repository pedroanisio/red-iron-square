"""Public SDK result models and configuration types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DecisionResult(BaseModel):
    """JSON-safe result for one-shot action selection."""

    chosen_action: str
    probabilities: dict[str, float]
    utilities: dict[str, float]
    activations: dict[str, float]
    action_order: list[str]


class TickRecord(BaseModel):
    """JSON-safe temporal simulation record."""

    tick: int
    scenario: dict[str, Any]
    action: str
    outcome: float
    state_before: dict[str, float]
    state_after: dict[str, float]
    activations: dict[str, float]
    emotions: list[dict[str, Any]]
    probabilities: list[float]
    precision: dict[str, Any] | None = None
    prediction_errors: dict[str, Any] | None = None


class SimulationTrace(BaseModel):
    """JSON-safe sequence of temporal simulation ticks."""

    ticks: list[TickRecord]


class SelfAwareTickRecord(TickRecord):
    """JSON-safe self-aware simulation record."""

    self_emotions: list[dict[str, Any]]
    psi_hat: dict[str, float]
    behavioral_evidence: dict[str, float]
    self_coherence: float
    self_accuracy: float
    identity_drift: float
    prediction_error: float
    predicted_probabilities: list[float]


class SelfAwareSimulationTrace(BaseModel):
    """JSON-safe sequence of self-aware simulation ticks."""

    ticks: list[SelfAwareTickRecord]
