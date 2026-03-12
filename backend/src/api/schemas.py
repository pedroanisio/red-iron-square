"""Request and response schemas for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScenarioInput(BaseModel):
    """SDK-friendly scenario input payload."""

    values: dict[str, float]
    name: str = ""
    description: str = ""


class ActionInput(BaseModel):
    """SDK-friendly action input payload."""

    name: str
    modifiers: dict[str, float]
    description: str = ""


class DecisionRequest(BaseModel):
    """Request body for one-shot decisions."""

    personality: dict[str, float]
    scenario: ScenarioInput
    actions: list[ActionInput]
    temperature: float = 1.0
    bias: float = 0.0


class SimulationRequest(BaseModel):
    """Request body for temporal or self-aware simulations."""

    personality: dict[str, float]
    actions: list[ActionInput]
    scenarios: list[ScenarioInput]
    outcomes: list[float | None] | None = None
    temperature: float = 1.0
    self_model: dict[str, float] | None = None


class ProposalInput(BaseModel):
    """SDK-friendly action proposal input payload."""

    kind: str
    name: str
    description: str = ""
    modifiers: dict[str, float] | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    intent: str | None = None
    method: str | None = None
    url: str | None = None


class OpenEndedDecisionRequest(BaseModel):
    """Request body for open-ended decisions with action proposals."""

    personality: dict[str, float]
    scenario: ScenarioInput
    proposals: list[ProposalInput]
    temperature: float = 1.0


class ApiEnvelope(BaseModel):
    """Uniform API response wrapper."""

    data: dict[str, Any] = Field(default_factory=dict)
