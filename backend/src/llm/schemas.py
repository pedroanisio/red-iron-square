"""Typed schemas for LLM-driven orchestration outputs.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScenarioProposal(BaseModel):
    """Validated scenario proposal from an LLM agent."""

    name: str
    description: str = ""
    values: dict[str, float]
    rationale: str = ""


class NarrativeChunk(BaseModel):
    """Human-facing narrative summary grounded in trajectory data."""

    summary: str
    tick_start: int
    tick_end: int
    evidence: list[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """Structured analysis over a run window."""

    dominant_regime: str
    notable_emotions: list[str] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class InterventionRecommendation(BaseModel):
    """Allowed intervention recommendation from an LLM agent."""

    action: Literal[
        "continue", "probe", "narrate", "analyze", "patch_params", "pause", "terminate"
    ]
    reason: str
    temperature: float | None = None


class LLMInvocationMetadata(BaseModel):
    """Metadata captured for one LLM invocation."""

    model: str
    provider: str = "anthropic"
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMInvocationResult(BaseModel):
    """Structured response plus raw payload metadata."""

    raw_text: str
    metadata: LLMInvocationMetadata


class MatrixProposal(BaseModel):
    """LLM-proposed A/B matrices for narrative generative model (§10).

    A-matrix: observation likelihood p(o|s,a).
    B-matrix: state transitions p(s'|s,a).
    Both must be non-negative and row-normalized.
    """

    a_matrix: list[list[list[float]]]
    b_matrix: list[list[list[float]]]
    rationale: str = ""
    n_states: int = Field(gt=0)
    n_actions: int = Field(gt=0)


class EmotionConstructor(BaseModel):
    """LLM-constructed emotion from precision-weighted prediction errors.

    Constrained by System 1 valence/arousal signals to prevent
    narratively plausible but psychologically inconsistent categorizations.
    """

    label: str
    description: str
    valence_sign: Literal["positive", "negative", "neutral"]
    arousal_level: Literal["high", "low"]
    confidence: float = Field(ge=0.0, le=1.0)
