"""Data model for a single simulation tick result."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.personality.vectors import Action, Scenario
from src.precision.state import PrecisionState, PredictionErrors
from src.temporal.emotions import EmotionReading
from src.temporal.state import AgentState


class TickResult(BaseModel):
    """Complete result for a single simulation tick."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tick: int
    scenario: Scenario
    action: Action
    outcome: float
    state_before: AgentState
    state_after: AgentState
    activations: np.ndarray
    emotions: list[EmotionReading]
    probabilities: np.ndarray
    precision: PrecisionState | None = None
    prediction_errors: PredictionErrors | None = None
    affect_signal: Any = None
    efe_breakdown: dict[str, dict[str, float]] | None = None
