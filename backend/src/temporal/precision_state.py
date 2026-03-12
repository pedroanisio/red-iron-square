"""Precision-weighted state transitions (§9).

Replaces handcrafted coefficients in update_state() with:
    s(t+1) = decay * s(t) + (1 - decay) * setpoint - gain * Π⁰ * ε⁰ + outcome_drive

Precision Π⁰ scales how strongly prediction errors drive state changes,
so personality indirectly modulates state evolution via precision weights.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, model_validator

from src.personality.vectors import PersonalityVector, Scenario
from src.precision.state import PrecisionState, PredictionErrors
from src.shared.validators import validate_real
from src.temporal.state import AgentState


class PrecisionStateParams(BaseModel):
    """Parameters for precision-weighted state transitions."""

    decay: np.ndarray = np.array([0.92, 0.88, 0.90, 0.90, 0.85])
    """Per-channel decay rates: [mood, arousal, energy, satisfaction, frustration]."""

    setpoint: np.ndarray = np.array([0.0, 0.5, 0.80, 0.5, 0.0])
    """Resting set-points for each state channel."""

    precision_gain: float = 0.05
    """Scaling factor for precision-weighted error correction."""

    outcome_gain: np.ndarray = np.array([0.30, 0.15, 0.0, 0.20, 0.25])
    """Per-channel outcome drive: how outcome pushes each state variable."""

    energy_cost_per_effort: float = 0.015
    """Energy cost per unit of action effort."""

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _check_shapes(self) -> PrecisionStateParams:
        """Validate array shapes and finiteness."""
        for name in ("decay", "setpoint", "outcome_gain"):
            arr = getattr(self, name)
            if arr.shape != (5,):
                raise ValueError(f"{name} must have shape (5,), got {arr.shape}")
            for v in arr:
                validate_real(name, float(v))
        validate_real("precision_gain", self.precision_gain)
        return self


def update_state_precision(
    state: AgentState,
    outcome: float,
    personality: PersonalityVector,
    scenario: Scenario,
    precision: PrecisionState,
    errors: PredictionErrors,
    params: PrecisionStateParams | None = None,
    action_effort: float = 0.0,
) -> AgentState:
    """Evolve state using precision-weighted prediction errors.

    s(t+1) = decay * s(t) + (1-decay) * setpoint
             - gain * Π⁰ * ε⁰
             + outcome_drive * outcome
             - energy_cost
    """
    p = params or PrecisionStateParams()
    s = state.to_array()

    # Decay toward setpoint
    new_s = p.decay * s + (1.0 - p.decay) * p.setpoint

    # Precision-weighted error correction
    # Π⁰ * ε⁰ pushes state to reduce prediction errors
    precision_correction = p.precision_gain * precision.level_0 * errors.level_0
    new_s -= precision_correction

    # Outcome drive (channel-specific sign logic)
    outcome_drive = p.outcome_gain * outcome
    if outcome > 0:
        outcome_drive[4] = 0.0  # no frustration from positive outcomes
    else:
        outcome_drive[0] *= 1.5  # mood more sensitive to negative outcomes
        outcome_drive[3] = 0.0  # no satisfaction loss from negative
        outcome_drive[4] = p.outcome_gain[4] * abs(outcome)  # frustration rises

    new_s += outcome_drive

    # Energy cost
    new_s[2] -= p.energy_cost_per_effort * action_effort

    # Clamp
    new_s[0] = float(np.clip(new_s[0], -1.0, 1.0))
    for i in range(1, 5):
        new_s[i] = float(np.clip(new_s[i], 0.0, 1.0))

    return AgentState(
        mood=float(new_s[0]),
        arousal=float(new_s[1]),
        energy=float(new_s[2]),
        satisfaction=float(new_s[3]),
        frustration=float(new_s[4]),
    )
