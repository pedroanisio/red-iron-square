"""Mutable agent state and state transition logic."""

import numpy as np
from pydantic import BaseModel, model_validator

from src.personality.vectors import PersonalityVector, Scenario
from src.shared.validators import validate_real, validate_unit_interval


class AgentState(BaseModel):
    """
    Mutable internal state that evolves across simulation ticks.

    mood:         Valence axis.  -1 = deeply negative, +1 = elated.
    arousal:      Activation axis.  0 = sluggish, 1 = wired.
    energy:       Cognitive/physical resource.  0 = depleted, 1 = full.
    satisfaction: Recent need fulfillment.  0 = deprived, 1 = satiated.
    frustration:  Accumulated blocked-goal signal.  0 = none, 1 = max.
    """

    mood: float = 0.0
    arousal: float = 0.5
    energy: float = 1.0
    satisfaction: float = 0.5
    frustration: float = 0.0

    @model_validator(mode="after")
    def _validate_ranges(self) -> "AgentState":
        if not (-1.0 <= self.mood <= 1.0):
            raise ValueError(f"mood={self.mood} outside [-1, 1]")
        for name in ("arousal", "energy", "satisfaction", "frustration"):
            validate_unit_interval(name, getattr(self, name))
        return self

    def to_array(self) -> np.ndarray:
        """Return state as [mood, arousal, energy, satisfaction, frustration]."""
        return np.array([self.mood, self.arousal, self.energy,
                         self.satisfaction, self.frustration])

    def snapshot(self) -> "AgentState":
        """Return an independent copy of the current state."""
        return AgentState(
            mood=self.mood, arousal=self.arousal, energy=self.energy,
            satisfaction=self.satisfaction, frustration=self.frustration,
        )

    def __repr__(self) -> str:
        return (f"State(mood={self.mood:+.2f}, arousal={self.arousal:.2f}, "
                f"energy={self.energy:.2f}, sat={self.satisfaction:.2f}, "
                f"frust={self.frustration:.2f})")


class StateTransitionParams(BaseModel):
    """Tunable parameters for how AgentState evolves between ticks."""

    mood_decay: float = 0.92
    arousal_decay: float = 0.88
    energy_decay: float = 0.90
    satisfaction_decay: float = 0.90
    frustration_decay: float = 0.85

    outcome_mood_gain: float = 0.25
    outcome_mood_loss: float = 0.35
    outcome_arousal_spike: float = 0.15
    outcome_satisfaction: float = 0.20
    outcome_frustration: float = 0.25

    energy_cost_per_effort: float = 0.015
    energy_cost_stress: float = 0.03
    energy_resting_level: float = 0.80

    N_mood_sensitivity: float = 0.5
    R_frustration_damping: float = 0.4
    E_arousal_baseline: float = 0.15

    @model_validator(mode="after")
    def _check_finite(self) -> "StateTransitionParams":
        for name, val in self.model_dump().items():
            validate_real(name, val)
        return self


def update_state(
    state: AgentState,
    outcome: float,
    personality: PersonalityVector,
    scenario: Scenario,
    params: StateTransitionParams = StateTransitionParams(),
    action_effort: float = 0.0,
) -> AgentState:
    """Evolve agent state based on tick outcome, personality, and scenario."""
    p = params
    keys = set(personality.registry.keys)
    N = personality["N"] if "N" in keys else 0.5
    R = personality["R"] if "R" in keys else 0.5
    E = personality["E"] if "E" in keys else 0.5
    stress = scenario["N"] if "N" in keys else 0.5

    mood = state.mood * p.mood_decay
    if outcome > 0:
        mood += outcome * p.outcome_mood_gain
    else:
        mood += outcome * p.outcome_mood_loss * (1.0 + p.N_mood_sensitivity * N)

    resting_arousal = 0.4 + p.E_arousal_baseline * E
    arousal = state.arousal * p.arousal_decay + (1 - p.arousal_decay) * resting_arousal
    arousal += abs(outcome) * p.outcome_arousal_spike

    resting_energy = p.energy_resting_level
    energy = state.energy * p.energy_decay + (1 - p.energy_decay) * resting_energy
    energy -= p.energy_cost_per_effort * action_effort + p.energy_cost_stress * stress

    satisfaction = state.satisfaction * p.satisfaction_decay + (1 - p.satisfaction_decay) * 0.5
    if outcome > 0:
        satisfaction += outcome * p.outcome_satisfaction

    frustration = state.frustration * p.frustration_decay
    if outcome < 0:
        damping = 1.0 - p.R_frustration_damping * R
        frustration += abs(outcome) * p.outcome_frustration * damping

    return AgentState(
        mood=float(np.clip(mood, -1.0, 1.0)),
        arousal=float(np.clip(arousal, 0.0, 1.0)),
        energy=float(np.clip(energy, 0.0, 1.0)),
        satisfaction=float(np.clip(satisfaction, 0.0, 1.0)),
        frustration=float(np.clip(frustration, 0.0, 1.0)),
    )
