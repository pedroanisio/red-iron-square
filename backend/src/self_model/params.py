"""Tunable parameters for self-model dynamics."""

from pydantic import BaseModel, model_validator

from src.shared.validators import validate_real


class SelfModelParams(BaseModel):
    """Tunable parameters for the self-model dynamics.

    Controls behavioral evidence accumulation, self-model update rates,
    emotion thresholds, and personality modulation.
    """

    evidence_memory: float = 0.85
    sigmoid_scale: float = 2.0
    learning_rate: float = 0.08
    identity_inertia: float = 0.04
    coherence_threat_threshold: float = 0.20
    coherence_threat_window: int = 6
    drift_crisis_threshold: float = 0.35
    N_shame_amplification: float = 0.5
    C_prediction_sharpening: float = 0.3
    shame_scaling: float = 3.0

    @model_validator(mode="after")
    def _check_finite(self) -> "SelfModelParams":
        for name, val in self.model_dump().items():
            if isinstance(val, float):
                validate_real(name, val)
        return self
