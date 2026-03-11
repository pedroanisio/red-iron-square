"""Hyperparameters for self-evidencing precision modulation."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from src.shared.validators import validate_real


class SelfEvidencingParams(BaseModel):
    """Tunable parameters for self-evidencing feedback (L2 -> L1).

    Controls how strongly the self-model's predictions modulate
    policy precision via mechanisms A (cap) and C (normalization)
    from section 5.1 of the research doc.
    """

    beta_0: float = 1.0
    """Base self-evidencing strength. Scaled by T trait."""

    pi_max: float = 3.0
    """Precision cap (Mechanism A). Self-model can at most triple base precision."""

    lambda_beta: float = 0.95
    """Temporal decay rate for beta (Mechanism B)."""

    t_beta_scale: float = 1.0
    """How strongly T trait scales beta: beta_min = T * t_beta_scale * beta_0."""

    o_beta_reduction: float = 0.5
    """How strongly O trait reduces beta: high O -> flexible identity."""

    @model_validator(mode="after")
    def _check_finite(self) -> SelfEvidencingParams:
        for name, val in self.model_dump().items():
            if isinstance(val, float):
                validate_real(name, val)
        return self
