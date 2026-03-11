"""Hyperparameters for Expected Free Energy computation."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from src.shared.validators import validate_real


class EFEParams(BaseModel):
    """Tunable parameters for the EFE decision engine.

    Controls C-vector shaping (kappa values), epistemic-pragmatic
    balance (w_base), and outcome prediction.
    """

    w_base: float = 0.5
    """Base weight for epistemic value. Scaled by Openness: w_O = O * w_base."""

    c_pragmatic_scale: float = 1.5
    """Pragmatic value scaling by Conscientiousness. High-C -> exploit more."""

    kappa_mood: float = 1.0
    """Base preference gradient for positive mood changes."""

    kappa_arousal: float = 1.0
    """Sharpness of arousal preference around E-dependent center."""

    kappa_energy: float = 0.8
    """Base preference gradient for energy stability/increase."""

    kappa_satisfaction: float = 1.0
    """Base preference gradient for satisfaction increase."""

    kappa_frustration: float = 1.2
    """Aversion gradient for frustration increase."""

    c_stability_bonus: float = 0.5
    """Bonus for 'stable' energy bin for high-C agents."""

    bin_sharpness: float = 3.0
    """Sharpness of soft-binning when mapping predictions to bins."""

    memory_window: int = 50
    """Window size for epistemic value (outcome variance) lookup."""

    n_bins: int = 5
    """Number of discretization bins per interoceptive dimension."""

    default_epistemic: float = 1.0
    """Default epistemic value when no memory exists for an action."""

    oc_temperature_scale: float = 2.0
    """Exponential temperature scaling by O - C difference."""

    @model_validator(mode="after")
    def _check_finite(self) -> EFEParams:
        for name, val in self.model_dump().items():
            if isinstance(val, float):
                validate_real(name, val)
        return self
