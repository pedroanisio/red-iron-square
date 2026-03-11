"""Tunable hyperparameters for the precision computation."""

from pydantic import BaseModel, model_validator

from src.shared.validators import validate_real


class PrecisionParams(BaseModel):
    """Hyperparameters controlling precision generation.

    Weight scale factors are seeded from existing handcrafted coefficients
    in ``StateTransitionParams`` so that precision trajectories reproduce
    known personality-dependent patterns from the start.
    """

    # Dimension counts
    n_personality: int = 8
    n_state: int = 5
    n_context: int = 8
    n_interoceptive: int = 5

    # Default bias: softplus(0.54) ≈ 1.0
    default_bias: float = 0.54

    # Supervised init weights — Level 0 (interoceptive)
    n_mood_precision_weight: float = 0.5
    e_arousal_precision_weight: float = 0.3
    r_frustration_precision_weight: float = 0.4
    n_satisfaction_precision_weight: float = 0.2
    energy_precision_base: float = 0.1

    # Supervised init weights — Level 1 (policy)
    e_policy_precision_weight: float = 0.4

    # Supervised init weights — Level 2 (narrative)
    t_narrative_precision_weight: float = 0.4

    @model_validator(mode="after")
    def _check_finite(self) -> "PrecisionParams":
        """Ensure all float fields are finite."""
        for name, val in self.model_dump().items():
            if isinstance(val, float):
                validate_real(name, val)
        return self
