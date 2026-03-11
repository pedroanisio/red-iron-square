"""Hyperparameters for constructed emotion computation."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from src.shared.validators import validate_real


class ConstructedEmotionParams(BaseModel):
    """Tunable parameters for the constructed affective engine.

    Controls free energy computation, surprise spike detection,
    and mood EMA dynamics per Barrett (2017) and Hesp et al. (2021).
    """

    mood_ema_alpha: float = 0.90
    """Base EMA coefficient for mood. High R lowers alpha (faster recovery)."""

    r_alpha_scale: float = 0.08
    """How much Resilience trait reduces mood EMA alpha."""

    surprise_window: int = 50
    """Rolling window for adaptive surprise threshold (mu + 2*sigma)."""

    surprise_n_sigma: float = 2.0
    """Number of standard deviations above mean for surprise spike."""

    surprise_sigma_min: float = 0.05
    """Floor on sigma to prevent trivial triggers in monotonous periods."""

    surprise_warmup_threshold: float = 0.20
    """Fixed surprise threshold during warmup (first `surprise_window` ticks)."""

    @model_validator(mode="after")
    def _check_finite(self) -> ConstructedEmotionParams:
        for name, val in self.model_dump().items():
            if isinstance(val, float):
                validate_real(name, val)
        return self
