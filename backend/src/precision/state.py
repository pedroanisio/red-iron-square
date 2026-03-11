"""Precision state and prediction error models."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from src.precision.setpoints import INTEROCEPTIVE_KEYS


class PrecisionState(BaseModel):
    """Precision values at all three hierarchy levels.

    Level 0: 5D vector — one per interoceptive variable.
    Level 1: scalar  — policy precision gamma.
    Level 2: scalar  — narrative precision.

    All values must be strictly positive (precision = inverse variance).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    level_0: np.ndarray
    level_1: float
    level_2: float

    @model_validator(mode="after")
    def _validate_positive(self) -> PrecisionState:
        """Ensure all precision values are positive."""
        if self.level_0.shape != (len(INTEROCEPTIVE_KEYS),):
            raise ValueError(
                f"level_0 must have shape ({len(INTEROCEPTIVE_KEYS)},), "
                f"got {self.level_0.shape}"
            )
        if np.any(self.level_0 <= 0):
            raise ValueError(f"level_0 must be positive, got {self.level_0}")
        if self.level_1 <= 0:
            raise ValueError(f"level_1 must be positive, got {self.level_1}")
        if self.level_2 <= 0:
            raise ValueError(f"level_2 must be positive, got {self.level_2}")
        return self


class PredictionErrors(BaseModel):
    """Level-0 allostatic prediction errors.

    Each element is ``s_i(t) - s_hat_i(theta)`` where ``s_hat`` is the
    personality-dependent allostatic set-point.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    level_0: np.ndarray

    @model_validator(mode="after")
    def _validate_shape(self) -> PredictionErrors:
        """Ensure correct shape."""
        if self.level_0.shape != (len(INTEROCEPTIVE_KEYS),):
            raise ValueError(
                f"level_0 must have shape ({len(INTEROCEPTIVE_KEYS)},), "
                f"got {self.level_0.shape}"
            )
        return self


class PrecisionSnapshot(BaseModel):
    """JSON-serializable snapshot of precision state."""

    level_0: dict[str, float]
    level_1: float
    level_2: float

    @classmethod
    def from_state(cls, state: PrecisionState) -> PrecisionSnapshot:
        """Create snapshot from internal state."""
        return cls(
            level_0={
                k: float(state.level_0[i]) for i, k in enumerate(INTEROCEPTIVE_KEYS)
            },
            level_1=state.level_1,
            level_2=state.level_2,
        )


class PredictionErrorSnapshot(BaseModel):
    """JSON-serializable snapshot of prediction errors."""

    level_0: dict[str, float]

    @classmethod
    def from_errors(cls, errors: PredictionErrors) -> PredictionErrorSnapshot:
        """Create snapshot from internal errors."""
        return cls(
            level_0={
                k: float(errors.level_0[i]) for i, k in enumerate(INTEROCEPTIVE_KEYS)
            },
        )
