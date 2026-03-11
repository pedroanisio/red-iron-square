"""Tunable hyperparameters and enums for activation functions."""

from enum import Enum

from pydantic import BaseModel, model_validator

from src.shared.validators import validate_real


class HyperParameters(BaseModel):
    """
    Tunable constants for the activation functions.

    Separated from the functions so they can be configured per simulation
    run, per personality, or per scenario.
    """

    alpha: float = 3.0
    beta: float = 5.0
    c_threshold: float = 0.3
    gamma: float = 4.0
    delta: float = 3.0
    rho: float = 4.0

    @model_validator(mode="after")
    def _check_finite(self) -> "HyperParameters":
        for name, val in self.model_dump().items():
            validate_real(name, val)
        return self


class ResilienceMode(Enum):
    """
    Controls how resilience activation interprets adversity.

    ACTIVATION: Adversity mobilizes the resilient person (output rises).
    BUFFER:     Adversity is a penalty; resilience reduces it.
    """

    ACTIVATION = "activation"
    BUFFER = "buffer"
