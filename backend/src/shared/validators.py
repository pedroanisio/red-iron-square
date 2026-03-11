"""Reusable validation functions for bounded numeric ranges."""

import numpy as np


def validate_unit_interval(name: str, value: float) -> None:
    """Raise ValueError if value is not in [0, 1]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name}={value} is outside the required [0, 1] interval.")


def validate_real(name: str, value: float) -> None:
    """Raise ValueError if value is not finite."""
    if not np.isfinite(value):
        raise ValueError(f"{name}={value} is not finite.")
