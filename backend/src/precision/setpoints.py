"""Allostatic set-points derived from personality."""

import numpy as np

from src.personality.vectors import PersonalityVector

INTEROCEPTIVE_KEYS: tuple[str, ...] = (
    "mood",
    "arousal",
    "energy",
    "satisfaction",
    "frustration",
)


class AllostaticSetPoints:
    """Compute allostatic set-points from personality.

    Set-points define the predicted resting value for each interoceptive
    variable.  Deviations from set-points produce prediction errors.

    Only arousal has a personality-dependent set-point (via Extraversion).
    The remaining variables use fixed defaults matching the current
    ``StateTransitionParams`` values.
    """

    _MOOD_SP: float = 0.0
    _ENERGY_SP: float = 0.80
    _SATISFACTION_SP: float = 0.5
    _FRUSTRATION_SP: float = 0.0
    _AROUSAL_BASE: float = 0.4
    _AROUSAL_E_WEIGHT: float = 0.15

    def compute(self, personality: PersonalityVector) -> np.ndarray:
        """Return 5D set-point vector aligned with ``INTEROCEPTIVE_KEYS``."""
        keys = set(personality.registry.keys)
        e_val = personality["E"] if "E" in keys else 0.5
        arousal_sp = self._AROUSAL_BASE + self._AROUSAL_E_WEIGHT * e_val
        return np.array(
            [
                self._MOOD_SP,
                arousal_sp,
                self._ENERGY_SP,
                self._SATISFACTION_SP,
                self._FRUSTRATION_SP,
            ]
        )
