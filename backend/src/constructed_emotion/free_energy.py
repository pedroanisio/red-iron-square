"""Total variational free energy computation.

F_total = sum_i (Pi_0,i * epsilon_0,i^2 - ln Pi_0,i)

References:
    Feldman & Friston (2010), Frontiers in Human Neuroscience.
    Hesp et al. (2021), Neural Computation, 33(2):398-446.
"""

from __future__ import annotations

import numpy as np

from src.precision.state import PrecisionState, PredictionErrors


def compute_free_energy(
    precision: PrecisionState,
    errors: PredictionErrors,
) -> float:
    """Compute total variational free energy across interoceptive channels.

    F = sum_i (Pi_i * eps_i^2 - ln Pi_i)

    The -ln(Pi) term is the Occam factor preventing trivial minimization
    by setting all precisions to zero.
    """
    pi_0 = precision.level_0
    eps_0 = errors.level_0
    weighted_errors = pi_0 * eps_0**2
    occam_term = np.log(pi_0)
    result: float = float(np.sum(weighted_errors - occam_term))
    return result


def compute_valence(f_prev: float, f_curr: float) -> float:
    """Valence as rate of free energy decrease.

    valence(t) = F(t-1) - F(t)

    Positive = free energy decreasing (things improving).
    Negative = free energy increasing (things worsening).
    """
    return f_prev - f_curr


def compute_arousal_signal(
    precision: PrecisionState,
    errors: PredictionErrors,
) -> float:
    """Arousal as norm of precision-weighted prediction errors.

    arousal_signal(t) = ||Pi_0 * epsilon_0||
    """
    weighted: np.ndarray = precision.level_0 * errors.level_0
    return float(np.linalg.norm(weighted))
