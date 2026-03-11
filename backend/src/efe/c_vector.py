"""C-vector: prior preference distributions derived from personality.

Implements the factored C-vector from the PHP architecture (section 2.5).
Each interoceptive dimension has an independent categorical distribution
over discretized outcome bins, shaped by personality traits.
"""

from __future__ import annotations

import numpy as np

from src.efe.params import EFEParams
from src.personality.vectors import PersonalityVector

INTEROCEPTIVE_KEYS = ("mood", "arousal", "energy", "satisfaction", "frustration")
N_INTEROCEPTIVE = len(INTEROCEPTIVE_KEYS)


class CVector:
    """Prior preference distributions over 5D interoceptive outcomes.

    Each dimension is discretized into ``n_bins`` bins representing
    outcome quality from very negative to very positive.
    The C-vector is factored as 5 independent categorical distributions.

    Personality traits shape preferences:
        - N, I -> mood preferences (asymmetric aversion)
        - E -> arousal preferences (preferred arousal level)
        - C -> energy preferences (stability bonus)
        - A -> satisfaction preferences (gradient strength)
        - R -> frustration preferences (tolerance)
    """

    def __init__(
        self,
        personality: PersonalityVector,
        params: EFEParams | None = None,
    ) -> None:
        self._params = params or EFEParams()
        self._n_bins = self._params.n_bins
        self._log_prefs = self._build_log_preferences(personality)

    @property
    def log_preferences(self) -> np.ndarray:
        """Log-probability matrix of shape (5, n_bins)."""
        return self._log_prefs.copy()

    @property
    def preferences(self) -> np.ndarray:
        """Probability matrix of shape (5, n_bins)."""
        result: np.ndarray = np.exp(self._log_prefs)
        return result

    def _build_log_preferences(self, psi: PersonalityVector) -> np.ndarray:
        """Construct log-preference matrix from personality."""
        keys = set(psi.registry.keys)
        N = psi["N"] if "N" in keys else 0.5
        E = psi["E"] if "E" in keys else 0.5
        C = psi["C"] if "C" in keys else 0.5
        A = psi["A"] if "A" in keys else 0.5
        R = psi["R"] if "R" in keys else 0.5
        I = psi["I"] if "I" in keys else 0.5  # noqa: E741

        p = self._params
        log_prefs = np.zeros((N_INTEROCEPTIVE, self._n_bins))

        log_prefs[0] = self._mood_preferences(N, I, p)
        log_prefs[1] = self._arousal_preferences(E, p)
        log_prefs[2] = self._energy_preferences(C, p)
        log_prefs[3] = self._satisfaction_preferences(A, p)
        log_prefs[4] = self._frustration_preferences(R, p)

        return self._normalize_log(log_prefs)

    def _mood_preferences(self, N: float, idealism: float, p: EFEParams) -> np.ndarray:
        """C_mood(k): asymmetric — high N penalizes negative bins more."""
        k = np.arange(self._n_bins, dtype=float)
        k0 = (self._n_bins - 1) / 2.0
        kappa = p.kappa_mood + 0.3 * idealism
        negative_penalty = N * np.maximum(k0 - k, 0.0)
        return kappa * (k - k0) - negative_penalty

    def _arousal_preferences(self, E: float, p: EFEParams) -> np.ndarray:
        """C_arousal(k) proportional to exp(-kappa * (k - k*_E)^2)."""
        k = np.arange(self._n_bins, dtype=float)
        k_star = 1.0 + 2.0 * E
        return -p.kappa_arousal * (k - k_star) ** 2

    def _energy_preferences(self, C: float, p: EFEParams) -> np.ndarray:
        """C_energy(k) proportional to exp(kappa * (k - mid) + C * 1[k == mid])."""
        k = np.arange(self._n_bins, dtype=float)
        mid = (self._n_bins - 1) / 2.0
        log_unnorm = p.kappa_energy * (k - mid)
        stable_idx = self._n_bins // 2
        log_unnorm[stable_idx] += p.c_stability_bonus * C
        return log_unnorm

    def _satisfaction_preferences(self, A: float, p: EFEParams) -> np.ndarray:
        """C_satisfaction(k) proportional to exp((kappa + A) * (k - mid))."""
        k = np.arange(self._n_bins, dtype=float)
        mid = (self._n_bins - 1) / 2.0
        return (p.kappa_satisfaction + A) * (k - mid)

    def _frustration_preferences(self, R: float, p: EFEParams) -> np.ndarray:
        """C_frustration(k) proportional to exp(-kappa * (1 - R) * k)."""
        k = np.arange(self._n_bins, dtype=float)
        return -p.kappa_frustration * (1.0 - R) * k

    @staticmethod
    def _normalize_log(log_unnorm: np.ndarray) -> np.ndarray:
        """Row-wise log-softmax normalization."""
        log_max = log_unnorm.max(axis=1, keepdims=True)
        shifted = log_unnorm - log_max
        log_sum = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        result: np.ndarray = shifted - log_sum
        return result
