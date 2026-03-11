"""Self-evidencing precision modulator (L2 -> L1 feedback).

Implements: Pi_1,pi^self = Pi_1^base * min(Pi_max, exp(-beta * d(pi, psi_hat)))
with policy normalization (Mechanism C) to conserve total precision budget.

References:
    Friston (2024), National Science Review, 11(5).
    Fisher et al. (2024), Entropy, 26(6):518.
    Laukkonen, Friston & Chandaria (2025), Neuroscience and Biobehavioral Reviews.
"""

from __future__ import annotations

import numpy as np

from src.self_evidencing.params import SelfEvidencingParams
from src.shared.logging import get_logger

_log = get_logger(module="self_evidencing.modulator")


class SelfEvidencingModulator:
    """Modulates per-action policy precision via self-model predictions.

    Uses mechanisms A (precision cap) and C (policy normalization)
    from section 5.1. Mechanism B (temporal decay) runs each tick.
    """

    def __init__(
        self,
        params: SelfEvidencingParams | None = None,
    ) -> None:
        self._params = params or SelfEvidencingParams()
        self._beta = self._params.beta_0

    @property
    def beta(self) -> float:
        """Current self-evidencing strength."""
        return self._beta

    def reset_beta(self, value: float | None = None) -> None:
        """Reset beta (e.g. after System 2 narrative refresh)."""
        self._beta = value if value is not None else self._params.beta_0

    def compute_precision_weights(
        self,
        predicted_probs: np.ndarray,
        base_precision: float,
    ) -> np.ndarray:
        """Compute per-action precision with self-evidencing modulation.

        Args:
            predicted_probs: Self-model's predicted action distribution.
            base_precision: Base policy precision (Pi_1^base from PrecisionState).

        Returns:
            Per-action precision array of same length as predicted_probs.
        """
        divergences = self._action_divergences(predicted_probs)
        raw_boosts = np.exp(-self._beta * divergences)

        capped = np.minimum(raw_boosts, self._params.pi_max)

        normalized = self._normalize_boosts(capped)

        result: np.ndarray = base_precision * normalized
        return result

    def decay_beta(self, personality_t: float = 0.5) -> None:
        """Apply temporal decay (Mechanism B).

        beta(t+1) = beta_min + (beta(t) - beta_min) * lambda_beta
        """
        p = self._params
        beta_min = personality_t * p.t_beta_scale * p.beta_0
        self._beta = beta_min + (self._beta - beta_min) * p.lambda_beta

    @staticmethod
    def _action_divergences(predicted_probs: np.ndarray) -> np.ndarray:
        """d(pi, psi_hat) = -log(p_hat_pi) for each action.

        Low predicted probability -> high divergence from self-model.
        """
        eps = 1e-10
        safe_probs = np.maximum(predicted_probs, eps)
        result: np.ndarray = -np.log(safe_probs)
        return result

    @staticmethod
    def _normalize_boosts(boosts: np.ndarray) -> np.ndarray:
        """Mechanism C: policy-normalized precision (conserves budget).

        norm_i = boost_i / mean(boosts)
        """
        mean_boost = np.mean(boosts)
        if mean_boost < 1e-10:
            return np.ones_like(boosts)
        result: np.ndarray = boosts / mean_boost
        return result
