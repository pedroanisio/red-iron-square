"""Activation functions mapping (stimulus, trait) -> [0, 1]."""

from typing import Callable

import numpy as np

from src.personality.hyperparameters import HyperParameters, ResilienceMode


class ActivationFunctions:
    """
    Activation functions f_i : [0,1] x [0,1] -> [0,1].

    Every function maps (stimulus, trait) to a value in [0, 1].
    Downstream code depends on this guarantee.

    Centering note: bipolar traits (E, C) use t_centered = 2*trait - 1
    so both poles are active.  Magnitude traits (O) scale directly.
    """

    @staticmethod
    def f_openness(stimulus: float, trait: float, hp: HyperParameters) -> float:
        """f_O(s, O) = O * tanh(alpha * s).  Magnitude scaling, saturating."""
        return trait * np.tanh(hp.alpha * stimulus)

    @staticmethod
    def f_conscientiousness(stimulus: float, trait: float, hp: HyperParameters) -> float:
        """f_C(s, C) = sigma(beta * (2C-1) * (s - theta)).  Bipolar sigmoid."""
        t_centered = 2.0 * trait - 1.0
        return 1.0 / (1.0 + np.exp(-hp.beta * t_centered * (stimulus - hp.c_threshold)))

    @staticmethod
    def f_extraversion(stimulus: float, trait: float, hp: HyperParameters) -> float:
        """f_E(s, E) = sigma(gamma * (2E-1) * (s - 0.5)).  Bipolar sigmoid."""
        t_centered = 2.0 * trait - 1.0
        return 1.0 / (1.0 + np.exp(-hp.gamma * t_centered * (stimulus - 0.5)))

    @staticmethod
    def _f_linear_interpolation(stimulus: float, trait: float, _hp: HyperParameters) -> float:
        """
        f(s, T) = T*s + (1-T)*(1-s).

        Shared implementation for agreeableness, idealism, and tradition.
        High trait prefers high stimulus; low trait prefers low stimulus.
        """
        return trait * stimulus + (1.0 - trait) * (1.0 - stimulus)

    f_agreeableness = _f_linear_interpolation
    f_idealism = _f_linear_interpolation
    f_tradition = _f_linear_interpolation

    @staticmethod
    def f_neuroticism(stimulus: float, trait: float, hp: HyperParameters) -> float:
        """f_N(s, N) = exp(-delta * N * s^2).  Gaussian decay under stress."""
        return np.exp(-hp.delta * trait * stimulus ** 2)

    @staticmethod
    def f_resilience(
        stimulus: float,
        trait: float,
        hp: HyperParameters,
        mode: ResilienceMode = ResilienceMode.ACTIVATION,
    ) -> float:
        """
        ACTIVATION: f_R = R * (1 - exp(-rho * s)).  Adversity mobilizes.
        BUFFER:     f_R = 1 - s * (1 - R).  Resilience absorbs penalty.
        """
        if mode == ResilienceMode.ACTIVATION:
            return trait * (1.0 - np.exp(-hp.rho * stimulus))
        return 1.0 - stimulus * (1.0 - trait)


DEFAULT_ACTIVATION_REGISTRY: dict[str, Callable[..., float]] = {
    "O": ActivationFunctions.f_openness,
    "C": ActivationFunctions.f_conscientiousness,
    "E": ActivationFunctions.f_extraversion,
    "A": ActivationFunctions.f_agreeableness,
    "N": ActivationFunctions.f_neuroticism,
    "R": ActivationFunctions.f_resilience,
    "I": ActivationFunctions.f_idealism,
    "T": ActivationFunctions.f_tradition,
}
