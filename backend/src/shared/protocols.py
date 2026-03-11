"""Structural protocols for cross-boundary duck typing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from src.personality.dimensions import DimensionRegistry
from src.personality.hyperparameters import HyperParameters, ResilienceMode
from src.personality.vectors import Action, PersonalityVector, Scenario


class DecisionEngineProtocol(Protocol):
    """Structural interface for decision engines.

    Both ``DecisionEngine`` and ``EFEEngine`` satisfy this protocol,
    allowing the temporal simulator to accept either without coupling
    to a concrete implementation.
    """

    registry: DimensionRegistry
    hp: HyperParameters
    resilience_mode: ResilienceMode
    activations: dict[str, Callable[..., float]]

    def compute_activations(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
    ) -> np.ndarray:
        """Compute N-dimensional activation vector."""
        ...

    def utility(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        action: Action,
        bias: float = 0.0,
        activations_override: np.ndarray | None = None,
    ) -> float:
        """Compute scalar utility for an action."""
        ...

    def decide(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        actions: Sequence[Action],
        temperature: float = 1.0,
        bias: float = 0.0,
        rng: np.random.Generator | None = None,
        activations_override: np.ndarray | None = None,
    ) -> tuple[Action, np.ndarray]:
        """Select an action and return probabilities."""
        ...
