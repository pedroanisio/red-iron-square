"""Structural protocols for cross-boundary duck typing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

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


class System2RuntimeProtocol(Protocol):
    """Structural interface for System 2 LLM integration.

    Satisfied by ``AgentRuntime`` without coupling the simulation
    layer to the full LLM adapter surface.
    """

    def propose_matrices(
        self,
        *,
        personality: dict[str, float],
        trajectory_window: list[dict[str, Any]],
        n_states: int,
        n_actions: int,
    ) -> tuple[Any, Any]:
        """Propose A/B matrices for narrative generative model."""
        ...


class ActionEncoderProtocol(Protocol):
    """Structural interface for action-to-modifier encoding."""

    def encode_modifiers(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Return estimated personality-dimension modifiers in [-1, 1]."""
        ...
