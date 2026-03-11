"""SDK facade for constructing and running agent simulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.personality.decision import DecisionEngine
from src.personality.dimensions import DEFAULT_DIMENSIONS, Dimension, DimensionRegistry
from src.personality.hyperparameters import HyperParameters, ResilienceMode
from src.personality.vectors import Action, PersonalityVector, Scenario
from src.sdk.builders import (
    build_action,
    build_initial_self_model,
    build_personality,
    build_registry,
    build_scenario,
)
from src.sdk.decision_client import DecisionClient
from src.sdk.self_model_client import SelfModelSimulationClient
from src.sdk.simulation_client import TemporalSimulationClient
from src.sdk.types import (
    DecisionResult,
    SelfAwareSimulationTrace,
    SelfAwareTickRecord,
    SimulationTrace,
    TickRecord,
)


class AgentSDK:
    """Public SDK facade over the domain-level simulation components."""

    def __init__(
        self,
        registry: DimensionRegistry | None = None,
        *,
        hyperparameters: HyperParameters | None = None,
        resilience_mode: ResilienceMode = ResilienceMode.ACTIVATION,
    ) -> None:
        self.registry = registry or build_registry()
        self.engine = DecisionEngine(
            registry=self.registry,
            hyperparameters=hyperparameters or HyperParameters(),
            resilience_mode=resilience_mode,
        )

    @classmethod
    def default(cls) -> AgentSDK:
        """Create the default OCEAN+RIT SDK."""
        return cls()

    @classmethod
    def from_dimensions(cls, dimensions: Sequence[Dimension]) -> AgentSDK:
        """Create an SDK for a custom dimension registry."""
        return cls(registry=build_registry(dimensions))

    def personality(self, values: Mapping[str, float]) -> PersonalityVector:
        """Build a personality vector from sparse values."""
        return build_personality(values, self.registry)

    def scenario(
        self,
        values: Mapping[str, float],
        *,
        name: str = "",
        description: str = "",
    ) -> Scenario:
        """Build a scenario from sparse values."""
        return build_scenario(
            values,
            self.registry,
            name=name,
            description=description,
        )

    def action(
        self,
        name: str,
        modifiers: Mapping[str, float],
        *,
        description: str = "",
    ) -> Action:
        """Build an action from sparse modifiers."""
        return build_action(
            name=name,
            modifiers=modifiers,
            registry=self.registry,
            description=description,
        )

    def initial_self_model(self, values: Mapping[str, float]) -> np.ndarray:
        """Build an initial self-model vector from sparse values."""
        return build_initial_self_model(values, self.registry)

    def decide(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        actions: Sequence[Action],
        *,
        temperature: float = 1.0,
        bias: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> DecisionResult:
        """Run a one-shot decision through the SDK."""
        return DecisionClient(self.engine, self.registry).decide(
            personality,
            scenario,
            actions,
            temperature=temperature,
            bias=bias,
            rng=rng,
        )

    def simulator(
        self,
        personality: PersonalityVector,
        actions: Sequence[Action],
        **simulator_kwargs: object,
    ) -> TemporalSimulationClient:
        """Create a temporal simulation client."""
        return TemporalSimulationClient(
            personality,
            actions,
            self.engine,
            self.registry,
            **simulator_kwargs,
        )

    def self_aware_simulator(
        self,
        personality: PersonalityVector,
        initial_self_model: np.ndarray,
        actions: Sequence[Action],
        **simulator_kwargs: object,
    ) -> SelfModelSimulationClient:
        """Create a self-aware simulation client."""
        return SelfModelSimulationClient(
            personality,
            initial_self_model,
            actions,
            self.engine,
            self.registry,
            **simulator_kwargs,
        )


__all__ = [
    "AgentSDK",
    "DEFAULT_DIMENSIONS",
    "DecisionResult",
    "SelfAwareSimulationTrace",
    "SelfAwareTickRecord",
    "SimulationTrace",
    "TickRecord",
]
