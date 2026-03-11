"""High-level SDK client for self-aware simulations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.personality.decision import DecisionEngine
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action, PersonalityVector, Scenario
from src.sdk.mappers import self_aware_tick_result_to_payload
from src.sdk.types import SelfAwareSimulationTrace, SelfAwareTickRecord
from src.self_model.simulator import SelfAwareSimulator


class SelfModelSimulationClient:
    """Convenience wrapper over the self-aware simulator."""

    def __init__(
        self,
        personality: PersonalityVector,
        initial_self_model: np.ndarray,
        actions: Sequence[Action],
        engine: DecisionEngine,
        registry: DimensionRegistry,
        **simulator_kwargs: Any,
    ) -> None:
        self.registry = registry
        self.simulator = SelfAwareSimulator(
            personality,
            initial_self_model,
            actions,
            engine,
            **simulator_kwargs,
        )

    def tick(self, scenario: Scenario, outcome: float | None = None) -> SelfAwareTickRecord:
        """Run one self-aware tick."""
        result = self.simulator.tick(scenario, outcome=outcome)
        return SelfAwareTickRecord(**self_aware_tick_result_to_payload(result, self.registry))

    def run(
        self,
        scenarios: Sequence[Scenario],
        outcomes: Sequence[float | None] | None = None,
    ) -> SelfAwareSimulationTrace:
        """Run a self-aware scenario sequence and collect JSON-safe records."""
        outcome_list = list(outcomes) if outcomes is not None else [None] * len(scenarios)
        if len(outcome_list) != len(scenarios):
            raise ValueError("Length of `outcomes` must match length of `scenarios`.")
        ticks = [
            self.tick(scenario, outcome=outcome)
            for scenario, outcome in zip(scenarios, outcome_list)
        ]
        return SelfAwareSimulationTrace(ticks=ticks)
