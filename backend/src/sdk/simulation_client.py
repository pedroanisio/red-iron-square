"""High-level SDK client for temporal simulations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action, PersonalityVector, Scenario
from src.sdk.mappers import tick_result_to_payload
from src.sdk.types import SimulationTrace, TickRecord
from src.shared.protocols import DecisionEngineProtocol
from src.temporal.simulator import TemporalSimulator


class TemporalSimulationClient:
    """Convenience wrapper over the temporal simulator."""

    def __init__(
        self,
        personality: PersonalityVector,
        actions: Sequence[Action],
        engine: DecisionEngineProtocol,
        registry: DimensionRegistry,
        **simulator_kwargs: Any,
    ) -> None:
        self.registry = registry
        self.simulator = TemporalSimulator(
            personality,
            actions,
            engine,
            **simulator_kwargs,
        )

    def tick(self, scenario: Scenario, outcome: float | None = None) -> TickRecord:
        """Run one temporal tick."""
        result = self.simulator.tick(scenario, outcome=outcome)
        return TickRecord(**tick_result_to_payload(result, self.registry))

    def run(
        self,
        scenarios: Sequence[Scenario],
        outcomes: Sequence[float | None] | None = None,
    ) -> SimulationTrace:
        """Run a scenario sequence and collect JSON-safe records."""
        outcome_list = (
            list(outcomes) if outcomes is not None else [None] * len(scenarios)
        )
        if len(outcome_list) != len(scenarios):
            raise ValueError("Length of `outcomes` must match length of `scenarios`.")
        ticks = [
            self.tick(scenario, outcome=outcome)
            for scenario, outcome in zip(scenarios, outcome_list)
        ]
        return SimulationTrace(ticks=ticks)
