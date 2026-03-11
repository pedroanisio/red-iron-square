"""Builder for reconstructing simulation clients from persisted run configs.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.api.run_models import TickEventRecord
from src.sdk import AgentSDK
from src.sdk.self_model_client import SelfModelSimulationClient
from src.sdk.simulation_client import TemporalSimulationClient


class RunClientBuilder:
    """Reconstruct a simulation client from a persisted run config and tick history."""

    def __init__(self, sdk: AgentSDK | None = None) -> None:
        self._sdk = sdk or AgentSDK.with_precision()

    def build(
        self,
        config: dict[str, Any],
        prior_ticks: list[TickEventRecord],
    ) -> TemporalSimulationClient | SelfModelSimulationClient:
        """Build a simulation client, replaying prior ticks to restore state."""
        personality = self._sdk.personality(config["personality"])
        actions = [
            self._sdk.action(
                action["name"],
                action["modifiers"],
                description=action.get("description", ""),
            )
            for action in config["actions"]
        ]
        rng = (
            np.random.default_rng(config["seed"])
            if config.get("seed") is not None
            else np.random.default_rng()
        )
        client = self._build_client(config, personality, actions, rng)
        self._replay_ticks(client, prior_ticks)
        return client

    def _build_client(
        self,
        config: dict[str, Any],
        personality: Any,
        actions: list[Any],
        rng: np.random.Generator,
    ) -> TemporalSimulationClient | SelfModelSimulationClient:
        """Construct the appropriate client type from config."""
        temperature = config.get("temperature", 1.0)
        if config.get("self_model") is None:
            return self._sdk.simulator(
                personality, actions, temperature=temperature, rng=rng
            )
        return self._sdk.self_aware_simulator(
            personality,
            self._sdk.initial_self_model(config["self_model"]),
            actions,
            temperature=temperature,
            rng=rng,
        )

    def _replay_ticks(
        self,
        client: TemporalSimulationClient | SelfModelSimulationClient,
        prior_ticks: list[TickEventRecord],
    ) -> None:
        """Replay persisted ticks to restore simulator state."""
        for tick in prior_ticks:
            scenario = self._sdk.scenario(
                tick.scenario["values"],
                name=tick.scenario.get("name", ""),
                description=tick.scenario.get("description", ""),
            )
            client.tick(scenario, outcome=tick.requested_outcome)
