"""Builder for reconstructing simulation clients from persisted run configs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.api.run_models import TickEventRecord
from src.sdk import AgentSDK
from src.sdk.self_model_client import SelfModelSimulationClient
from src.sdk.simulation_client import TemporalSimulationClient

SDK_FACTORIES: dict[str, Callable[[], AgentSDK]] = {
    "default": AgentSDK.default,
    "precision": AgentSDK.with_precision,
    "efe": AgentSDK.with_efe,
    "constructed_emotion": AgentSDK.with_constructed_emotion,
    "self_evidencing": AgentSDK.with_self_evidencing,
}


class RunClientBuilder:
    """Reconstruct a simulation client from a persisted run config and tick history."""

    def build(
        self,
        config: dict[str, Any],
        prior_ticks: list[TickEventRecord],
    ) -> TemporalSimulationClient | SelfModelSimulationClient:
        """Build a simulation client, replaying prior ticks to restore state."""
        sdk = self._resolve_sdk(config)
        personality = sdk.personality(config["personality"])
        actions = [
            sdk.action(
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
        client = self._build_client(config, sdk, personality, actions, rng)
        self._replay_ticks(client, sdk, prior_ticks)
        return client

    @staticmethod
    def _resolve_sdk(config: dict[str, Any]) -> AgentSDK:
        """Resolve SDK factory from config's sdk_mode field."""
        sdk_mode = config.get("sdk_mode", "efe")
        factory = SDK_FACTORIES.get(sdk_mode, AgentSDK.with_efe)
        return factory()

    @staticmethod
    def _build_client(
        config: dict[str, Any],
        sdk: AgentSDK,
        personality: Any,
        actions: list[Any],
        rng: np.random.Generator,
    ) -> TemporalSimulationClient | SelfModelSimulationClient:
        """Construct the appropriate client type from config."""
        temperature = config.get("temperature", 1.0)
        if config.get("self_model") is None:
            return sdk.simulator(personality, actions, temperature=temperature, rng=rng)
        return sdk.self_aware_simulator(
            personality,
            sdk.initial_self_model(config["self_model"]),
            actions,
            temperature=temperature,
            rng=rng,
        )

    @staticmethod
    def _replay_ticks(
        client: TemporalSimulationClient | SelfModelSimulationClient,
        sdk: AgentSDK,
        prior_ticks: list[TickEventRecord],
    ) -> None:
        """Replay persisted ticks to restore simulator state."""
        for tick in prior_ticks:
            scenario = sdk.scenario(
                tick.scenario["values"],
                name=tick.scenario.get("name", ""),
                description=tick.scenario.get("description", ""),
            )
            client.tick(scenario, outcome=tick.requested_outcome)
