"""High-level SDK client for one-shot decisions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action, PersonalityVector, Scenario
from src.sdk.mappers import action_list_to_names, probabilities_to_dict, vector_to_dict
from src.sdk.types import DecisionResult
from src.shared.protocols import DecisionEngineProtocol


class DecisionClient:
    """Task-oriented wrapper around the decision engine."""

    def __init__(
        self, engine: DecisionEngineProtocol, registry: DimensionRegistry
    ) -> None:
        self.engine = engine
        self.registry = registry

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
        """Choose an action and expose JSON-safe diagnostics."""
        action_list = list(actions)
        activations = self.engine.compute_activations(personality, scenario)
        chosen_action, probabilities = self.engine.decide(
            personality,
            scenario,
            action_list,
            temperature=temperature,
            bias=bias,
            rng=rng,
            activations_override=activations,
        )
        utilities = {
            action.name: self.engine.utility(
                personality,
                scenario,
                action,
                bias=bias,
                activations_override=activations,
            )
            for action in action_list
        }
        return DecisionResult(
            chosen_action=chosen_action.name,
            probabilities=probabilities_to_dict(action_list, probabilities),
            utilities=utilities,
            activations=vector_to_dict(activations, self.registry),
            action_order=action_list_to_names(action_list),
        )
