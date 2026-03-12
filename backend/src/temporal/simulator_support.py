"""Support helpers for the temporal simulator.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.personality.vectors import Action, PersonalityVector, Scenario
from src.precision.state import PrecisionState, PredictionErrors
from src.shared.protocols import DecisionEngineProtocol
from src.temporal.state import AgentState

if TYPE_CHECKING:
    from src.constructed_emotion.affect import AffectSignal, ConstructedAffectiveEngine
    from src.narrative.model import NarrativeGenerativeModel
    from src.self_evidencing.modulator import SelfEvidencingModulator
    from src.shared.protocols import System2RuntimeProtocol
    from src.temporal.system2 import System2Orchestrator


def build_system2(
    narrative_model: NarrativeGenerativeModel | None,
    self_evidencing: SelfEvidencingModulator | None,
    agent_runtime: System2RuntimeProtocol | None,
    personality: PersonalityVector,
) -> System2Orchestrator | None:
    """Create a System 2 orchestrator when narrative support is enabled."""
    if narrative_model is None:
        return None
    from src.temporal.system2 import System2Orchestrator

    pvals = dict(zip(personality.registry.keys, personality.to_array(), strict=False))
    return System2Orchestrator(
        agent_runtime=agent_runtime,
        narrative_model=narrative_model,
        self_evidencing=self_evidencing,
        personality=pvals,
    )


def resolve_outcome(
    engine: DecisionEngineProtocol,
    personality: PersonalityVector,
    scenario: Scenario,
    chosen_action: Action,
    activations: np.ndarray,
    rng: np.random.Generator,
    outcome: float | None,
) -> float:
    """Use the provided outcome or sample one from the utility model."""
    if outcome is not None:
        return outcome
    utility = engine.utility(
        personality,
        scenario,
        chosen_action,
        activations_override=activations,
    )
    return float(np.clip(rng.normal(0.2 * utility, 0.3), -1.0, 1.0))


def compute_counterfactual(
    engine: DecisionEngineProtocol,
    personality: PersonalityVector,
    scenario: Scenario,
    actions: Sequence[Action],
    chosen_action: Action,
    activations: np.ndarray,
) -> float:
    """Return best unchosen utility minus chosen utility."""
    utilities = np.array(
        [
            engine.utility(
                personality,
                scenario,
                action,
                activations_override=activations,
            )
            for action in actions
        ]
    )
    chosen_idx = next(
        idx for idx, action in enumerate(actions) if action.name == chosen_action.name
    )
    unchosen = np.delete(utilities, chosen_idx)
    if len(unchosen) == 0:
        return 0.0
    return float(max(unchosen) - utilities[chosen_idx])


def determine_is_still_acting(
    withdraw_action_name: str,
    chosen_action: Action,
    new_state: AgentState,
) -> bool:
    """Return whether the agent continues acting after the tick."""
    if chosen_action.name == withdraw_action_name:
        return False
    return new_state.energy > 1e-9 and new_state.frustration < 1.0 - 1e-9


def compute_action_effort(action: Action) -> float:
    """Compute action effort as the L2 norm of the modifier vector."""
    return float(np.linalg.norm(action.modifiers))


def run_constructed_affect(
    constructed_affect: ConstructedAffectiveEngine | None,
    precision: PrecisionState | None,
    errors: PredictionErrors | None,
    personality: PersonalityVector,
) -> AffectSignal | None:
    """Run the constructed-affect engine when all dependencies are present."""
    if constructed_affect is None or precision is None or errors is None:
        return None
    return constructed_affect.process_tick(
        precision,
        errors,
        personality,
    )
