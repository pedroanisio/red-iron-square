"""Temporal simulator: the main tick loop integrating state, memory, and emotions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.personality.vectors import Action, PersonalityVector, Scenario
from src.precision.state import PrecisionState, PredictionErrors
from src.shared.logging import get_logger
from src.shared.protocols import DecisionEngineProtocol
from src.temporal.affective_engine import AffectiveEngine
from src.temporal.emotions import EmotionReading, EmotionThresholds
from src.temporal.memory import MemoryBank, MemoryEntry
from src.temporal.state import AgentState, StateTransitionParams, update_state

if TYPE_CHECKING:
    from src.constructed_emotion.affect import AffectSignal, ConstructedAffectiveEngine
    from src.precision.engine import PrecisionEngine

_log = get_logger(module="temporal.simulator")


class TickResult(BaseModel):
    """Complete result for a single simulation tick."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tick: int
    scenario: Scenario
    action: Action
    outcome: float
    state_before: AgentState
    state_after: AgentState
    activations: np.ndarray
    emotions: list[EmotionReading]
    probabilities: np.ndarray
    precision: PrecisionState | None = None
    prediction_errors: PredictionErrors | None = None
    affect_signal: Any = None


class TemporalSimulator:
    """The main simulation loop: personality x scenarios x time -> emotion traces.

    Pipeline per tick:
        1. Compute state-modulated activations
        2. Compute effective temperature (mood-modulated)
        3. Select action via Boltzmann
        4. Resolve outcome
        5. Compute counterfactual
        6. Determine withdrawal
        7. Compute action effort and update state
        8. Store memory
        9. Detect emotions
        10. (Optional) Compute shadow precision and prediction errors
    """

    WITHDRAW_ACTION_NAME = "Withdraw"

    def __init__(
        self,
        personality: PersonalityVector,
        actions: Sequence[Action],
        engine: DecisionEngineProtocol,
        *,
        initial_state: AgentState | None = None,
        state_params: StateTransitionParams = StateTransitionParams(),
        emotion_thresholds: EmotionThresholds = EmotionThresholds(),
        memory_size: int = 500,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
        precision_engine: PrecisionEngine | None = None,
        constructed_affect: ConstructedAffectiveEngine | None = None,
    ) -> None:
        self.personality = personality
        self.actions = list(actions)
        self.engine = engine
        self.registry = engine.registry
        self.state = initial_state or AgentState()
        self.state_params = state_params
        self.memory = MemoryBank(max_size=memory_size)
        self.affect = AffectiveEngine(thresholds=emotion_thresholds)
        self.temperature = temperature
        self.rng = rng or np.random.default_rng()
        self._tick_counter = 0
        self._precision_engine = precision_engine
        self._constructed_affect = constructed_affect
        self._bind_engine_memory()

    def _bind_engine_memory(self) -> None:
        """Bind memory bank to engine if it supports it (e.g. EFEEngine)."""
        if hasattr(self.engine, "bind_memory"):
            self.engine.bind_memory(self.memory)

    def _compute_modulated_activations(self, scenario: Scenario) -> np.ndarray:
        """Compute raw activations gated by energy."""
        raw = self.engine.compute_activations(self.personality, scenario)
        energy_factor = 0.5 + 0.5 * self.state.energy
        return raw * energy_factor

    def _compute_effective_temperature(
        self,
        precision: PrecisionState | None = None,
    ) -> float:
        """Compute temperature, using precision-derived gamma when available.

        When a ``PrecisionState`` is provided, policy precision gamma
        (level_1) sets the temperature as ``1/gamma``. Otherwise falls
        back to the mood-modulated base temperature.
        """
        if precision is not None:
            gamma = precision.level_1
            return 1.0 / gamma
        return self.temperature * (1.0 + 0.3 * max(0, -self.state.mood))

    def _resolve_outcome(
        self,
        outcome: float | None,
        activations: np.ndarray,
        chosen_action: Action,
        scenario: Scenario,
    ) -> float:
        """Use provided outcome or stochastic model."""
        if outcome is not None:
            return outcome
        u = self.engine.utility(
            self.personality,
            scenario,
            chosen_action,
            activations_override=activations,
        )
        return float(np.clip(self.rng.normal(0.2 * u, 0.3), -1.0, 1.0))

    def _compute_counterfactual(
        self,
        activations: np.ndarray,
        chosen_action: Action,
        scenario: Scenario,
    ) -> float:
        """Best unchosen utility minus chosen utility."""
        utilities = np.array(
            [
                self.engine.utility(
                    self.personality, scenario, a, activations_override=activations
                )
                for a in self.actions
            ]
        )
        chosen_idx = next(
            i for i, a in enumerate(self.actions) if a.name == chosen_action.name
        )
        unchosen = np.delete(utilities, chosen_idx)
        if len(unchosen) == 0:
            return 0.0
        return float(max(unchosen) - utilities[chosen_idx])

    def _determine_is_still_acting(
        self,
        chosen_action: Action,
        new_state: AgentState,
    ) -> bool:
        """Return False if agent withdrew, energy depleted, or frustration maxed."""
        if chosen_action.name == self.WITHDRAW_ACTION_NAME:
            return False
        if new_state.energy <= 1e-9:
            return False
        if new_state.frustration >= 1.0 - 1e-9:
            return False
        return True

    @staticmethod
    def _compute_action_effort(action: Action) -> float:
        """Compute action effort as L2 norm of the modifier vector."""
        return float(np.linalg.norm(action.modifiers))

    def _store_memory(
        self,
        tick: int,
        state_before: AgentState,
        scenario: Scenario,
        chosen_action: Action,
        outcome: float,
        counterfactual: float,
    ) -> None:
        """Create and store a memory entry."""
        valence = float(np.clip(outcome * 0.6 + state_before.mood * 0.4, -1, 1))
        entry = MemoryEntry(
            tick=tick,
            scenario_name=scenario.name,
            action_name=chosen_action.name,
            outcome=outcome,
            counterfactual=counterfactual,
            state_snapshot=state_before,
            valence=valence,
        )
        self.memory.store(entry)

    def _compute_precision(
        self,
        state: AgentState,
        scenario: Scenario,
    ) -> tuple[PrecisionState | None, PredictionErrors | None]:
        """Compute shadow precision and prediction errors if engine present."""
        if self._precision_engine is None:
            return None, None
        precision = self._precision_engine.compute(self.personality, state, scenario)
        errors = self._precision_engine.compute_errors(state, self.personality)
        return precision, errors

    def _run_constructed_affect(
        self,
        precision: PrecisionState | None,
        errors: PredictionErrors | None,
    ) -> AffectSignal | None:
        """Run constructed affective engine if available and precision exists."""
        if self._constructed_affect is None or precision is None or errors is None:
            return None
        return self._constructed_affect.process_tick(
            precision,
            errors,
            self.personality,
        )

    @property
    def _uses_efe(self) -> bool:
        """Check if the engine supports EFE-based decisions."""
        return hasattr(self.engine, "bind_memory")

    def tick(self, scenario: Scenario, outcome: float | None = None) -> TickResult:
        """Execute one simulation tick."""
        state_before = self.state.snapshot()
        activations = self._compute_modulated_activations(scenario)

        pre_precision: PrecisionState | None = None
        if self._uses_efe and self._precision_engine is not None:
            pre_precision, _ = self._compute_precision(self.state, scenario)

        temperature = self._compute_effective_temperature(pre_precision)

        chosen_action, probs = self.engine.decide(
            self.personality,
            scenario,
            self.actions,
            temperature=temperature,
            rng=self.rng,
            activations_override=activations,
        )

        resolved_outcome = self._resolve_outcome(
            outcome,
            activations,
            chosen_action,
            scenario,
        )
        counterfactual = self._compute_counterfactual(
            activations,
            chosen_action,
            scenario,
        )

        action_effort = self._compute_action_effort(chosen_action)
        new_state = update_state(
            self.state,
            resolved_outcome,
            self.personality,
            scenario,
            self.state_params,
            action_effort=action_effort,
        )
        is_acting = self._determine_is_still_acting(chosen_action, new_state)

        self._store_memory(
            self._tick_counter,
            state_before,
            scenario,
            chosen_action,
            resolved_outcome,
            counterfactual,
        )

        emotions = self.affect.detect_all(
            activations,
            new_state,
            self.personality,
            self.memory,
            self.registry,
            is_still_acting=is_acting,
        )

        precision, pred_errors = self._compute_precision(new_state, scenario)
        affect_sig = self._run_constructed_affect(precision, pred_errors)

        self.state = new_state
        self._tick_counter += 1

        if not is_acting:
            _log.warning(
                "agent_withdrawal",
                tick=self._tick_counter - 1,
                action=chosen_action.name,
                energy=new_state.energy,
                frustration=new_state.frustration,
            )

        _log.info(
            "tick_complete",
            tick=self._tick_counter - 1,
            scenario=scenario.name,
            action=chosen_action.name,
            outcome=round(resolved_outcome, 4),
            mood=round(new_state.mood, 4),
            energy=round(new_state.energy, 4),
            emotions_detected=len(emotions),
        )

        return TickResult(
            tick=self._tick_counter - 1,
            scenario=scenario,
            action=chosen_action,
            outcome=resolved_outcome,
            state_before=state_before,
            state_after=new_state,
            activations=activations,
            emotions=emotions,
            probabilities=probs,
            precision=precision,
            prediction_errors=pred_errors,
            affect_signal=affect_sig,
        )

    @property
    def tick_count(self) -> int:
        """Number of ticks executed so far."""
        return self._tick_counter

    @property
    def current_state(self) -> AgentState:
        """Current agent state."""
        return self.state
