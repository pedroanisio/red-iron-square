"""Self-aware simulator: extends TemporalSimulator with self-model hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.personality.vectors import Action, PersonalityVector, Scenario
from src.precision.state import PrecisionState
from src.self_model.emotions import SelfEmotionDetector, SelfEmotionReading
from src.self_model.model import SelfModel
from src.self_model.params import SelfModelParams
from src.shared.logging import get_logger
from src.shared.protocols import DecisionEngineProtocol
from src.temporal.simulator import TemporalSimulator
from src.temporal.state import AgentState
from src.temporal.tick_result import TickResult

if TYPE_CHECKING:
    from src.constructed_emotion.affect import ConstructedAffectiveEngine
    from src.narrative.model import NarrativeGenerativeModel
    from src.precision.engine import PrecisionEngine
    from src.self_evidencing.modulator import SelfEvidencingModulator
    from src.shared.protocols import System2RuntimeProtocol

_log = get_logger(module="self_model.simulator")


class SelfAwareTickResult(TickResult):
    """Extends TickResult with self-model data."""

    self_emotions: list[SelfEmotionReading]
    psi_hat: np.ndarray
    behavioral_evidence: np.ndarray
    self_coherence: float
    self_accuracy: float
    identity_drift: float
    prediction_error: float
    predicted_probs: np.ndarray
    self_evidencing_weights: np.ndarray | None = None


class SelfAwareSimulator(TemporalSimulator):
    """Extends TemporalSimulator with a self-model via inheritance.

    All temporal logic is inherited.  This class adds only:
        - Pre-decision: self-model predicts (state-aware).
        - Post-decision: self-model updates + self-emotions detected.
    """

    def __init__(
        self,
        personality: PersonalityVector,
        initial_self_model: np.ndarray,
        actions: Sequence[Action],
        engine: DecisionEngineProtocol,
        *,
        self_model_params: SelfModelParams = SelfModelParams(),
        initial_state: AgentState | None = None,
        memory_size: int = 500,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
        precision_engine: PrecisionEngine | None = None,
        constructed_affect: ConstructedAffectiveEngine | None = None,
        self_evidencing: SelfEvidencingModulator | None = None,
        narrative_model: NarrativeGenerativeModel | None = None,
        agent_runtime: System2RuntimeProtocol | None = None,
    ) -> None:
        super().__init__(
            personality,
            actions,
            engine,
            initial_state=initial_state,
            memory_size=memory_size,
            temperature=temperature,
            rng=rng,
            precision_engine=precision_engine,
            constructed_affect=constructed_affect,
            narrative_model=narrative_model,
            agent_runtime=agent_runtime,
            self_evidencing=self_evidencing,
        )
        self.self_model = SelfModel(
            initial_self_model,
            self.registry,
            params=self_model_params,
        )
        self.self_emotion_detector = SelfEmotionDetector(params=self_model_params)
        self._self_evidencing = self_evidencing
        self._se_temp_scale: float = 1.0

    def _compute_effective_temperature(
        self,
        precision: PrecisionState | None = None,
    ) -> float:
        """Override to apply self-evidencing temperature scaling."""
        base_temp = super()._compute_effective_temperature(precision)
        return base_temp * self._se_temp_scale

    def tick(
        self,
        scenario: Scenario,
        outcome: float | None = None,
    ) -> SelfAwareTickResult:
        """One tick with self-model integration."""
        true_psi = self.personality.to_array()

        predicted_probs = self.self_model.predict_action_distribution(
            scenario,
            self.actions,
            self.engine,
            self.temperature,
            state=self.state,
        )

        self._apply_self_evidencing_scale(predicted_probs)
        base: TickResult = super().tick(scenario, outcome)
        self._se_temp_scale = 1.0

        modifier_list = [a.modifiers for a in self.actions]
        sm_metrics = self.self_model.update(base.probabilities, modifier_list)

        pred_error = self.self_model.compute_prediction_error(
            base.probabilities,
            predicted_probs,
        )
        self_accuracy = self.self_model.compute_self_accuracy(true_psi)

        self_emotions = self.self_emotion_detector.detect_all(
            self.self_model,
            pred_error,
            base.outcome,
            self.personality,
            self.registry,
        )

        se_weights = self._compute_self_evidencing(predicted_probs, base)

        _log.debug(
            "self_aware_tick",
            tick=base.tick,
            prediction_error=round(pred_error, 4),
            self_accuracy=round(self_accuracy, 4),
        )

        return SelfAwareTickResult(
            tick=base.tick,
            scenario=base.scenario,
            action=base.action,
            outcome=base.outcome,
            state_before=base.state_before,
            state_after=base.state_after,
            activations=base.activations,
            emotions=base.emotions,
            self_emotions=self_emotions,
            probabilities=base.probabilities,
            psi_hat=self.self_model.psi_hat,
            behavioral_evidence=self.self_model.behavioral_evidence,
            self_coherence=sm_metrics["self_coherence"],
            self_accuracy=self_accuracy,
            identity_drift=sm_metrics["identity_drift"],
            prediction_error=pred_error,
            predicted_probs=predicted_probs,
            precision=base.precision,
            prediction_errors=base.prediction_errors,
            affect_signal=base.affect_signal,
            self_evidencing_weights=se_weights,
        )

    def _apply_self_evidencing_scale(
        self,
        predicted_probs: np.ndarray,
    ) -> None:
        """Compute temperature scale from self-evidencing concentration.

        Higher beta (high-T) -> more concentrated predicted probs
        -> lower scale -> lower effective temperature -> narrower distribution.
        """
        if self._self_evidencing is None:
            return
        concentration = float(np.max(predicted_probs))
        n_actions = len(self.actions)
        excess = concentration - 1.0 / n_actions
        self._se_temp_scale = max(
            0.3,
            1.0 / (1.0 + self._self_evidencing.beta * excess),
        )

    def _compute_self_evidencing(
        self,
        predicted_probs: np.ndarray,
        base: TickResult,
    ) -> np.ndarray | None:
        """Compute self-evidencing precision weights and decay beta."""
        if self._self_evidencing is None or base.precision is None:
            return None
        weights = self._self_evidencing.compute_precision_weights(
            predicted_probs,
            base.precision.level_1,
        )
        keys = set(self.personality.registry.keys)
        t_val = self.personality["T"] if "T" in keys else 0.5
        self._self_evidencing.decay_beta(t_val)
        return weights
