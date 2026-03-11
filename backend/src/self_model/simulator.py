"""Self-aware simulator: extends TemporalSimulator with self-model hooks."""

from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.personality.vectors import PersonalityVector, Scenario, Action
from src.personality.decision import DecisionEngine
from src.temporal.state import AgentState
from src.temporal.emotions import EmotionReading
from src.temporal.simulator import TickResult, TemporalSimulator
from src.shared.logging import get_logger
from src.self_model.params import SelfModelParams
from src.self_model.model import SelfModel
from src.self_model.emotions import SelfEmotionReading, SelfEmotionDetector

_log = get_logger(module="self_model.simulator")


class SelfAwareTickResult(BaseModel):
    """Extends TickResult with self-model data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tick: int
    scenario: Scenario
    action: Action
    outcome: float
    state_before: AgentState
    state_after: AgentState
    activations: np.ndarray
    emotions: list[EmotionReading]
    self_emotions: list[SelfEmotionReading]
    probabilities: np.ndarray
    psi_hat: np.ndarray
    behavioral_evidence: np.ndarray
    self_coherence: float
    self_accuracy: float
    identity_drift: float
    prediction_error: float
    predicted_probs: np.ndarray


class SelfAwareSimulator(TemporalSimulator):
    """
    Extends TemporalSimulator with a self-model via inheritance.

    All temporal logic is inherited.  This class adds only:
        - Pre-decision: self-model predicts (state-aware).
        - Post-decision: self-model updates + self-emotions detected.
    """

    def __init__(
        self,
        personality: PersonalityVector,
        initial_self_model: np.ndarray,
        actions: Sequence[Action],
        engine: DecisionEngine,
        *,
        self_model_params: SelfModelParams = SelfModelParams(),
        initial_state: Optional[AgentState] = None,
        memory_size: int = 500,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            personality, actions, engine,
            initial_state=initial_state,
            memory_size=memory_size,
            temperature=temperature,
            rng=rng,
        )
        self.self_model = SelfModel(
            initial_self_model, self.registry, params=self_model_params,
        )
        self.self_emotion_detector = SelfEmotionDetector(params=self_model_params)

    def tick(
        self, scenario: Scenario, outcome: Optional[float] = None,
    ) -> SelfAwareTickResult:
        """One tick with self-model integration."""
        true_psi = self.personality.to_array()

        predicted_probs = self.self_model.predict_action_distribution(
            scenario, self.actions, self.engine, self.temperature, state=self.state,
        )

        base: TickResult = super().tick(scenario, outcome)

        modifier_list = [a.modifiers for a in self.actions]
        sm_metrics = self.self_model.update(base.probabilities, modifier_list)

        pred_error = self.self_model.compute_prediction_error(
            base.probabilities, predicted_probs,
        )
        self_accuracy = self.self_model.compute_self_accuracy(true_psi)

        self_emotions = self.self_emotion_detector.detect_all(
            self.self_model, pred_error, base.outcome,
            self.personality, self.registry,
        )

        _log.debug(
            "self_aware_tick",
            tick=base.tick,
            prediction_error=round(pred_error, 4),
            self_accuracy=round(self_accuracy, 4),
        )

        return SelfAwareTickResult(
            tick=base.tick, scenario=base.scenario, action=base.action,
            outcome=base.outcome, state_before=base.state_before,
            state_after=base.state_after, activations=base.activations,
            emotions=base.emotions, self_emotions=self_emotions,
            probabilities=base.probabilities, psi_hat=self.self_model.psi_hat,
            behavioral_evidence=self.self_model.behavioral_evidence,
            self_coherence=sm_metrics["self_coherence"],
            self_accuracy=self_accuracy,
            identity_drift=sm_metrics["identity_drift"],
            prediction_error=pred_error, predicted_probs=predicted_probs,
        )
