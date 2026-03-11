"""Self-model: the agent's internal estimate of its own personality."""

from typing import Optional, Sequence
from collections import deque

import numpy as np

from src.shared.validators import validate_unit_interval
from src.shared.logging import get_logger
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import PersonalityVector, Scenario, Action
from src.personality.decision import DecisionEngine
from src.temporal.state import AgentState
from src.self_model.params import SelfModelParams

_log = get_logger(module="self_model.model")


class SelfModel:
    """
    The agent's internal model of its own personality psi_hat.

    psi_hat can diverge from true psi, creating self-deception,
    identity threat, and self-related emotions.

    Update rule:
        psi_hat_i(t+1) = psi_hat_i(t)
            + eta * [B_i(t) - psi_hat_i(t)]
            - lambda * [psi_hat_i(t) - psi_hat_0_i]
    """

    def __init__(
        self,
        initial_self_model: np.ndarray,
        registry: DimensionRegistry,
        params: SelfModelParams = SelfModelParams(),
    ) -> None:
        n = registry.size
        if initial_self_model.shape != (n,):
            raise ValueError(
                f"Self-model must have shape ({n},), got {initial_self_model.shape}"
            )
        for i, key in enumerate(registry.keys):
            validate_unit_interval(f"psi_hat_0[{key}]", initial_self_model[i])

        self._registry = registry
        self.params = params
        self._psi_hat = initial_self_model.copy()
        self._psi_hat_0 = initial_self_model.copy()
        self._B = initial_self_model.copy()
        self._coherence_history: deque[float] = deque(maxlen=50)
        self._psi_hat_history: deque[np.ndarray] = deque(maxlen=50)
        self._psi_hat_history.append(initial_self_model.copy())

    @property
    def psi_hat(self) -> np.ndarray:
        """Current self-model estimate."""
        return self._psi_hat.copy()

    @property
    def anchor(self) -> np.ndarray:
        """Identity anchor psi_hat_0."""
        return self._psi_hat_0.copy()

    @property
    def behavioral_evidence(self) -> np.ndarray:
        """Current behavioral evidence accumulator."""
        return self._B.copy()

    @property
    def coherence_history(self) -> list[float]:
        """History of coherence gap values."""
        return list(self._coherence_history)

    @property
    def registry(self) -> DimensionRegistry:
        """The dimension registry."""
        return self._registry

    def current_coherence_gap(self) -> float:
        """epsilon_coh = ||psi_hat - B|| / sqrt(N)."""
        n = self._registry.size
        return float(np.linalg.norm(self._psi_hat - self._B) / np.sqrt(n))

    def current_identity_drift(self) -> float:
        """delta = ||psi_hat - psi_hat_0|| / sqrt(N)."""
        n = self._registry.size
        return float(np.linalg.norm(self._psi_hat - self._psi_hat_0) / np.sqrt(n))

    def sustained_coherence_threat(self) -> bool:
        """Has coherence gap exceeded threshold for enough consecutive ticks?"""
        p = self.params
        if len(self._coherence_history) < p.coherence_threat_window:
            return False
        recent = list(self._coherence_history)[-p.coherence_threat_window:]
        return all(g > p.coherence_threat_threshold for g in recent)

    def compute_self_accuracy(self, true_psi: np.ndarray) -> float:
        """epsilon_acc = ||psi_hat - psi||_2 / sqrt(N).  Observer-only metric."""
        n = self._registry.size
        return float(np.linalg.norm(self._psi_hat - true_psi) / np.sqrt(n))

    def update(
        self, action_probs: np.ndarray, action_modifiers: list[np.ndarray],
    ) -> dict[str, float]:
        """Full self-model update cycle: evidence -> psi_hat revision."""
        p = self.params

        fingerprint = self._compute_behavioral_fingerprint(action_probs, action_modifiers)
        self._update_evidence(fingerprint)

        evidence_pull = p.learning_rate * (self._B - self._psi_hat)
        anchor_pull = p.identity_inertia * (self._psi_hat - self._psi_hat_0)
        delta = evidence_pull - anchor_pull

        self._psi_hat = np.clip(self._psi_hat + delta, 0.0, 1.0)
        self._psi_hat_history.append(self._psi_hat.copy())

        coherence = self.current_coherence_gap()
        drift = self.current_identity_drift()
        update_mag = float(np.linalg.norm(delta) / np.sqrt(self._registry.size))

        self._coherence_history.append(coherence)

        _log.debug(
            "self_model_updated",
            coherence_gap=round(coherence, 4),
            identity_drift=round(drift, 4),
            update_magnitude=round(update_mag, 4),
        )

        if self.sustained_coherence_threat():
            _log.warning("sustained_coherence_threat")

        return {
            "self_coherence": coherence,
            "identity_drift": drift,
            "update_magnitude": update_mag,
        }

    def predict_action_distribution(
        self,
        scenario: Scenario,
        actions: Sequence[Action],
        engine: DecisionEngine,
        temperature: float = 1.0,
        state: Optional[AgentState] = None,
    ) -> np.ndarray:
        """
        Predict action distribution using psi_hat, with optional state modulation.

        When state is provided, applies the same energy gating and
        mood-temperature modulation as the actual decision pipeline.
        """
        psi_hat_pv = PersonalityVector(array=self._psi_hat, registry=self._registry)
        raw_activations = engine.compute_activations(psi_hat_pv, scenario)

        if state is not None:
            energy_factor = 0.5 + 0.5 * state.energy
            raw_activations = raw_activations * energy_factor
            temperature = temperature * (1.0 + 0.3 * max(0, -state.mood))

        utilities = np.array([
            engine.utility(psi_hat_pv, scenario, a, activations_override=raw_activations)
            for a in actions
        ])

        logits = utilities / max(temperature, 1e-8)
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()

    def compute_prediction_error(
        self, actual_probs: np.ndarray, predicted_probs: np.ndarray,
    ) -> float:
        """H(q, p_hat) normalized to [0, 1]."""
        eps = 1e-10
        K = len(actual_probs)
        cross_entropy = -np.sum(actual_probs * np.log(predicted_probs + eps))
        max_entropy = np.log(K)
        return float(np.clip(cross_entropy / max(max_entropy, eps), 0.0, 1.0))

    def _compute_behavioral_fingerprint(
        self, action_probs: np.ndarray, action_modifiers: list[np.ndarray],
    ) -> np.ndarray:
        """Probability-weighted average of action modifiers, through sigmoid."""
        m_bar = np.zeros(self._registry.size)
        for prob, mod in zip(action_probs, action_modifiers):
            m_bar += prob * mod
        return 1.0 / (1.0 + np.exp(-self.params.sigmoid_scale * m_bar))

    def _update_evidence(self, fingerprint: np.ndarray) -> None:
        """EMA update of behavioral evidence."""
        alpha = self.params.evidence_memory
        self._B = alpha * self._B + (1.0 - alpha) * fingerprint
