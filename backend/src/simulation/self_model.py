"""
Self-Model Layer (v2)
=====================

Adds a functional self-model to the temporal simulation: the agent
maintains an internal estimate ψ̂ of its own personality, which can
diverge from the true personality ψ.  The gap between ψ̂ and ψ is
the source of self-deception, identity threat, and self-related
emotions (pride, shame, authenticity, alienation).

Mathematical Framework
----------------------

Let N = number of personality dimensions.

Objects:
    ψ ∈ [0,1]^N         True personality (fixed).  Known to the simulation,
                         NOT known to the agent.
    ψ̂(t) ∈ [0,1]^N     Self-model at tick t.  The agent's belief about its
                         own personality.  Initialized to ψ̂₀ (possibly ≠ ψ).
    B(t) ∈ [0,1]^N      Behavioral evidence accumulator (EMA of behavioral
                         fingerprints inferred from the action distribution).
    ψ̂₀ ∈ [0,1]^N       Identity anchor.  The initial self-concept that
                         resists revision.

Self-Model Update:

    ψ̂_i(t+1) = ψ̂_i(t) + η · [ B_i(t) − ψ̂_i(t) ]
                        − λ · [ ψ̂_i(t) − ψ̂₀_i ]

    η = learning rate.  λ = identity inertia.
    η > λ → identity-flexible.  λ > η → identity-rigid.

Prediction (state-aware, v2 fix):

    The self-model predicts the action distribution THROUGH the same
    state-coupled pipeline as the actual decision: energy-modulated
    activations and mood-modulated temperature.  This ensures that
    prediction error measures ONLY self-knowledge error, not confusion
    from being tired or stressed.

Theoretical Grounding:
    - Metzinger (2003), "Being No One", MIT Press.
    - Higgins (1987), "Self-Discrepancy", Psych Review 94(3).
    - Festinger (1957), "Cognitive Dissonance", Stanford.
    - Swann (2012), "Self-Verification Theory".
    - Kernis & Goldman (2006), "Authenticity".

Changelog (v2):
    - SelfAwareSimulator inherits from TemporalSimulator (no duplication).
    - Self-prediction is now state-aware (energy + mood modulate the
      predicted distribution, so prediction error is a clean signal).
    - Shame scaling constant extracted to SelfModelParams.shame_scaling.
    - Private attributes exposed via public properties/methods.
    - SelfEmotionDetector no longer accesses private members directly.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence
from collections import deque
from enum import Enum

from src.simulation.personality_framework import (
    DimensionRegistry,
    PersonalityVector,
    Scenario,
    Action,
    DecisionEngine,
    _validate_unit_interval,
    _validate_real,
)
from src.simulation.temporal_engine import (
    AgentState,
    MemoryBank,
    MemoryEntry,
    TickResult,
    TemporalSimulator,
    StateTransitionParams,
    EmotionThresholds,
    AffectiveEngine,
    update_state,
    EmotionLabel,
    EmotionReading,
)


# =============================================================================
# 1. SELF-MODEL PARAMETERS
# =============================================================================

@dataclass
class SelfModelParams:
    """
    Tunable parameters for the self-model dynamics.
    """
    # Behavioral evidence accumulation
    evidence_memory: float = 0.85    # α — EMA decay for behavioral evidence
    sigmoid_scale: float = 2.0       # κ — sigmoid scaling for modifier → [0,1]

    # Self-model update
    learning_rate: float = 0.08      # η — how fast ψ̂ tracks behavior
    identity_inertia: float = 0.04   # λ — pull toward identity anchor ψ̂₀

    # Emotion thresholds
    coherence_threat_threshold: float = 0.20
    coherence_threat_window: int = 6
    drift_crisis_threshold: float = 0.35

    # Personality modulation
    N_shame_amplification: float = 0.5
    C_prediction_sharpening: float = 0.3

    # Shame scaling — controls how strongly prediction_error × coherence_gap
    # maps to shame intensity.  Previously hardcoded as 3.0.
    shame_scaling: float = 3.0

    def __post_init__(self):
        for name, val in vars(self).items():
            if isinstance(val, float):
                _validate_real(name, val)


# =============================================================================
# 2. SELF-MODEL STATE
# =============================================================================

class SelfModel:
    """
    The agent's internal model of its own personality.

    All state is accessible via public properties and methods.
    SelfEmotionDetector uses ONLY public interfaces.
    """

    def __init__(
        self,
        initial_self_model: np.ndarray,
        registry: DimensionRegistry,
        params: SelfModelParams = SelfModelParams(),
    ):
        n = registry.size
        if initial_self_model.shape != (n,):
            raise ValueError(
                f"Self-model must have shape ({n},), got {initial_self_model.shape}"
            )
        for i, key in enumerate(registry.keys):
            _validate_unit_interval(f"ψ̂₀[{key}]", initial_self_model[i])

        self._registry = registry
        self.params = params  # Public.
        self._psi_hat = initial_self_model.copy()
        self._psi_hat_0 = initial_self_model.copy()
        self._B = initial_self_model.copy()
        self._coherence_history: deque[float] = deque(maxlen=50)
        self._psi_hat_history: deque[np.ndarray] = deque(maxlen=50)
        self._psi_hat_history.append(initial_self_model.copy())

    # ── Public properties ────────────────────────────────────────────

    @property
    def psi_hat(self) -> np.ndarray:
        return self._psi_hat.copy()

    @property
    def anchor(self) -> np.ndarray:
        return self._psi_hat_0.copy()

    @property
    def behavioral_evidence(self) -> np.ndarray:
        return self._B.copy()

    @property
    def coherence_history(self) -> list[float]:
        return list(self._coherence_history)

    @property
    def registry(self) -> DimensionRegistry:
        return self._registry

    # ── Public computed metrics ──────────────────────────────────────

    def current_coherence_gap(self) -> float:
        """ε_coh = ||ψ̂ − B|| / √N."""
        n = self._registry.size
        return float(np.linalg.norm(self._psi_hat - self._B) / np.sqrt(n))

    def current_identity_drift(self) -> float:
        """δ = ||ψ̂ − ψ̂₀|| / √N."""
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
        """ε_acc = ||ψ̂ − ψ||₂ / √N.  Observer-only."""
        n = self._registry.size
        return float(np.linalg.norm(self._psi_hat - true_psi) / np.sqrt(n))

    # ── Behavioral Inference ─────────────────────────────────────────

    def _compute_behavioral_fingerprint(
        self, action_probs: np.ndarray, action_modifiers: list[np.ndarray],
    ) -> np.ndarray:
        m_bar = np.zeros(self._registry.size)
        for prob, mod in zip(action_probs, action_modifiers):
            m_bar += prob * mod
        return 1.0 / (1.0 + np.exp(-self.params.sigmoid_scale * m_bar))

    def _update_evidence(self, fingerprint: np.ndarray) -> None:
        alpha = self.params.evidence_memory
        self._B = alpha * self._B + (1.0 - alpha) * fingerprint

    # ── Self-Model Update ────────────────────────────────────────────

    def update(
        self, action_probs: np.ndarray, action_modifiers: list[np.ndarray],
    ) -> dict[str, float]:
        """Full self-model update cycle."""
        p = self.params

        fingerprint = self._compute_behavioral_fingerprint(
            action_probs, action_modifiers,
        )
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

        return {
            "self_coherence": coherence,
            "identity_drift": drift,
            "update_magnitude": update_mag,
        }

    # ── Self-Prediction (STATE-AWARE — v2 fix) ──────────────────────

    def predict_action_distribution(
        self,
        scenario: Scenario,
        actions: Sequence[Action],
        engine: DecisionEngine,
        temperature: float = 1.0,
        state: Optional[AgentState] = None,
    ) -> np.ndarray:
        """
        Predict the action distribution using ψ̂, with optional state modulation.

        v2 FIX: when state is provided, applies the SAME energy gating and
        mood-temperature modulation as the actual decision pipeline.  This
        ensures prediction error measures self-knowledge error only, not
        state-ignorance.

        Args:
            state: If provided, prediction uses state-modulated activations
                   and mood-adjusted temperature.
        """
        psi_hat_pv = PersonalityVector(
            array=self._psi_hat, registry=self._registry,
        )

        raw_activations = engine.compute_activations(psi_hat_pv, scenario)

        if state is not None:
            energy_factor = 0.5 + 0.5 * state.energy
            raw_activations = raw_activations * energy_factor
            temperature = temperature * (1.0 + 0.3 * max(0, -state.mood))

        utilities = np.array([
            engine.utility(psi_hat_pv, scenario, a,
                           activations_override=raw_activations)
            for a in actions
        ])

        logits = utilities / max(temperature, 1e-8)
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()

    def compute_prediction_error(
        self,
        actual_probs: np.ndarray,
        predicted_probs: np.ndarray,
    ) -> float:
        """H(q, p̂) normalized to [0, 1]."""
        eps = 1e-10
        K = len(actual_probs)
        cross_entropy = -np.sum(actual_probs * np.log(predicted_probs + eps))
        max_entropy = np.log(K)
        return float(np.clip(cross_entropy / max(max_entropy, eps), 0.0, 1.0))


# =============================================================================
# 3. SELF-RELATED EMOTION DETECTORS
# =============================================================================

class SelfEmotionLabel(Enum):
    PRIDE = "pride"
    SHAME = "shame"
    AUTHENTICITY = "authenticity"
    IDENTITY_THREAT = "identity_threat"
    IDENTITY_CRISIS = "identity_crisis"


@dataclass
class SelfEmotionReading:
    label: SelfEmotionLabel
    intensity: float
    description: str

    def __repr__(self) -> str:
        return f"{self.label.value}: {self.intensity:.2f}"


class SelfEmotionDetector:
    """
    Detects self-related emotions from self-model metrics.

    Uses ONLY public properties/methods of SelfModel (no private access).
    """

    def __init__(self, params: SelfModelParams = SelfModelParams()):
        self.params = params

    def detect_all(
        self,
        self_model: SelfModel,
        prediction_error: float,
        outcome: float,
        personality: PersonalityVector,
        registry: DimensionRegistry,
    ) -> list[SelfEmotionReading]:
        readings = [
            self._detect_pride(prediction_error, outcome),
            self._detect_shame(prediction_error, self_model, personality, registry),
            self._detect_authenticity(self_model),
            self._detect_identity_threat(self_model),
            self._detect_identity_crisis(self_model),
        ]
        return [r for r in readings if r.intensity >= 0.10]

    def _detect_pride(
        self, prediction_error: float, outcome: float,
    ) -> SelfEmotionReading:
        accuracy = max(0, 1.0 - prediction_error)
        intensity = float(np.clip(accuracy * max(0, outcome), 0, 1))
        return SelfEmotionReading(
            SelfEmotionLabel.PRIDE, intensity,
            "Action consistent with self-concept and succeeded.",
        )

    def _detect_shame(
        self, prediction_error: float, self_model: SelfModel,
        personality: PersonalityVector, registry: DimensionRegistry,
    ) -> SelfEmotionReading:
        p = self.params
        N = personality["N"] if "N" in registry.keys else 0.5
        coherence_gap = self_model.current_coherence_gap()
        n_amplifier = 1.0 + p.N_shame_amplification * N
        intensity = float(np.clip(
            prediction_error * coherence_gap * p.shame_scaling * n_amplifier, 0, 1
        ))
        return SelfEmotionReading(
            SelfEmotionLabel.SHAME, intensity,
            "Behavior violated self-concept; self-discrepancy.",
        )

    def _detect_authenticity(
        self, self_model: SelfModel,
    ) -> SelfEmotionReading:
        threshold = self_model.params.coherence_threat_threshold
        gap = self_model.current_coherence_gap()
        intensity = float(np.clip(1.0 - gap / threshold, 0, 1))
        return SelfEmotionReading(
            SelfEmotionLabel.AUTHENTICITY, intensity,
            "Self-concept aligns with behavioral evidence.",
        )

    def _detect_identity_threat(
        self, self_model: SelfModel,
    ) -> SelfEmotionReading:
        if not self_model.sustained_coherence_threat():
            return SelfEmotionReading(
                SelfEmotionLabel.IDENTITY_THREAT, 0.0,
                "No sustained coherence threat.",
            )
        p = self_model.params
        history = self_model.coherence_history
        recent = history[-p.coherence_threat_window:]
        mean_excess = np.mean([max(0, g - p.coherence_threat_threshold) for g in recent])
        intensity = float(np.clip(mean_excess * 5.0, 0, 1))
        return SelfEmotionReading(
            SelfEmotionLabel.IDENTITY_THREAT, intensity,
            "Sustained evidence contradicts self-concept.",
        )

    def _detect_identity_crisis(
        self, self_model: SelfModel,
    ) -> SelfEmotionReading:
        p = self_model.params
        drift = self_model.current_identity_drift()
        if drift < p.drift_crisis_threshold:
            return SelfEmotionReading(
                SelfEmotionLabel.IDENTITY_CRISIS, 0.0,
                "Self-model stable relative to anchor.",
            )
        intensity = float(np.clip(
            (drift - p.drift_crisis_threshold) / (1.0 - p.drift_crisis_threshold), 0, 1,
        ))
        return SelfEmotionReading(
            SelfEmotionLabel.IDENTITY_CRISIS, intensity,
            "Self-concept has drifted far from original identity.",
        )


# =============================================================================
# 4. SELF-AWARE SIMULATION LOOP (inherits from TemporalSimulator)
# =============================================================================

@dataclass
class SelfAwareTickResult:
    """Extends TickResult with self-model data."""
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

    v2 FIX: no tick-loop duplication.  All temporal logic (state
    modulation, energy gating, mood temperature, withdrawal,
    action effort, memory, emotion detection) is inherited from
    TemporalSimulator.  This class adds only self-model hooks:
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
    ):
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
        self,
        scenario: Scenario,
        outcome: Optional[float] = None,
    ) -> SelfAwareTickResult:
        """
        One tick with self-model integration.

            1. Self-model predicts (state-aware) — pre-decision hook.
            2. super().tick() — full temporal pipeline.
            3. Self-model updates — post-decision hook.
            4. Self-emotions detected.
        """
        true_psi = self.personality.to_array()

        # 1. Pre-decision: self-model predicts using current state.
        predicted_probs = self.self_model.predict_action_distribution(
            scenario, self.actions, self.engine, self.temperature,
            state=self.state,
        )

        # 2. Full temporal tick (inherited).
        base: TickResult = super().tick(scenario, outcome)

        # 3. Post-decision: self-model update.
        modifier_list = [a.modifiers for a in self.actions]
        sm_metrics = self.self_model.update(base.probabilities, modifier_list)

        # 4. Prediction error + self-emotions.
        pred_error = self.self_model.compute_prediction_error(
            base.probabilities, predicted_probs,
        )
        self_accuracy = self.self_model.compute_self_accuracy(true_psi)

        self_emotions = self.self_emotion_detector.detect_all(
            self.self_model, pred_error, base.outcome,
            self.personality, self.registry,
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
        )
