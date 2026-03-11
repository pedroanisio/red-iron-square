"""
Temporal Affective Engine (Layers 1–3)
=======================================

Extends the stateless personality framework (Layer 0) with:

    Layer 1 — Temporal State:   Mutable internal state (mood, arousal, energy)
                                that evolves across simulation ticks.
    Layer 2 — Memory:           Decision history, outcomes, counterfactuals,
                                and attachment traces for retrospection.
    Layer 3 — Affective Engine: Emergent emotion detection from the combination
                                of activations, internal state, and memory.

Architecture:

    ┌───────────────────────────────────────────────────────────┐
    │  for t in timesteps:                                      │
    │    1. scenario(t) arrives                                 │
    │    2. state modulates activations  (state → activation)   │
    │    3. action selected              (Layer 0 engine)       │
    │    4. outcome observed             (environment)          │
    │    5. state updated                (Layer 1)              │
    │    6. memory stored                (Layer 2)              │
    │    7. emotions detected            (Layer 3)              │
    └───────────────────────────────────────────────────────────┘

Theoretical grounding:
    - Russell's Circumplex Model: emotions as valence × arousal coordinates.
      Russell, "A Circumplex Model of Affect", J. Personality and Social
      Psychology 39(6), 1980, pp. 1161–1178.
    - Appraisal Theory: emotions arise from cognitive evaluation of events
      relative to goals and coping capacity.  Lazarus, "Emotion and
      Adaptation", Oxford University Press, 1991.
    - Counterfactual Regret: Loomes & Sugden, "Regret Theory: An Alternative
      Theory of Rational Choice Under Uncertainty", Economic Journal 92,
      1982, pp. 805–824.
    - Grit: Duckworth et al., "Grit: Perseverance and Passion for Long-Term
      Goals", J. Personality and Social Psychology 92(6), 2007, pp. 1087–1101.
    - Saudade: modeled as nostalgic longing via memory-present valence gap;
      see Neto, "Saudade: The Portuguese Word for Longing", in The Handbook
      of Solitude (Coplan & Bowker, eds.), 2013.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence
from collections import deque
from enum import Enum

from src.simulation.personality_framework import (
    DimensionRegistry,
    PersonalityVector,
    Scenario,
    Action,
    DecisionEngine,
    HyperParameters,
    _validate_unit_interval,
    _validate_real,
)


# =============================================================================
# LAYER 1 — TEMPORAL STATE
# =============================================================================

@dataclass
class AgentState:
    """
    Mutable internal state that evolves across simulation ticks.

    Dimensions (all in [-1, 1] or [0, 1]):
        mood:         Valence axis.  -1 = deeply negative, +1 = elated.
        arousal:      Activation axis.  0 = sluggish, 1 = wired.
        energy:       Cognitive/physical resource.  0 = depleted, 1 = full.
        satisfaction: Recent need fulfillment.  0 = deprived, 1 = satiated.
        frustration:  Accumulated blocked-goal signal.  0 = none, 1 = max.

    These are NOT personality traits (those are fixed in ψ).
    These are transient states that change every tick based on outcomes,
    scenarios, personality, and decay dynamics.
    """
    mood: float = 0.0           # [-1, 1]
    arousal: float = 0.5        # [0, 1]
    energy: float = 1.0         # [0, 1]
    satisfaction: float = 0.5   # [0, 1]
    frustration: float = 0.0    # [0, 1]

    def __post_init__(self):
        if not (-1.0 <= self.mood <= 1.0):
            raise ValueError(f"mood={self.mood} outside [-1, 1]")
        for name in ("arousal", "energy", "satisfaction", "frustration"):
            _validate_unit_interval(name, getattr(self, name))

    def to_array(self) -> np.ndarray:
        return np.array([self.mood, self.arousal, self.energy,
                         self.satisfaction, self.frustration])

    def copy(self) -> "AgentState":
        return AgentState(
            mood=self.mood, arousal=self.arousal, energy=self.energy,
            satisfaction=self.satisfaction, frustration=self.frustration,
        )

    def __repr__(self) -> str:
        return (f"State(mood={self.mood:+.2f}, arousal={self.arousal:.2f}, "
                f"energy={self.energy:.2f}, sat={self.satisfaction:.2f}, "
                f"frust={self.frustration:.2f})")


@dataclass
class StateTransitionParams:
    """
    Tunable parameters for how AgentState evolves between ticks.

    All decay rates are per-tick multipliers in (0, 1].
    A decay of 0.95 means 5% regression toward baseline per tick.
    """
    # Decay rates (per tick, toward baseline)
    mood_decay: float = 0.92         # Mood regresses toward 0
    arousal_decay: float = 0.88      # Arousal regresses toward resting level
    energy_decay: float = 0.90       # Energy regresses toward resting level
    satisfaction_decay: float = 0.90  # Satisfaction regresses toward 0.5
    frustration_decay: float = 0.85  # Frustration decays toward 0

    # Outcome sensitivity (how much a +1 or -1 outcome shifts state)
    outcome_mood_gain: float = 0.25     # Positive outcome → mood boost
    outcome_mood_loss: float = 0.35     # Negative outcome → mood drop (asymmetric)
    outcome_arousal_spike: float = 0.15  # Any outcome → arousal spike
    outcome_satisfaction: float = 0.20   # Positive outcome → satisfaction
    outcome_frustration: float = 0.25    # Negative outcome → frustration

    # Energy cost (proportional to action effort + stress)
    energy_cost_per_effort: float = 0.015  # Scaled by action modifier L2 norm
    energy_cost_stress: float = 0.03       # Additional cost under high-stress scenarios
    energy_resting_level: float = 0.80     # Level energy decays toward when idle

    # Personality modulation
    N_mood_sensitivity: float = 0.5   # High N amplifies mood drops
    R_frustration_damping: float = 0.4  # High R dampens frustration accumulation
    E_arousal_baseline: float = 0.15  # High E raises resting arousal

    def __post_init__(self):
        for name, val in vars(self).items():
            _validate_real(name, val)


def update_state(
    state: AgentState,
    outcome: float,
    personality: PersonalityVector,
    scenario: Scenario,
    params: StateTransitionParams = StateTransitionParams(),
    action_effort: float = 0.0,
) -> AgentState:
    """
    Evolve the agent's internal state based on the tick's outcome.

    Args:
        state:         Current internal state.
        outcome:       How well the action went, in [-1, 1].
        personality:   The agent's fixed personality vector.
        scenario:      The scenario that was faced this tick.
        params:        Transition parameters.
        action_effort: Effort magnitude of the chosen action (e.g.,
                       L2 norm of action modifiers).  Higher effort
                       drains more energy.

    Returns:
        New AgentState for the next tick.
    """
    p = params

    # Extract relevant personality traits (safe fallback if dimension missing).
    keys = set(personality.registry.keys)
    N = personality["N"] if "N" in keys else 0.5
    R = personality["R"] if "R" in keys else 0.5
    E = personality["E"] if "E" in keys else 0.5

    # Extract stress from scenario (dimension N's stimulus, if present).
    stress = scenario["N"] if "N" in keys else 0.5

    # ── Mood ──────────────────────────────────────────────────────────
    mood = state.mood * p.mood_decay
    if outcome > 0:
        mood += outcome * p.outcome_mood_gain
    else:
        # Loss aversion in mood: negative outcomes hit harder.
        # High N amplifies the drop.
        mood += outcome * p.outcome_mood_loss * (1.0 + p.N_mood_sensitivity * N)

    # ── Arousal ───────────────────────────────────────────────────────
    resting_arousal = 0.4 + p.E_arousal_baseline * E  # Extraverts rest higher
    arousal = state.arousal * p.arousal_decay + (1 - p.arousal_decay) * resting_arousal
    arousal += abs(outcome) * p.outcome_arousal_spike  # Any strong outcome spikes arousal

    # ── Energy ────────────────────────────────────────────────────────
    # EMA toward resting level, minus effort and stress costs.
    resting_energy = p.energy_resting_level
    energy = state.energy * p.energy_decay + (1 - p.energy_decay) * resting_energy
    effort_cost = p.energy_cost_per_effort * action_effort
    stress_cost = p.energy_cost_stress * stress
    energy -= effort_cost + stress_cost

    # ── Satisfaction ──────────────────────────────────────────────────
    # EMA toward 0.5 (neutral), boosted by positive outcomes.
    satisfaction = state.satisfaction * p.satisfaction_decay + (1 - p.satisfaction_decay) * 0.5
    if outcome > 0:
        satisfaction += outcome * p.outcome_satisfaction

    # ── Frustration ───────────────────────────────────────────────────
    frustration = state.frustration * p.frustration_decay
    if outcome < 0:
        # High R dampens frustration accumulation.
        damping = 1.0 - p.R_frustration_damping * R
        frustration += abs(outcome) * p.outcome_frustration * damping

    return AgentState(
        mood=float(np.clip(mood, -1.0, 1.0)),
        arousal=float(np.clip(arousal, 0.0, 1.0)),
        energy=float(np.clip(energy, 0.0, 1.0)),
        satisfaction=float(np.clip(satisfaction, 0.0, 1.0)),
        frustration=float(np.clip(frustration, 0.0, 1.0)),
    )


# =============================================================================
# LAYER 2 — MEMORY
# =============================================================================

@dataclass
class MemoryEntry:
    """
    A single episodic memory: what happened at tick t.

    Fields:
        tick:            Simulation timestep.
        scenario:        The scenario that was faced.
        action:          The action that was chosen.
        outcome:         How well it went, in [-1, 1].
        counterfactual:  Best utility among *unchosen* actions minus chosen
                         action's utility.  Positive = "I could have done better."
        state_snapshot:  The agent's internal state at that tick.
        valence:         Subjective emotional valence of this memory, in [-1, 1].
                         Computed as f(outcome, mood_at_time).
    """
    tick: int
    scenario_name: str
    action_name: str
    outcome: float
    counterfactual: float
    state_snapshot: AgentState
    valence: float


class MemoryBank:
    """
    Stores and queries episodic memories.

    Supports:
        - Chronological retrieval
        - Valence-weighted queries (for saudade, regret)
        - Rolling statistics (for boredom, melancholy detection)
    """

    def __init__(self, max_size: int = 500):
        self._entries: deque[MemoryEntry] = deque(maxlen=max_size)

    def store(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def recent(self, n: int) -> list[MemoryEntry]:
        """Last n memories, most recent first."""
        return list(reversed(list(self._entries)))[:n]

    def mean_outcome(self, window: int = 10) -> float:
        """Mean outcome over the last `window` entries."""
        recent = self.recent(window)
        if not recent:
            return 0.0
        return float(np.mean([m.outcome for m in recent]))

    def mean_valence(self, window: int = 10) -> float:
        """Mean emotional valence over the last `window` entries."""
        recent = self.recent(window)
        if not recent:
            return 0.0
        return float(np.mean([m.valence for m in recent]))

    def mean_arousal(self, window: int = 10) -> float:
        """Mean arousal over the last `window` entries."""
        recent = self.recent(window)
        if not recent:
            return 0.5
        return float(np.mean([m.state_snapshot.arousal for m in recent]))

    def peak_valence(self, window: int = 50) -> float:
        """Highest valence in the last `window` entries (for saudade baseline)."""
        recent = self.recent(window)
        if not recent:
            return 0.0
        return float(max(m.valence for m in recent))

    def total_regret(self, window: int = 10) -> float:
        """Sum of positive counterfactuals ('I should have...') in recent window."""
        recent = self.recent(window)
        return float(sum(max(0, m.counterfactual) for m in recent))

    def consecutive_failures(self) -> int:
        """Count of consecutive negative outcomes from the most recent tick."""
        count = 0
        for m in reversed(list(self._entries)):
            if m.outcome < 0:
                count += 1
            else:
                break
        return count

    def outcome_variance(self, window: int = 10) -> float:
        """Variance of recent outcomes (low variance + low arousal → boredom)."""
        recent = self.recent(window)
        if len(recent) < 2:
            return 0.0
        return float(np.var([m.outcome for m in recent]))


# =============================================================================
# LAYER 3 — AFFECTIVE ENGINE (EMOTION DETECTION)
# =============================================================================

class EmotionLabel(Enum):
    """
    Named emotional states that can emerge from the simulation.

    Each is defined by its position in the valence × arousal space
    plus specific computational prerequisites.
    """
    # Valence+, Arousal+
    EXCITEMENT = "excitement"
    ENTHUSIASM = "enthusiasm"

    # Valence+, Arousal−
    CONTENTMENT = "contentment"

    # Valence−, Arousal+
    FRUSTRATION_EMO = "frustration"   # Distinct from state.frustration (signal vs label)
    ANXIETY = "anxiety"

    # Valence−, Arousal−
    BOREDOM = "boredom"
    MELANCHOLY = "melancholy"
    SAUDADE = "saudade"

    # Requires memory
    REGRET = "regret"

    # Requires persistence
    PERSEVERANCE = "perseverance"
    GRIT = "grit"

    # Trait-derived (always available)
    CREATIVITY = "creativity"
    FOCUS = "focus"


@dataclass
class EmotionReading:
    """A single detected emotion with its intensity."""
    label: EmotionLabel
    intensity: float   # [0, 1]
    description: str   # Human-readable explanation

    def __repr__(self) -> str:
        return f"{self.label.value}: {self.intensity:.2f}"


@dataclass
class EmotionThresholds:
    """
    Tunable thresholds for emotion emergence.

    These control how sensitive the detection is.  Lower thresholds
    mean the emotion is detected more easily / frequently.
    """
    # Minimum intensity to report an emotion
    report_threshold: float = 0.15

    # Boredom: arousal must be below this AND outcome variance must be low
    boredom_arousal_ceiling: float = 0.35
    boredom_variance_ceiling: float = 0.05

    # Melancholy: mood must be below this for sustained period
    melancholy_mood_floor: float = -0.20
    melancholy_window: int = 8  # How many ticks of low mood

    # Saudade: gap between peak past valence and current mood
    saudade_gap_threshold: float = 0.40

    # Regret: cumulative counterfactual threshold
    regret_counterfactual_threshold: float = 0.30

    # Perseverance: consecutive failures while still acting
    perseverance_failure_threshold: int = 3

    # Grit: same as perseverance but over longer horizon
    grit_failure_threshold: int = 8


class AffectiveEngine:
    """
    Detects emergent emotions from the combination of:
        - Current activation vector (from Layer 0)
        - Current internal state (from Layer 1)
        - Memory bank (from Layer 2)
        - Personality vector (fixed traits)

    Emotions are NOT hardcoded outputs.  They EMERGE from specific
    patterns in the data, analogous to how a doctor reads symptoms.

    Each detect_* method returns an intensity in [0, 1] or 0 if the
    conditions are not met.
    """

    def __init__(self, thresholds: EmotionThresholds = EmotionThresholds()):
        self.th = thresholds

    def detect_all(
        self,
        activations: np.ndarray,
        state: AgentState,
        personality: PersonalityVector,
        memory: MemoryBank,
        registry: DimensionRegistry,
        is_still_acting: bool = True,
    ) -> list[EmotionReading]:
        """
        Run all emotion detectors and return those above report_threshold.
        """
        readings = [
            self._detect_excitement(state),
            self._detect_enthusiasm(state),
            self._detect_contentment(state),
            self._detect_frustration_emo(state),
            self._detect_anxiety(state, personality, registry),
            self._detect_boredom(state, memory, personality, registry),
            self._detect_melancholy(state, memory, personality, registry),
            self._detect_saudade(state, memory, personality, registry),
            self._detect_regret(memory),
            self._detect_perseverance(memory, is_still_acting),
            self._detect_grit(memory, is_still_acting),
            self._detect_creativity(activations, state, registry),
            self._detect_focus(activations, state, registry),
        ]
        return [r for r in readings if r.intensity >= self.th.report_threshold]

    # ── Valence+, Arousal+ ──────────────────────────────────────────

    def _detect_excitement(self, state: AgentState) -> EmotionReading:
        """Excitement: high positive mood + high arousal."""
        intensity = max(0, state.mood) * state.arousal
        return EmotionReading(
            EmotionLabel.EXCITEMENT, intensity,
            "High-energy positive affect; anticipation or thrill.",
        )

    def _detect_enthusiasm(self, state: AgentState) -> EmotionReading:
        """Enthusiasm: moderate-to-high positive mood + moderate arousal."""
        # Enthusiasm is broader/warmer than excitement; doesn't need peak arousal.
        mood_component = max(0, state.mood)
        arousal_component = min(state.arousal, 0.8)  # Caps out; enthusiasm isn't frantic
        intensity = mood_component * arousal_component * state.energy
        return EmotionReading(
            EmotionLabel.ENTHUSIASM, intensity,
            "Warm, energized positive engagement.",
        )

    # ── Valence+, Arousal− ──────────────────────────────────────────

    def _detect_contentment(self, state: AgentState) -> EmotionReading:
        """Contentment: positive mood + low arousal + high satisfaction."""
        intensity = max(0, state.mood) * (1 - state.arousal) * state.satisfaction
        return EmotionReading(
            EmotionLabel.CONTENTMENT, intensity,
            "Calm, satisfied state; things are good and quiet.",
        )

    # ── Valence−, Arousal+ ──────────────────────────────────────────

    def _detect_frustration_emo(self, state: AgentState) -> EmotionReading:
        """Frustration (emotion): high frustration signal + arousal."""
        intensity = state.frustration * state.arousal
        return EmotionReading(
            EmotionLabel.FRUSTRATION_EMO, intensity,
            "Goals are blocked; agitated negative state.",
        )

    def _detect_anxiety(
        self, state: AgentState, psi: PersonalityVector, reg: DimensionRegistry
    ) -> EmotionReading:
        """Anxiety: high arousal + negative mood + high N."""
        N = psi["N"] if "N" in reg.keys else 0.5
        intensity = state.arousal * max(0, -state.mood) * N
        return EmotionReading(
            EmotionLabel.ANXIETY, intensity,
            "Tense anticipation of negative outcomes.",
        )

    # ── Valence−, Arousal− ──────────────────────────────────────────

    def _detect_boredom(
        self, state: AgentState, memory: MemoryBank,
        psi: PersonalityVector, reg: DimensionRegistry,
    ) -> EmotionReading:
        """
        Boredom: low arousal + low outcome variance + understimulation.

        High O and high E agents have a higher desired stimulation level,
        so they get bored more easily.
        """
        O = psi["O"] if "O" in reg.keys else 0.5
        E = psi["E"] if "E" in reg.keys else 0.5
        desired_arousal = 0.3 + 0.25 * O + 0.2 * E  # Personality sets the bar

        arousal_deficit = max(0, desired_arousal - state.arousal)
        variance = memory.outcome_variance(window=8)
        low_variance = max(0, self.th.boredom_variance_ceiling - variance) / self.th.boredom_variance_ceiling

        # Boredom = understimulated + monotonous
        intensity = float(np.clip(arousal_deficit * 2 * low_variance, 0, 1))
        return EmotionReading(
            EmotionLabel.BOREDOM, intensity,
            "Understimulated; craving novelty or challenge.",
        )

    def _detect_melancholy(
        self, state: AgentState, memory: MemoryBank,
        psi: PersonalityVector, reg: DimensionRegistry,
    ) -> EmotionReading:
        """
        Melancholy: sustained negative mood + low arousal + reflective traits.

        Requires mood to have been negative for several ticks (not a single
        bad outcome — that's frustration, not melancholy).
        """
        O = psi["O"] if "O" in reg.keys else 0.5
        I = psi["I"] if "I" in reg.keys else 0.5

        recent_valence = memory.mean_valence(window=self.th.melancholy_window)
        sustained_negativity = max(0, -recent_valence)  # 0 if recent valence ≥ 0

        mood_negativity = max(0, -state.mood)
        low_arousal = max(0, 0.5 - state.arousal)
        reflectiveness = (O + I) / 2  # Reflective personality amplifies

        intensity = float(np.clip(
            sustained_negativity * mood_negativity * (1 + low_arousal) * (0.5 + 0.5 * reflectiveness),
            0, 1
        ))
        return EmotionReading(
            EmotionLabel.MELANCHOLY, intensity,
            "Persistent reflective sadness; wistful low energy.",
        )

    def _detect_saudade(
        self, state: AgentState, memory: MemoryBank,
        psi: PersonalityVector, reg: DimensionRegistry,
    ) -> EmotionReading:
        """
        Saudade: longing for a valued past state that is now absent.

        Operationalized as the gap between the peak past valence and
        the current mood, modulated by Tradition (attachment to the past)
        and Idealism (yearning for what should be).

        Requires enough memories to have established a "golden past."
        """
        T = psi["T"] if "T" in reg.keys else 0.5
        I_val = psi["I"] if "I" in reg.keys else 0.5

        if len(memory) < 5:
            return EmotionReading(EmotionLabel.SAUDADE, 0.0,
                                  "Not enough memories for saudade.")

        peak = memory.peak_valence(window=50)
        current = state.mood  # [-1, 1]
        gap = peak - current  # Positive if the past was better

        if gap < self.th.saudade_gap_threshold:
            return EmotionReading(EmotionLabel.SAUDADE, 0.0,
                                  "Past is not much better than present.")

        # Tradition amplifies attachment to what was lost.
        # Idealism amplifies the yearning.
        personality_amplifier = 0.5 + 0.3 * T + 0.2 * I_val
        low_arousal = max(0, 0.6 - state.arousal)  # Saudade is quiet, not frantic

        intensity = float(np.clip(
            (gap - self.th.saudade_gap_threshold) * personality_amplifier * (1 + low_arousal),
            0, 1
        ))
        return EmotionReading(
            EmotionLabel.SAUDADE, intensity,
            "Nostalgic longing for a valued past that is now absent.",
        )

    # ── Memory-dependent ────────────────────────────────────────────

    def _detect_regret(self, memory: MemoryBank) -> EmotionReading:
        """
        Regret: accumulated counterfactual evidence that unchosen actions
        would have been better.

        counterfactual > 0 means "I could have done better."
        """
        total_regret = memory.total_regret(window=10)
        intensity = float(np.clip(
            (total_regret - self.th.regret_counterfactual_threshold) * 2, 0, 1
        ))
        return EmotionReading(
            EmotionLabel.REGRET, intensity,
            "Counterfactual awareness: unchosen paths were better.",
        )

    # ── Persistence-dependent ───────────────────────────────────────

    def _detect_perseverance(
        self, memory: MemoryBank, still_acting: bool
    ) -> EmotionReading:
        """
        Perseverance: continuing to act despite consecutive failures.

        The agent must have failed recently AND still be choosing to engage
        (not withdrawing or fleeing).
        """
        failures = memory.consecutive_failures()
        if not still_acting or failures < self.th.perseverance_failure_threshold:
            return EmotionReading(EmotionLabel.PERSEVERANCE, 0.0, "Not persevering.")

        intensity = float(np.clip(
            (failures - self.th.perseverance_failure_threshold + 1) * 0.2, 0, 1
        ))
        return EmotionReading(
            EmotionLabel.PERSEVERANCE, intensity,
            "Continuing despite repeated failure.",
        )

    def _detect_grit(
        self, memory: MemoryBank, still_acting: bool
    ) -> EmotionReading:
        """
        Grit: perseverance over a longer horizon.

        Same mechanism as perseverance but requires more sustained failure
        (Duckworth's distinction: grit = perseverance + passion over time).
        """
        failures = memory.consecutive_failures()
        if not still_acting or failures < self.th.grit_failure_threshold:
            return EmotionReading(EmotionLabel.GRIT, 0.0, "Not showing grit.")

        intensity = float(np.clip(
            (failures - self.th.grit_failure_threshold + 1) * 0.15, 0, 1
        ))
        return EmotionReading(
            EmotionLabel.GRIT, intensity,
            "Sustained perseverance over extended adversity.",
        )

    # ── Trait-derived (always computable) ───────────────────────────

    def _detect_creativity(
        self, activations: np.ndarray, state: AgentState,
        reg: DimensionRegistry,
    ) -> EmotionReading:
        """
        Creativity: high O-activation + adequate energy + moderate arousal.

        Creativity peaks at moderate arousal (Yerkes-Dodson) — too wired
        or too sluggish both reduce it.  Uses a Gaussian bell curve
        centered at arousal = 0.5, not a piecewise-linear triangle.
        """
        if "O" not in reg.keys:
            return EmotionReading(EmotionLabel.CREATIVITY, 0.0, "No O dimension.")

        o_act = activations[reg.index("O")]
        # Smooth Gaussian Yerkes-Dodson: peak at arousal=0.5, width controlled by k.
        arousal_factor = float(np.exp(-10.0 * (state.arousal - 0.5) ** 2))

        intensity = float(np.clip(o_act * arousal_factor * state.energy, 0, 1))
        return EmotionReading(
            EmotionLabel.CREATIVITY, intensity,
            "Creative capacity: open activation + optimal arousal + energy.",
        )

    def _detect_focus(
        self, activations: np.ndarray, state: AgentState,
        reg: DimensionRegistry,
    ) -> EmotionReading:
        """
        Focus: high C-activation + high energy + low frustration.

        Focus degrades with low energy (fatigue) and high frustration
        (attentional hijacking).
        """
        if "C" not in reg.keys:
            return EmotionReading(EmotionLabel.FOCUS, 0.0, "No C dimension.")

        c_act = activations[reg.index("C")]
        intensity = float(np.clip(
            c_act * state.energy * (1 - state.frustration * 0.7), 0, 1
        ))
        return EmotionReading(
            EmotionLabel.FOCUS, intensity,
            "Attentional control: conscientious activation + energy − frustration.",
        )


# =============================================================================
# SIMULATION LOOP
# =============================================================================

@dataclass
class TickResult:
    """Complete result for a single simulation tick."""
    tick: int
    scenario: Scenario
    action: Action
    outcome: float
    state_before: AgentState
    state_after: AgentState
    activations: np.ndarray
    emotions: list[EmotionReading]
    probabilities: np.ndarray


class TemporalSimulator:
    """
    The main simulation loop: personality × scenarios × time → emotion traces.

    The tick pipeline:
        1. Compute state-modulated activations.
        2. Compute effective temperature (mood-modulated).
        3. Select action via Boltzmann (using modulated activations).
        4. Resolve outcome.
        5. Compute counterfactual.
        6. Determine withdrawal (is_still_acting).
        7. Compute action effort and update state.
        8. Store memory.
        9. Detect emotions.

    Helper methods (_compute_*, _resolve_*, _store_*) are exposed as
    protected methods so subclasses (e.g., SelfAwareSimulator) can
    reuse them without duplicating the tick pipeline.
    """

    # Name of the canonical withdrawal action.  If an action with this
    # name exists in the action set AND is chosen, is_still_acting=False.
    WITHDRAW_ACTION_NAME = "Withdraw"

    def __init__(
        self,
        personality: PersonalityVector,
        actions: Sequence[Action],
        engine: DecisionEngine,
        *,
        initial_state: Optional[AgentState] = None,
        state_params: StateTransitionParams = StateTransitionParams(),
        emotion_thresholds: EmotionThresholds = EmotionThresholds(),
        memory_size: int = 500,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
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

    # -----------------------------------------------------------------
    # Protected helpers (reusable by subclasses)
    # -----------------------------------------------------------------

    def _compute_modulated_activations(self, scenario: Scenario) -> np.ndarray:
        """Compute raw activations and gate by energy."""
        raw = self.engine.compute_activations(self.personality, scenario)
        energy_factor = 0.5 + 0.5 * self.state.energy
        return raw * energy_factor

    def _compute_effective_temperature(self) -> float:
        """Mood-modulated temperature: negative mood → more random."""
        return self.temperature * (1.0 + 0.3 * max(0, -self.state.mood))

    def _resolve_outcome(
        self, outcome: Optional[float], activations: np.ndarray,
        chosen_action: Action, scenario: Scenario,
    ) -> float:
        """Resolve outcome: use provided value or stochastic model."""
        if outcome is not None:
            return outcome
        u = self.engine.utility(
            self.personality, scenario, chosen_action,
            activations_override=activations,
        )
        return float(np.clip(self.rng.normal(0.2 * u, 0.3), -1.0, 1.0))

    def _compute_counterfactual(
        self, activations: np.ndarray, chosen_action: Action, scenario: Scenario,
    ) -> float:
        """Counterfactual: best unchosen utility minus chosen utility."""
        utilities = np.array([
            self.engine.utility(self.personality, scenario, a,
                                activations_override=activations)
            for a in self.actions
        ])
        chosen_idx = next(
            i for i, a in enumerate(self.actions) if a.name == chosen_action.name
        )
        unchosen = np.delete(utilities, chosen_idx)
        if len(unchosen) == 0:
            return 0.0
        return float(max(unchosen) - utilities[chosen_idx])

    def _determine_is_still_acting(
        self, chosen_action: Action, new_state: AgentState,
    ) -> bool:
        """
        Determine whether the agent is still actively engaging.

        Returns False (withdrawn) if:
          - The agent chose the canonical Withdraw action, OR
          - Energy is fully depleted (0.0), OR
          - Frustration is maxed (1.0).

        This makes perseverance/grit detectors behaviorally meaningful:
        they can only trigger when the agent COULD withdraw but doesn't.
        """
        if chosen_action.name == self.WITHDRAW_ACTION_NAME:
            return False
        if new_state.energy <= 0.0 + 1e-9:
            return False
        if new_state.frustration >= 1.0 - 1e-9:
            return False
        return True

    @staticmethod
    def _compute_action_effort(action: Action) -> float:
        """Action effort = L2 norm of the modifier vector."""
        return float(np.linalg.norm(action.modifiers))

    def _store_memory(
        self, tick: int, state_before: AgentState, scenario: Scenario,
        chosen_action: Action, outcome: float, counterfactual: float,
    ) -> None:
        """Create and store a memory entry."""
        valence = float(np.clip(
            outcome * 0.6 + state_before.mood * 0.4, -1, 1
        ))
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

    # -----------------------------------------------------------------
    # Main tick
    # -----------------------------------------------------------------

    def tick(
        self,
        scenario: Scenario,
        outcome: Optional[float] = None,
    ) -> TickResult:
        """Execute one simulation tick."""

        state_before = self.state.copy()

        # 1-2. Activations and temperature.
        activations = self._compute_modulated_activations(scenario)
        temperature = self._compute_effective_temperature()

        # 3. Decide.
        chosen_action, probs = self.engine.decide(
            self.personality, scenario, self.actions,
            temperature=temperature,
            rng=self.rng,
            activations_override=activations,
        )

        # 4. Outcome.
        outcome = self._resolve_outcome(outcome, activations, chosen_action, scenario)

        # 5. Counterfactual.
        counterfactual = self._compute_counterfactual(activations, chosen_action, scenario)

        # 6-7. State update with action effort.
        action_effort = self._compute_action_effort(chosen_action)
        new_state = update_state(
            self.state, outcome, self.personality, scenario,
            self.state_params, action_effort=action_effort,
        )

        # 6b. Withdrawal detection.
        is_acting = self._determine_is_still_acting(chosen_action, new_state)

        # 8. Memory.
        self._store_memory(
            self._tick_counter, state_before, scenario,
            chosen_action, outcome, counterfactual,
        )

        # 9. Emotions.
        emotions = self.affect.detect_all(
            activations, new_state, self.personality, self.memory,
            self.registry,
            is_still_acting=is_acting,
        )

        self.state = new_state
        self._tick_counter += 1

        return TickResult(
            tick=self._tick_counter - 1,
            scenario=scenario,
            action=chosen_action,
            outcome=outcome,
            state_before=state_before,
            state_after=new_state,
            activations=activations,
            emotions=emotions,
            probabilities=probs,
        )

    @property
    def tick_count(self) -> int:
        return self._tick_counter

    @property
    def current_state(self) -> AgentState:
        return self.state


# =============================================================================
# CONVENIENCE: SCENARIO GENERATORS
# =============================================================================

def generate_scenario_sequence(
    registry: DimensionRegistry,
    n_ticks: int,
    pattern: str = "crisis_recovery",
    rng: Optional[np.random.Generator] = None,
) -> list[Scenario]:
    """
    Generate a sequence of scenarios for simulation.

    Patterns:
        'stable':          Low-variance, moderate stimulation throughout.
        'crisis_recovery': Calm → crisis → slow recovery → calm.
        'monotony':        Increasingly boring (stimulation decays).
        'random':          i.i.d. uniform random scenarios.
        'loss':            Good period → sudden loss → aftermath.
    """
    rng = rng or np.random.default_rng()
    n = registry.size
    scenarios = []

    # Registry-aware stress index (not hardcoded to position 4).
    stress_idx = registry.index("N") if "N" in set(registry.keys) else None

    for t in range(n_ticks):
        frac = t / max(1, n_ticks - 1)

        if pattern == "stable":
            base = 0.4 + 0.1 * rng.random(n)

        elif pattern == "crisis_recovery":
            if frac < 0.3:
                base = 0.3 + 0.1 * rng.random(n)  # Calm
            elif frac < 0.5:
                base = 0.7 + 0.2 * rng.random(n)  # Crisis
                if stress_idx is not None:
                    base[stress_idx] = 0.9  # High stress
            elif frac < 0.7:
                base = 0.5 + 0.15 * rng.random(n)  # Turbulence
            else:
                base = 0.3 + 0.1 * rng.random(n)  # Recovery

        elif pattern == "monotony":
            decay = max(0.05, 1.0 - frac * 0.9)
            base = decay * (0.3 + 0.1 * rng.random(n))

        elif pattern == "loss":
            if frac < 0.4:
                base = 0.5 + 0.2 * rng.random(n)  # Good times
            elif frac < 0.5:
                base = 0.8 + 0.15 * rng.random(n)  # Sudden crisis
                if stress_idx is not None:
                    base[stress_idx] = 0.95  # Extreme stress
            else:
                base = 0.25 + 0.1 * rng.random(n)  # Low-stimulation aftermath

        else:  # random
            base = rng.random(n)

        base = np.clip(base, 0, 1)
        scenarios.append(Scenario(
            array=base, registry=registry,
            name=f"tick_{t}_{pattern}",
        ))

    return scenarios


def generate_outcome_sequence(
    n_ticks: int,
    pattern: str = "crisis_recovery",
    rng: Optional[np.random.Generator] = None,
) -> list[float]:
    """
    Generate outcome values matching a scenario pattern.

    Returns list of floats in [-1, 1].
    """
    rng = rng or np.random.default_rng()
    outcomes = []

    for t in range(n_ticks):
        frac = t / max(1, n_ticks - 1)

        if pattern == "stable":
            o = rng.normal(0.2, 0.2)
        elif pattern == "crisis_recovery":
            if frac < 0.3:
                o = rng.normal(0.3, 0.15)
            elif frac < 0.5:
                o = rng.normal(-0.5, 0.25)  # Failure during crisis
            elif frac < 0.7:
                o = rng.normal(-0.1, 0.3)
            else:
                o = rng.normal(0.3, 0.15)
        elif pattern == "monotony":
            o = rng.normal(0.1, 0.05)  # Always meh
        elif pattern == "loss":
            if frac < 0.4:
                o = rng.normal(0.4, 0.15)  # Good outcomes
            elif frac < 0.5:
                o = rng.normal(-0.7, 0.15)  # Devastating loss
            else:
                o = rng.normal(-0.1, 0.2)
        else:
            o = rng.normal(0, 0.4)

        outcomes.append(float(np.clip(o, -1, 1)))
    return outcomes
