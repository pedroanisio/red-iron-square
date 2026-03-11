"""Affective engine: emergent emotion detection from activations, state, and memory."""

import numpy as np

from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import PersonalityVector
from src.shared.logging import get_logger
from src.temporal.emotions import (
    EmotionLabel,
    EmotionReading,
    EmotionThresholds,
)
from src.temporal.memory import MemoryBank
from src.temporal.state import AgentState

_log = get_logger(module="temporal.affective_engine")


class AffectiveEngine:
    """Detects emergent emotions from activations, state, memory, and personality.

    Emotions are NOT hardcoded — they EMERGE from specific patterns,
    analogous to how a doctor reads symptoms.
    """

    def __init__(
        self,
        thresholds: EmotionThresholds = EmotionThresholds(),
    ) -> None:
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
        """Run all detectors and return readings above report_threshold."""
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
        detected = [r for r in readings if r.intensity >= self.th.report_threshold]
        if detected:
            _log.debug(
                "emotions_detected",
                labels=[r.label.value for r in detected],
                count=len(detected),
            )
        return detected

    def _detect_excitement(
        self,
        state: AgentState,
    ) -> EmotionReading:
        """High positive mood + high arousal."""
        intensity = max(0, state.mood) * state.arousal
        return EmotionReading(
            label=EmotionLabel.EXCITEMENT,
            intensity=intensity,
            description=("High-energy positive affect; anticipation or thrill."),
        )

    def _detect_enthusiasm(
        self,
        state: AgentState,
    ) -> EmotionReading:
        """Moderate-to-high positive mood + moderate arousal."""
        mood_component = max(0, state.mood)
        arousal_component = min(state.arousal, 0.8)
        intensity = mood_component * arousal_component * state.energy
        return EmotionReading(
            label=EmotionLabel.ENTHUSIASM,
            intensity=intensity,
            description=("Warm, energized positive engagement."),
        )

    def _detect_contentment(
        self,
        state: AgentState,
    ) -> EmotionReading:
        """Positive mood + low arousal + high satisfaction."""
        intensity = max(0, state.mood) * (1 - state.arousal) * state.satisfaction
        return EmotionReading(
            label=EmotionLabel.CONTENTMENT,
            intensity=intensity,
            description=("Calm, satisfied state; things are good and quiet."),
        )

    def _detect_frustration_emo(
        self,
        state: AgentState,
    ) -> EmotionReading:
        """High frustration signal + arousal."""
        intensity = state.frustration * state.arousal
        return EmotionReading(
            label=EmotionLabel.FRUSTRATION_EMO,
            intensity=intensity,
            description=("Goals are blocked; agitated negative state."),
        )

    def _detect_anxiety(
        self,
        state: AgentState,
        psi: PersonalityVector,
        reg: DimensionRegistry,
    ) -> EmotionReading:
        """High arousal + negative mood + high N."""
        trait_n = psi["N"] if "N" in reg.keys else 0.5
        intensity = state.arousal * max(0, -state.mood) * trait_n
        return EmotionReading(
            label=EmotionLabel.ANXIETY,
            intensity=intensity,
            description=("Tense anticipation of negative outcomes."),
        )

    def _detect_boredom(
        self,
        state: AgentState,
        memory: MemoryBank,
        psi: PersonalityVector,
        reg: DimensionRegistry,
    ) -> EmotionReading:
        """Low arousal + low outcome variance + understimulation."""
        trait_o = psi["O"] if "O" in reg.keys else 0.5
        trait_e = psi["E"] if "E" in reg.keys else 0.5
        desired_arousal = 0.3 + 0.25 * trait_o + 0.2 * trait_e
        arousal_deficit = max(0, desired_arousal - state.arousal)
        variance = memory.outcome_variance(window=8)
        ceiling = self.th.boredom_variance_ceiling
        low_variance = max(0, ceiling - variance) / ceiling
        intensity = float(np.clip(arousal_deficit * 2 * low_variance, 0, 1))
        return EmotionReading(
            label=EmotionLabel.BOREDOM,
            intensity=intensity,
            description=("Understimulated; craving novelty or challenge."),
        )

    def _detect_melancholy(
        self,
        state: AgentState,
        memory: MemoryBank,
        psi: PersonalityVector,
        reg: DimensionRegistry,
    ) -> EmotionReading:
        """Sustained negative mood + low arousal + reflective traits."""
        trait_o = psi["O"] if "O" in reg.keys else 0.5
        trait_i = psi["I"] if "I" in reg.keys else 0.5
        recent_valence = memory.mean_valence(
            window=self.th.melancholy_window,
        )
        sustained_negativity = max(0, -recent_valence)
        mood_negativity = max(0, -state.mood)
        low_arousal = max(0, 0.5 - state.arousal)
        reflectiveness = (trait_o + trait_i) / 2
        intensity = float(
            np.clip(
                sustained_negativity
                * mood_negativity
                * (1 + low_arousal)
                * (0.5 + 0.5 * reflectiveness),
                0,
                1,
            )
        )
        return EmotionReading(
            label=EmotionLabel.MELANCHOLY,
            intensity=intensity,
            description=("Persistent reflective sadness; wistful low energy."),
        )

    def _detect_saudade(
        self,
        state: AgentState,
        memory: MemoryBank,
        psi: PersonalityVector,
        reg: DimensionRegistry,
    ) -> EmotionReading:
        """Longing for a valued past state that is now absent."""
        trait_t = psi["T"] if "T" in reg.keys else 0.5
        trait_i = psi["I"] if "I" in reg.keys else 0.5
        if len(memory) < 5:
            return EmotionReading(
                label=EmotionLabel.SAUDADE,
                intensity=0.0,
                description="Not enough memories for saudade.",
            )
        peak = memory.peak_valence(window=50)
        gap = peak - state.mood
        if gap < self.th.saudade_gap_threshold:
            return EmotionReading(
                label=EmotionLabel.SAUDADE,
                intensity=0.0,
                description=("Past is not much better than present."),
            )
        personality_amplifier = 0.5 + 0.3 * trait_t + 0.2 * trait_i
        low_arousal = max(0, 0.6 - state.arousal)
        gap_delta = gap - self.th.saudade_gap_threshold
        intensity = float(
            np.clip(
                gap_delta * personality_amplifier * (1 + low_arousal),
                0,
                1,
            )
        )
        return EmotionReading(
            label=EmotionLabel.SAUDADE,
            intensity=intensity,
            description=("Nostalgic longing for a valued past that is now absent."),
        )

    def _detect_regret(
        self,
        memory: MemoryBank,
    ) -> EmotionReading:
        """Accumulated counterfactual evidence that unchosen actions were better."""
        total_regret = memory.total_regret(window=10)
        threshold = self.th.regret_counterfactual_threshold
        intensity = float(
            np.clip(
                (total_regret - threshold) * 2,
                0,
                1,
            )
        )
        return EmotionReading(
            label=EmotionLabel.REGRET,
            intensity=intensity,
            description=("Counterfactual awareness: unchosen paths were better."),
        )

    def _detect_perseverance(
        self,
        memory: MemoryBank,
        still_acting: bool,
    ) -> EmotionReading:
        """Continuing to act despite consecutive failures."""
        failures = memory.consecutive_failures()
        threshold = self.th.perseverance_failure_threshold
        if not still_acting or failures < threshold:
            return EmotionReading(
                label=EmotionLabel.PERSEVERANCE,
                intensity=0.0,
                description="Not persevering.",
            )
        intensity = float(
            np.clip(
                (failures - threshold + 1) * 0.2,
                0,
                1,
            )
        )
        return EmotionReading(
            label=EmotionLabel.PERSEVERANCE,
            intensity=intensity,
            description=("Continuing despite repeated failure."),
        )

    def _detect_grit(
        self,
        memory: MemoryBank,
        still_acting: bool,
    ) -> EmotionReading:
        """Sustained perseverance over extended adversity."""
        failures = memory.consecutive_failures()
        threshold = self.th.grit_failure_threshold
        if not still_acting or failures < threshold:
            return EmotionReading(
                label=EmotionLabel.GRIT,
                intensity=0.0,
                description="Not showing grit.",
            )
        intensity = float(
            np.clip(
                (failures - threshold + 1) * 0.15,
                0,
                1,
            )
        )
        return EmotionReading(
            label=EmotionLabel.GRIT,
            intensity=intensity,
            description=("Sustained perseverance over extended adversity."),
        )

    def _detect_creativity(
        self,
        activations: np.ndarray,
        state: AgentState,
        reg: DimensionRegistry,
    ) -> EmotionReading:
        """High O-activation + adequate energy + moderate arousal (Yerkes-Dodson)."""
        if "O" not in reg.keys:
            return EmotionReading(
                label=EmotionLabel.CREATIVITY,
                intensity=0.0,
                description="No O dimension.",
            )
        o_act = activations[reg.index("O")]
        arousal_factor = float(np.exp(-10.0 * (state.arousal - 0.5) ** 2))
        intensity = float(
            np.clip(
                o_act * arousal_factor * state.energy,
                0,
                1,
            )
        )
        return EmotionReading(
            label=EmotionLabel.CREATIVITY,
            intensity=intensity,
            description=(
                "Creative capacity: open activation + optimal arousal + energy."
            ),
        )

    def _detect_focus(
        self,
        activations: np.ndarray,
        state: AgentState,
        reg: DimensionRegistry,
    ) -> EmotionReading:
        """High C-activation + high energy + low frustration."""
        if "C" not in reg.keys:
            return EmotionReading(
                label=EmotionLabel.FOCUS,
                intensity=0.0,
                description="No C dimension.",
            )
        c_act = activations[reg.index("C")]
        frustration_penalty = 1 - state.frustration * 0.7
        intensity = float(
            np.clip(
                c_act * state.energy * frustration_penalty,
                0,
                1,
            )
        )
        return EmotionReading(
            label=EmotionLabel.FOCUS,
            intensity=intensity,
            description=(
                "Attentional control: conscientious activation + energy - frustration."
            ),
        )
