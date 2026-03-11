"""Emotion detector functions: each maps simulation signals to an EmotionReading.

Detectors are plain functions grouped by their input signature.
The AffectiveEngine composes them via detect_all().
"""

import numpy as np

from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import PersonalityVector
from src.temporal.emotions import EmotionLabel, EmotionReading, EmotionThresholds
from src.temporal.memory import MemoryBank
from src.temporal.state import AgentState

# --- State-only detectors ---------------------------------------------------


def detect_excitement(state: AgentState) -> EmotionReading:
    """High positive mood + high arousal."""
    intensity = max(0, state.mood) * state.arousal
    return EmotionReading(
        label=EmotionLabel.EXCITEMENT,
        intensity=intensity,
        description="High-energy positive affect; anticipation or thrill.",
    )


def detect_enthusiasm(state: AgentState) -> EmotionReading:
    """Moderate-to-high positive mood + moderate arousal."""
    mood_component = max(0, state.mood)
    arousal_component = min(state.arousal, 0.8)
    intensity = mood_component * arousal_component * state.energy
    return EmotionReading(
        label=EmotionLabel.ENTHUSIASM,
        intensity=intensity,
        description="Warm, energized positive engagement.",
    )


def detect_contentment(state: AgentState) -> EmotionReading:
    """Positive mood + low arousal + high satisfaction."""
    intensity = max(0, state.mood) * (1 - state.arousal) * state.satisfaction
    return EmotionReading(
        label=EmotionLabel.CONTENTMENT,
        intensity=intensity,
        description="Calm, satisfied state; things are good and quiet.",
    )


def detect_frustration(state: AgentState) -> EmotionReading:
    """High frustration signal + arousal."""
    intensity = state.frustration * state.arousal
    return EmotionReading(
        label=EmotionLabel.FRUSTRATION_EMO,
        intensity=intensity,
        description="Goals are blocked; agitated negative state.",
    )


# --- State + personality detectors -------------------------------------------


def detect_anxiety(
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
        description="Tense anticipation of negative outcomes.",
    )


# --- State + memory + personality detectors ----------------------------------


def detect_boredom(
    state: AgentState,
    memory: MemoryBank,
    psi: PersonalityVector,
    reg: DimensionRegistry,
    th: EmotionThresholds,
) -> EmotionReading:
    """Low arousal + low outcome variance + understimulation."""
    trait_o = psi["O"] if "O" in reg.keys else 0.5
    trait_e = psi["E"] if "E" in reg.keys else 0.5
    desired_arousal = 0.3 + 0.25 * trait_o + 0.2 * trait_e
    arousal_deficit = max(0, desired_arousal - state.arousal)
    variance = memory.outcome_variance(window=8)
    ceiling = th.boredom_variance_ceiling
    low_variance = max(0, ceiling - variance) / ceiling
    intensity = float(np.clip(arousal_deficit * 2 * low_variance, 0, 1))
    return EmotionReading(
        label=EmotionLabel.BOREDOM,
        intensity=intensity,
        description="Understimulated; craving novelty or challenge.",
    )


def detect_melancholy(
    state: AgentState,
    memory: MemoryBank,
    psi: PersonalityVector,
    reg: DimensionRegistry,
    th: EmotionThresholds,
) -> EmotionReading:
    """Sustained negative mood + low arousal + reflective traits."""
    trait_o = psi["O"] if "O" in reg.keys else 0.5
    trait_i = psi["I"] if "I" in reg.keys else 0.5
    recent_valence = memory.mean_valence(window=th.melancholy_window)
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
        description="Persistent reflective sadness; wistful low energy.",
    )


def detect_saudade(
    state: AgentState,
    memory: MemoryBank,
    psi: PersonalityVector,
    reg: DimensionRegistry,
    th: EmotionThresholds,
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
    if gap < th.saudade_gap_threshold:
        return EmotionReading(
            label=EmotionLabel.SAUDADE,
            intensity=0.0,
            description="Past is not much better than present.",
        )
    personality_amplifier = 0.5 + 0.3 * trait_t + 0.2 * trait_i
    low_arousal = max(0, 0.6 - state.arousal)
    gap_delta = gap - th.saudade_gap_threshold
    intensity = float(
        np.clip(gap_delta * personality_amplifier * (1 + low_arousal), 0, 1)
    )
    return EmotionReading(
        label=EmotionLabel.SAUDADE,
        intensity=intensity,
        description="Nostalgic longing for a valued past that is now absent.",
    )


# --- Memory-only detectors --------------------------------------------------


def detect_regret(memory: MemoryBank, th: EmotionThresholds) -> EmotionReading:
    """Accumulated counterfactual evidence that unchosen actions were better."""
    total_regret = memory.total_regret(window=10)
    threshold = th.regret_counterfactual_threshold
    intensity = float(np.clip((total_regret - threshold) * 2, 0, 1))
    return EmotionReading(
        label=EmotionLabel.REGRET,
        intensity=intensity,
        description="Counterfactual awareness: unchosen paths were better.",
    )


def detect_perseverance(
    memory: MemoryBank,
    still_acting: bool,
    th: EmotionThresholds,
) -> EmotionReading:
    """Continuing to act despite consecutive failures."""
    failures = memory.consecutive_failures()
    threshold = th.perseverance_failure_threshold
    if not still_acting or failures < threshold:
        return EmotionReading(
            label=EmotionLabel.PERSEVERANCE,
            intensity=0.0,
            description="Not persevering.",
        )
    intensity = float(np.clip((failures - threshold + 1) * 0.2, 0, 1))
    return EmotionReading(
        label=EmotionLabel.PERSEVERANCE,
        intensity=intensity,
        description="Continuing despite repeated failure.",
    )


def detect_grit(
    memory: MemoryBank,
    still_acting: bool,
    th: EmotionThresholds,
) -> EmotionReading:
    """Sustained perseverance over extended adversity."""
    failures = memory.consecutive_failures()
    threshold = th.grit_failure_threshold
    if not still_acting or failures < threshold:
        return EmotionReading(
            label=EmotionLabel.GRIT,
            intensity=0.0,
            description="Not showing grit.",
        )
    intensity = float(np.clip((failures - threshold + 1) * 0.15, 0, 1))
    return EmotionReading(
        label=EmotionLabel.GRIT,
        intensity=intensity,
        description="Sustained perseverance over extended adversity.",
    )


# --- Activation-based detectors ---------------------------------------------


def detect_creativity(
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
    intensity = float(np.clip(o_act * arousal_factor * state.energy, 0, 1))
    return EmotionReading(
        label=EmotionLabel.CREATIVITY,
        intensity=intensity,
        description="Creative capacity: open activation + optimal arousal + energy.",
    )


def detect_focus(
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
    intensity = float(np.clip(c_act * state.energy * frustration_penalty, 0, 1))
    return EmotionReading(
        label=EmotionLabel.FOCUS,
        intensity=intensity,
        description=(
            "Attentional control: conscientious activation + energy - frustration."
        ),
    )
