"""Affective engine: emergent emotion detection from activations, state, and memory."""

import numpy as np

from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import PersonalityVector
from src.shared.logging import get_logger
from src.temporal.detectors import (
    detect_anxiety,
    detect_boredom,
    detect_contentment,
    detect_creativity,
    detect_enthusiasm,
    detect_excitement,
    detect_focus,
    detect_frustration,
    detect_grit,
    detect_melancholy,
    detect_perseverance,
    detect_regret,
    detect_saudade,
)
from src.temporal.emotions import (
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
            detect_excitement(state),
            detect_enthusiasm(state),
            detect_contentment(state),
            detect_frustration(state),
            detect_anxiety(state, personality, registry),
            detect_boredom(state, memory, personality, registry, self.th),
            detect_melancholy(state, memory, personality, registry, self.th),
            detect_saudade(state, memory, personality, registry, self.th),
            detect_regret(memory, self.th),
            detect_perseverance(memory, is_still_acting, self.th),
            detect_grit(memory, is_still_acting, self.th),
            detect_creativity(activations, state, registry),
            detect_focus(activations, state, registry),
        ]
        detected = [r for r in readings if r.intensity >= self.th.report_threshold]
        if detected:
            _log.debug(
                "emotions_detected",
                labels=[r.label.value for r in detected],
                count=len(detected),
            )
        return detected
