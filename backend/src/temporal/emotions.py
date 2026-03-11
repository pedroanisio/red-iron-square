"""Emotion types, readings, and detection thresholds."""

from enum import Enum

from pydantic import BaseModel


class EmotionLabel(Enum):
    """Named emotional states that can emerge from the simulation.

    Each is defined by its position in the valence x arousal space
    plus specific computational prerequisites.
    """

    EXCITEMENT = "excitement"
    ENTHUSIASM = "enthusiasm"
    CONTENTMENT = "contentment"
    FRUSTRATION_EMO = "frustration"
    ANXIETY = "anxiety"
    BOREDOM = "boredom"
    MELANCHOLY = "melancholy"
    SAUDADE = "saudade"
    REGRET = "regret"
    PERSEVERANCE = "perseverance"
    GRIT = "grit"
    CREATIVITY = "creativity"
    FOCUS = "focus"


class EmotionReading(BaseModel):
    """A single detected emotion with its intensity."""

    label: EmotionLabel
    intensity: float
    description: str

    def __repr__(self) -> str:
        return f"{self.label.value}: {self.intensity:.2f}"


class EmotionThresholds(BaseModel):
    """Tunable thresholds for emotion emergence."""

    report_threshold: float = 0.15
    boredom_arousal_ceiling: float = 0.35
    boredom_variance_ceiling: float = 0.05
    melancholy_mood_floor: float = -0.20
    melancholy_window: int = 8
    saudade_gap_threshold: float = 0.40
    regret_counterfactual_threshold: float = 0.30
    perseverance_failure_threshold: int = 3
    grit_failure_threshold: int = 8
