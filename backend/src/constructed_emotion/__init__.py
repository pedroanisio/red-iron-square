"""Constructed emotion: precision-weighted affect from prediction errors."""

from src.constructed_emotion.affect import AffectSignal, ConstructedAffectiveEngine
from src.constructed_emotion.free_energy import (
    compute_arousal_signal,
    compute_free_energy,
    compute_valence,
)
from src.constructed_emotion.params import ConstructedEmotionParams
from src.constructed_emotion.surprise import SurpriseSpikeDetector

__all__ = [
    "AffectSignal",
    "ConstructedAffectiveEngine",
    "ConstructedEmotionParams",
    "SurpriseSpikeDetector",
    "compute_arousal_signal",
    "compute_free_energy",
    "compute_valence",
]
