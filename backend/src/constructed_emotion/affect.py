"""Constructed affective engine: System 1 (every tick) + System 2 (surprise spikes).

System 1 computes valence from delta-F and arousal from weighted prediction errors.
System 2 triggers on surprise spikes for LLM emotion categorization
(with heuristic fallback).

References:
    Barrett (2017), SCAN, 12(1):1-23.
    Hesp et al. (2021), Neural Computation, 33(2):398-446.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.constructed_emotion.free_energy import (
    compute_arousal_signal,
    compute_free_energy,
    compute_valence,
)
from src.constructed_emotion.params import ConstructedEmotionParams
from src.constructed_emotion.surprise import SurpriseSpikeDetector
from src.precision.state import PrecisionState, PredictionErrors
from src.shared.logging import get_logger
from src.temporal.emotions import EmotionLabel, EmotionReading

if TYPE_CHECKING:
    from src.personality.vectors import PersonalityVector

_log = get_logger(module="constructed_emotion.affect")


class AffectSignal(BaseModel):
    """Per-tick constructed affect signal (System 1 output)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    valence: float
    arousal_signal: float
    free_energy: float
    is_surprise_spike: bool
    mood: float
    constructed_emotions: list[EmotionReading]


class ConstructedAffectiveEngine:
    """Constructs affect from precision-weighted prediction errors.

    System 1 (every tick): valence from delta-F, arousal from ||eps_tilde||.
    System 2 (surprise spikes): heuristic emotion categorization from
    valence/arousal quadrant (LLM integration point for future extension).
    """

    def __init__(
        self,
        params: ConstructedEmotionParams | None = None,
    ) -> None:
        self._params = params or ConstructedEmotionParams()
        self._spike_detector = SurpriseSpikeDetector(self._params)
        self._prev_free_energy: float | None = None
        self._mood: float = 0.0

    @property
    def mood(self) -> float:
        """Current mood (slow hyperprior over valence)."""
        return self._mood

    @property
    def spike_detector(self) -> SurpriseSpikeDetector:
        """Expose detector for inspection."""
        return self._spike_detector

    def process_tick(
        self,
        precision: PrecisionState,
        errors: PredictionErrors,
        personality: PersonalityVector,
    ) -> AffectSignal:
        """Run System 1 (+ conditional System 2) for one tick."""
        f_total = compute_free_energy(precision, errors)
        arousal = compute_arousal_signal(precision, errors)

        valence = self._compute_valence(f_total)
        self._update_mood(valence, personality)
        is_spike = self._spike_detector.observe(arousal)

        emotions: list[EmotionReading] = []
        if is_spike:
            emotions = self._categorize_emotion(valence, arousal)
            _log.info(
                "surprise_spike",
                arousal=round(arousal, 4),
                valence=round(valence, 4),
                n_emotions=len(emotions),
            )

        self._prev_free_energy = f_total

        return AffectSignal(
            valence=valence,
            arousal_signal=arousal,
            free_energy=f_total,
            is_surprise_spike=is_spike,
            mood=self._mood,
            constructed_emotions=emotions,
        )

    def _compute_valence(self, f_total: float) -> float:
        """Valence = F(t-1) - F(t). Zero on first tick."""
        if self._prev_free_energy is None:
            return 0.0
        return compute_valence(self._prev_free_energy, f_total)

    def _update_mood(self, valence: float, personality: PersonalityVector) -> None:
        """EMA mood update: mood(t+1) = alpha * mood(t) + (1-alpha) * valence.

        Alpha is personality-dependent: high R -> smaller alpha (faster recovery).
        """
        keys = set(personality.registry.keys)
        r_val = personality["R"] if "R" in keys else 0.5
        p = self._params
        alpha = p.mood_ema_alpha - p.r_alpha_scale * r_val
        self._mood = float(
            np.clip(
                alpha * self._mood + (1.0 - alpha) * valence,
                -1.0,
                1.0,
            )
        )

    def _categorize_emotion(
        self,
        valence: float,
        arousal: float,
    ) -> list[EmotionReading]:
        """Heuristic valence/arousal quadrant categorization.

        This is the System 2 fallback. In future, an LLM adapter
        would provide context-dependent categorization here.
        """
        emotions: list[EmotionReading] = []
        intensity = min(1.0, abs(valence) * 0.5 + arousal * 0.5)

        if valence > 0 and arousal > 0.5:
            emotions.append(
                EmotionReading(
                    label=EmotionLabel.EXCITEMENT,
                    intensity=intensity,
                    description="Constructed: positive valence, high arousal",
                )
            )
        elif valence > 0 and arousal <= 0.5:
            emotions.append(
                EmotionReading(
                    label=EmotionLabel.CONTENTMENT,
                    intensity=intensity,
                    description="Constructed: positive valence, low arousal",
                )
            )
        elif valence < 0 and arousal > 0.5:
            emotions.append(
                EmotionReading(
                    label=EmotionLabel.ANXIETY,
                    intensity=intensity,
                    description="Constructed: negative valence, high arousal",
                )
            )
        elif valence < 0 and arousal <= 0.5:
            emotions.append(
                EmotionReading(
                    label=EmotionLabel.MELANCHOLY,
                    intensity=intensity,
                    description="Constructed: negative valence, low arousal",
                )
            )

        return emotions
