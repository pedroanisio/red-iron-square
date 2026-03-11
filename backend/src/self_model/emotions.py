"""Self-related emotion detection.

Covers pride, shame, authenticity, identity threat/crisis.
"""

from enum import Enum

import numpy as np
from pydantic import BaseModel

from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import PersonalityVector
from src.self_model.model import SelfModel
from src.self_model.params import SelfModelParams


class SelfEmotionLabel(Enum):
    """Named self-related emotional states."""

    PRIDE = "pride"
    SHAME = "shame"
    AUTHENTICITY = "authenticity"
    IDENTITY_THREAT = "identity_threat"
    IDENTITY_CRISIS = "identity_crisis"


class SelfEmotionReading(BaseModel):
    """A detected self-related emotion with its intensity."""

    label: SelfEmotionLabel
    intensity: float
    description: str

    def __repr__(self) -> str:
        return f"{self.label.value}: {self.intensity:.2f}"


class SelfEmotionDetector:
    """Detects self-related emotions from self-model metrics.

    Uses ONLY public properties/methods of SelfModel.
    """

    def __init__(self, params: SelfModelParams = SelfModelParams()) -> None:
        self.params = params

    def detect_all(
        self,
        self_model: SelfModel,
        prediction_error: float,
        outcome: float,
        personality: PersonalityVector,
        registry: DimensionRegistry,
    ) -> list[SelfEmotionReading]:
        """Run all self-emotion detectors, return those with intensity >= 0.10."""
        readings = [
            self._detect_pride(prediction_error, outcome),
            self._detect_shame(prediction_error, self_model, personality, registry),
            self._detect_authenticity(self_model),
            self._detect_identity_threat(self_model),
            self._detect_identity_crisis(self_model),
        ]
        return [r for r in readings if r.intensity >= 0.10]

    def _detect_pride(
        self, prediction_error: float, outcome: float
    ) -> SelfEmotionReading:
        """Action consistent with self-concept and succeeded."""
        accuracy = max(0, 1.0 - prediction_error)
        intensity = float(np.clip(accuracy * max(0, outcome), 0, 1))
        return SelfEmotionReading(
            label=SelfEmotionLabel.PRIDE,
            intensity=intensity,
            description="Action consistent with self-concept and succeeded.",
        )

    def _detect_shame(
        self,
        prediction_error: float,
        self_model: SelfModel,
        personality: PersonalityVector,
        registry: DimensionRegistry,
    ) -> SelfEmotionReading:
        """Behavior violated self-concept; self-discrepancy."""
        p = self.params
        N = personality["N"] if "N" in registry.keys else 0.5
        coherence_gap = self_model.current_coherence_gap()
        n_amplifier = 1.0 + p.N_shame_amplification * N
        intensity = float(
            np.clip(
                prediction_error * coherence_gap * p.shame_scaling * n_amplifier, 0, 1
            )
        )
        return SelfEmotionReading(
            label=SelfEmotionLabel.SHAME,
            intensity=intensity,
            description="Behavior violated self-concept; self-discrepancy.",
        )

    def _detect_authenticity(self, self_model: SelfModel) -> SelfEmotionReading:
        """Self-concept aligns with behavioral evidence."""
        threshold = self_model.params.coherence_threat_threshold
        gap = self_model.current_coherence_gap()
        intensity = float(np.clip(1.0 - gap / threshold, 0, 1))
        return SelfEmotionReading(
            label=SelfEmotionLabel.AUTHENTICITY,
            intensity=intensity,
            description="Self-concept aligns with behavioral evidence.",
        )

    def _detect_identity_threat(self, self_model: SelfModel) -> SelfEmotionReading:
        """Sustained evidence contradicts self-concept."""
        if not self_model.sustained_coherence_threat():
            return SelfEmotionReading(
                label=SelfEmotionLabel.IDENTITY_THREAT,
                intensity=0.0,
                description="No sustained coherence threat.",
            )
        p = self_model.params
        history = self_model.coherence_history
        recent = history[-p.coherence_threat_window :]
        mean_excess = np.mean(
            [max(0, g - p.coherence_threat_threshold) for g in recent]
        )
        intensity = float(np.clip(mean_excess * 5.0, 0, 1))
        return SelfEmotionReading(
            label=SelfEmotionLabel.IDENTITY_THREAT,
            intensity=intensity,
            description="Sustained evidence contradicts self-concept.",
        )

    def _detect_identity_crisis(self, self_model: SelfModel) -> SelfEmotionReading:
        """Self-concept has drifted far from original identity."""
        p = self_model.params
        drift = self_model.current_identity_drift()
        if drift < p.drift_crisis_threshold:
            return SelfEmotionReading(
                label=SelfEmotionLabel.IDENTITY_CRISIS,
                intensity=0.0,
                description="Self-model stable relative to anchor.",
            )
        intensity = float(
            np.clip(
                (drift - p.drift_crisis_threshold) / (1.0 - p.drift_crisis_threshold),
                0,
                1,
            )
        )
        return SelfEmotionReading(
            label=SelfEmotionLabel.IDENTITY_CRISIS,
            intensity=intensity,
            description="Self-concept has drifted far from original identity.",
        )
