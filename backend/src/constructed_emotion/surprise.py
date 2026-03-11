"""Surprise spike detection with adaptive thresholding.

Threshold: ||eps_tilde(t)|| > mu + n_sigma * sigma
where mu and sigma are computed from the last `window` ticks,
with sigma_min floor and warmup fallback.

Reference: §3 of the research doc.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.constructed_emotion.params import ConstructedEmotionParams


class SurpriseSpikeDetector:
    """Adaptive surprise spike detection over arousal signal history.

    During warmup (fewer than `window` samples), uses a fixed threshold.
    After warmup, uses mu + n_sigma * max(sigma, sigma_min).
    """

    def __init__(self, params: ConstructedEmotionParams | None = None) -> None:
        self._params = params or ConstructedEmotionParams()
        self._history: deque[float] = deque(maxlen=self._params.surprise_window)

    @property
    def history(self) -> list[float]:
        """Return the current arousal signal history."""
        return list(self._history)

    @property
    def is_warmed_up(self) -> bool:
        """Whether enough samples exist for adaptive thresholding."""
        return len(self._history) >= self._params.surprise_window

    def current_threshold(self) -> float:
        """Compute the current spike threshold."""
        if not self.is_warmed_up:
            return self._params.surprise_warmup_threshold
        p = self._params
        arr = np.array(self._history)
        mu = float(np.mean(arr))
        sigma = max(float(np.std(arr)), p.surprise_sigma_min)
        return mu + p.surprise_n_sigma * sigma

    def observe(self, arousal_signal: float) -> bool:
        """Record an arousal signal and return True if it is a surprise spike."""
        threshold = self.current_threshold()
        is_spike = arousal_signal > threshold
        self._history.append(arousal_signal)
        return is_spike

    def reset(self) -> None:
        """Clear the history."""
        self._history.clear()
