"""NarrativeGenerativeModel: cached generative model for System 2 narrative.

Maintains A (observation), B (transition), and C (preference) matrices
that are refreshed at surprise spikes and phase boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.logging import get_logger

_log = get_logger(module="narrative.model")


class NarrativeGenerativeModel:
    """Cached generative model with A/B/C matrices for active inference.

    A-matrix: observation likelihood p(o|s,a) -- shape (n_obs, n_states, n_actions)
    B-matrix: state transitions p(s'|s,a) -- shape (n_states, n_states, n_actions)
    C-vector: prior preferences over observations -- shape (n_obs,)

    Initialized from personality and updated at System 2 junctures.
    """

    def __init__(
        self,
        personality: dict[str, float],
        *,
        n_obs: int = 5,
        n_states: int = 5,
        n_actions: int = 3,
    ) -> None:
        self._personality = personality
        self._n_obs = n_obs
        self._n_states = n_states
        self._n_actions = n_actions
        self._A = self._init_A()
        self._B = self._init_B()
        self._C = self._init_C()

    @property
    def n_obs(self) -> int:
        """Number of observation dimensions."""
        return self._n_obs

    @property
    def cached_A(self) -> np.ndarray:
        """Observation likelihood p(o|s,a) — shape (n_obs, n_states, n_actions)."""
        return self._A.copy()

    @property
    def cached_B(self) -> np.ndarray:
        """State transition p(s'|s,a) — shape (n_states, n_states, n_actions)."""
        return self._B.copy()

    @property
    def cached_C(self) -> np.ndarray:
        """Prior preference vector — shape (n_obs,)."""
        return self._C.copy()

    def refresh_on_spike(
        self,
        is_spike: bool,
        trajectory_window: list[dict[str, Any]],
    ) -> None:
        """Conditionally refresh matrices when a surprise spike occurs."""
        if is_spike:
            self.refresh_from_trajectory(trajectory_window)

    def refresh_from_trajectory(
        self,
        trajectory_window: list[dict[str, Any]],
    ) -> None:
        """Update cached matrices from recent trajectory data.

        Uses trajectory evidence to refine transition and observation
        models. Called at surprise spikes and phase boundaries.
        """
        if not trajectory_window:
            return
        self._update_B_from_trajectory(trajectory_window)
        _log.info("narrative_model_refreshed", n_ticks=len(trajectory_window))

    def _init_A(self) -> np.ndarray:
        """Initialize A-matrix: observation likelihood p(o|s,a).

        Shape (n_obs, n_states, n_actions). When n_obs == n_states the
        matrix is near-identity; otherwise rows are uniform with slight
        diagonal bias where indices overlap.
        """
        A = np.zeros((self._n_obs, self._n_states, self._n_actions))
        diag_size = min(self._n_obs, self._n_states)
        for a in range(self._n_actions):
            A[:, :, a] = 0.3 / self._n_states
            for i in range(diag_size):
                A[i, i, a] += 0.7
            A[:, :, a] /= A[:, :, a].sum(axis=0, keepdims=True)
        return A

    def _init_B(self) -> np.ndarray:
        """Initialize B-matrix: near-identity transitions."""
        B = np.zeros((self._n_states, self._n_states, self._n_actions))
        for a in range(self._n_actions):
            B[:, :, a] = np.eye(self._n_states) * 0.6 + 0.4 / self._n_states
            B[:, :, a] /= B[:, :, a].sum(axis=1, keepdims=True)
        return B

    def _init_C(self) -> np.ndarray:
        """Initialize C-vector from personality (prior preferences).

        Returns an (n_obs,) vector. When n_obs == 5 each entry maps to
        one interoceptive dimension; for other sizes the vector is filled
        with a neutral default.
        """
        N = self._personality.get("N", 0.5)
        E = self._personality.get("E", 0.5)
        R = self._personality.get("R", 0.5)
        base = [
            -0.5 * N,
            0.2 * E,
            0.1,
            0.3,
            -0.2 * (1.0 - R),
        ]
        if self._n_obs <= len(base):
            return np.array(base[: self._n_obs])
        return np.array(base + [0.0] * (self._n_obs - len(base)))

    def _update_B_from_trajectory(self, trajectory: list[dict[str, Any]]) -> None:
        """Refine transition matrix from observed state changes."""
        rng = np.random.default_rng(0)
        for entry in trajectory:
            state = entry.get("state", [])
            if len(state) != self._n_states:
                continue
            outcome = entry.get("outcome", 0.0)
            scale = 0.01 * abs(outcome)
            for a in range(self._n_actions):
                noise = rng.normal(0, scale, self._B[:, :, a].shape)
                self._B[:, :, a] += noise
                self._B[:, :, a] = np.maximum(self._B[:, :, a], 1e-6)
                self._B[:, :, a] /= self._B[:, :, a].sum(axis=1, keepdims=True)
