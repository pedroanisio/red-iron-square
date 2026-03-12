"""Tests for information-gain epistemic value (§10 pymdp bridge)."""

import numpy as np
import pytest
from src.efe.info_gain import compute_all_info_gains, compute_info_gain


class TestInfoGain:
    """Information-gain computation from A-matrix."""

    def test_identity_observation_gives_max_info_gain(self) -> None:
        """Perfect observation model gives maximum information gain."""
        n = 3
        A = np.zeros((n, n, 2))
        A[:, :, 0] = np.eye(n)
        A[:, :, 1] = np.eye(n)
        belief = np.ones(n) / n  # uniform belief

        gain = compute_info_gain(A, belief, action_idx=0)
        # With identity A and uniform belief, gain = H(belief) - 0 = log(n)
        assert gain == pytest.approx(np.log(n), abs=1e-6)

    def test_uniform_observation_gives_zero_info_gain(self) -> None:
        """Uniform observation model gives zero information gain."""
        n = 3
        A = np.ones((n, n, 1)) / n
        belief = np.ones(n) / n

        gain = compute_info_gain(A, belief, action_idx=0)
        assert gain == pytest.approx(0.0, abs=1e-6)

    def test_info_gain_always_non_negative(self) -> None:
        """Information gain is non-negative by definition."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            n_s, n_o, n_a = 4, 4, 3
            A = rng.dirichlet(np.ones(n_o), size=(n_s, n_a)).transpose(1, 0, 2)
            belief = rng.dirichlet(np.ones(n_s))
            for a in range(n_a):
                assert compute_info_gain(A, belief, a) >= -1e-10

    def test_compute_all_returns_correct_shape(self) -> None:
        n = 3
        A = np.ones((n, n, 2)) / n
        belief = np.ones(n) / n
        gains = compute_all_info_gains(A, belief)
        assert gains.shape == (2,)

    def test_asymmetric_observation_differentiates_actions(self) -> None:
        """Actions with different observation models yield different info gains."""
        n = 3
        A = np.zeros((n, n, 2))
        # Action 0: identity (perfect observation)
        A[:, :, 0] = np.eye(n)
        # Action 1: uniform (no information)
        A[:, :, 1] = 1.0 / n

        belief = np.ones(n) / n
        gains = compute_all_info_gains(A, belief)
        assert gains[0] > gains[1]

    def test_peaked_belief_reduces_info_gain(self) -> None:
        """When belief is already certain, info gain is lower."""
        n = 3
        A = np.zeros((n, n, 1))
        A[:, :, 0] = np.eye(n)

        uniform = np.ones(n) / n
        peaked = np.array([0.98, 0.01, 0.01])

        gain_uniform = compute_info_gain(A, uniform, 0)
        gain_peaked = compute_info_gain(A, peaked, 0)
        assert gain_uniform > gain_peaked
