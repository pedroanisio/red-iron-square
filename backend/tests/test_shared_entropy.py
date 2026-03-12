"""Tests for src.shared.entropy — Shannon entropy computation."""

import numpy as np
import pytest
from src.shared.entropy import compute_action_entropy


class TestComputeActionEntropy:
    """Shannon entropy from action frequency counts."""

    def test_uniform_three_actions(self) -> None:
        """Uniform distribution over 3 actions = ln(3)."""
        counts = {"A": 10, "B": 10, "C": 10}
        expected = float(np.log(3))
        assert compute_action_entropy(counts) == pytest.approx(expected, abs=0.01)

    def test_degenerate_single_action(self) -> None:
        """Single action = zero entropy."""
        assert compute_action_entropy({"X": 50}) == pytest.approx(0.0)

    def test_empty_counts(self) -> None:
        """Empty dict returns zero."""
        assert compute_action_entropy({}) == 0.0

    def test_zero_total(self) -> None:
        """All-zero counts returns zero."""
        assert compute_action_entropy({"A": 0, "B": 0}) == 0.0

    def test_binary_skewed(self) -> None:
        """Skewed binary distribution has lower entropy than uniform."""
        uniform = compute_action_entropy({"A": 50, "B": 50})
        skewed = compute_action_entropy({"A": 90, "B": 10})
        assert skewed < uniform

    def test_entropy_non_negative(self) -> None:
        """Entropy is always non-negative."""
        assert compute_action_entropy({"A": 3, "B": 7, "C": 1}) >= 0.0
