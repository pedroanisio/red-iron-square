"""Tests for NarrativeGenerativeModel with cached A/B/C matrices."""

import numpy as np
import pytest
from src.narrative.model import NarrativeGenerativeModel


def _profile(overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Return a balanced profile with optional overrides."""
    vals = {k: 0.5 for k in "OCEANRIT"}
    if overrides:
        vals.update(overrides)
    return vals


class TestNarrativeGenerativeModel:
    """Cached generative model for System 2 narrative maintenance."""

    def test_initial_matrices_from_personality(self) -> None:
        """Model initializes default A/B/C matrices from personality."""
        model = NarrativeGenerativeModel(_profile(), n_states=5, n_actions=3)
        assert model.cached_A.shape == (5, 5, 3)
        assert model.cached_B.shape == (5, 5, 3)
        assert model.cached_C.shape == (5,)

    def test_a_matrix_shape_with_different_n_obs(self) -> None:
        """A-matrix first dim is n_obs, distinct from n_states."""
        model = NarrativeGenerativeModel(
            _profile(),
            n_obs=3,
            n_states=5,
            n_actions=2,
        )
        assert model.cached_A.shape == (3, 5, 2)
        assert model.cached_B.shape == (5, 5, 2)
        assert model.cached_C.shape == (3,)

    def test_a_matrix_columns_sum_to_one(self) -> None:
        """A-matrix columns (over obs) normalized: p(o|s,a)."""
        model = NarrativeGenerativeModel(_profile(), n_states=5, n_actions=3)
        A = model.cached_A
        for action_idx in range(A.shape[2]):
            col_sums = np.sum(A[:, :, action_idx], axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-6)

    def test_b_matrix_rows_sum_to_one(self) -> None:
        """B-matrix rows are valid probability distributions."""
        model = NarrativeGenerativeModel(_profile(), n_states=5, n_actions=3)
        B = model.cached_B
        for action_idx in range(B.shape[2]):
            for row in range(B.shape[0]):
                assert np.sum(B[row, :, action_idx]) == pytest.approx(
                    1.0,
                    abs=1e-6,
                )

    def test_refresh_updates_matrices(self) -> None:
        """Refresh method updates cached matrices."""
        model = NarrativeGenerativeModel(_profile(), n_states=5, n_actions=3)
        B_before = model.cached_B.copy()
        model.refresh_from_trajectory(
            trajectory_window=[
                {
                    "action": "Engage",
                    "outcome": 0.5,
                    "state": [0.1, 0.5, 0.8, 0.5, 0.1],
                },
            ],
        )
        assert not np.allclose(model.cached_B, B_before)

    def test_c_vector_reflects_personality(self) -> None:
        """C-vector preferences depend on personality."""
        high_n = NarrativeGenerativeModel(
            _profile({"N": 0.9}),
            n_states=5,
            n_actions=3,
        )
        low_n = NarrativeGenerativeModel(
            _profile({"N": 0.1}),
            n_states=5,
            n_actions=3,
        )
        assert not np.allclose(high_n.cached_C, low_n.cached_C)

    def test_c_vector_size_matches_n_obs(self) -> None:
        """C-vector length equals n_obs, not n_states."""
        model = NarrativeGenerativeModel(
            _profile(),
            n_obs=7,
            n_states=5,
            n_actions=3,
        )
        assert model.cached_C.shape == (7,)

    def test_empty_trajectory_no_change(self) -> None:
        """Empty trajectory should not change matrices."""
        model = NarrativeGenerativeModel(_profile(), n_states=5, n_actions=3)
        B_before = model.cached_B.copy()
        model.refresh_from_trajectory(trajectory_window=[])
        np.testing.assert_array_equal(model.cached_B, B_before)

    def test_n_obs_property(self) -> None:
        """n_obs property returns configured observation count."""
        model = NarrativeGenerativeModel(
            _profile(),
            n_obs=4,
            n_states=5,
            n_actions=3,
        )
        assert model.n_obs == 4

    def test_refresh_on_spike_true_updates(self) -> None:
        """refresh_on_spike with is_spike=True triggers B update."""
        model = NarrativeGenerativeModel(_profile(), n_states=5, n_actions=3)
        B_before = model.cached_B.copy()
        entries = [
            {"state": [0.1, 0.5, 0.8, 0.5, 0.1], "outcome": 0.7, "action": "A"},
        ]
        model.refresh_on_spike(True, entries)
        assert not np.allclose(model.cached_B, B_before)

    def test_refresh_on_spike_false_no_change(self) -> None:
        """refresh_on_spike with is_spike=False leaves matrices unchanged."""
        model = NarrativeGenerativeModel(_profile(), n_states=5, n_actions=3)
        B_before = model.cached_B.copy()
        entries = [
            {"state": [0.1, 0.5, 0.8, 0.5, 0.1], "outcome": 0.9, "action": "A"},
        ]
        model.refresh_on_spike(False, entries)
        np.testing.assert_array_equal(model.cached_B, B_before)

    def test_update_b_skips_wrong_state_length(self) -> None:
        """Trajectory entries with wrong state length are skipped."""
        model = NarrativeGenerativeModel(_profile(), n_states=5, n_actions=3)
        B_before = model.cached_B.copy()
        model.refresh_from_trajectory(
            [
                {"state": [0.1, 0.2], "outcome": 0.5, "action": "A"},
            ]
        )
        np.testing.assert_array_equal(model.cached_B, B_before)

    def test_update_b_multiple_entries(self) -> None:
        """Multiple trajectory entries accumulate B-matrix updates."""
        model = NarrativeGenerativeModel(_profile(), n_states=5, n_actions=3)
        entries = [
            {"state": [0.1, 0.2, 0.3, 0.4, 0.5], "outcome": 0.8, "action": "A"},
            {"state": [0.5, 0.4, 0.3, 0.2, 0.1], "outcome": -0.3, "action": "B"},
            {"state": [0.3, 0.3, 0.3, 0.3, 0.3], "outcome": 0.0, "action": "C"},
        ]
        model.refresh_from_trajectory(entries)
        B = model.cached_B
        for a in range(B.shape[2]):
            row_sums = np.sum(B[:, :, a], axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_c_vector_truncated_for_small_n_obs(self) -> None:
        """C-vector truncated to n_obs when n_obs < 5."""
        model = NarrativeGenerativeModel(_profile(), n_obs=2, n_states=5, n_actions=3)
        assert model.cached_C.shape == (2,)

    def test_c_vector_padded_for_large_n_obs(self) -> None:
        """C-vector padded with zeros when n_obs > 5."""
        model = NarrativeGenerativeModel(_profile(), n_obs=8, n_states=5, n_actions=3)
        C = model.cached_C
        assert C.shape == (8,)
        assert C[5] == 0.0
        assert C[7] == 0.0
