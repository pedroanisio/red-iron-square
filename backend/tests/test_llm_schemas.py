"""Tests for LLM structured output schemas."""

import pytest
from src.llm.schemas import EmotionConstructor, MatrixProposal


class TestEmotionConstructor:
    """EmotionConstructor schema with valence/arousal constraints."""

    def test_valid_construction(self) -> None:
        ec = EmotionConstructor(
            label="excitement",
            description="Positive surprise at outcome",
            valence_sign="positive",
            arousal_level="high",
            confidence=0.85,
        )
        assert ec.label == "excitement"
        assert ec.confidence == pytest.approx(0.85)

    def test_rejects_invalid_valence_sign(self) -> None:
        with pytest.raises(ValueError):
            EmotionConstructor(
                label="test",
                description="test",
                valence_sign="wrong",
                arousal_level="high",
                confidence=0.5,
            )

    def test_rejects_invalid_arousal_level(self) -> None:
        with pytest.raises(ValueError):
            EmotionConstructor(
                label="test",
                description="test",
                valence_sign="positive",
                arousal_level="wrong",
                confidence=0.5,
            )

    def test_confidence_clamped_to_unit(self) -> None:
        ec = EmotionConstructor(
            label="test",
            description="test",
            valence_sign="positive",
            arousal_level="high",
            confidence=0.5,
        )
        assert 0.0 <= ec.confidence <= 1.0


class TestMatrixProposal:
    """MatrixProposal schema for LLM-proposed A/B matrices."""

    def test_valid_construction(self) -> None:
        """Valid 2x2x2 matrices."""
        mp = MatrixProposal(
            a_matrix=[[[0.6, 0.4], [0.3, 0.7]], [[0.5, 0.5], [0.4, 0.6]]],
            b_matrix=[[[0.7, 0.3], [0.2, 0.8]], [[0.6, 0.4], [0.3, 0.7]]],
            rationale="test",
            n_states=2,
            n_actions=2,
        )
        assert mp.n_states == 2
        assert mp.n_actions == 2

    def test_rejects_zero_states(self) -> None:
        with pytest.raises(ValueError):
            MatrixProposal(
                a_matrix=[],
                b_matrix=[],
                n_states=0,
                n_actions=1,
            )
