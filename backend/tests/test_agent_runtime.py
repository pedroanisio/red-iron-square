"""Tests for AgentRuntime LLM task methods."""

from unittest.mock import MagicMock

from src.llm.agent_runtime import AgentRuntime
from src.llm.schemas import (
    EmotionConstructor,
    LLMInvocationMetadata,
    LLMInvocationResult,
    MatrixProposal,
)


def _make_runtime_with_mock(response_obj: object) -> AgentRuntime:
    """Create an AgentRuntime with a mocked adapter."""
    adapter = MagicMock()
    meta = LLMInvocationMetadata(model="test", provider="test")
    invocation = LLMInvocationResult(raw_text="{}", metadata=meta)
    adapter.complete_json.return_value = (response_obj, invocation)
    return AgentRuntime(adapter)


class TestConstructEmotion:
    """AgentRuntime.construct_emotion() method."""

    def test_returns_emotion_constructor(self) -> None:
        emotion = EmotionConstructor(
            label="excitement",
            description="Positive surprise",
            valence_sign="positive",
            arousal_level="high",
            confidence=0.9,
        )
        runtime = _make_runtime_with_mock(emotion)
        result, _ = runtime.construct_emotion(
            valence=0.5,
            arousal=0.8,
            prediction_errors=[0.1, -0.2, 0.0, 0.3, -0.1],
            context="Agent encountered unexpected positive outcome",
        )
        assert isinstance(result, EmotionConstructor)
        assert result.label == "excitement"

    def test_passes_valence_arousal_constraints(self) -> None:
        emotion = EmotionConstructor(
            label="anxiety",
            description="test",
            valence_sign="negative",
            arousal_level="high",
            confidence=0.7,
        )
        runtime = _make_runtime_with_mock(emotion)
        result, invocation = runtime.construct_emotion(
            valence=-0.3,
            arousal=0.7,
            prediction_errors=[0.0] * 5,
            context="test",
        )
        assert result.valence_sign == "negative"


class TestProposeMatrices:
    """AgentRuntime.propose_matrices() method."""

    def test_returns_matrix_proposal(self) -> None:
        proposal = MatrixProposal(
            a_matrix=[[[0.6, 0.4], [0.3, 0.7]], [[0.5, 0.5], [0.4, 0.6]]],
            b_matrix=[[[0.7, 0.3], [0.2, 0.8]], [[0.6, 0.4], [0.3, 0.7]]],
            rationale="Based on personality traits",
            n_states=2,
            n_actions=2,
        )
        runtime = _make_runtime_with_mock(proposal)
        result, _ = runtime.propose_matrices(
            personality={k: 0.5 for k in "OCEANRIT"},
            trajectory_window=[{"action": "Act", "outcome": 0.5}],
            n_states=2,
            n_actions=2,
        )
        assert isinstance(result, MatrixProposal)
        assert result.n_states == 2
