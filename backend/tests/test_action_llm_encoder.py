"""Tests for LLM-backed action encoding."""

import pytest
from src.action_space.llm_encoder import LLMEncoderBackend
from src.action_space.proposal import TextActionProposal, ToolActionProposal
from src.llm.schemas import ActionEncoding, LLMInvocationMetadata, LLMInvocationResult


class FakeLLMAdapter:
    """Returns canned ActionEncoding responses."""

    def __init__(self, modifiers: dict[str, float] | None = None) -> None:
        self._modifiers = modifiers or {"O": 0.5, "E": 0.3}

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, response_model: type
    ) -> tuple[ActionEncoding, LLMInvocationResult]:
        """Return a pre-built encoding and metadata pair."""
        encoding = ActionEncoding(
            modifiers=self._modifiers,
            confidence=0.8,
            rationale="test",
        )
        meta = LLMInvocationResult(
            raw_text="{}",
            metadata=LLMInvocationMetadata(model="test", provider="test"),
        )
        return encoding, meta


class TestLLMEncoderBackend:
    """LLM backend uses structured output for modifier estimation."""

    def test_encodes_tool_proposal(self) -> None:
        """Tool proposals are encoded via the LLM adapter."""
        adapter = FakeLLMAdapter(modifiers={"O": 0.7, "C": 0.4})
        backend = LLMEncoderBackend(adapter=adapter)
        proposal = ToolActionProposal(
            name="search",
            description="search papers",
            tool_name="web_search",
            tool_args={"query": "test"},
        )
        modifiers = backend.estimate(proposal)
        assert modifiers["O"] == pytest.approx(0.7)
        assert modifiers["C"] == pytest.approx(0.4)

    def test_encodes_text_proposal(self) -> None:
        """Text proposals are encoded via the LLM adapter."""
        adapter = FakeLLMAdapter(modifiers={"A": 0.9})
        backend = LLMEncoderBackend(adapter=adapter)
        proposal = TextActionProposal(
            name="comfort",
            description="comfort someone",
            intent="comfort",
        )
        modifiers = backend.estimate(proposal)
        assert modifiers["A"] == pytest.approx(0.9)

    def test_graceful_degradation_on_error(self) -> None:
        """Backend returns empty dict when the LLM adapter raises."""

        class FailingAdapter:
            """Always raises on complete_json."""

            def complete_json(self, **kwargs: object) -> None:
                """Simulate an LLM failure."""
                raise RuntimeError("LLM unavailable")

        backend = LLMEncoderBackend(adapter=FailingAdapter())
        proposal = ToolActionProposal(
            name="search",
            description="search",
            tool_name="web_search",
            tool_args={},
        )
        modifiers = backend.estimate(proposal)
        assert isinstance(modifiers, dict)
