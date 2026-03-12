"""Tests for LLM-backed action proposer."""

from src.action_space.llm_proposer import LLMProposerBackend
from src.action_space.registry import ToolCapability, ToolRegistry
from src.llm.schemas import (
    ActionSetProposal,
    LLMInvocationMetadata,
    LLMInvocationResult,
)


class FakeLLMAdapter:
    """Returns canned ActionSetProposal responses."""

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, response_model: type
    ) -> tuple[ActionSetProposal, LLMInvocationResult]:
        """Return a fixed proposal with two actions."""
        proposal = ActionSetProposal(
            actions=[
                {
                    "kind": "text",
                    "name": "explain_concept",
                    "description": "explain it",
                    "intent": "explain",
                },
                {
                    "kind": "tool",
                    "name": "search",
                    "description": "search web",
                    "tool_name": "web_search",
                    "tool_args": {"query": "test"},
                },
            ],
            rationale="context suggests information gathering",
        )
        meta = LLMInvocationResult(
            raw_text="{}",
            metadata=LLMInvocationMetadata(model="test", provider="test"),
        )
        return proposal, meta


class TestLLMProposerBackend:
    """LLM proposer generates context-aware candidate actions."""

    def test_proposes_actions_from_llm(self) -> None:
        """Verify proposer parses LLM output into typed proposals."""
        adapter = FakeLLMAdapter()
        tool_reg = ToolRegistry()
        tool_reg.register(
            ToolCapability(
                name="web_search",
                description="search",
                parameter_schema={},
            )
        )
        backend = LLMProposerBackend(adapter=adapter, tool_registry=tool_reg)
        proposals = backend.propose(
            state={"energy": 0.8, "mood": 0.3},
            trajectory=[],
            goals=["understand the topic"],
        )
        assert len(proposals) >= 1
        names = [p.name for p in proposals]
        assert "explain_concept" in names or "search" in names

    def test_graceful_degradation_returns_empty(self) -> None:
        """When the LLM adapter fails, propose() returns an empty list."""

        class FailingAdapter:
            """Always raises on complete_json."""

            def complete_json(self, **kwargs: object) -> None:
                """Simulate LLM failure."""
                raise RuntimeError("LLM down")

        backend = LLMProposerBackend(adapter=FailingAdapter())
        proposals = backend.propose(state={}, trajectory=[], goals=[])
        assert proposals == []
