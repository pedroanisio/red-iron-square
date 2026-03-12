"""Tests for action execution dispatch."""

import pytest
from src.action_space.executor import (
    ActionExecutor,
    ActionResult,
    NoopToolHandler,
)
from src.action_space.proposal import (
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
)


class TestActionResult:
    """ActionResult carries typed execution output."""

    def test_success_result(self) -> None:
        result = ActionResult(
            success=True,
            output={"data": [1, 2, 3]},
            outcome_signal=0.6,
        )
        assert result.success
        assert result.outcome_signal == pytest.approx(0.6)

    def test_failure_result(self) -> None:
        result = ActionResult(
            success=False,
            output={},
            error="timeout",
            outcome_signal=-0.3,
        )
        assert not result.success
        assert result.error == "timeout"


class TestActionExecutor:
    """ActionExecutor dispatches to the correct handler."""

    def test_classic_actions_return_noop_result(self) -> None:
        executor = ActionExecutor()
        proposal = ClassicActionProposal(
            name="bold", description="bold", modifiers={"O": 1.0}
        )
        result = executor.execute(proposal)
        assert result.success
        assert result.outcome_signal is None

    def test_tool_action_dispatches_to_handler(self) -> None:
        handler = NoopToolHandler()
        executor = ActionExecutor(tool_handlers={"web_search": handler})
        proposal = ToolActionProposal(
            name="search",
            description="search",
            tool_name="web_search",
            tool_args={"query": "test"},
        )
        result = executor.execute(proposal)
        assert result.success

    def test_missing_tool_handler_returns_failure(self) -> None:
        executor = ActionExecutor()
        proposal = ToolActionProposal(
            name="search",
            description="search",
            tool_name="missing_tool",
            tool_args={},
        )
        result = executor.execute(proposal)
        assert not result.success
        assert "no handler" in (result.error or "").lower()

    def test_text_action_returns_placeholder(self) -> None:
        executor = ActionExecutor()
        proposal = TextActionProposal(
            name="explain",
            description="explain a concept",
            intent="explain",
        )
        result = executor.execute(proposal)
        assert result.success
        assert "text" in str(result.output).lower() or result.output is not None
