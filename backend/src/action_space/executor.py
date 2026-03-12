"""Action executor: dispatches chosen actions for real-world effects.

Classic actions require no execution (they are personality-space abstractions).
Tool, API, and text actions are dispatched to registered handlers.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.action_space.proposal import (
    ApiActionProposal,
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
    _ProposalBase,
)
from src.shared.logging import get_logger

_log = get_logger(module="action_space.executor")


class ActionResult(BaseModel):
    """Typed result from executing an action."""

    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    outcome_signal: float | None = None


class ToolHandler(Protocol):
    """Interface for tool execution handlers."""

    def execute(self, tool_args: dict[str, Any]) -> ActionResult:
        """Execute the tool with the given arguments."""
        ...


class NoopToolHandler:
    """Placeholder handler that always succeeds with empty output."""

    def execute(self, tool_args: dict[str, Any]) -> ActionResult:
        """Return a successful no-op result."""
        return ActionResult(success=True, output={"handler": "noop", "args": tool_args})


class ActionExecutor:
    """Dispatches action proposals to appropriate handlers."""

    def __init__(
        self,
        tool_handlers: dict[str, ToolHandler] | None = None,
    ) -> None:
        self._tool_handlers = tool_handlers or {}

    def execute(self, proposal: _ProposalBase) -> ActionResult:
        """Execute the given action proposal and return a typed result."""
        if isinstance(proposal, ClassicActionProposal):
            return self._execute_classic(proposal)
        if isinstance(proposal, ToolActionProposal):
            return self._execute_tool(proposal)
        if isinstance(proposal, ApiActionProposal):
            return self._execute_api(proposal)
        if isinstance(proposal, TextActionProposal):
            return self._execute_text(proposal)
        msg = f"Unknown proposal type: {type(proposal)}"
        return ActionResult(success=False, error=msg)

    def _execute_classic(self, proposal: ClassicActionProposal) -> ActionResult:
        """Pass through classic actions with no side effects."""
        _log.debug("classic_action_executed", name=proposal.name)
        return ActionResult(success=True, output={"kind": "classic"})

    def _execute_tool(self, proposal: ToolActionProposal) -> ActionResult:
        """Dispatch to the registered tool handler or fail if missing."""
        handler = self._tool_handlers.get(proposal.tool_name)
        if handler is None:
            msg = f"No handler registered for tool '{proposal.tool_name}'"
            _log.warning("tool_handler_missing", tool=proposal.tool_name)
            return ActionResult(success=False, error=msg)
        return handler.execute(proposal.tool_args)

    def _execute_api(self, proposal: ApiActionProposal) -> ActionResult:
        """Dispatch an API action as a placeholder."""
        _log.debug("api_action_placeholder", method=proposal.method, url=proposal.url)
        return ActionResult(
            success=True,
            output={"kind": "api", "method": proposal.method, "url": proposal.url},
        )

    def _execute_text(self, proposal: TextActionProposal) -> ActionResult:
        """Dispatch a text action as a placeholder."""
        _log.debug("text_action_placeholder", intent=proposal.intent)
        return ActionResult(
            success=True,
            output={"kind": "text", "intent": proposal.intent},
        )
