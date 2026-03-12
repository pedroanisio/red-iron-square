"""Context-aware action proposal for open-ended action spaces.

The ActionProposer generates candidate actions from:
1. A pluggable backend (static defaults or LLM-generated)
2. Available tools from the ToolRegistry
3. A mandatory Withdraw action (when enabled)
"""

from __future__ import annotations

from typing import Any, Protocol

from src.action_space.proposal import (
    ClassicActionProposal,
    ToolActionProposal,
    _ProposalBase,
)
from src.action_space.registry import ToolRegistry
from src.shared.logging import get_logger

_log = get_logger(module="action_space.proposer")

_WITHDRAW = ClassicActionProposal(
    name="Withdraw",
    description="Disengage from the current scenario",
    modifiers={"E": -0.8, "R": -0.3},
)


class ProposerBackend(Protocol):
    """Pluggable backend for generating action candidates."""

    def propose(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> list[_ProposalBase]:
        """Return candidate proposals given current context."""
        ...


class StaticProposerBackend:
    """Returns a fixed set of classic action proposals (fallback)."""

    def __init__(self, defaults: list[_ProposalBase] | None = None) -> None:
        self._defaults = list(defaults) if defaults else []

    def propose(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> list[_ProposalBase]:
        """Return the static defaults regardless of context."""
        return list(self._defaults)


class ActionProposer:
    """Merges tool-based proposals with backend-generated candidates."""

    def __init__(
        self,
        backend: ProposerBackend,
        tool_registry: ToolRegistry | None = None,
        max_proposals: int = 10,
        include_withdraw: bool = False,
    ) -> None:
        self._backend = backend
        self._tools = tool_registry
        self._max = max_proposals
        self._withdraw = include_withdraw

    def propose(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> list[_ProposalBase]:
        """Generate candidate actions from context."""
        candidates: list[_ProposalBase] = []

        if self._tools:
            candidates.extend(self._tool_proposals())

        candidates.extend(self._backend.propose(state, trajectory, goals))

        if self._withdraw:
            candidates.append(_WITHDRAW)

        result = candidates[: self._max]

        _log.debug(
            "actions_proposed",
            count=len(result),
            kinds=[getattr(p, "kind", "unknown") for p in result],
        )
        return result

    def _tool_proposals(self) -> list[_ProposalBase]:
        """Create one proposal per registered tool with default args."""
        if not self._tools:
            return []
        proposals: list[_ProposalBase] = []
        for tool in self._tools.list_tools():
            proposals.append(
                ToolActionProposal(
                    name=f"use_{tool.name}",
                    description=tool.description,
                    tool_name=tool.name,
                    tool_args={},
                )
            )
        return proposals
