"""LLM-backed action proposer backend.

Asks the LLM to generate contextually appropriate action candidates,
then parses them into typed ActionProposal objects.
"""

from __future__ import annotations

import json
from typing import Any

from src.action_space.proposal import (
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
    _ProposalBase,
)
from src.action_space.registry import ToolRegistry
from src.shared.logging import get_logger

_log = get_logger(module="action_space.llm_proposer")


class LLMProposerBackend:
    """Proposes actions via LLM structured output."""

    def __init__(
        self,
        adapter: Any,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._adapter = adapter
        self._tools = tool_registry

    def propose(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> list[_ProposalBase]:
        """Ask the LLM to propose candidate actions."""
        try:
            from src.llm.schemas import ActionSetProposal

            result, _meta = self._adapter.complete_json(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(state, trajectory, goals),
                response_model=ActionSetProposal,
            )
            return self._parse_actions(result.actions)
        except Exception:
            _log.warning("llm_proposal_failed", exc_info=True)
            return []

    def _system_prompt(self) -> str:
        """Build the system prompt, including tool context if available."""
        tools_ctx = ""
        if self._tools:
            tools_ctx = f"\nAvailable tools:\n{self._tools.to_prompt_context()}\n"
        return (
            "You propose candidate actions for a personality-driven agent. "
            "Return JSON with keys: `actions` (list of action objects), `rationale`. "
            "Each action object must have: "
            "`kind` (tool|text|classic), `name`, `description`. "
            "Tool actions need: `tool_name`, `tool_args`. "
            "Text actions need: `intent`. "
            "Classic actions need: `modifiers` (dict[str, float] in [-1,1])."
            f"{tools_ctx}"
        )

    def _user_prompt(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> str:
        """Build the user prompt with state, recent trajectory, and goals."""
        return json.dumps(
            {
                "current_state": state,
                "recent_trajectory": trajectory[-5:] if trajectory else [],
                "goals": goals,
                "output_schema": "ActionSetProposal",
            }
        )

    def _parse_actions(self, raw: list[dict[str, Any]]) -> list[_ProposalBase]:
        """Convert raw dicts to typed proposal objects."""
        proposals: list[_ProposalBase] = []
        for item in raw:
            try:
                kind = item.get("kind", "classic")
                if kind == "tool":
                    proposals.append(ToolActionProposal(**item))
                elif kind == "text":
                    proposals.append(TextActionProposal(**item))
                else:
                    proposals.append(ClassicActionProposal(**item))
            except Exception:
                _log.debug("skipping_invalid_proposal", item=item, exc_info=True)
        return proposals
