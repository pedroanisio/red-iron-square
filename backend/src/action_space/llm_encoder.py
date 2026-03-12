"""LLM-backed action encoder backend.

Uses structured output to estimate personality-dimension modifiers.
Falls back to empty modifiers on LLM failure.
"""

from __future__ import annotations

from typing import Any

from src.action_space.proposal import _ProposalBase
from src.shared.logging import get_logger

_log = get_logger(module="action_space.llm_encoder")


class LLMEncoderBackend:
    """Encodes action proposals via LLM structured output."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def estimate(self, proposal: _ProposalBase) -> dict[str, float]:
        """Ask the LLM to estimate personality-dimension modifiers."""
        try:
            from src.llm.schemas import ActionEncoding

            encoding, _meta = self._adapter.complete_json(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(proposal),
                response_model=ActionEncoding,
            )
            return dict(encoding.modifiers)
        except Exception:
            _log.warning("llm_encoding_failed", action=proposal.name, exc_info=True)
            return {}

    def _system_prompt(self) -> str:
        """Return the system prompt for action encoding."""
        return (
            "You estimate how an action aligns with personality dimensions. "
            "Return JSON only. Keys: `modifiers` (dict[str, float] in [-1,1]), "
            "`confidence` (float 0-1), `rationale`. "
            "Dimensions: O(penness), C(onscientiousness), E(xtraversion), "
            "A(greeableness), N(euroticism), R(esilience), I(dealism), T(radition)."
        )

    def _user_prompt(self, proposal: _ProposalBase) -> str:
        """Build the user prompt from proposal fields."""
        import json

        return json.dumps(
            {
                "action_name": proposal.name,
                "action_description": proposal.description,
                "action_kind": getattr(proposal, "kind", "unknown"),
                "output_schema": "ActionEncoding",
            }
        )
