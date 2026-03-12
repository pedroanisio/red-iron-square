"""Action encoder: maps open-ended proposals to personality-dimension modifiers.

Classic proposals pass through directly. Tool, API, and text proposals
are encoded via a pluggable backend (heuristic or LLM-backed).
"""

from __future__ import annotations

from typing import Protocol

from src.action_space.params import ActionEncoderParams
from src.action_space.proposal import (
    ApiActionProposal,
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
    _ProposalBase,
)
from src.action_space.registry import ToolRegistry
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action
from src.sdk.builders import build_action
from src.shared.logging import get_logger

_log = get_logger(module="action_space.encoder")

# Heuristic intent-to-modifier mappings for text actions.
_INTENT_HINTS: dict[str, dict[str, float]] = {
    "explain": {"O": 0.6, "A": 0.4, "E": 0.3},
    "persuade": {"E": 0.7, "O": 0.5, "A": -0.2},
    "comfort": {"A": 0.8, "E": 0.3, "N": -0.4},
    "challenge": {"O": 0.6, "E": 0.5, "A": -0.4, "N": 0.3},
    "withdraw": {"E": -0.6, "N": 0.5, "R": -0.3},
    "organize": {"C": 0.8, "T": 0.4},
    "create": {"O": 0.9, "I": 0.5, "C": -0.2},
    "analyze": {"O": 0.5, "C": 0.7, "I": 0.4},
}


class EncoderBackend(Protocol):
    """Pluggable backend for modifier estimation."""

    def estimate(self, proposal: _ProposalBase) -> dict[str, float]:
        """Return estimated modifiers for a non-classic proposal."""
        ...


class HeuristicEncoderBackend:
    """Deterministic heuristic encoding (no LLM calls).

    Uses tool personality hints and intent-to-modifier lookup tables.
    Falls back to zero modifiers for unknown actions.
    """

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        self._tools = tool_registry

    def estimate(self, proposal: _ProposalBase) -> dict[str, float]:
        """Estimate modifiers from tool hints or intent keywords."""
        if isinstance(proposal, ToolActionProposal):
            return self._encode_tool(proposal)
        if isinstance(proposal, ApiActionProposal):
            return self._encode_api(proposal)
        if isinstance(proposal, TextActionProposal):
            return self._encode_text(proposal)
        return {}

    def _encode_tool(self, proposal: ToolActionProposal) -> dict[str, float]:
        """Encode a tool proposal using registry personality hints."""
        if self._tools and self._tools.has(proposal.tool_name):
            return dict(self._tools.get(proposal.tool_name).personality_hint)
        return {"O": 0.3}

    def _encode_api(self, proposal: ApiActionProposal) -> dict[str, float]:
        """Encode an API proposal with method-dependent modifiers."""
        base: dict[str, float] = {"O": 0.4, "C": 0.3}
        if proposal.method in ("POST", "PUT", "DELETE"):
            base["E"] = 0.3
        return base

    def _encode_text(self, proposal: TextActionProposal) -> dict[str, float]:
        """Encode a text proposal via intent keyword matching."""
        intent = proposal.intent.lower().strip()
        for key, hint in _INTENT_HINTS.items():
            if key in intent:
                return dict(hint)
        return {"O": 0.3, "E": 0.2}


class ActionEncoder:
    """Maps ActionProposals to Action objects with modifier vectors.

    Classic proposals pass through directly (zero encoding cost).
    Other kinds delegate to an EncoderBackend.
    """

    def __init__(
        self,
        dimension_registry: DimensionRegistry,
        backend: EncoderBackend,
        params: ActionEncoderParams | None = None,
    ) -> None:
        self._dim_reg = dimension_registry
        self._backend = backend
        self._params = params or ActionEncoderParams()

    def encode(self, proposal: _ProposalBase) -> Action:
        """Encode a single proposal into an Action with modifier vector."""
        if isinstance(proposal, ClassicActionProposal):
            return build_action(
                proposal.name,
                proposal.modifiers,
                self._dim_reg,
                description=proposal.description,
            )

        modifiers = self._backend.estimate(proposal)
        _log.debug(
            "action_encoded",
            name=proposal.name,
            kind=getattr(proposal, "kind", "unknown"),
            modifiers=modifiers,
        )
        return build_action(
            proposal.name,
            modifiers,
            self._dim_reg,
            description=proposal.description,
        )

    def encode_batch(self, proposals: list[_ProposalBase]) -> list[Action]:
        """Encode multiple proposals into Action objects."""
        return [self.encode(p) for p in proposals]
