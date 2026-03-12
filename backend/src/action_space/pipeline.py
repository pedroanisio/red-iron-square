"""Action pipeline: propose and encode actions in a single step.

Wraps ActionProposer and ActionEncoder into a callable unit that
the temporal simulator can use to dynamically generate actions per tick.
"""

from __future__ import annotations

from typing import Any

from __future__ import annotations

from typing import TYPE_CHECKING

from src.action_space.encoder import ActionEncoder
from src.action_space.proposal import _ProposalBase
from src.action_space.proposer import ActionProposer
from src.personality.vectors import Action
from src.shared.logging import get_logger

if TYPE_CHECKING:
    from src.action_space.executor import ActionExecutor, ActionResult

_log = get_logger(module="action_space.pipeline")


class ActionPipeline:
    """Combines proposal, encoding, and optional execution into a single unit."""

    def __init__(
        self,
        proposer: ActionProposer,
        encoder: ActionEncoder,
        executor: ActionExecutor | None = None,
    ) -> None:
        self._proposer = proposer
        self._encoder = encoder
        self._executor = executor

    def propose_and_encode(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> tuple[list[Action], list[_ProposalBase]]:
        """Generate candidate actions and encode them to modifier vectors.

        Returns:
            Tuple of (encoded Action objects, raw proposals for metadata).
        """
        proposals = self._proposer.propose(
            state=state,
            trajectory=trajectory,
            goals=goals,
        )
        if not proposals:
            _log.warning("no_proposals_generated")
            return [], []

        actions = self._encoder.encode_batch(proposals)
        _log.debug(
            "pipeline_complete",
            n_proposals=len(proposals),
            n_actions=len(actions),
        )
        return actions, proposals

    def execute(self, proposal: _ProposalBase) -> ActionResult:
        """Execute a chosen action proposal via the registered executor.

        Raises:
            RuntimeError: If no executor is registered.
        """
        if self._executor is None:
            msg = "No ActionExecutor registered on this pipeline"
            raise RuntimeError(msg)
        result = self._executor.execute(proposal)
        _log.debug(
            "action_executed",
            proposal=proposal.name,
            success=result.success,
        )
        return result
