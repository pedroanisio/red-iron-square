"""System 2 orchestrator: coordinates LLM calls on surprise spikes.

Flow on spike:
  1. Call agent_runtime.propose_matrices() for narrative model update
  2. Apply proposal via NarrativeGenerativeModel.update_from_proposal()
  3. Reset self-evidencing beta for post-refresh recalibration
  4. Fall back to heuristic refresh_from_trajectory() on any failure
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.shared.logging import get_logger

if TYPE_CHECKING:
    from src.constructed_emotion.affect import AffectSignal
    from src.narrative.model import NarrativeGenerativeModel
    from src.self_evidencing.modulator import SelfEvidencingModulator
    from src.shared.protocols import System2RuntimeProtocol

_log = get_logger(module="temporal.system2")


class System2Orchestrator:
    """Coordinates System 2 narrative refresh and beta reset on surprise spikes.

    When an LLM runtime is available, proposes new generative model matrices
    grounded in personality and trajectory. Falls back to heuristic noise
    injection when the runtime is absent or errors.
    """

    def __init__(
        self,
        agent_runtime: System2RuntimeProtocol | None,
        narrative_model: NarrativeGenerativeModel | None,
        self_evidencing: SelfEvidencingModulator | None,
        personality: dict[str, float],
    ) -> None:
        self._agent_runtime = agent_runtime
        self._narrative_model = narrative_model
        self._self_evidencing = self_evidencing
        self._personality = personality

    def on_tick(
        self,
        affect_signal: AffectSignal | None,
        trajectory_window: list[dict[str, Any]],
    ) -> None:
        """Trigger System 2 coordination on surprise spikes.

        Does nothing when affect signal is absent or not a spike.
        """
        if affect_signal is None or not affect_signal.is_surprise_spike:
            return
        if self._narrative_model is None:
            return

        if self._agent_runtime is not None:
            self._try_llm_refresh(trajectory_window)
        else:
            self._heuristic_refresh(trajectory_window)

    def _try_llm_refresh(self, trajectory_window: list[dict[str, Any]]) -> None:
        """Attempt LLM-driven matrix proposal with heuristic fallback."""
        assert self._narrative_model is not None
        try:
            proposal, _meta = self._agent_runtime.propose_matrices(  # type: ignore[union-attr]
                personality=self._personality,
                trajectory_window=trajectory_window,
                n_states=self._narrative_model._n_states,
                n_actions=self._narrative_model._n_actions,
            )
            applied = self._narrative_model.update_from_proposal(proposal)
            if applied:
                _log.info("system2_llm_refresh_applied")
                self._reset_beta()
                return
            _log.warning("system2_llm_proposal_rejected_falling_back")
        except Exception:
            _log.warning("system2_llm_error_falling_back", exc_info=True)
        self._heuristic_refresh(trajectory_window)

    def _heuristic_refresh(self, trajectory_window: list[dict[str, Any]]) -> None:
        """Fall back to noise-injection B-matrix update."""
        assert self._narrative_model is not None
        self._narrative_model.refresh_from_trajectory(trajectory_window)

    def _reset_beta(self) -> None:
        """Reset self-evidencing beta after narrative refresh."""
        if self._self_evidencing is not None:
            self._self_evidencing.reset_beta()
            _log.info("system2_beta_reset")

    @staticmethod
    def build_trajectory_window(
        memory_entries: list[Any],
        state: Any,
    ) -> list[dict[str, Any]]:
        """Extract trajectory window from recent memory entries.

        Returns a list of dicts with state, outcome, and action keys
        suitable for both heuristic refresh and LLM proposal.
        """
        return [
            {
                "state": list(state.to_array()) if hasattr(state, "to_array") else [],
                "outcome": e.outcome,
                "action": e.action_name,
            }
            for e in memory_entries
        ]
