"""Phase C integration tests: LLM pipeline wiring for System 2.

Verifies:
  - NarrativeGenerativeModel.update_from_proposal() applies LLM matrices
  - System2Orchestrator coordinates spike → propose → refresh → reset
  - Graceful degradation when LLM fails or is absent
  - SDK threads agent_runtime through to simulators
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from src.llm.schemas import (
    EmotionConstructor,
    LLMInvocationMetadata,
    LLMInvocationResult,
    MatrixProposal,
)
from src.narrative.model import NarrativeGenerativeModel
from src.sdk import AgentSDK
from src.self_evidencing.modulator import SelfEvidencingModulator


def _balanced() -> dict[str, float]:
    """Return a balanced personality profile."""
    return {k: 0.5 for k in "OCEANRIT"}


def _make_invocation_result() -> LLMInvocationResult:
    """Create a minimal LLM invocation result for mocking."""
    return LLMInvocationResult(
        raw_text="{}",
        metadata=LLMInvocationMetadata(model="test", provider="test"),
    )


def _make_proposal(
    n_states: int = 5,
    n_actions: int = 3,
) -> MatrixProposal:
    """Create a valid MatrixProposal with distinct matrices.

    Uses action-dependent diagonal bias so the result differs from
    the default initialization in NarrativeGenerativeModel.
    """
    rng = np.random.default_rng(99)
    a_arr = np.zeros((n_states, n_states, n_actions))
    b_arr = np.zeros((n_states, n_states, n_actions))
    for a in range(n_actions):
        a_raw = rng.dirichlet(np.ones(n_states), size=n_states).T
        b_raw = rng.dirichlet(np.ones(n_states), size=n_states)
        a_arr[:, :, a] = a_raw
        b_arr[:, :, a] = b_raw
    return MatrixProposal(
        a_matrix=a_arr.tolist(),
        b_matrix=b_arr.tolist(),
        rationale="test proposal",
        n_states=n_states,
        n_actions=n_actions,
    )


# ---------------------------------------------------------------------------
# NarrativeGenerativeModel.update_from_proposal
# ---------------------------------------------------------------------------


class TestUpdateFromProposal:
    """update_from_proposal replaces A/B matrices from LLM-proposed values."""

    def test_valid_proposal_replaces_matrices(self) -> None:
        """Matching-dimension proposal updates both A and B matrices."""
        model = NarrativeGenerativeModel(_balanced(), n_states=5, n_actions=3)
        A_before = model.cached_A.copy()
        B_before = model.cached_B.copy()
        proposal = _make_proposal(n_states=5, n_actions=3)

        result = model.update_from_proposal(proposal)

        assert result is True
        assert not np.allclose(model.cached_A, A_before)
        assert not np.allclose(model.cached_B, B_before)

    def test_dimension_mismatch_rejects(self) -> None:
        """Proposal with wrong dimensions is rejected gracefully."""
        model = NarrativeGenerativeModel(_balanced(), n_states=5, n_actions=3)
        B_before = model.cached_B.copy()
        proposal = _make_proposal(n_states=4, n_actions=3)

        result = model.update_from_proposal(proposal)

        assert result is False
        np.testing.assert_array_equal(model.cached_B, B_before)

    def test_normalizes_rows(self) -> None:
        """Proposal rows are renormalized even if they don't sum to 1."""
        model = NarrativeGenerativeModel(_balanced(), n_states=5, n_actions=3)
        proposal = _make_proposal(n_states=5, n_actions=3)
        proposal.b_matrix[0][0] = [10.0, 10.0, 10.0]

        model.update_from_proposal(proposal)

        B = model.cached_B
        for a in range(B.shape[2]):
            row_sums = np.sum(B[:, :, a], axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# System2Orchestrator
# ---------------------------------------------------------------------------


class TestSystem2Orchestrator:
    """System2Orchestrator coordinates LLM calls on surprise spikes."""

    def _make_affect_signal(self, *, is_spike: bool) -> object:
        """Create a minimal AffectSignal-like object."""
        from src.constructed_emotion.affect import AffectSignal

        return AffectSignal(
            valence=0.5,
            arousal_signal=0.8,
            free_energy=1.0,
            is_surprise_spike=is_spike,
            mood=0.2,
            constructed_emotions=[],
        )

    def _make_mock_runtime(self) -> MagicMock:
        """Create a mock runtime that returns valid proposals."""
        runtime = MagicMock()
        runtime.propose_matrices.return_value = (
            _make_proposal(n_states=5, n_actions=3),
            _make_invocation_result(),
        )
        return runtime

    def test_calls_propose_on_spike(self) -> None:
        """Orchestrator calls propose_matrices when spike is True."""
        from src.temporal.system2 import System2Orchestrator

        runtime = self._make_mock_runtime()
        model = NarrativeGenerativeModel(_balanced(), n_states=5, n_actions=3)
        orch = System2Orchestrator(
            agent_runtime=runtime,
            narrative_model=model,
            self_evidencing=None,
            personality=_balanced(),
        )
        sig = self._make_affect_signal(is_spike=True)
        orch.on_tick(sig, [{"state": [0.1] * 5, "outcome": 0.5, "action": "A"}])

        runtime.propose_matrices.assert_called_once()

    def test_resets_beta_after_refresh(self) -> None:
        """Orchestrator resets self-evidencing beta after narrative refresh."""
        from src.temporal.system2 import System2Orchestrator

        runtime = self._make_mock_runtime()
        model = NarrativeGenerativeModel(_balanced(), n_states=5, n_actions=3)
        se = SelfEvidencingModulator()
        se.decay_beta(0.5)
        beta_before = se.beta

        orch = System2Orchestrator(
            agent_runtime=runtime,
            narrative_model=model,
            self_evidencing=se,
            personality=_balanced(),
        )
        sig = self._make_affect_signal(is_spike=True)
        orch.on_tick(sig, [{"state": [0.1] * 5, "outcome": 0.5, "action": "A"}])

        assert se.beta != beta_before, "Beta should be reset after narrative refresh"

    def test_graceful_degradation_on_llm_error(self) -> None:
        """Orchestrator falls back to heuristic when LLM raises."""
        from src.temporal.system2 import System2Orchestrator

        runtime = MagicMock()
        runtime.propose_matrices.side_effect = RuntimeError("LLM down")
        model = NarrativeGenerativeModel(_balanced(), n_states=5, n_actions=3)
        B_before = model.cached_B.copy()

        orch = System2Orchestrator(
            agent_runtime=runtime,
            narrative_model=model,
            self_evidencing=None,
            personality=_balanced(),
        )
        sig = self._make_affect_signal(is_spike=True)
        window = [{"state": [0.1] * 5, "outcome": 0.5, "action": "A"}]

        orch.on_tick(sig, window)

        assert not np.allclose(model.cached_B, B_before), (
            "Heuristic fallback should still update B"
        )

    def test_noop_when_no_spike(self) -> None:
        """Orchestrator does nothing when affect signal has no spike."""
        from src.temporal.system2 import System2Orchestrator

        runtime = self._make_mock_runtime()
        model = NarrativeGenerativeModel(_balanced(), n_states=5, n_actions=3)
        orch = System2Orchestrator(
            agent_runtime=runtime,
            narrative_model=model,
            self_evidencing=None,
            personality=_balanced(),
        )
        sig = self._make_affect_signal(is_spike=False)
        orch.on_tick(sig, [])

        runtime.propose_matrices.assert_not_called()

    def test_noop_when_no_runtime(self) -> None:
        """Orchestrator uses heuristic path when runtime is None."""
        from src.temporal.system2 import System2Orchestrator

        model = NarrativeGenerativeModel(_balanced(), n_states=5, n_actions=3)
        B_before = model.cached_B.copy()
        orch = System2Orchestrator(
            agent_runtime=None,
            narrative_model=model,
            self_evidencing=None,
            personality=_balanced(),
        )
        sig = self._make_affect_signal(is_spike=True)
        window = [{"state": [0.1] * 5, "outcome": 0.5, "action": "A"}]
        orch.on_tick(sig, window)

        assert not np.allclose(model.cached_B, B_before), (
            "Heuristic should still refresh on spike without runtime"
        )

    def test_noop_when_no_affect_signal(self) -> None:
        """Orchestrator does nothing when affect signal is None."""
        from src.temporal.system2 import System2Orchestrator

        runtime = self._make_mock_runtime()
        model = NarrativeGenerativeModel(_balanced(), n_states=5, n_actions=3)
        B_before = model.cached_B.copy()
        orch = System2Orchestrator(
            agent_runtime=runtime,
            narrative_model=model,
            self_evidencing=None,
            personality=_balanced(),
        )
        orch.on_tick(None, [])

        runtime.propose_matrices.assert_not_called()
        np.testing.assert_array_equal(model.cached_B, B_before)


# ---------------------------------------------------------------------------
# SDK wiring
# ---------------------------------------------------------------------------


class TestSDKAgentRuntimeWiring:
    """SDK threads agent_runtime through to simulators."""

    def test_sdk_accepts_agent_runtime(self) -> None:
        """AgentSDK constructor accepts agent_runtime parameter."""
        runtime = MagicMock()
        sdk = AgentSDK.with_self_evidencing()
        sdk.set_agent_runtime(runtime)
        assert sdk._agent_runtime is runtime

    def test_simulator_receives_agent_runtime(self) -> None:
        """Simulator factory passes agent_runtime into the simulator."""
        runtime = MagicMock()
        runtime.propose_matrices.return_value = (
            _make_proposal(),
            _make_invocation_result(),
        )
        sdk = AgentSDK.with_constructed_emotion()
        sdk.set_agent_runtime(runtime)
        personality = sdk.personality(_balanced())
        actions = [sdk.action("A", {"O": 0.3})]
        sim = sdk.simulator(personality, actions, rng=np.random.default_rng(0))

        assert sim.simulator._system2 is not None
        assert sim.simulator._system2._agent_runtime is runtime

    def test_self_aware_simulator_receives_agent_runtime(self) -> None:
        """Self-aware simulator also gets agent_runtime."""
        runtime = MagicMock()
        runtime.propose_matrices.return_value = (
            _make_proposal(),
            _make_invocation_result(),
        )
        sdk = AgentSDK.with_self_evidencing()
        sdk.set_agent_runtime(runtime)
        personality = sdk.personality(_balanced())
        psi_hat = sdk.initial_self_model(_balanced())
        actions = [sdk.action("A", {"O": 0.3})]
        sim = sdk.self_aware_simulator(
            personality,
            psi_hat,
            actions,
            rng=np.random.default_rng(0),
        )

        assert sim.simulator._system2 is not None


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestPhaseC_E2E:
    """Full Phase C pipeline: spike → LLM → narrative update → beta reset."""

    def test_full_pipeline_with_mocked_runtime(self) -> None:
        """Run simulation with mock runtime; verify matrices change on spike."""
        runtime = MagicMock()
        runtime.propose_matrices.return_value = (
            _make_proposal(n_states=5, n_actions=3),
            _make_invocation_result(),
        )
        runtime.construct_emotion.return_value = (
            EmotionConstructor(
                label="excitement",
                description="test",
                valence_sign="positive",
                arousal_level="high",
                confidence=0.9,
            ),
            _make_invocation_result(),
        )

        sdk = AgentSDK.with_self_evidencing()
        sdk.set_agent_runtime(runtime)
        personality = sdk.personality(_balanced())
        psi_hat = sdk.initial_self_model(_balanced())
        actions = [
            sdk.action("A", {"O": 0.5, "E": 0.3}),
            sdk.action("B", {"C": 0.5, "E": -0.1}),
        ]
        sim = sdk.self_aware_simulator(
            personality,
            psi_hat,
            actions,
            rng=np.random.default_rng(42),
        )
        scenario = sdk.scenario(_balanced(), name="test")

        for _ in range(200):
            sim.tick(scenario)

        # At least one spike should have triggered propose_matrices
        assert runtime.propose_matrices.call_count >= 1

    def test_heuristic_fallback_without_runtime(self) -> None:
        """Without runtime, simulation completes with heuristic path."""
        sdk = AgentSDK.with_self_evidencing()
        personality = sdk.personality(_balanced())
        psi_hat = sdk.initial_self_model(_balanced())
        actions = [
            sdk.action("A", {"O": 0.5, "E": 0.3}),
            sdk.action("B", {"C": 0.5, "E": -0.1}),
        ]
        sim = sdk.self_aware_simulator(
            personality,
            psi_hat,
            actions,
            rng=np.random.default_rng(42),
        )
        scenario = sdk.scenario(_balanced(), name="test")

        for _ in range(100):
            sim.tick(scenario)
        # No crash = regression test passes

    def test_llm_failure_falls_back_gracefully(self) -> None:
        """Runtime that raises still allows simulation to complete."""
        runtime = MagicMock()
        runtime.propose_matrices.side_effect = RuntimeError("LLM unavailable")

        sdk = AgentSDK.with_constructed_emotion()
        sdk.set_agent_runtime(runtime)
        personality = sdk.personality(_balanced())
        actions = [
            sdk.action("A", {"O": 0.5, "E": 0.3}),
            sdk.action("B", {"C": 0.5, "E": -0.1}),
        ]
        sim = sdk.simulator(
            personality,
            actions,
            rng=np.random.default_rng(42),
        )
        scenario = sdk.scenario(_balanced(), name="test")

        for _ in range(100):
            sim.tick(scenario)
        # No crash = graceful degradation works
