"""Tests for dynamic action space integration in TemporalSimulator."""

from __future__ import annotations

import pytest
from src.action_space.proposal import ClassicActionProposal
from src.action_space.proposer import StaticProposerBackend
from src.personality.vectors import Action
from src.sdk import AgentSDK


@pytest.fixture()
def sdk():
    """Default SDK instance."""
    return AgentSDK.default()


@pytest.fixture()
def personality(sdk):
    """Neutral personality vector."""
    return sdk.personality({"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5})


@pytest.fixture()
def scenario(sdk):
    """Build a basic test scenario."""
    return sdk.scenario({"O": 0.0, "C": 0.0}, name="test_scenario")


@pytest.fixture()
def actions(sdk):
    """Two basic actions."""
    return [
        sdk.action("Act", {"O": 0.1, "C": 0.1}),
        sdk.action("Wait", {"O": -0.1, "C": -0.1}),
    ]


def _default_proposals() -> list[ClassicActionProposal]:
    """Classic proposals for the static backend."""
    return [
        ClassicActionProposal(
            name="Engage",
            description="engage actively",
            modifiers={"O": 0.3, "E": 0.2},
        ),
        ClassicActionProposal(
            name="Observe",
            description="observe passively",
            modifiers={"O": -0.1, "C": 0.2},
        ),
    ]


class TestSimulatorStaticActions:
    """Verify simulator still works with static actions (backward compat)."""

    def test_tick_with_static_actions(self, sdk, personality, actions, scenario):
        """Static actions path is unchanged."""
        client = sdk.simulator(personality, actions, temperature=1.0)
        result = client.tick(scenario)
        assert result.action in {"Act", "Wait"}

    def test_no_pipeline_uses_static(self, sdk, personality, actions):
        """Without pipeline, _resolve_tick_actions returns self.actions."""
        client = sdk.simulator(personality, actions)
        sim = client.simulator
        assert sim._action_pipeline is None
        resolved = sim._resolve_tick_actions()
        assert resolved == list(actions)


class TestSimulatorDynamicActions:
    """Verify dynamic action pipeline integration."""

    def test_pipeline_set_via_sdk(self, personality, actions):
        """with_open_actions wires pipeline into simulator."""
        backend = StaticProposerBackend(defaults=_default_proposals())
        sdk = AgentSDK.with_open_actions(proposer_backend=backend)
        client = sdk.simulator(personality, actions)
        assert client.simulator._action_pipeline is not None

    def test_pipeline_proposes_actions(self, personality, scenario):
        """Pipeline generates actions when static list is empty."""
        backend = StaticProposerBackend(defaults=_default_proposals())
        sdk = AgentSDK.with_open_actions(proposer_backend=backend)
        empty_actions: list[Action] = []
        client = sdk.simulator(personality, empty_actions)
        sim = client.simulator
        resolved = sim._resolve_tick_actions()
        assert len(resolved) == 2

    def test_pipeline_overrides_static_actions(self, personality, actions):
        """When pipeline is set, it takes precedence over static actions."""
        backend = StaticProposerBackend(defaults=_default_proposals())
        sdk = AgentSDK.with_open_actions(proposer_backend=backend)
        client = sdk.simulator(personality, actions)
        sim = client.simulator
        resolved = sim._resolve_tick_actions()
        names = {a.name for a in resolved}
        assert names == {"Engage", "Observe"}

    def test_tick_with_pipeline(self, personality, scenario):
        """Full tick works with pipeline-proposed actions."""
        backend = StaticProposerBackend(defaults=_default_proposals())
        sdk = AgentSDK.with_open_actions(proposer_backend=backend)
        empty_actions: list[Action] = []
        client = sdk.simulator(personality, empty_actions, temperature=1.0)
        result = client.tick(scenario)
        assert result.action in {"Engage", "Observe"}


class TestEfeBreakdownWithDynamic:
    """Verify EFE breakdown passes tick_actions correctly."""

    def test_efe_breakdown_receives_actions(self, sdk, personality, actions, scenario):
        """_compute_efe_breakdown receives tick_actions parameter."""
        client = sdk.simulator(personality, actions)
        sim = client.simulator
        activations = sim._compute_modulated_activations(scenario)
        result = sim._compute_efe_breakdown(scenario, activations, actions)
        # Non-EFE engine returns None
        assert result is None
