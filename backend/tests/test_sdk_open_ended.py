"""Tests for open-ended action space via AgentSDK."""

import pytest
from src.action_space.proposal import ClassicActionProposal
from src.action_space.proposer import StaticProposerBackend
from src.sdk import AgentSDK

_PERSONALITY = {
    "O": 0.8,
    "C": 0.5,
    "E": 0.3,
    "A": 0.7,
    "N": 0.4,
    "R": 0.9,
    "I": 0.6,
    "T": 0.2,
}


class TestSDKOpenEnded:
    """AgentSDK supports open-ended action proposal and decision."""

    def test_with_open_actions_factory(self) -> None:
        """Factory creates a valid SDK instance."""
        sdk = AgentSDK.with_open_actions()
        assert sdk is not None

    def test_propose_and_decide_with_classics(self) -> None:
        """Propose and decide works with classic action proposals."""
        classics = [
            ClassicActionProposal(
                name="bold",
                description="bold",
                modifiers={"O": 1.0, "R": 0.8},
            ),
            ClassicActionProposal(
                name="safe",
                description="safe",
                modifiers={"C": 0.9, "T": 0.8},
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        sdk = AgentSDK.with_open_actions(proposer_backend=backend)
        personality = sdk.personality(_PERSONALITY)
        scenario = sdk.scenario({"O": 0.9, "N": 0.7}, name="pitch")
        result = sdk.propose_and_decide(personality, scenario)
        assert result.chosen_action in ("bold", "safe")
        assert len(result.probabilities) == 2

    def test_backward_compat_decide_unchanged(self) -> None:
        """Classic decide() path still works on default SDK."""
        sdk = AgentSDK.default()
        personality = sdk.personality(_PERSONALITY)
        scenario = sdk.scenario({"O": 0.9}, name="test")
        actions = [
            sdk.action("bold", {"O": 1.0}),
            sdk.action("safe", {"C": 0.9}),
        ]
        result = sdk.decide(personality, scenario, actions)
        assert result.chosen_action in ("bold", "safe")

    def test_propose_and_decide_returns_proposal_metadata(self) -> None:
        """Result includes serialized proposal data."""
        classics = [
            ClassicActionProposal(
                name="bold",
                description="bold move",
                modifiers={"O": 1.0},
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        sdk = AgentSDK.with_open_actions(proposer_backend=backend)
        personality = sdk.personality(_PERSONALITY)
        scenario = sdk.scenario({"O": 0.9}, name="test")
        result = sdk.propose_and_decide(personality, scenario)
        assert result.proposals is not None
        assert len(result.proposals) >= 1

    def test_propose_and_decide_requires_open_actions(self) -> None:
        """Calling propose_and_decide on default SDK raises RuntimeError."""
        sdk = AgentSDK.default()
        personality = sdk.personality(_PERSONALITY)
        scenario = sdk.scenario({"O": 0.9}, name="test")
        with pytest.raises(RuntimeError, match="with_open_actions"):
            sdk.propose_and_decide(personality, scenario)
