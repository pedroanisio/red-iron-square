"""Tests for action-to-modifier encoding."""

import numpy as np
import pytest
from src.action_space.encoder import ActionEncoder, HeuristicEncoderBackend
from src.action_space.proposal import (
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
)
from src.action_space.registry import ToolCapability, ToolRegistry
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action


class TestClassicEncoding:
    """Classic proposals pass through without LLM call."""

    def test_classic_becomes_action_directly(self) -> None:
        """Classic proposal is converted to Action with exact modifiers."""
        registry = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=registry,
            backend=HeuristicEncoderBackend(),
        )
        proposal = ClassicActionProposal(
            name="bold",
            description="bold move",
            modifiers={"O": 1.0, "R": 0.8},
        )
        action = encoder.encode(proposal)
        assert isinstance(action, Action)
        assert action.name == "bold"
        o_idx = registry.index("O")
        assert action.modifiers[o_idx] == pytest.approx(1.0)

    def test_classic_preserves_all_modifiers(self) -> None:
        """Every modifier in a classic proposal survives round-trip."""
        registry = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=registry,
            backend=HeuristicEncoderBackend(),
        )
        proposal = ClassicActionProposal(
            name="safe",
            description="safe choice",
            modifiers={"C": 0.9, "T": 0.8, "N": -0.2},
        )
        action = encoder.encode(proposal)
        c_idx = registry.index("C")
        t_idx = registry.index("T")
        n_idx = registry.index("N")
        assert action.modifiers[c_idx] == pytest.approx(0.9)
        assert action.modifiers[t_idx] == pytest.approx(0.8)
        assert action.modifiers[n_idx] == pytest.approx(-0.2)


class TestToolEncoding:
    """Tool proposals encode via personality hint + heuristic."""

    def test_tool_with_personality_hint(self) -> None:
        """Registered tool hint is used as modifier source."""
        dim_reg = DimensionRegistry()
        tool_reg = ToolRegistry()
        tool_reg.register(
            ToolCapability(
                name="web_search",
                description="search the web",
                parameter_schema={"query": {"type": "string"}},
                personality_hint={"O": 0.8, "E": 0.4},
            )
        )
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(tool_registry=tool_reg),
        )
        proposal = ToolActionProposal(
            name="search_papers",
            description="search for papers",
            tool_name="web_search",
            tool_args={"query": "active inference"},
        )
        action = encoder.encode(proposal)
        o_idx = dim_reg.index("O")
        assert action.modifiers[o_idx] == pytest.approx(0.8)

    def test_tool_without_hint_uses_defaults(self) -> None:
        """Unknown tool falls back to default modifier vector."""
        dim_reg = DimensionRegistry()
        tool_reg = ToolRegistry()
        tool_reg.register(
            ToolCapability(
                name="noop",
                description="does nothing",
                parameter_schema={},
            )
        )
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(tool_registry=tool_reg),
        )
        proposal = ToolActionProposal(
            name="do_nothing",
            description="nothing",
            tool_name="noop",
            tool_args={},
        )
        action = encoder.encode(proposal)
        assert action.modifiers.shape == (dim_reg.size,)
        assert np.all(np.abs(action.modifiers) <= 1.0)


class TestTextEncoding:
    """Text proposals encode via heuristic intent mapping."""

    def test_explain_intent_maps_to_openness(self) -> None:
        """Explain intent yields positive openness modifier."""
        dim_reg = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(),
        )
        proposal = TextActionProposal(
            name="explain",
            description="explain a concept",
            intent="explain",
        )
        action = encoder.encode(proposal)
        o_idx = dim_reg.index("O")
        assert action.modifiers[o_idx] > 0.3

    def test_output_always_bounded(self) -> None:
        """All modifiers stay within [-1, 1] regardless of intent."""
        dim_reg = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(),
        )
        proposal = TextActionProposal(
            name="rant",
            description="go on a rant",
            intent="express_frustration",
        )
        action = encoder.encode(proposal)
        assert np.all(action.modifiers >= -1.0)
        assert np.all(action.modifiers <= 1.0)


class TestBatchEncoding:
    """Batch encoding for multiple proposals."""

    def test_encode_batch(self) -> None:
        """Multiple proposals are encoded in order."""
        dim_reg = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(),
        )
        proposals = [
            ClassicActionProposal(name="a", description="a", modifiers={"O": 0.5}),
            ClassicActionProposal(name="b", description="b", modifiers={"C": 0.9}),
        ]
        actions = encoder.encode_batch(proposals)
        assert len(actions) == 2
        assert actions[0].name == "a"
        assert actions[1].name == "b"
