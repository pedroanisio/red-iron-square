"""Tests for context-aware action proposal."""

from src.action_space.proposal import ClassicActionProposal
from src.action_space.proposer import ActionProposer, StaticProposerBackend
from src.action_space.registry import ToolCapability, ToolRegistry


class TestStaticProposerBackend:
    """Static backend returns predefined actions (fallback path)."""

    def test_returns_classic_proposals(self) -> None:
        classics = [
            ClassicActionProposal(
                name="bold", description="bold", modifiers={"O": 1.0}
            ),
            ClassicActionProposal(
                name="safe", description="safe", modifiers={"C": 0.9}
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposals = backend.propose(state={}, trajectory=[], goals=[])
        assert len(proposals) == 2
        assert proposals[0].name == "bold"


class TestActionProposer:
    """ActionProposer merges tool-based and backend proposals."""

    def test_includes_tool_proposals_from_registry(self) -> None:
        tool_reg = ToolRegistry()
        tool_reg.register(
            ToolCapability(
                name="web_search",
                description="search the web",
                parameter_schema={"query": {"type": "string"}},
            )
        )
        classics = [
            ClassicActionProposal(
                name="safe", description="safe", modifiers={"C": 0.9}
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposer = ActionProposer(
            backend=backend,
            tool_registry=tool_reg,
        )
        proposals = proposer.propose(
            state={"energy": 0.8},
            trajectory=[],
            goals=["find information"],
        )
        kinds = [getattr(p, "kind", None) for p in proposals]
        assert "tool" in kinds
        assert "classic" in kinds

    def test_no_tools_returns_backend_only(self) -> None:
        classics = [
            ClassicActionProposal(
                name="bold", description="bold", modifiers={"O": 1.0}
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposer = ActionProposer(backend=backend)
        proposals = proposer.propose(state={}, trajectory=[], goals=[])
        assert len(proposals) == 1
        assert proposals[0].name == "bold"

    def test_max_proposals_limit(self) -> None:
        classics = [
            ClassicActionProposal(name=f"a{i}", description="", modifiers={"O": 0.1})
            for i in range(20)
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposer = ActionProposer(backend=backend, max_proposals=5)
        proposals = proposer.propose(state={}, trajectory=[], goals=[])
        assert len(proposals) == 5

    def test_always_includes_withdraw(self) -> None:
        classics = [
            ClassicActionProposal(
                name="bold", description="bold", modifiers={"O": 1.0}
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposer = ActionProposer(backend=backend, include_withdraw=True)
        proposals = proposer.propose(state={}, trajectory=[], goals=[])
        names = [p.name for p in proposals]
        assert "Withdraw" in names
