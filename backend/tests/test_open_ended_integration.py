"""Integration test: full open-ended action pipeline."""

from __future__ import annotations

from src.action_space.executor import ActionExecutor
from src.action_space.proposal import ClassicActionProposal
from src.action_space.proposer import StaticProposerBackend
from src.action_space.registry import ToolCapability, ToolRegistry
from src.sdk import AgentSDK


class TestFullPipeline:
    """End-to-end: propose, encode, decide, execute."""

    def test_tool_plus_classic_pipeline(self) -> None:
        """Run full pipeline with tool and classic proposals."""
        tool_reg = ToolRegistry()
        tool_reg.register(
            ToolCapability(
                name="web_search",
                description="search the web",
                parameter_schema={"query": {"type": "string"}},
                personality_hint={"O": 0.8, "E": 0.3},
            )
        )

        classics = [
            ClassicActionProposal(
                name="safe",
                description="safe choice",
                modifiers={"C": 0.9, "T": 0.8},
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)

        sdk = AgentSDK.with_open_actions(
            proposer_backend=backend,
            tool_registry=tool_reg,
            include_withdraw=True,
        )

        personality = sdk.personality(
            {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7, "N": 0.4}
        )
        scenario = sdk.scenario({"O": 0.9, "N": 0.7}, name="research_meeting")

        result = sdk.propose_and_decide(personality, scenario)
        assert result.chosen_action in ("use_web_search", "safe", "Withdraw")
        assert len(result.proposals) == 3

    def test_executor_dispatches_classic(self) -> None:
        """Executor handles classic proposals as no-op."""
        executor = ActionExecutor()
        proposal = ClassicActionProposal(
            name="bold",
            description="bold",
            modifiers={"O": 1.0},
        )
        result = executor.execute(proposal)
        assert result.success is True
        assert result.outcome_signal is None

    def test_backward_compat_full_simulation(self) -> None:
        """Existing simulation path works unchanged."""
        sdk = AgentSDK.default()
        personality = sdk.personality(
            {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7, "N": 0.4}
        )
        actions = [
            sdk.action("bold", {"O": 1.0}),
            sdk.action("safe", {"C": 0.9}),
        ]
        scenarios = [
            sdk.scenario({"O": 0.9}, name="s1"),
            sdk.scenario({"C": 0.8}, name="s2"),
        ]
        trace = sdk.simulator(personality, actions).run(
            scenarios,
            outcomes=[0.4, 0.2],
        )
        assert len(trace.ticks) == 2

    def test_open_ended_simulation(self) -> None:
        """Open-ended actions work through the simulator."""
        classics = [
            ClassicActionProposal(
                name="engage",
                description="engage",
                modifiers={"E": 0.5},
            ),
            ClassicActionProposal(
                name="retreat",
                description="retreat",
                modifiers={"E": -0.5},
            ),
        ]
        sdk = AgentSDK.with_open_actions(
            proposer_backend=StaticProposerBackend(defaults=classics),
        )
        personality = sdk.personality({"O": 0.5, "E": 0.5})
        scenarios = [
            sdk.scenario({"O": 0.3}, name="s1"),
            sdk.scenario({"E": 0.7}, name="s2"),
        ]
        # Pass empty actions — pipeline should propose
        trace = sdk.simulator(personality, []).run(
            scenarios,
            outcomes=[0.3, 0.6],
        )
        assert len(trace.ticks) == 2
        for tick in trace.ticks:
            assert tick.action in ("engage", "retreat")
