"""Tests for the Anthropic integration boundary.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from src.llm import AgentRuntime, AnthropicAdapter


@dataclass
class _FakeUsage:
    input_tokens: int = 11
    output_tokens: int = 7


@dataclass
class _FakeBlock:
    text: str


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]
    stop_reason: str = "end_turn"
    usage: _FakeUsage = field(default_factory=_FakeUsage)


class _FakeMessages:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(content=[_FakeBlock(text=self._outputs.pop(0))])


class _FakeAnthropicClient:
    def __init__(self, outputs: list[str]) -> None:
        self.messages = _FakeMessages(outputs)


class TestAnthropicAdapter:
    def test_complete_json_validates_structured_output(self) -> None:
        adapter = AnthropicAdapter(
            _FakeAnthropicClient(
                [
                    '{"name":"probe_social",'
                    '"description":"Push E and A.",'
                    '"values":{"E":0.8,"A":0.6},'
                    '"rationale":"Check response to social demand."}'
                ]
            )
        )
        runtime = AgentRuntime(adapter)

        proposal, invocation = runtime.propose_scenario(
            current_state={"mood": 0.1},
            trajectory_window=[],
            goals=["probe extraversion"],
        )

        assert proposal.name == "probe_social"
        assert proposal.values["E"] == pytest.approx(0.8)
        assert invocation.metadata.model == "claude-3-7-sonnet-latest"
        assert invocation.metadata.input_tokens == 11

    def test_runtime_exposes_analysis_and_intervention_tasks(self) -> None:
        adapter = AnthropicAdapter(
            _FakeAnthropicClient(
                [
                    '{"dominant_regime":"recovery",'
                    '"notable_emotions":["focus"],'
                    '"anomalies":[],'
                    '"recommendations":'
                    '["continue observation"]}',
                    '{"action":"patch_params",'
                    '"reason":"Lower temperature to '
                    'reduce randomness.",'
                    '"temperature":0.7}',
                ]
            )
        )
        runtime = AgentRuntime(adapter)

        analysis, _ = runtime.analyze_window(ticks=[{"tick": 0, "outcome": 0.4}])
        intervention, _ = runtime.recommend_intervention(
            current_state={"tick": 1},
            ticks=[{"tick": 0, "outcome": 0.4}],
            goals=["stabilize behavior"],
        )

        assert analysis.dominant_regime == "recovery"
        assert intervention.action == "patch_params"
        assert intervention.temperature == pytest.approx(0.7)

    def test_invalid_json_raises(self) -> None:
        adapter = AnthropicAdapter(_FakeAnthropicClient(["not-json"]))
        runtime = AgentRuntime(adapter)

        with pytest.raises(ValueError):
            runtime.summarize_window(ticks=[{"tick": 0}])
