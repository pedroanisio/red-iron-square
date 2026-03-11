"""Tests for the LLM integration boundary.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from src.llm import AgentRuntime, AnthropicAdapter, OpenAIAdapter, build_default_runtime
from src.llm.anthropic_adapter import LLMConfigurationError


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


@dataclass
class _FakeOpenAIUsage:
    prompt_tokens: int = 13
    completion_tokens: int = 9


@dataclass
class _FakeOpenAIMessage:
    parsed: object | None = None
    refusal: str | None = None
    content: str | None = None


@dataclass
class _FakeOpenAIChoice:
    message: _FakeOpenAIMessage


@dataclass
class _FakeOpenAIResponse:
    choices: list[_FakeOpenAIChoice]
    usage: _FakeOpenAIUsage = field(default_factory=_FakeOpenAIUsage)


class _FakeOpenAICompletions:
    def __init__(self, outputs: list[object]) -> None:
        self._outputs = outputs
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> _FakeOpenAIResponse:
        self.calls.append(kwargs)
        parsed = self._outputs.pop(0)
        content = parsed if isinstance(parsed, str) else None
        message = _FakeOpenAIMessage(parsed=parsed, content=content)
        return _FakeOpenAIResponse(choices=[_FakeOpenAIChoice(message=message)])


class _FakeOpenAIChat:
    def __init__(self, outputs: list[object]) -> None:
        self.completions = _FakeOpenAICompletions(outputs)


class _FakeOpenAIBeta:
    def __init__(self, outputs: list[object]) -> None:
        self.chat = _FakeOpenAIChat(outputs)


class _FakeOpenAIClient:
    def __init__(self, outputs: list[object]) -> None:
        self.beta = _FakeOpenAIBeta(outputs)


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
        assert invocation.metadata.model == "claude-sonnet-4-20250514"
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

    def test_complete_json_accepts_fenced_single_item_wrappers(self) -> None:
        adapter = AnthropicAdapter(
            _FakeAnthropicClient(
                [
                    "```json\n"
                    "{\n"
                    '  "scenarios": [\n'
                    "    {\n"
                    '      "name": "probe_social",\n'
                    '      "description": "Push E and A.",\n'
                    '      "values": {"E": 0.8, "A": 0.6},\n'
                    '      "rationale": "Check response to social demand."\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "```"
                ]
            )
        )
        runtime = AgentRuntime(adapter)

        proposal, _ = runtime.propose_scenario(
            current_state={"mood": 0.1},
            trajectory_window=[],
            goals=["probe extraversion"],
        )

        assert proposal.name == "probe_social"
        assert proposal.values["A"] == pytest.approx(0.6)

    def test_default_client_requires_anthropic_credentials(self) -> None:
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""},
            clear=False,
        ):
            with pytest.raises(LLMConfigurationError):
                AnthropicAdapter()


class TestOpenAIAdapter:
    def test_complete_json_uses_native_structured_parsing(self) -> None:
        adapter = OpenAIAdapter(
            _FakeOpenAIClient(
                [
                    {
                        "name": "probe_social",
                        "description": "Push E and A.",
                        "values": {"E": 0.8, "A": 0.6},
                        "rationale": "Check response to social demand.",
                    }
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
        assert proposal.values["A"] == pytest.approx(0.6)
        assert invocation.metadata.model == "gpt-4.1"
        assert invocation.metadata.provider == "openai"
        assert invocation.metadata.input_tokens == 13

    def test_refusal_raises(self) -> None:
        adapter = OpenAIAdapter(_FakeOpenAIClient([]))
        adapter._client.beta.chat.completions.parse = lambda **kwargs: (
            _FakeOpenAIResponse(  # type: ignore[method-assign]
                choices=[
                    _FakeOpenAIChoice(
                        message=_FakeOpenAIMessage(
                            parsed=None, refusal="safety refusal"
                        )
                    )
                ]
            )
        )

        runtime = AgentRuntime(adapter)

        with pytest.raises(ValueError, match="OpenAI refusal"):
            runtime.summarize_window(ticks=[{"tick": 0}])

    def test_default_client_requires_openai_credentials(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(LLMConfigurationError):
                OpenAIAdapter()


def test_factory_selects_openai_provider_from_env() -> None:
    with patch.dict(
        "os.environ",
        {"RED_IRON_SQUARE_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"},
        clear=False,
    ):
        with patch(
            "src.llm.openai_adapter.OpenAIAdapter._build_default_client"
        ) as build:
            build.return_value = _FakeOpenAIClient(
                [
                    {
                        "name": "probe_social",
                        "description": "Push E and A.",
                        "values": {"E": 0.8, "A": 0.6},
                        "rationale": "Check response to social demand.",
                    }
                ]
            )
            runtime = build_default_runtime()

    proposal, _ = runtime.propose_scenario(
        current_state={"mood": 0.1},
        trajectory_window=[],
        goals=["probe extraversion"],
    )
    assert proposal.name == "probe_social"
