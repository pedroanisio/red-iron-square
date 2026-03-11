"""Tests for demo-specific LLM behavior.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.demo.llm_service import DemoEmotionLabel, DemoLLMService, DemoNarrative
from src.demo.models import DemoAgentSnapshot
from src.demo.personas import DEFAULT_PERSONAS
from src.llm import AgentRuntime, LLMInvocationMetadata, LLMInvocationResult
from src.llm.schemas import ScenarioProposal


@dataclass
class FakeAdapter:
    outputs: list[object]

    def complete_json(self, **kwargs: object) -> tuple[object, LLMInvocationResult]:
        payload = self.outputs.pop(0)
        response_model = kwargs["response_model"]
        parsed = response_model.model_validate(payload)
        return (
            parsed,
            LLMInvocationResult(
                raw_text="{}",
                metadata=LLMInvocationMetadata(model="fake-model", provider="test"),
            ),
        )


class FakeRuntime:
    def __init__(self, proposal: ScenarioProposal | Exception) -> None:
        self._proposal = proposal

    def propose_scenario(
        self,
        *,
        current_state: dict[str, object],
        trajectory_window: list[dict[str, object]],
        goals: list[str],
    ) -> tuple[ScenarioProposal, LLMInvocationResult]:
        if isinstance(self._proposal, Exception):
            raise self._proposal
        return (
            self._proposal,
            LLMInvocationResult(
                raw_text="{}",
                metadata=LLMInvocationMetadata(model="fake-model", provider="test"),
            ),
        )


def test_enrich_scenario_uses_runtime_output() -> None:
    service = DemoLLMService(
        runtime=FakeRuntime(
            ScenarioProposal(
                name="workplace_conflict",
                description="You get into a heated argument with a coworker.",
                values={"E": 0.7, "A": 0.8},
                rationale="High social demand and strain.",
            )
        ),
    )

    scenario, invocation = service.enrich_scenario("Argument at work", {}, [])

    assert scenario.name == "workplace_conflict"
    assert scenario.values["E"] == 0.7
    assert scenario.values["O"] == 0.5
    assert invocation is not None


def test_enrich_scenario_falls_back_to_neutral_payload_on_invalid_runtime() -> None:
    service = DemoLLMService(runtime=FakeRuntime(ValueError("bad output")))

    scenario, invocation = service.enrich_scenario("Luna's cat gets sick", {}, [])

    assert scenario.name == "Audience Scenario"
    assert all(value == 0.5 for value in scenario.values.values())
    assert invocation is None


def test_build_narrative_uses_adapter_and_respects_label_valence() -> None:
    service = DemoLLMService(
        runtime=AgentRuntime(FakeAdapter([])),
        adapter=FakeAdapter(
            [
                DemoNarrative(
                    text="I can feel how much this changes the ground under me."
                ),
                DemoEmotionLabel(label="Ambivalence", valence="negative"),
            ]
        ),
    )
    agent = DemoAgentSnapshot(
        key="luna",
        name="Luna",
        summary=DEFAULT_PERSONAS["luna"].summary,
        traits=DEFAULT_PERSONAS["luna"].traits,
    )

    result = service.build_narrative(
        agent,
        scenario=service._fallback_scenario("Promotion"),
        tick={
            "state_after": {"mood": -0.3, "energy": 0.4},
            "emotions": [{"label": "ANXIETY", "intensity": 0.8}],
        },
    )

    assert "changes the ground" in result.text
    assert result.emotion_label == "Ambivalence"


def test_build_narrative_falls_back_when_label_valence_conflicts() -> None:
    service = DemoLLMService(
        runtime=AgentRuntime(FakeAdapter([])),
        adapter=FakeAdapter(
            [
                DemoNarrative(text="I think I can make something of this."),
                DemoEmotionLabel(label="Excitement", valence="positive"),
            ]
        ),
    )
    agent = DemoAgentSnapshot(
        key="luna",
        name="Luna",
        summary=DEFAULT_PERSONAS["luna"].summary,
        traits=DEFAULT_PERSONAS["luna"].traits,
    )

    result = service.build_narrative(
        agent,
        scenario=service._fallback_scenario("Promotion"),
        tick={
            "state_after": {"mood": -0.4, "energy": 0.4},
            "emotions": [{"label": "ANXIETY", "intensity": 0.8}],
        },
    )

    assert result.emotion_label == "Anxiety"
    assert result.text == "I think I can make something of this."
