"""LLM-backed helpers for the Two Minds demo.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.demo.models import DemoAgentSnapshot, DemoScenario
from src.demo.personas import DISPLAY_EMOTION_LABELS
from src.llm import AgentRuntime, LLMInvocationMetadata, LLMInvocationResult
from src.llm.agent_runtime import StructuredLLMAdapter


class DemoNarrative(BaseModel):
    """Structured spoken response for one demo agent."""

    text: str = Field(min_length=1)


class DemoEmotionLabel(BaseModel):
    """Context-specific display label candidate."""

    label: str = Field(min_length=1)
    valence: str = Field(pattern="^(positive|negative|neutral)$")


@dataclass(frozen=True, slots=True)
class DemoNarrativeResult:
    """Narrative text plus display label plus invocation metadata."""

    text: str
    emotion_label: str
    invocation: LLMInvocationResult | None = None


class DemoLLMService:
    """Encapsulate demo-specific LLM prompting with safe fallbacks."""

    def __init__(
        self,
        runtime: AgentRuntime | None = None,
        adapter: StructuredLLMAdapter | None = None,
    ) -> None:
        self._runtime = runtime
        self._adapter = adapter or self._resolve_adapter(runtime)

    def enrich_scenario(
        self,
        text: str,
        current_state: dict[str, Any],
        trajectory_window: list[dict[str, Any]],
    ) -> tuple[DemoScenario, LLMInvocationResult | None]:
        """Convert free text into a validated demo scenario."""
        if self._runtime is None:
            return self._fallback_scenario(text), None
        try:
            proposal, invocation = self._runtime.propose_scenario(
                current_state=current_state,
                trajectory_window=trajectory_window,
                goals=[text],
            )
            return (
                DemoScenario(
                    key="custom",
                    name=proposal.name or "Audience Scenario",
                    description=proposal.description or text,
                    values=self._normalize_values(proposal.values),
                ),
                invocation,
            )
        except Exception:
            return self._fallback_scenario(text), None

    def build_narrative(
        self,
        agent: DemoAgentSnapshot,
        scenario: DemoScenario,
        tick: dict[str, Any],
    ) -> DemoNarrativeResult:
        """Generate a first-person narrative and display label."""
        fallback_label = self._heuristic_emotion_label(tick)
        fallback_text = self._fallback_narrative(agent, scenario, tick)
        if self._adapter is None:
            return DemoNarrativeResult(text=fallback_text, emotion_label=fallback_label)
        state_after = tick.get("state_after", {})
        prompt_payload = {
            "agent_name": agent.name,
            "personality_summary": agent.summary,
            "scenario": scenario.description,
            "state": self._describe_state(state_after),
            "emotions": tick.get("self_emotions") or tick.get("emotions") or [],
            "recent_transcript": agent.transcript[-3:],
        }
        try:
            narrative, invocation = self._adapter.complete_json(
                system_prompt=(
                    "You write short first-person spoken dialogue. "
                    "Return JSON only with a `text` key. "
                    "Keep it to 2-4 spoken sentences. "
                    "Be natural and emotionally honest."
                ),
                user_prompt=json.dumps(prompt_payload),
                response_model=DemoNarrative,
            )
            emotion_label = self._resolve_display_label(
                tick=tick,
                narrative_text=narrative.text,
                fallback=fallback_label,
            )
            return DemoNarrativeResult(
                text=narrative.text,
                emotion_label=emotion_label,
                invocation=invocation,
            )
        except Exception:
            return DemoNarrativeResult(text=fallback_text, emotion_label=fallback_label)

    def _resolve_display_label(
        self,
        *,
        tick: dict[str, Any],
        narrative_text: str,
        fallback: str,
    ) -> str:
        if self._adapter is None:
            return fallback
        try:
            label, _ = self._adapter.complete_json(
                system_prompt=(
                    "You choose one family-friendly emotion label. "
                    "Return JSON only with `label` and `valence`. "
                    "Use plain language."
                ),
                user_prompt=json.dumps(
                    {
                        "narrative_text": narrative_text,
                        "emotions": tick.get("self_emotions")
                        or tick.get("emotions")
                        or [],
                        "mood": tick.get("state_after", {}).get("mood", 0.0),
                    }
                ),
                response_model=DemoEmotionLabel,
            )
        except Exception:
            return fallback
        expected = self._expected_valence(
            float(tick.get("state_after", {}).get("mood", 0.0))
        )
        if expected != "neutral" and label.valence != expected:
            return fallback
        return label.label

    @staticmethod
    def _resolve_adapter(runtime: AgentRuntime | None) -> StructuredLLMAdapter | None:
        if runtime is None:
            return None
        return getattr(runtime, "_adapter", None)

    @staticmethod
    def _fallback_scenario(text: str) -> DemoScenario:
        return DemoScenario(
            key="custom",
            name="Audience Scenario",
            description=text,
            values={key: 0.5 for key in ("O", "C", "E", "A", "N", "R", "I", "T")},
        )

    @staticmethod
    def _normalize_values(values: dict[str, float]) -> dict[str, float]:
        keys = ("O", "C", "E", "A", "N", "R", "I", "T")
        return {key: max(0.0, min(1.0, float(values.get(key, 0.5)))) for key in keys}

    @staticmethod
    def _describe_state(state_after: dict[str, Any]) -> str:
        mood = float(state_after.get("mood", 0.0))
        energy = float(state_after.get("energy", 0.5))
        if mood < -0.35 and energy < 0.4:
            return "You feel drained and uneasy."
        if mood > 0.35 and energy > 0.6:
            return "You feel energized and optimistic."
        return "You feel mixed and are still processing it."

    @staticmethod
    def _fallback_narrative(
        agent: DemoAgentSnapshot,
        scenario: DemoScenario,
        tick: dict[str, Any],
    ) -> str:
        mood = float(tick.get("state_after", {}).get("mood", 0.0))
        if mood < -0.2:
            return (
                f"I can't stop thinking about {scenario.name.lower()}. "
                f"It leaves me unsettled, and I'm still trying to steady myself."
            )
        if mood > 0.2:
            return (
                f"I keep coming back to {scenario.name.lower()}. "
                f"It feels like something I can step toward as {agent.name}."
            )
        return (
            f"I'm still sorting out what {scenario.name.lower()} means for me. "
            "Part of me sees the risk, and part of me wants to keep going."
        )

    @staticmethod
    def _heuristic_emotion_label(tick: dict[str, Any]) -> str:
        readings = tick.get("self_emotions") or tick.get("emotions") or []
        if not readings:
            return "Neutral"
        strongest = max(readings, key=lambda item: float(item.get("intensity", 0.0)))
        raw = str(strongest.get("label", "Neutral"))
        return DISPLAY_EMOTION_LABELS.get(raw, raw.replace("_", " ").title())

    @staticmethod
    def _expected_valence(mood: float) -> str:
        if mood > 0.15:
            return "positive"
        if mood < -0.15:
            return "negative"
        return "neutral"


def empty_invocation() -> LLMInvocationResult:
    """Return a deterministic invocation placeholder for local fallbacks."""
    return LLMInvocationResult(
        raw_text="{}",
        metadata=LLMInvocationMetadata(model="fallback", provider="local"),
    )
