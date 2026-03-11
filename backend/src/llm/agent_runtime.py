"""Small runtime for typed structured-output agent tasks.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from src.llm.schemas import (
    AnalysisReport,
    InterventionRecommendation,
    LLMInvocationResult,
    NarrativeChunk,
    ScenarioProposal,
)

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class StructuredLLMAdapter(Protocol):
    """Common typed LLM adapter surface used by the runtime."""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModelT],
    ) -> tuple[StructuredModelT, LLMInvocationResult]:
        """Return a validated structured response and invocation metadata."""


class AgentRuntime:
    """Task-oriented wrapper around a typed LLM adapter."""

    def __init__(self, adapter: StructuredLLMAdapter) -> None:
        self._adapter = adapter

    def propose_scenario(
        self,
        *,
        current_state: dict[str, Any],
        trajectory_window: list[dict[str, Any]],
        goals: list[str],
    ) -> tuple[ScenarioProposal, LLMInvocationResult]:
        """Produce a validated scenario proposal."""
        system_prompt = (
            "You propose simulation scenarios. "
            "Return JSON only. Return exactly one object with keys "
            "`name`, `description`, `values`, and `rationale`. "
            "Do not wrap the object in arrays or extra keys. "
            "Respect [0,1] value bounds."
        )
        user_prompt = json.dumps(
            {
                "current_state": current_state,
                "trajectory_window": trajectory_window,
                "goals": goals,
                "output_schema": "ScenarioProposal",
            }
        )
        return self._adapter.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ScenarioProposal,
        )

    def summarize_window(
        self,
        *,
        ticks: list[dict[str, Any]],
    ) -> tuple[NarrativeChunk, LLMInvocationResult]:
        """Produce a narrative summary grounded in recent ticks."""
        system_prompt = (
            "You narrate simulation dynamics. "
            "Return JSON only. Return exactly one object with keys "
            "`summary`, `tick_start`, `tick_end`, and `evidence`. "
            "Do not wrap the object in arrays or extra keys. "
            "Do not invent events outside the supplied ticks."
        )
        user_prompt = json.dumps({"ticks": ticks, "output_schema": "NarrativeChunk"})
        return self._adapter.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=NarrativeChunk,
        )

    def analyze_window(
        self,
        *,
        ticks: list[dict[str, Any]],
    ) -> tuple[AnalysisReport, LLMInvocationResult]:
        """Produce a structured analysis report."""
        system_prompt = (
            "You analyze simulation dynamics. "
            "Return JSON only. Return exactly one object with keys "
            "`dominant_regime`, `notable_emotions`, `anomalies`, "
            "and `recommendations`. "
            "Do not wrap the object in arrays or extra keys. "
            "Produce concise structured findings."
        )
        user_prompt = json.dumps({"ticks": ticks, "output_schema": "AnalysisReport"})
        return self._adapter.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=AnalysisReport,
        )

    def recommend_intervention(
        self,
        *,
        current_state: dict[str, Any],
        ticks: list[dict[str, Any]],
        goals: list[str],
    ) -> tuple[InterventionRecommendation, LLMInvocationResult]:
        """Produce a validated intervention recommendation."""
        system_prompt = (
            "You recommend one allowed intervention for the simulation. "
            "Return JSON only. Return exactly one object with keys "
            "`action`, `reason`, and optional `temperature`. "
            "Do not wrap the object in arrays or extra keys."
        )
        user_prompt = json.dumps(
            {
                "current_state": current_state,
                "ticks": ticks,
                "goals": goals,
                "allowed_actions": [
                    "continue",
                    "probe",
                    "narrate",
                    "analyze",
                    "patch_params",
                    "pause",
                    "terminate",
                ],
                "output_schema": "InterventionRecommendation",
            }
        )
        return self._adapter.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=InterventionRecommendation,
        )
