"""Small runtime for typed Anthropic-backed agent tasks.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
from typing import Any

from src.llm.anthropic_adapter import AnthropicAdapter
from src.llm.schemas import (
    AnalysisReport,
    InterventionRecommendation,
    LLMInvocationResult,
    NarrativeChunk,
    ScenarioProposal,
)


class AgentRuntime:
    """Task-oriented wrapper around the Anthropic adapter."""

    def __init__(self, adapter: AnthropicAdapter) -> None:
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
            "Return JSON only. Respect [0,1] value bounds."
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
            "Return JSON only. Do not invent events outside the supplied ticks."
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
            "Return JSON only. Produce concise structured findings."
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
            "Return JSON only."
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
