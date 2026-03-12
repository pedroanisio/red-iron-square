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
    EmotionConstructor,
    InterventionRecommendation,
    LLMInvocationResult,
    MatrixProposal,
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

    def construct_emotion(
        self,
        *,
        valence: float,
        arousal: float,
        prediction_errors: list[float],
        context: str,
    ) -> tuple[EmotionConstructor, LLMInvocationResult]:
        """Construct an emotion label from prediction error pattern (§4 Step 3b).

        Constrains the LLM to produce labels consistent with System 1
        valence/arousal signals.
        """
        valence_sign = (
            "positive" if valence > 0 else ("negative" if valence < 0 else "neutral")
        )
        arousal_level = "high" if arousal > 0.5 else "low"
        system_prompt = (
            "You categorize emotional states from interoceptive prediction errors. "
            "Return JSON only. Return exactly one object with keys "
            "`label`, `description`, `valence_sign`, `arousal_level`, `confidence`. "
            f"CONSTRAINT: valence_sign MUST be '{valence_sign}'. "
            f"CONSTRAINT: arousal_level MUST be '{arousal_level}'. "
            "Do not wrap the object in arrays or extra keys."
        )
        user_prompt = json.dumps(
            {
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "prediction_errors": [round(e, 4) for e in prediction_errors],
                "context": context,
                "output_schema": "EmotionConstructor",
            }
        )
        return self._adapter.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=EmotionConstructor,
        )

    def propose_matrices(
        self,
        *,
        personality: dict[str, float],
        trajectory_window: list[dict[str, Any]],
        n_states: int,
        n_actions: int,
    ) -> tuple[MatrixProposal, LLMInvocationResult]:
        """Propose A/B matrices for narrative generative model (§10).

        The LLM proposes observation likelihood and transition matrices
        grounded in the agent's personality and recent trajectory.
        """
        system_prompt = (
            "You propose generative model matrices for active inference. "
            "Return JSON only. Return exactly one object with keys "
            "`a_matrix`, `b_matrix`, `rationale`, `n_states`, `n_actions`. "
            "Matrices must be 3D arrays with shape (n_states, n_states, n_actions). "
            "Each row must sum to 1.0 (valid probability distributions). "
            "Do not wrap the object in arrays or extra keys."
        )
        user_prompt = json.dumps(
            {
                "personality": personality,
                "trajectory_window": trajectory_window,
                "n_states": n_states,
                "n_actions": n_actions,
                "output_schema": "MatrixProposal",
            }
        )
        return self._adapter.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=MatrixProposal,
        )
