"""LLM integration boundary for Anthropic-backed agent tasks.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from src.llm.agent_runtime import AgentRuntime
from src.llm.anthropic_adapter import AnthropicAdapter
from src.llm.schemas import (
    AnalysisReport,
    InterventionRecommendation,
    LLMInvocationMetadata,
    LLMInvocationResult,
    NarrativeChunk,
    ScenarioProposal,
)

__all__ = [
    "AgentRuntime",
    "AnthropicAdapter",
    "AnalysisReport",
    "InterventionRecommendation",
    "LLMInvocationMetadata",
    "LLMInvocationResult",
    "NarrativeChunk",
    "ScenarioProposal",
]
