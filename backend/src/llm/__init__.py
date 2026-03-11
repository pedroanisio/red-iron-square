"""LLM integration boundary for typed agent tasks.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from src.llm.agent_runtime import AgentRuntime
from src.llm.anthropic_adapter import AnthropicAdapter, LLMConfigurationError
from src.llm.factory import build_default_runtime
from src.llm.openai_adapter import OpenAIAdapter
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
    "OpenAIAdapter",
    "build_default_runtime",
    "LLMConfigurationError",
    "AnalysisReport",
    "InterventionRecommendation",
    "LLMInvocationMetadata",
    "LLMInvocationResult",
    "NarrativeChunk",
    "ScenarioProposal",
]
