"""LLM runtime factory and provider selection.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.llm.agent_runtime import AgentRuntime
from src.llm.anthropic_adapter import AnthropicAdapter, LLMConfigurationError
from src.llm.openai_adapter import OpenAIAdapter


def build_default_runtime(provider: str | None = None) -> AgentRuntime:
    """Build the configured default LLM runtime."""
    selected_provider = _resolve_provider(provider)
    if selected_provider == "anthropic":
        return AgentRuntime(AnthropicAdapter())
    if selected_provider == "openai":
        return AgentRuntime(OpenAIAdapter())
    raise LLMConfigurationError(
        "Unsupported LLM provider. Set `RED_IRON_SQUARE_LLM_PROVIDER` to "
        "`anthropic` or `openai`."
    )


def _resolve_provider(provider: str | None) -> str:
    """Resolve the configured provider name from explicit or env input."""
    load_dotenv()
    return (
        provider or os.getenv("RED_IRON_SQUARE_LLM_PROVIDER") or "anthropic"
    ).lower()
