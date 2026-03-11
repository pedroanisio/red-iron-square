"""Anthropic adapter boundary for typed structured outputs.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
from typing import Protocol, TypeVar

from pydantic import BaseModel

from src.llm.schemas import LLMInvocationMetadata, LLMInvocationResult

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class AnthropicMessagesProtocol(Protocol):
    """Subset of the Anthropic messages API used by the adapter."""

    def create(self, **kwargs: object) -> object:
        """Create one model response."""


class AnthropicClientProtocol(Protocol):
    """Protocol for the Anthropic client object."""

    messages: AnthropicMessagesProtocol


class AnthropicAdapter:
    """Typed wrapper over the Anthropic Python client."""

    def __init__(
        self,
        client: AnthropicClientProtocol | None = None,
        *,
        model: str = "claude-3-7-sonnet-latest",
        max_tokens: int = 1000,
    ) -> None:
        self._client = client or self._build_default_client()
        self._model = model
        self._max_tokens = max_tokens

    @staticmethod
    def _build_default_client() -> AnthropicClientProtocol:
        """Construct the default Anthropic client lazily."""
        from anthropic import Anthropic

        return Anthropic()  # type: ignore[no-any-return]

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModelT],
    ) -> tuple[StructuredModelT, LLMInvocationResult]:
        """Request JSON output and validate it against a Pydantic model."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = self._extract_text(response)
        parsed = response_model.model_validate(json.loads(raw_text))
        metadata = LLMInvocationMetadata(
            model=self._model,
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=getattr(
                getattr(response, "usage", None), "input_tokens", None
            ),
            output_tokens=getattr(
                getattr(response, "usage", None), "output_tokens", None
            ),
        )
        return parsed, LLMInvocationResult(raw_text=raw_text, metadata=metadata)

    @staticmethod
    def _extract_text(response: object) -> str:
        """Extract text from Anthropic-style content blocks."""
        content = getattr(response, "content", None)
        if not isinstance(content, list):
            raise ValueError("Anthropic response does not contain a content list.")
        text_blocks: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_blocks.append(text)
        if not text_blocks:
            raise ValueError("Anthropic response does not contain any text blocks.")
        return "".join(text_blocks)
