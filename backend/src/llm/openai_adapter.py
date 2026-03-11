"""OpenAI adapter boundary for typed structured outputs.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
import os
from typing import Protocol, TypeVar, cast

from dotenv import load_dotenv
from pydantic import BaseModel

from src.llm.anthropic_adapter import LLMConfigurationError
from src.llm.schemas import LLMInvocationMetadata, LLMInvocationResult

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class OpenAIParsedMessageProtocol(Protocol):
    """Subset of one parsed OpenAI message used by the adapter."""

    parsed: object | None
    refusal: str | None
    content: str | None


class OpenAIChoiceProtocol(Protocol):
    """Subset of one OpenAI choice wrapper used by the adapter."""

    message: OpenAIParsedMessageProtocol


class OpenAIUsageProtocol(Protocol):
    """Subset of OpenAI token usage metadata used by the adapter."""

    prompt_tokens: int | None
    completion_tokens: int | None


class OpenAIParsedResponseProtocol(Protocol):
    """Subset of one parsed OpenAI response used by the adapter."""

    choices: list[OpenAIChoiceProtocol]
    usage: OpenAIUsageProtocol | None


class OpenAICompletionsProtocol(Protocol):
    """Subset of the OpenAI beta completions API used by the adapter."""

    def parse(self, **kwargs: object) -> OpenAIParsedResponseProtocol:
        """Request one parsed structured response."""


class OpenAIChatProtocol(Protocol):
    """Subset of the OpenAI beta chat API used by the adapter."""

    completions: OpenAICompletionsProtocol


class OpenAIBetaProtocol(Protocol):
    """Subset of the OpenAI beta API used by the adapter."""

    chat: OpenAIChatProtocol


class OpenAIClientProtocol(Protocol):
    """Protocol for the OpenAI client object."""

    beta: OpenAIBetaProtocol


class OpenAIAdapter:
    """Typed wrapper over the OpenAI Python client structured parser."""

    def __init__(
        self,
        client: OpenAIClientProtocol | None = None,
        *,
        model: str = "gpt-4.1",
    ) -> None:
        self._client = client or self._build_default_client()
        self._model = model

    @staticmethod
    def _build_default_client() -> OpenAIClientProtocol:
        """Construct the default OpenAI client lazily."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "OpenAI credentials are not configured. Set `OPENAI_API_KEY`."
            )
        from openai import OpenAI

        return cast(OpenAIClientProtocol, OpenAI(api_key=api_key))

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModelT],
    ) -> tuple[StructuredModelT, LLMInvocationResult]:
        """Request a native structured response and return typed output."""
        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_model,
        )
        if not response.choices:
            raise ValueError("OpenAI response did not contain any choices.")
        message = response.choices[0].message
        if message.refusal:
            raise ValueError(f"OpenAI refusal: {message.refusal}")
        if message.parsed is None:
            raise ValueError("OpenAI response did not contain a parsed payload.")
        parsed = response_model.model_validate(message.parsed)
        raw_text = self._serialize_raw_message(message)
        metadata = LLMInvocationMetadata(
            model=self._model,
            provider="openai",
            stop_reason=None,
            input_tokens=getattr(
                getattr(response, "usage", None), "prompt_tokens", None
            ),
            output_tokens=getattr(
                getattr(response, "usage", None), "completion_tokens", None
            ),
        )
        return parsed, LLMInvocationResult(raw_text=raw_text, metadata=metadata)

    @staticmethod
    def _serialize_raw_message(message: OpenAIParsedMessageProtocol) -> str:
        """Serialize one parsed OpenAI message for audit storage."""
        if isinstance(message.content, str) and message.content:
            return message.content
        if message.parsed is not None:
            if isinstance(message.parsed, BaseModel):
                return message.parsed.model_dump_json()
            return json.dumps(message.parsed)
        return ""
