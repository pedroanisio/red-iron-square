"""Anthropic adapter boundary for typed structured outputs.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol, TypeVar, cast

from dotenv import load_dotenv
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
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1000,
    ) -> None:
        self._client = client or self._build_default_client()
        self._model = model
        self._max_tokens = max_tokens

    @staticmethod
    def _build_default_client() -> AnthropicClientProtocol:
        """Construct the default Anthropic client lazily."""
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTROPHIC_API_KEY")
        auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        if not api_key and not auth_token:
            raise LLMConfigurationError(
                "Anthropic credentials are not configured. "
                "Set `ANTHROPIC_API_KEY` (or legacy `ANTROPHIC_API_KEY`) "
                "or `ANTHROPIC_AUTH_TOKEN`."
            )
        from anthropic import Anthropic

        return cast(
            AnthropicClientProtocol,
            Anthropic(
                api_key=api_key,
                auth_token=auth_token,
            ),
        )

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
        parsed_payload = self._decode_json_payload(raw_text)
        parsed = response_model.model_validate(
            self._normalize_payload(parsed_payload, response_model)
        )
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

    @staticmethod
    def _decode_json_payload(raw_text: str) -> object:
        """Decode JSON from a raw model response, tolerating fenced code blocks."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(
            r"```(?:json)?\s*(?P<body>[\s\S]*?)\s*```",
            raw_text,
            flags=re.IGNORECASE,
        )
        if fenced_match is not None:
            return json.loads(fenced_match.group("body"))

        stripped = raw_text.strip()
        object_start = stripped.find("{")
        object_end = stripped.rfind("}")
        if object_start != -1 and object_end != -1 and object_end > object_start:
            return json.loads(stripped[object_start : object_end + 1])

        array_start = stripped.find("[")
        array_end = stripped.rfind("]")
        if array_start != -1 and array_end != -1 and array_end > array_start:
            return json.loads(stripped[array_start : array_end + 1])

        raise ValueError("Anthropic response did not contain valid JSON.")

    @staticmethod
    def _normalize_payload(
        payload: object,
        response_model: type[StructuredModelT],
    ) -> object:
        """Normalize common wrapper objects into the expected schema payload."""
        if not isinstance(payload, dict):
            return payload

        for key in (
            "scenario",
            "scenarios",
            "narrative",
            "narratives",
            "recommendation",
            "recommendations",
            "analysis",
            "analyses",
        ):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return nested
            if isinstance(nested, list) and nested and isinstance(nested[0], dict):
                return nested[0]

        if response_model.__name__ == "NarrativeChunk":
            summary = payload.get("summary") or payload.get("narrative")
            if isinstance(summary, str):
                tick_start = payload.get("tick_start")
                tick_end = payload.get("tick_end")
                return {
                    "summary": summary,
                    "tick_start": tick_start if isinstance(tick_start, int) else 0,
                    "tick_end": tick_end if isinstance(tick_end, int) else 0,
                    "evidence": (
                        payload.get("evidence")
                        if isinstance(payload.get("evidence"), list)
                        else []
                    ),
                }

        return payload


class LLMConfigurationError(RuntimeError):
    """Raise when the configured LLM provider credentials are unavailable."""
