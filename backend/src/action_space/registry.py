"""Tool and capability registry for open-ended action spaces.

Each tool declares its parameter schema and an optional personality hint
mapping personality dimensions to affinity scores. The hint guides the
ActionEncoder when estimating modifier vectors for tool-based actions.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from src.shared.logging import get_logger

_log = get_logger(module="action_space.registry")


class ToolCapability(BaseModel):
    """A registered tool or capability the agent can invoke."""

    name: str = Field(min_length=1)
    description: str
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    personality_hint: dict[str, float] = Field(default_factory=dict)


class ToolRegistry:
    """Manages the set of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        """Register a tool. Raises ValueError on duplicate names."""
        if capability.name in self._tools:
            msg = f"Tool '{capability.name}' already registered"
            raise ValueError(msg)
        self._tools[capability.name] = capability
        _log.debug("tool_registered", tool=capability.name)

    def get(self, name: str) -> ToolCapability:
        """Look up a tool by name. Raises KeyError if missing."""
        return self._tools[name]

    def list_tools(self) -> list[ToolCapability]:
        """Return all registered tools."""
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    def to_prompt_context(self) -> str:
        """Serialize registry as a prompt-friendly string for LLM proposers."""
        entries = []
        for tool in self._tools.values():
            entries.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameter_schema,
                }
            )
        return json.dumps(entries, indent=2)
