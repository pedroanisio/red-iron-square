"""Tests for the tool/capability registry."""

import pytest
from src.action_space.registry import ToolCapability, ToolRegistry


class TestToolRegistry:
    """ToolRegistry manages available tools."""

    def test_register_and_lookup(self) -> None:
        registry = ToolRegistry()
        cap = ToolCapability(
            name="web_search",
            description="Search the web for information",
            parameter_schema={"query": {"type": "string"}},
            personality_hint={"O": 0.7, "E": 0.3},
        )
        registry.register(cap)
        assert registry.get("web_search") is cap

    def test_lookup_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("missing")

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        cap1 = ToolCapability(
            name="search",
            description="search",
            parameter_schema={},
        )
        cap2 = ToolCapability(
            name="calculate",
            description="calculate",
            parameter_schema={},
        )
        registry.register(cap1)
        registry.register(cap2)
        names = [t.name for t in registry.list_tools()]
        assert "search" in names
        assert "calculate" in names

    def test_duplicate_name_raises(self) -> None:
        registry = ToolRegistry()
        cap = ToolCapability(name="a", description="a", parameter_schema={})
        registry.register(cap)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(cap)

    def test_personality_hint_default_empty(self) -> None:
        cap = ToolCapability(name="t", description="t", parameter_schema={})
        assert cap.personality_hint == {}

    def test_to_prompt_context(self) -> None:
        registry = ToolRegistry()
        cap = ToolCapability(
            name="search",
            description="Search for things",
            parameter_schema={"query": {"type": "string"}},
        )
        registry.register(cap)
        ctx = registry.to_prompt_context()
        assert "search" in ctx
        assert "query" in ctx
