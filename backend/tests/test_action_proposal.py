"""Tests for action proposal types."""

import pytest
from src.action_space.proposal import (
    ActionProposal,
    ApiActionProposal,
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
)


class TestToolActionProposal:
    """ToolActionProposal validation."""

    def test_valid_tool_call(self) -> None:
        proposal = ToolActionProposal(
            name="search_web",
            description="Search for recent papers on active inference",
            tool_name="web_search",
            tool_args={"query": "active inference precision 2025"},
        )
        assert proposal.kind == "tool"
        assert proposal.tool_name == "web_search"

    def test_tool_name_required(self) -> None:
        with pytest.raises(ValueError):
            ToolActionProposal(
                name="search",
                description="search",
                tool_name="",
                tool_args={},
            )


class TestApiActionProposal:
    """ApiActionProposal validation."""

    def test_valid_api_request(self) -> None:
        proposal = ApiActionProposal(
            name="fetch_weather",
            description="Get current weather data",
            method="GET",
            url="https://api.weather.example/current",
            headers={"Authorization": "Bearer tok"},
        )
        assert proposal.kind == "api"
        assert proposal.method == "GET"

    def test_method_constrained(self) -> None:
        with pytest.raises(ValueError):
            ApiActionProposal(
                name="bad",
                description="bad",
                method="PATCH",
                url="https://example.com",
            )


class TestTextActionProposal:
    """TextActionProposal validation."""

    def test_valid_text_generation(self) -> None:
        proposal = TextActionProposal(
            name="explain_concept",
            description="Explain active inference to the user",
            intent="explain",
            prompt_hint="Describe precision weighting in plain language",
        )
        assert proposal.kind == "text"

    def test_intent_required(self) -> None:
        with pytest.raises(ValueError):
            TextActionProposal(
                name="bad",
                description="bad",
                intent="",
            )


class TestClassicActionProposal:
    """ClassicActionProposal wraps predefined modifiers."""

    def test_wraps_existing_modifiers(self) -> None:
        proposal = ClassicActionProposal(
            name="bold",
            description="Take a bold approach",
            modifiers={"O": 1.0, "R": 0.8, "N": -0.3},
        )
        assert proposal.kind == "classic"
        assert proposal.modifiers["O"] == 1.0

    def test_modifier_bounds_enforced(self) -> None:
        with pytest.raises(ValueError):
            ClassicActionProposal(
                name="bad",
                description="bad",
                modifiers={"O": 1.5},
            )


class TestActionProposalDiscriminator:
    """Union type resolves correctly."""

    def test_discriminator_routing(self) -> None:
        data = {
            "kind": "tool",
            "name": "search",
            "description": "search",
            "tool_name": "web_search",
            "tool_args": {"query": "test"},
        }
        proposal = ActionProposal.model_validate(data)
        assert isinstance(proposal.root, ToolActionProposal)
