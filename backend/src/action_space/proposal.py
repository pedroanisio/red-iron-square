"""Action proposal types for open-ended action spaces.

Each proposal kind represents a different action modality:
- tool: invoke a registered tool/capability
- api: make an HTTP request to an external service
- text: generate free-text output
- classic: wrap predefined modifier vectors (backward compat)
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, RootModel, field_validator


class _ProposalBase(BaseModel):
    """Shared fields for all action proposals."""

    name: str = Field(min_length=1)
    description: str = ""
    rationale: str = ""


class ToolActionProposal(_ProposalBase):
    """Invoke a registered tool or capability."""

    kind: Literal["tool"] = "tool"
    tool_name: str = Field(min_length=1)
    tool_args: dict[str, object] = Field(default_factory=dict)
    timeout_ms: int = Field(default=30_000, gt=0)


class ApiActionProposal(_ProposalBase):
    """Make an HTTP request to an external service."""

    kind: Literal["api"] = "api"
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, object] | None = None


class TextActionProposal(_ProposalBase):
    """Generate free-text output."""

    kind: Literal["text"] = "text"
    intent: str = Field(min_length=1)
    prompt_hint: str = ""
    max_tokens: int = Field(default=1024, gt=0)


class ClassicActionProposal(_ProposalBase):
    """Wrap predefined personality-dimension modifiers (backward compat)."""

    kind: Literal["classic"] = "classic"
    modifiers: dict[str, float]

    @field_validator("modifiers")
    @classmethod
    def _check_bounds(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure all modifier values are within [-1, 1]."""
        for key, val in v.items():
            if not -1.0 <= val <= 1.0:
                msg = f"{key}={val} outside [-1, 1]"
                raise ValueError(msg)
        return v


_ProposalUnion = Annotated[
    ToolActionProposal | ApiActionProposal | TextActionProposal | ClassicActionProposal,
    Field(discriminator="kind"),
]


class ActionProposal(RootModel[_ProposalUnion]):
    """Discriminated union over action proposal types."""

    @property
    def kind(self) -> str:
        """Delegate to inner model."""
        return self.root.kind

    @property
    def name(self) -> str:
        """Delegate to inner model."""
        return self.root.name

    @property
    def description(self) -> str:
        """Delegate to inner model."""
        return self.root.description
