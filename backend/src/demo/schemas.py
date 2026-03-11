"""Pydantic schemas for the Two Minds demo API.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DemoSessionCreateRequest(BaseModel):
    """Create a fresh demo session."""

    act_number: int = Field(default=1, ge=1, le=3)


class DemoCustomScenarioRequest(BaseModel):
    """Submit a custom audience scenario."""

    text: str = Field(min_length=1)


class DemoScriptedScenarioResponse(BaseModel):
    """Return metadata for a scripted scenario turn."""

    session_id: str
    scenario_key: str
    turn_count: int


class DemoAgentResponse(BaseModel):
    """Frontend-safe snapshot of one agent."""

    key: str
    name: str
    summary: str
    mood: float
    energy: float
    calm: float
    emotion_label: str


class DemoSessionResponse(BaseModel):
    """Frontend-safe session payload."""

    session_id: str
    act_number: int
    turn_count: int
    agents: list[DemoAgentResponse]


class DemoSwapResponse(BaseModel):
    """Response payload for a swap/reset action."""

    session_id: str
    act_number: int
    swapped: bool = True
