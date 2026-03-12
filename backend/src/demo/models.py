"""Typed models for the Two Minds demo.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

TraitVector = dict[str, float]


@dataclass(frozen=True, slots=True)
class DemoPersona:
    """Audience-facing persona metadata and trait vector."""

    key: str
    name: str
    summary: str
    traits: TraitVector


@dataclass(frozen=True, slots=True)
class DemoScenario:
    """Structured scenario consumed by both demo agents."""

    key: str
    name: str
    description: str
    values: TraitVector
    forced_outcome: float | None = None


@dataclass(slots=True)
class DemoAgentSnapshot:
    """Minimal frontend snapshot for one agent."""

    key: str
    name: str
    summary: str
    traits: TraitVector
    mood: float = 0.0
    energy: float = 0.5
    calm: float = 0.5
    emotion_label: str = "Neutral"
    transcript: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DemoSessionState:
    """In-memory session state for the demo flow."""

    session_id: str
    act_number: int
    turn_count: int
    agents: dict[str, DemoAgentSnapshot]
    scripted_index: int = 0

    def reset_for_swap(
        self,
        new_agents: Mapping[str, DemoPersona | DemoAgentSnapshot],
    ) -> None:
        """Reset session state when the audience swaps personalities."""
        self.turn_count = 0
        self.scripted_index = 0
        self.agents = {
            key: _coerce_agent_snapshot(agent) for key, agent in new_agents.items()
        }


def _coerce_agent_snapshot(
    agent: DemoPersona | DemoAgentSnapshot,
) -> DemoAgentSnapshot:
    """Normalize persona data into a reset frontend-facing snapshot."""
    if isinstance(agent, DemoAgentSnapshot):
        return DemoAgentSnapshot(
            key=agent.key,
            name=agent.name,
            summary=agent.summary,
            traits=agent.traits,
        )
    return DemoAgentSnapshot(
        key=agent.key,
        name=agent.name,
        summary=agent.summary,
        traits=agent.traits,
    )


@dataclass(frozen=True, slots=True)
class DemoEvent:
    """A single event delivered to the frontend."""

    event_type: str
    session_id: str
    payload: dict[str, object]
