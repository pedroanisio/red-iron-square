"""Run persistence models.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    """Return the current UTC timestamp as ISO 8601."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class PhaseRecord:
    """Persist one phase annotation."""

    start_tick: int
    label: str
    end_tick: int | None = None
    notes: str = ""
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class TickEventRecord:
    """Persist one realized tick event."""

    tick: int
    scenario: dict[str, Any]
    requested_outcome: float | None
    result: dict[str, Any]
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class AgentInvocationRecord:
    """Persist one LLM invocation tied to a run."""

    agent_name: str
    purpose: str
    input_json: dict[str, Any]
    output_json: dict[str, Any]
    raw_text: str
    metadata_json: dict[str, Any]
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class InterventionDecisionRecord:
    """Persist one intervention decision."""

    action: str
    reason: str
    payload_json: dict[str, Any]
    applied: bool
    created_at: str = field(default_factory=utc_now)
