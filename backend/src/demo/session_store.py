"""In-memory store for active Two Minds demo sessions.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.demo.models import DemoEvent, DemoSessionState


@dataclass(slots=True)
class DemoSessionRecord:
    """One active demo session and its transient delivery state."""

    session: DemoSessionState
    run_ids: dict[str, str]
    swapped: bool = False
    event_history: list[DemoEvent] = field(default_factory=list)
    subscribers: list[asyncio.Queue[DemoEvent]] = field(default_factory=list)


class DemoSessionStore:
    """Keep active sessions in memory for the live demo transport."""

    def __init__(self) -> None:
        self._sessions: dict[str, DemoSessionRecord] = {}

    def create(self, record: DemoSessionRecord) -> DemoSessionRecord:
        """Store one active session record."""
        self._sessions[record.session.session_id] = record
        return record

    def get(self, session_id: str) -> DemoSessionRecord:
        """Return one active session record."""
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(session_id) from exc

    def add_event(self, session_id: str, event: DemoEvent) -> None:
        """Persist one event and fan it out to subscribers."""
        record = self.get(session_id)
        record.event_history.append(event)
        for queue in record.subscribers:
            queue.put_nowait(event)

    def register(self, session_id: str) -> asyncio.Queue[DemoEvent]:
        """Register one subscriber queue for websocket delivery."""
        queue: asyncio.Queue[DemoEvent] = asyncio.Queue()
        self.get(session_id).subscribers.append(queue)
        return queue

    def unregister(self, session_id: str, queue: asyncio.Queue[DemoEvent]) -> None:
        """Remove one subscriber queue from the active session."""
        record = self.get(session_id)
        if queue in record.subscribers:
            record.subscribers.remove(queue)
