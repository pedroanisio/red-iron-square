"""Episodic memory: storage and query for decision history."""

from collections import deque

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.temporal.state import AgentState


class MemoryEntry(BaseModel):
    """A single episodic memory: what happened at tick t.

    counterfactual: best unchosen utility minus chosen utility.
    valence: subjective emotional valence in [-1, 1].
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tick: int
    scenario_name: str
    action_name: str
    outcome: float
    counterfactual: float
    state_snapshot: AgentState
    valence: float


class MemoryBank:
    """Stores and queries episodic memories.

    Supports chronological retrieval, valence-weighted queries,
    and rolling statistics for emotion detection.
    """

    def __init__(self, max_size: int = 500) -> None:
        self._entries: deque[MemoryEntry] = deque(maxlen=max_size)

    def store(self, entry: MemoryEntry) -> None:
        """Append a memory entry."""
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[MemoryEntry]:
        """All entries in chronological order."""
        return list(self._entries)

    def recent(self, n: int) -> list[MemoryEntry]:
        """Last n memories, most recent first."""
        return list(reversed(list(self._entries)))[:n]

    def mean_outcome(self, window: int = 10) -> float:
        """Mean outcome over the last `window` entries."""
        recent = self.recent(window)
        if not recent:
            return 0.0
        return float(np.mean([m.outcome for m in recent]))

    def mean_valence(self, window: int = 10) -> float:
        """Mean emotional valence over the last `window` entries."""
        recent = self.recent(window)
        if not recent:
            return 0.0
        return float(np.mean([m.valence for m in recent]))

    def mean_arousal(self, window: int = 10) -> float:
        """Mean arousal over the last `window` entries."""
        recent = self.recent(window)
        if not recent:
            return 0.5
        return float(np.mean([m.state_snapshot.arousal for m in recent]))

    def peak_valence(self, window: int = 50) -> float:
        """Highest valence in the last `window` entries (for saudade baseline)."""
        recent = self.recent(window)
        if not recent:
            return 0.0
        return float(max(m.valence for m in recent))

    def total_regret(self, window: int = 10) -> float:
        """Sum of positive counterfactuals in recent window."""
        recent = self.recent(window)
        return float(sum(max(0, m.counterfactual) for m in recent))

    def consecutive_failures(self) -> int:
        """Count of consecutive negative outcomes from the most recent tick."""
        count = 0
        for m in reversed(list(self._entries)):
            if m.outcome < 0:
                count += 1
            else:
                break
        return count

    def outcome_variance(self, window: int = 10) -> float:
        """Variance of recent outcomes."""
        recent = self.recent(window)
        if len(recent) < 2:
            return 0.0
        return float(np.var([m.outcome for m in recent]))

    def action_outcome_variance(
        self, action_name: str, window: int = 50
    ) -> float | None:
        """Variance of recent outcomes for a specific action.

        Returns None when fewer than 2 entries exist for the action.
        """
        recent = self.recent(window)
        outcomes = [m.outcome for m in recent if m.action_name == action_name]
        if len(outcomes) < 2:
            return None
        return float(np.var(outcomes))
