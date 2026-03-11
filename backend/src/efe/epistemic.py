"""Epistemic value computation from episodic memory.

Epistemic value for an action is approximated as the outcome variance
from recent memory entries where that action was taken (section 2.2).
High variance = high uncertainty = high epistemic value = explore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.temporal.memory import MemoryBank


def compute_epistemic_value(
    action_name: str,
    memory: MemoryBank,
    window: int = 50,
    default: float = 1.0,
) -> float:
    """Outcome variance for a specific action from recent memory.

    Returns ``default`` when fewer than 2 entries exist for the action,
    encouraging exploration of unknown actions.
    """
    recent = memory.recent(window)
    outcomes = [m.outcome for m in recent if m.action_name == action_name]
    if len(outcomes) < 2:
        return default
    return float(np.var(outcomes))
