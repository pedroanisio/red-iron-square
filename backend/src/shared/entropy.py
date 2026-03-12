"""Shared Shannon entropy computation for action frequency analysis."""

from __future__ import annotations

import numpy as np


def compute_action_entropy(action_counts: dict[str, int]) -> float:
    """Compute Shannon entropy in nats from action frequency counts.

    Args:
        action_counts: Mapping of action names to their occurrence counts.

    Returns:
        Shannon entropy in nats. Returns 0.0 when total count is zero.
    """
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in action_counts.values()])
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))
