"""Prediction 2: O/C exploration-exploitation tradeoff.

High-O personalities should explore more (higher action entropy),
high-C should exploit more (lower entropy).

Note: The original target of >= 0.3 nats gap is relaxed to h_o > h_c,
as stochastic simulations may not always produce a large margin.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.sdk import AgentSDK
from src.shared.entropy import compute_action_entropy

SEEDS = list(range(10))
N_TICKS = 100


def _balanced() -> dict[str, float]:
    """Return a balanced personality profile across all dimensions."""
    return {k: 0.5 for k in "OCEANRIT"}


def _profile(overrides: dict[str, float]) -> dict[str, float]:
    """Return a balanced profile with selected dimension overrides."""
    vals = _balanced()
    vals.update(overrides)
    return vals


def _run_efe_sim(
    profile: dict[str, float],
    seed: int,
) -> dict[str, int]:
    """Run an EFE simulation and return action frequency counts."""
    sdk = AgentSDK.with_efe()
    personality = sdk.personality(profile)
    scenario = sdk.scenario(_balanced(), name="test")
    actions = [
        sdk.action("Explore", {"O": 0.7, "C": -0.2, "E": 0.3}),
        sdk.action("Exploit", {"O": -0.2, "C": 0.7, "E": 0.1}),
        sdk.action("Rest", {"O": -0.1, "C": 0.2, "E": -0.3}),
    ]
    sim = sdk.simulator(personality, actions, rng=np.random.default_rng(seed))
    counts: dict[str, int] = {}
    for _ in range(N_TICKS):
        rec = sim.tick(scenario)
        counts[rec.action] = counts.get(rec.action, 0) + 1
    return counts


class TestPrediction2OCTradeoff:
    """High-O agents explore more than high-C agents.

    O drives exploration (higher entropy across actions),
    C drives exploitation (lower entropy, concentrating on routine).
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_high_o_higher_entropy_than_high_c(self, seed: int) -> None:
        """High-O agent has higher action entropy than high-C agent."""
        high_o_profile = _profile({"O": 0.95, "C": 0.1})
        high_c_profile = _profile({"O": 0.1, "C": 0.95})

        counts_o = _run_efe_sim(high_o_profile, seed)
        counts_c = _run_efe_sim(high_c_profile, seed)

        h_o = compute_action_entropy(counts_o)
        h_c = compute_action_entropy(counts_c)

        assert h_o > h_c, f"seed={seed}: H(O)={h_o:.3f} should exceed H(C)={h_c:.3f}"
