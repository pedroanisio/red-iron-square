"""Prediction 5: narrative coherence recovery shape after disruption.

After a sudden scenario change (disruption), the coherence gap
(self_coherence = ||psi_hat - B|| / sqrt(N)) should spike and then
recover (decrease) when the normal scenario is restored.

Note: self_coherence is a *gap* metric -- lower is better.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.sdk import AgentSDK

SEEDS = list(range(5))
BASELINE_TICKS = 30
DISRUPTION_TICKS = 10
RECOVERY_TICKS = 60


def _balanced() -> dict[str, float]:
    """Return a balanced personality profile across all dimensions."""
    return {k: 0.5 for k in "OCEANRIT"}


def _disruption_scenario() -> dict[str, float]:
    """Return an extreme scenario that disrupts coherence."""
    return {k: (0.9 if k in "ON" else 0.1) for k in "OCEANRIT"}


def _build_self_aware_sim(
    sdk: AgentSDK,
    seed: int,
) -> object:
    """Build a self-aware simulator with standard config."""
    profile = _balanced()
    personality = sdk.personality(profile)
    psi_hat = sdk.initial_self_model(profile)
    actions = [
        sdk.action("Act", {"O": 0.3, "E": 0.2}),
        sdk.action("Wait", {"C": 0.3, "E": -0.1}),
    ]
    return sdk.self_aware_simulator(
        personality,
        psi_hat,
        actions,
        rng=np.random.default_rng(seed),
    )


class TestPrediction5Recovery:
    """Narrative coherence gap recovers (decreases) after disruption.

    self_coherence = ||psi_hat - B|| / sqrt(N)  (lower = more coherent).

    A disruption shifts behavioral evidence away from the self-model,
    widening the gap.  Returning to normal should let the gap shrink
    back toward its pre-disruption level.
    """

    def test_aggregate_coherence_recovers(self) -> None:
        """Across seeds, mean late-recovery gap < mean disruption-peak gap."""
        sdk = AgentSDK.with_self_evidencing()
        normal = sdk.scenario(_balanced(), name="normal")
        disrupt = sdk.scenario(_disruption_scenario(), name="disruption")

        disruption_peaks: list[float] = []
        recovery_lates: list[float] = []

        for seed in SEEDS:
            sim = _build_self_aware_sim(sdk, seed)

            # Warm-up
            for _ in range(BASELINE_TICKS):
                sim.tick(normal)

            # Disruption: collect peak gap during disruption
            disrupt_gaps: list[float] = []
            for _ in range(DISRUPTION_TICKS):
                rec = sim.tick(disrupt)
                disrupt_gaps.append(rec.self_coherence)
            disruption_peaks.append(max(disrupt_gaps))

            # Recovery
            recovery_gaps: list[float] = []
            for _ in range(RECOVERY_TICKS):
                rec = sim.tick(normal)
                recovery_gaps.append(rec.self_coherence)
            recovery_lates.append(float(np.mean(recovery_gaps[-10:])))

        mean_peak = float(np.mean(disruption_peaks))
        mean_late = float(np.mean(recovery_lates))

        assert mean_late <= mean_peak, (
            f"Late recovery gap {mean_late:.4f} should be <= "
            f"disruption peak gap {mean_peak:.4f}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_per_seed_recovery_trend(self, seed: int) -> None:
        """Late-recovery gap does not exceed disruption peak gap."""
        sdk = AgentSDK.with_self_evidencing()
        sim = _build_self_aware_sim(sdk, seed)
        normal = sdk.scenario(_balanced(), name="normal")
        disrupt = sdk.scenario(_disruption_scenario(), name="disruption")

        for _ in range(BASELINE_TICKS):
            sim.tick(normal)

        disrupt_gaps: list[float] = []
        for _ in range(DISRUPTION_TICKS):
            rec = sim.tick(disrupt)
            disrupt_gaps.append(rec.self_coherence)
        peak_gap = max(disrupt_gaps)

        recovery_gaps: list[float] = []
        for _ in range(RECOVERY_TICKS):
            rec = sim.tick(normal)
            recovery_gaps.append(rec.self_coherence)

        late_mean = float(np.mean(recovery_gaps[-10:]))
        assert late_mean <= peak_gap, (
            f"seed={seed}: late gap {late_mean:.4f} should be <= "
            f"disruption peak {peak_gap:.4f}"
        )
