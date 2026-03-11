"""Exit criterion tests: precision trajectories show personality-dependent patterns.

Runs 8 personality profiles (extreme high/low for N, E, T, R) across
10 seeded runs of 50 ticks each, asserting that precision varies
meaningfully with personality.
"""

import numpy as np
from src.precision.state import PrecisionState
from src.sdk import AgentSDK

N_SEEDS = 10
N_TICKS = 50


def _balanced() -> dict[str, float]:
    return {k: 0.5 for k in "OCEANRIT"}


def _profile(trait: str, value: float) -> dict[str, float]:
    vals = _balanced()
    vals[trait] = value
    return vals


def _run_profile(profile: dict[str, float], seed: int) -> list[PrecisionState]:
    """Run one simulation and collect precision trajectory."""
    sdk = AgentSDK.with_precision()
    personality = sdk.personality(profile)
    scenario = sdk.scenario(_balanced(), name="standard")
    actions = [
        sdk.action("Engage", {"O": 0.6, "C": 0.3, "E": 0.4}),
        sdk.action("Reflect", {"O": 0.2, "C": 0.5, "E": -0.1}),
        sdk.action("Wait", {"O": -0.1, "C": 0.1, "E": -0.2}),
    ]
    sim = sdk.simulator(
        personality,
        actions,
        rng=np.random.default_rng(seed),
    )
    trajectory: list[PrecisionState] = []
    for _ in range(N_TICKS):
        rec = sim.tick(scenario)
        assert rec.precision is not None
        prec = rec.precision
        # Reconstruct PrecisionState from dict payload
        trajectory.append(
            PrecisionState(
                level_0=np.array(
                    [
                        prec["level_0"][k]
                        for k in (
                            "mood",
                            "arousal",
                            "energy",
                            "satisfaction",
                            "frustration",
                        )
                    ]
                ),
                level_1=prec["level_1"],
                level_2=prec["level_2"],
            )
        )
    return trajectory


def _mean_l0_channel(trajectories: list[list[PrecisionState]], idx: int) -> float:
    """Compute mean precision for one L0 channel across all runs and ticks."""
    values = [ps.level_0[idx] for traj in trajectories for ps in traj]
    return float(np.mean(values))


def _mean_l1(trajectories: list[list[PrecisionState]]) -> float:
    """Compute mean policy precision across all runs and ticks."""
    values = [ps.level_1 for traj in trajectories for ps in traj]
    return float(np.mean(values))


def _mean_l2(trajectories: list[list[PrecisionState]]) -> float:
    """Compute mean narrative precision across all runs and ticks."""
    values = [ps.level_2 for traj in trajectories for ps in traj]
    return float(np.mean(values))


class TestPrecisionTrajectories:
    """Exit criterion: precision shows personality-dependent patterns."""

    def test_high_N_higher_interoceptive_precision_than_low_N(self) -> None:
        """High-N agents produce higher mean mood precision (L0[0])."""
        high_n = [_run_profile(_profile("N", 0.9), s) for s in range(N_SEEDS)]
        low_n = [_run_profile(_profile("N", 0.1), s) for s in range(N_SEEDS)]
        mean_high = _mean_l0_channel(high_n, 0)
        mean_low = _mean_l0_channel(low_n, 0)
        assert mean_high > mean_low, (
            f"High-N mood precision {mean_high:.4f} should exceed low-N {mean_low:.4f}"
        )

    def test_high_E_higher_policy_precision_than_low_E(self) -> None:
        """High-E agents produce higher mean policy precision (L1)."""
        high_e = [_run_profile(_profile("E", 0.9), s) for s in range(N_SEEDS)]
        low_e = [_run_profile(_profile("E", 0.1), s) for s in range(N_SEEDS)]
        mean_high = _mean_l1(high_e)
        mean_low = _mean_l1(low_e)
        assert mean_high > mean_low, (
            f"High-E policy precision {mean_high:.4f} should exceed "
            f"low-E {mean_low:.4f}"
        )

    def test_high_T_higher_narrative_precision_than_low_T(self) -> None:
        """High-T agents produce higher mean narrative precision (L2)."""
        high_t = [_run_profile(_profile("T", 0.9), s) for s in range(N_SEEDS)]
        low_t = [_run_profile(_profile("T", 0.1), s) for s in range(N_SEEDS)]
        mean_high = _mean_l2(high_t)
        mean_low = _mean_l2(low_t)
        assert mean_high > mean_low, (
            f"High-T narrative precision {mean_high:.4f} should exceed "
            f"low-T {mean_low:.4f}"
        )

    def test_precision_varies_over_time(self) -> None:
        """Precision is not static — it changes as state evolves."""
        traj = _run_profile(_balanced(), seed=42)
        l0_values = np.array([ps.level_0 for ps in traj])
        # Standard deviation across ticks for at least one channel > 0
        stds = np.std(l0_values, axis=0)
        assert np.any(stds > 1e-6), (
            f"Precision should vary over time, but all stds are {stds}"
        )
