"""Exit criterion tests for Phase B: EFE surrogate decision engine.

Criteria from section 7.2 of the research doc:
  - Equivalence: balanced profiles -> KL divergence <= 0.1 nats
  - Differentiation: high-O vs high-C -> action entropy difference >= 0.3 nats
"""

import numpy as np
from src.sdk import AgentSDK
from src.shared.entropy import compute_action_entropy

N_SEEDS = 10
N_TICKS = 200


def _balanced() -> dict[str, float]:
    return {k: 0.5 for k in "OCEANRIT"}


def _profile(trait: str, value: float) -> dict[str, float]:
    vals = _balanced()
    vals[trait] = value
    return vals


def _run_action_distribution(
    sdk: AgentSDK,
    profile: dict[str, float],
    seed: int,
    n_ticks: int = N_TICKS,
) -> dict[str, int]:
    """Run a simulation and count action selections."""
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
    counts: dict[str, int] = {a.name: 0 for a in actions}
    for _ in range(n_ticks):
        rec = sim.tick(scenario)
        counts[rec.action] += 1
    return counts


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) in nats, with smoothing."""
    eps = 1e-10
    p_safe = p + eps
    q_safe = q + eps
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _aggregate_distribution(all_counts: list[dict[str, int]]) -> np.ndarray:
    """Aggregate action counts across seeds into a probability distribution."""
    keys = sorted(all_counts[0].keys())
    total_counts = np.zeros(len(keys))
    for counts in all_counts:
        for i, k in enumerate(keys):
            total_counts[i] += counts[k]
    total = total_counts.sum()
    if total == 0:
        return np.ones(len(keys)) / len(keys)
    return total_counts / total


class TestEFEEquivalence:
    """Balanced profiles: EFE and DecisionEngine produce similar distributions."""

    def test_balanced_kl_divergence_within_threshold(self) -> None:
        """KL divergence between EFE and DecisionEngine <= 0.1 nats."""
        sdk_base = AgentSDK.with_precision()
        sdk_efe = AgentSDK.with_efe()
        profile = _balanced()

        base_counts = [
            _run_action_distribution(sdk_base, profile, seed) for seed in range(N_SEEDS)
        ]
        efe_counts = [
            _run_action_distribution(sdk_efe, profile, seed) for seed in range(N_SEEDS)
        ]

        p_base = _aggregate_distribution(base_counts)
        p_efe = _aggregate_distribution(efe_counts)
        kl = _kl_divergence(p_efe, p_base)
        assert kl <= 0.1, (
            f"KL(EFE || base) = {kl:.4f} exceeds 0.1 nats. base={p_base}, efe={p_efe}"
        )


class TestEFEDifferentiation:
    """Extreme O/C profiles: EFE produces measurable exploration difference."""

    def test_high_O_more_entropic_than_high_C(self) -> None:
        """Action entropy difference >= 0.3 nats for O=0.9 vs C=0.9."""
        sdk = AgentSDK.with_efe()

        high_o_counts = [
            _run_action_distribution(
                sdk,
                _profile("O", 0.9),
                seed,
            )
            for seed in range(N_SEEDS)
        ]
        high_c_counts = [
            _run_action_distribution(
                sdk,
                _profile("C", 0.9),
                seed,
            )
            for seed in range(N_SEEDS)
        ]

        entropies_o = [compute_action_entropy(c) for c in high_o_counts]
        entropies_c = [compute_action_entropy(c) for c in high_c_counts]

        mean_o = float(np.mean(entropies_o))
        mean_c = float(np.mean(entropies_c))
        diff = mean_o - mean_c

        assert diff >= 0.3, (
            f"High-O entropy {mean_o:.4f} - high-C entropy {mean_c:.4f} = "
            f"{diff:.4f} < 0.3 nats required"
        )
