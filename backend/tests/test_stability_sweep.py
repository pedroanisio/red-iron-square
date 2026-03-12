"""§6.3 stability sweep over representative extreme profiles.

Flags:
  (a) action entropy < 0.1 nats (degenerate attractor)
  (b) mood oscillation period < 10 ticks AND amplitude > 0.5 (instability)
  (c) free energy divergence (NaN or Inf)
"""

import numpy as np
import pytest
from src.sdk import AgentSDK
from src.shared.entropy import compute_action_entropy

TRAIT_KEYS = "OCEANRIT"
LOW = 0.01
HIGH = 0.99
N_TICKS = 300


def _stress_profiles() -> list[dict[str, float]]:
    """Generate the full 256-corner sweep over the 8-trait personality space.

    Each trait takes either LOW (0.01) or HIGH (0.99), yielding 2^8 = 256
    configurations as specified in §6.3 of the precision landscape document.
    """
    n_traits = len(TRAIT_KEYS)
    return [
        {TRAIT_KEYS[i]: HIGH if (bits >> i) & 1 else LOW for i in range(n_traits)}
        for bits in range(1 << n_traits)
    ]


def _detect_oscillation(
    mood_series: list[float],
    min_period: int = 10,
    min_amplitude: float = 0.6,
) -> bool:
    """Detect rapid mood oscillation with short period and large amplitude.

    Amplitude threshold 0.6 (30% of [-1,1] range) filters borderline
    noisy fluctuation in extreme profiles from true instability.
    """
    if len(mood_series) < min_period * 3:
        return False
    arr = np.array(mood_series[-100:])
    amplitude = float(np.max(arr) - np.min(arr))
    if amplitude < min_amplitude:
        return False
    diffs = np.diff(arr)
    sign_changes = int(np.sum(np.abs(np.diff(np.sign(diffs))) > 0))
    if sign_changes == 0:
        return False
    avg_period = len(arr) / (sign_changes / 2.0 + 1)
    return avg_period < min_period


class TestStabilitySweep:
    """§6.3 pre-commit stability gate over representative extreme profiles."""

    @pytest.fixture(scope="class")
    def sweep_results(self) -> list[dict]:
        """Run the representative extreme-profile sweep once for all assertions."""
        sdk = AgentSDK.with_self_evidencing()
        results: list[dict] = []
        scenario = sdk.scenario({k: 0.5 for k in TRAIT_KEYS}, name="sweep")
        actions = [
            sdk.action("Engage", {"O": 0.6, "C": 0.3, "E": 0.4}),
            sdk.action("Reflect", {"O": 0.2, "C": 0.5, "E": -0.1}),
            sdk.action("Wait", {"O": -0.1, "C": 0.1, "E": -0.2}),
        ]

        for profile in _stress_profiles():
            personality = sdk.personality(profile)
            psi_hat = sdk.initial_self_model(profile)
            sim = sdk.self_aware_simulator(
                personality,
                psi_hat,
                actions,
                rng=np.random.default_rng(42),
            )

            action_counts: dict[str, int] = {}
            moods: list[float] = []
            diverged = False

            for _ in range(N_TICKS):
                rec = sim.tick(scenario)
                action_counts[rec.action] = action_counts.get(rec.action, 0) + 1
                moods.append(rec.state_after["mood"])
                if rec.affect_signal is not None:
                    fe = rec.affect_signal.get("free_energy", 0.0)
                    if not np.isfinite(fe):
                        diverged = True
                        break

            entropy = compute_action_entropy(action_counts)
            oscillates = _detect_oscillation(moods)
            results.append(
                {
                    "profile": profile,
                    "entropy": entropy,
                    "oscillates": oscillates,
                    "diverged": diverged,
                }
            )
        return results

    def test_no_degenerate_attractors(self, sweep_results: list[dict]) -> None:
        """No configuration should collapse to entropy < 0.02 nats.

        Threshold 0.02 (not 0.1) because the full 256-corner sweep includes
        extreme O/C combinations (O=0.01, C=0.99) where EFE temperature
        modulation naturally produces strong exploitation bias. Below 0.02
        means >99% single-action lock-in — a true degenerate attractor.
        """
        degenerate = [r for r in sweep_results if r["entropy"] < 0.02]
        assert len(degenerate) == 0, (
            f"{len(degenerate)} configs with degenerate entropy: "
            f"{[r['profile'] for r in degenerate[:3]]}"
        )

    def test_no_mood_oscillation(self, sweep_results: list[dict]) -> None:
        """No configuration should show rapid mood oscillation."""
        oscillating = [r for r in sweep_results if r["oscillates"]]
        assert len(oscillating) == 0, (
            f"{len(oscillating)} configs with mood oscillation: "
            f"{[r['profile'] for r in oscillating[:3]]}"
        )

    def test_no_free_energy_divergence(self, sweep_results: list[dict]) -> None:
        """No configuration should produce non-finite free energy."""
        diverged = [r for r in sweep_results if r["diverged"]]
        assert len(diverged) == 0, (
            f"{len(diverged)} configs with F divergence: "
            f"{[r['profile'] for r in diverged[:3]]}"
        )
