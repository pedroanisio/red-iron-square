"""Exit criterion tests for Phase C: constructed emotion + self-evidencing.

Criteria from section 6.1 of the research doc:
  - Prediction 1: High-N agents have higher arousal signals (amplified precision).
  - Prediction 4: High-N agents show greater mood disruption under stress.
  - Prediction 3: High-T agents converge to narrower action repertoire.
"""

import numpy as np
from src.sdk import AgentSDK
from src.self_evidencing.params import SelfEvidencingParams

N_SEEDS = 5
N_TICKS = 100


def _balanced() -> dict[str, float]:
    return {k: 0.5 for k in "OCEANRIT"}


def _profile(overrides: dict[str, float]) -> dict[str, float]:
    vals = _balanced()
    vals.update(overrides)
    return vals


def _run_constructed_emotion_sim(
    profile: dict[str, float],
    scenario_profile: dict[str, float],
    seed: int,
    n_ticks: int = N_TICKS,
) -> list[dict[str, float]]:
    """Run a simulation with constructed emotion, collecting affect signals."""
    sdk = AgentSDK.with_constructed_emotion()
    personality = sdk.personality(profile)
    scenario = sdk.scenario(scenario_profile, name="test")
    actions = [
        sdk.action("Engage", {"O": 0.6, "C": 0.3, "E": 0.4}),
        sdk.action("Reflect", {"O": 0.2, "C": 0.5, "E": -0.1}),
        sdk.action("Wait", {"O": -0.1, "C": 0.1, "E": -0.2}),
    ]
    sim = sdk.simulator(personality, actions, rng=np.random.default_rng(seed))
    results = []
    for _ in range(n_ticks):
        rec = sim.tick(scenario)
        if rec.affect_signal is not None:
            results.append(rec.affect_signal)
    return results


def _run_self_evidencing_sim(
    profile: dict[str, float],
    seed: int,
    n_ticks: int = N_TICKS,
) -> list[str]:
    """Run self-aware sim with self-evidencing, return action sequence."""
    se_params = SelfEvidencingParams(beta_0=2.0, t_beta_scale=2.0)
    sdk = AgentSDK.with_self_evidencing(self_evidencing_params=se_params)
    personality = sdk.personality(profile)
    scenario = sdk.scenario(_balanced(), name="standard")
    actions = [
        sdk.action("Engage", {"O": 0.6, "C": 0.3, "E": 0.4}),
        sdk.action("Reflect", {"O": 0.2, "C": 0.5, "E": -0.1}),
        sdk.action("Wait", {"O": -0.1, "C": 0.1, "E": -0.2}),
    ]
    psi_hat = sdk.initial_self_model(profile)
    sim = sdk.self_aware_simulator(
        personality,
        psi_hat,
        actions,
        rng=np.random.default_rng(seed),
    )
    action_seq = []
    for _ in range(n_ticks):
        rec = sim.tick(scenario)
        action_seq.append(rec.action)
    return action_seq


class TestPrediction1HighNAmplifiedArousal:
    """High-N agents produce higher mean arousal signals (Prediction 1).

    High-N amplifies interoceptive precision (Pi_0 on mood), so
    precision-weighted prediction errors are larger on average.
    """

    def test_high_n_higher_mean_arousal(self) -> None:
        """High-N (0.9) produces higher mean arousal than low-N (0.1)."""
        arousals_high: list[float] = []
        arousals_low: list[float] = []

        for seed in range(N_SEEDS):
            sigs_high = _run_constructed_emotion_sim(
                _profile({"N": 0.9}),
                _balanced(),
                seed,
            )
            sigs_low = _run_constructed_emotion_sim(
                _profile({"N": 0.1}),
                _balanced(),
                seed,
            )
            arousals_high.extend(s["arousal_signal"] for s in sigs_high)
            arousals_low.extend(s["arousal_signal"] for s in sigs_low)

        mean_high = float(np.mean(arousals_high))
        mean_low = float(np.mean(arousals_low))

        assert mean_high > mean_low, (
            f"High-N mean arousal {mean_high:.4f} should exceed "
            f"low-N mean arousal {mean_low:.4f}"
        )


class TestPrediction4NonlinearNStress:
    """High-N agents show greater mood disruption under stress (Prediction 4).

    The constructed mood EMA integrates valence over time. Under stress
    (high scenario-N), high-N agents have amplified precision on mood,
    producing larger free energy fluctuations and more negative mood.
    """

    def test_high_n_more_negative_mood_under_stress(self) -> None:
        """High-N agents reach more negative mean mood under stress."""
        stress_scenario = _profile({"N": 0.9})

        def _mean_mood(
            agent_profile: dict[str, float],
        ) -> float:
            moods: list[float] = []
            for seed in range(N_SEEDS):
                sigs = _run_constructed_emotion_sim(
                    agent_profile,
                    stress_scenario,
                    seed,
                )
                moods.extend(s["mood"] for s in sigs[-50:])
            return float(np.mean(moods))

        mood_high_n = _mean_mood(_profile({"N": 0.9}))
        mood_low_n = _mean_mood(_profile({"N": 0.1}))

        # High-N agents should have more negative mood under stress
        assert mood_high_n < mood_low_n, (
            f"High-N mood {mood_high_n:.4f} should be more negative than "
            f"low-N mood {mood_low_n:.4f} under stress"
        )


class TestPrediction3SelfEvidencingAttractors:
    """High-T agents converge to narrower behavioral repertoire (Prediction 3).

    Self-evidencing with high T -> higher beta floor -> stronger
    temperature reduction -> more concentrated action distribution.
    """

    def test_high_t_narrower_late_entropy(self) -> None:
        """High-T agents have lower action entropy in later ticks."""

        def _late_entropy(profile: dict[str, float]) -> float:
            entropies = []
            for seed in range(N_SEEDS * 2):
                actions = _run_self_evidencing_sim(
                    profile,
                    seed,
                    n_ticks=200,
                )
                late = actions[100:]
                counts: dict[str, int] = {}
                for a in late:
                    counts[a] = counts.get(a, 0) + 1
                total = sum(counts.values())
                if total == 0:
                    entropies.append(0.0)
                    continue
                probs = np.array([c / total for c in counts.values()])
                probs = probs[probs > 0]
                entropies.append(float(-np.sum(probs * np.log(probs))))
            return float(np.mean(entropies))

        entropy_high_t = _late_entropy(_profile({"T": 0.9}))
        entropy_low_t = _late_entropy(_profile({"T": 0.1}))

        assert entropy_high_t < entropy_low_t, (
            f"High-T entropy {entropy_high_t:.4f} should be less than "
            f"low-T entropy {entropy_low_t:.4f}"
        )
