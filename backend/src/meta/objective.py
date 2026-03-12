"""Meta-objective function for precision parameter optimization (§2.4).

The objective measures behavioral divergence between simulated profiles
with a diversity penalty against personality-collapsed solutions.

Minimization target:
    L(θ) = divergence_loss(θ) + λ * collapse_penalty(θ)

Where:
    - divergence_loss: how well different personalities produce different behaviors
    - collapse_penalty: penalizes when all personalities converge to same behavior
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from src.sdk import AgentSDK


class MetaObjectiveParams(BaseModel):
    """Parameters for the meta-objective function."""

    n_ticks: int = Field(default=50, gt=0)
    n_profiles: int = Field(default=8, gt=1)
    diversity_weight: float = Field(default=0.5, ge=0.0)
    seed: int = 42


class ProfileResult(BaseModel):
    """Behavioral summary for one personality profile."""

    model_config = {"arbitrary_types_allowed": True}

    action_distribution: dict[str, float]
    mean_mood: float
    mean_arousal: float
    entropy: float


class MetaObjective:
    """Computes behavioral divergence between personality profiles.

    Higher divergence = better (personalities produce distinct behaviors).
    Returns negative divergence as the loss to minimize.
    """

    def __init__(self, params: MetaObjectiveParams | None = None) -> None:
        self._params = params or MetaObjectiveParams()

    def evaluate(self, sdk: AgentSDK) -> float:
        """Run profiles through the SDK and return loss (lower = better).

        Loss = -mean_divergence + diversity_weight * collapse_penalty.
        """
        profiles = self._generate_profiles()
        results = [self._simulate_profile(sdk, p) for p in profiles]
        divergence = self._compute_divergence(results)
        collapse = self._compute_collapse_penalty(results)
        return -divergence + self._params.diversity_weight * collapse

    def _generate_profiles(self) -> list[dict[str, float]]:
        """Generate diverse personality profiles for evaluation."""
        rng = np.random.default_rng(self._params.seed)
        profiles: list[dict[str, float]] = []
        keys = list("OCEANRIT")
        for _ in range(self._params.n_profiles):
            profile = {k: float(rng.uniform(0.1, 0.9)) for k in keys}
            profiles.append(profile)
        return profiles

    def _simulate_profile(
        self,
        sdk: AgentSDK,
        profile: dict[str, float],
    ) -> ProfileResult:
        """Run simulation for one profile and collect behavioral metrics."""
        personality = sdk.personality(profile)
        scenario = sdk.scenario(
            {k: 0.5 for k in "OCEANRIT"},
            name="meta",
        )
        actions = [
            sdk.action("Engage", {"O": 0.5, "E": 0.3}),
            sdk.action("Reflect", {"C": 0.5, "E": -0.1}),
            sdk.action("Wait", {"O": -0.1, "E": -0.2}),
        ]
        rng = np.random.default_rng(self._params.seed)
        sim = sdk.simulator(personality, actions, rng=rng)

        counts: dict[str, int] = {}
        moods: list[float] = []
        arousals: list[float] = []

        for _ in range(self._params.n_ticks):
            rec = sim.tick(scenario)
            counts[rec.action] = counts.get(rec.action, 0) + 1
            moods.append(rec.state_after["mood"])
            arousals.append(rec.state_after["arousal"])

        total = sum(counts.values())
        dist = {k: v / total for k, v in counts.items()}
        probs = np.array(list(dist.values()))
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)))

        return ProfileResult(
            action_distribution=dist,
            mean_mood=float(np.mean(moods)),
            mean_arousal=float(np.mean(arousals)),
            entropy=entropy,
        )

    @staticmethod
    def _compute_divergence(results: list[ProfileResult]) -> float:
        """Mean pairwise Jensen-Shannon divergence of action distributions."""
        all_actions: set[str] = set()
        for r in results:
            all_actions.update(r.action_distribution.keys())
        action_list = sorted(all_actions)

        vectors = []
        for r in results:
            v = np.array([r.action_distribution.get(a, 0.0) for a in action_list])
            vectors.append(v)

        n = len(vectors)
        if n < 2:
            return 0.0

        divergences: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                divergences.append(_jsd(vectors[i], vectors[j]))
        return float(np.mean(divergences))

    @staticmethod
    def _compute_collapse_penalty(results: list[ProfileResult]) -> float:
        """Penalty for low variance in behavioral metrics across profiles."""
        entropies = [r.entropy for r in results]
        moods = [r.mean_mood for r in results]
        entropy_var = float(np.var(entropies))
        mood_var = float(np.var(moods))
        # Penalty is high when variance is low (all profiles behave the same)
        return 1.0 / (entropy_var + mood_var + 0.01)


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    m = np.maximum(m, 1e-16)
    p_safe = np.maximum(p, 1e-16)
    q_safe = np.maximum(q, 1e-16)
    kl_pm = float(np.sum(p_safe * np.log(p_safe / m)))
    kl_qm = float(np.sum(q_safe * np.log(q_safe / m)))
    return 0.5 * (kl_pm + kl_qm)
