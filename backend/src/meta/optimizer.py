"""CMA-ES optimizer for precision parameters (§2.4).

Gradient-free optimizer that tunes precision engine parameters to
maximize behavioral divergence between personality profiles while
penalizing personality-collapsed solutions.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from src.meta.objective import MetaObjective, MetaObjectiveParams
from src.precision.params import PrecisionParams
from src.sdk import AgentSDK
from src.shared.logging import get_logger

_log = get_logger(module="meta.optimizer")


class CMAESConfig(BaseModel):
    """Configuration for CMA-ES optimization."""

    population_size: int = Field(default=10, gt=2)
    max_generations: int = Field(default=20, gt=0)
    initial_sigma: float = Field(default=0.3, gt=0.0)
    seed: int = 42


class OptimizationResult(BaseModel):
    """Result of CMA-ES optimization run."""

    best_params: dict[str, float]
    best_loss: float
    generations_run: int
    loss_history: list[float]


class CMAESOptimizer:
    """CMA-ES optimizer for precision parameters.

    Tunes the supervised init weights that control how personality
    dimensions map to precision levels.
    """

    PARAM_KEYS = [
        "n_mood_precision_weight",
        "e_arousal_precision_weight",
        "r_frustration_precision_weight",
        "default_bias",
    ]

    def __init__(
        self,
        config: CMAESConfig | None = None,
        objective_params: MetaObjectiveParams | None = None,
    ) -> None:
        """Initialize optimizer with CMA-ES config and objective parameters."""
        self._config = config or CMAESConfig()
        self._objective = MetaObjective(objective_params)
        self._rng = np.random.default_rng(self._config.seed)

    def optimize(self) -> OptimizationResult:
        """Run CMA-ES optimization loop."""
        n_params = len(self.PARAM_KEYS)
        mean = self._default_params_vector()
        sigma = self._config.initial_sigma
        cov = np.eye(n_params) * sigma**2
        mu = max(2, self._config.population_size // 2)

        loss_history: list[float] = []
        best_loss = float("inf")
        best_params = mean.copy()

        for gen in range(self._config.max_generations):
            population = self._rng.multivariate_normal(
                mean,
                cov,
                size=self._config.population_size,
            )
            population = np.clip(population, 0.01, 5.0)

            losses = np.array([self._evaluate_candidate(c) for c in population])

            sorted_idx = np.argsort(losses)
            elite = population[sorted_idx[:mu]]
            elite_losses = losses[sorted_idx[:mu]]

            mean = np.mean(elite, axis=0)
            diff = elite - mean
            cov = (diff.T @ diff) / mu + 1e-4 * np.eye(n_params)

            gen_best = float(elite_losses[0])
            loss_history.append(gen_best)

            if gen_best < best_loss:
                best_loss = gen_best
                best_params = elite[0].copy()

            _log.info(
                "cmaes_generation",
                generation=gen,
                best_loss=round(gen_best, 6),
                mean_loss=round(float(np.mean(losses)), 6),
            )

        return OptimizationResult(
            best_params=dict(
                zip(
                    self.PARAM_KEYS,
                    best_params.tolist(),
                    strict=False,
                )
            ),
            best_loss=best_loss,
            generations_run=self._config.max_generations,
            loss_history=loss_history,
        )

    def _default_params_vector(self) -> np.ndarray:
        """Extract default precision params as a vector."""
        defaults = PrecisionParams()
        return np.array(
            [
                defaults.n_mood_precision_weight,
                defaults.e_arousal_precision_weight,
                defaults.r_frustration_precision_weight,
                defaults.default_bias,
            ]
        )

    def _evaluate_candidate(self, candidate: np.ndarray) -> float:
        """Evaluate one candidate parameter vector."""
        params = PrecisionParams(
            n_mood_precision_weight=float(candidate[0]),
            e_arousal_precision_weight=float(candidate[1]),
            r_frustration_precision_weight=float(candidate[2]),
            default_bias=float(candidate[3]),
        )
        sdk = AgentSDK.with_precision(params)
        return self._objective.evaluate(sdk)
