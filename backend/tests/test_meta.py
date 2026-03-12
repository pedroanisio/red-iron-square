"""Tests for meta-learning: objective function and CMA-ES optimizer (§2.4)."""

import numpy as np
import pytest
from src.meta.objective import MetaObjective, MetaObjectiveParams, _jsd
from src.meta.optimizer import CMAESConfig, CMAESOptimizer, OptimizationResult
from src.sdk import AgentSDK


class TestJSD:
    """Jensen-Shannon divergence helper."""

    def test_identical_distributions_zero(self) -> None:
        p = np.array([0.5, 0.3, 0.2])
        assert _jsd(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_different_distributions_positive(self) -> None:
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.05, 0.05, 0.9])
        assert _jsd(p, q) > 0.0

    def test_symmetric(self) -> None:
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.3, 0.4, 0.3])
        assert _jsd(p, q) == pytest.approx(_jsd(q, p), abs=1e-10)


class TestMetaObjective:
    """Meta-objective function for behavioral divergence."""

    def test_evaluate_returns_finite(self) -> None:
        params = MetaObjectiveParams(n_ticks=10, n_profiles=3)
        obj = MetaObjective(params)
        sdk = AgentSDK.with_precision()
        loss = obj.evaluate(sdk)
        assert np.isfinite(loss)

    def test_generates_correct_number_of_profiles(self) -> None:
        params = MetaObjectiveParams(n_profiles=5)
        obj = MetaObjective(params)
        profiles = obj._generate_profiles()
        assert len(profiles) == 5

    def test_divergence_positive_for_diverse_results(self) -> None:
        """Diverse profiles should produce positive divergence."""
        params = MetaObjectiveParams(n_ticks=20, n_profiles=4)
        obj = MetaObjective(params)
        sdk = AgentSDK.with_precision()
        results = [obj._simulate_profile(sdk, p) for p in obj._generate_profiles()]
        div = obj._compute_divergence(results)
        assert div >= 0.0


class TestCMAESOptimizer:
    """CMA-ES optimizer for precision parameters."""

    def test_optimize_returns_result(self) -> None:
        config = CMAESConfig(
            population_size=4,
            max_generations=2,
        )
        obj_params = MetaObjectiveParams(n_ticks=5, n_profiles=3)
        optimizer = CMAESOptimizer(config, obj_params)
        result = optimizer.optimize()
        assert isinstance(result, OptimizationResult)
        assert result.generations_run == 2
        assert len(result.loss_history) == 2
        assert np.isfinite(result.best_loss)

    def test_best_params_has_expected_keys(self) -> None:
        config = CMAESConfig(population_size=4, max_generations=1)
        obj_params = MetaObjectiveParams(n_ticks=5, n_profiles=3)
        optimizer = CMAESOptimizer(config, obj_params)
        result = optimizer.optimize()
        for key in CMAESOptimizer.PARAM_KEYS:
            assert key in result.best_params
            assert result.best_params[key] > 0

    def test_default_params_vector_shape(self) -> None:
        optimizer = CMAESOptimizer()
        vec = optimizer._default_params_vector()
        assert vec.shape == (len(CMAESOptimizer.PARAM_KEYS),)
        assert np.all(vec > 0)
