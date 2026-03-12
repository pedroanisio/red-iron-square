"""Tests for the ablation protocol (section 8).

Covers configuration validity, individual runs, the full sweep,
and behavioral differentiation between conditions.
"""

from __future__ import annotations

import pytest
from src.ablation.runner import (
    ABLATION_CONFIGS,
    AblationConfig,
    AblationResult,
    AblationRunner,
    _build_sdk,
)
from src.precision.params import PrecisionParams
from src.self_evidencing.params import SelfEvidencingParams
from src.shared.entropy import compute_action_entropy

_SHORT_TICKS = 10


def _short_configs() -> list[AblationConfig]:
    """Return all ablation configs with reduced tick count for speed."""
    return [
        cfg.model_copy(update={"n_ticks": _SHORT_TICKS}) for cfg in ABLATION_CONFIGS
    ]


class TestAblationConfig:
    """Validate ablation configuration invariants."""

    def test_ten_configs_defined(self) -> None:
        """Protocol specifies exactly ten conditions."""
        assert len(ABLATION_CONFIGS) == 10

    def test_config_names_unique(self) -> None:
        names = [cfg.name for cfg in ABLATION_CONFIGS]
        assert len(names) == len(set(names))

    def test_full_model_present(self) -> None:
        names = {cfg.name for cfg in ABLATION_CONFIGS}
        assert "full_model" in names

    def test_all_sdk_modes_valid(self) -> None:
        """Every config must reference a known SDK factory."""
        for cfg in ABLATION_CONFIGS:
            sdk = _build_sdk(cfg.sdk_mode)
            assert sdk is not None

    def test_invalid_sdk_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown sdk_mode"):
            _build_sdk("nonexistent_mode")


class TestAblationEntropy:
    """Unit tests for the entropy helper."""

    def test_uniform_distribution_entropy(self) -> None:
        counts = {"A": 10, "B": 10, "C": 10}
        entropy = compute_action_entropy(counts)
        assert entropy == pytest.approx(1.0986, abs=0.01)

    def test_degenerate_distribution_entropy(self) -> None:
        counts = {"A": 30}
        entropy = compute_action_entropy(counts)
        assert entropy == pytest.approx(0.0)


class TestAblationSingleRun:
    """Each ablation condition must run without errors."""

    @pytest.fixture()
    def runner(self) -> AblationRunner:
        return AblationRunner(seed=123)

    @pytest.mark.parametrize(
        "config",
        _short_configs(),
        ids=[c.name for c in ABLATION_CONFIGS],
    )
    def test_config_runs_without_error(
        self,
        runner: AblationRunner,
        config: AblationConfig,
    ) -> None:
        result = runner.run(config)
        assert isinstance(result, AblationResult)
        assert result.config_name == config.name
        assert result.n_ticks == _SHORT_TICKS

    @pytest.mark.parametrize(
        "config",
        _short_configs(),
        ids=[c.name for c in ABLATION_CONFIGS],
    )
    def test_result_metrics_finite(
        self,
        runner: AblationRunner,
        config: AblationConfig,
    ) -> None:
        result = runner.run(config)
        assert result.mean_entropy >= 0.0
        assert -1.0 <= result.mean_mood <= 1.0


class TestAblationNoSelfModel:
    """The no_self_model condition must omit coherence."""

    def test_no_self_model_coherence_is_none(self) -> None:
        runner = AblationRunner(seed=42)
        cfg = next(
            c.model_copy(update={"n_ticks": _SHORT_TICKS})
            for c in ABLATION_CONFIGS
            if c.name == "no_self_model"
        )
        result = runner.run(cfg)
        assert result.mean_coherence is None

    def test_full_model_coherence_is_present(self) -> None:
        runner = AblationRunner(seed=42)
        cfg = next(
            c.model_copy(update={"n_ticks": _SHORT_TICKS})
            for c in ABLATION_CONFIGS
            if c.name == "full_model"
        )
        result = runner.run(cfg)
        assert result.mean_coherence is not None


class TestNewAblationConditions:
    """Validate the three additional ablation conditions from section 8."""

    def test_no_allostatic_setpoints_present(self) -> None:
        """Config exists and uses self_evidencing mode."""
        cfg = next(c for c in ABLATION_CONFIGS if c.name == "no_allostatic_setpoints")
        assert cfg.sdk_mode == "self_evidencing"
        assert cfg.precision_params is None

    def test_learned_vs_fixed_precision_has_custom_params(self) -> None:
        """Config carries hand-tuned precision weights."""
        cfg = next(
            c for c in ABLATION_CONFIGS if c.name == "learned_vs_fixed_precision"
        )
        assert cfg.precision_params is not None
        assert cfg.precision_params.n_mood_precision_weight == 0.3

    def test_no_se_cap_has_high_pi_max(self) -> None:
        """Config removes the stability cap by setting pi_max very high."""
        cfg = next(c for c in ABLATION_CONFIGS if c.name == "no_se_cap")
        assert cfg.self_evidencing_params is not None
        assert cfg.self_evidencing_params.pi_max == 100.0

    def test_build_sdk_with_custom_params(self) -> None:
        """SDK construction respects optional param overrides."""
        sdk = _build_sdk(
            "self_evidencing",
            precision_params=PrecisionParams(default_bias=0.1),
            self_evidencing_params=SelfEvidencingParams(pi_max=50.0),
        )
        assert sdk is not None

    def test_new_configs_run_without_error(self) -> None:
        """All three new conditions execute successfully."""
        runner = AblationRunner(seed=99)
        new_names = {
            "no_allostatic_setpoints",
            "learned_vs_fixed_precision",
            "no_se_cap",
        }
        for cfg in ABLATION_CONFIGS:
            if cfg.name in new_names:
                result = runner.run(cfg.model_copy(update={"n_ticks": _SHORT_TICKS}))
                assert isinstance(result, AblationResult)
                assert result.mean_entropy >= 0.0


class TestAblationRunAll:
    """Integration test for the full ablation sweep."""

    def test_run_all_returns_ten_results(self) -> None:
        """All ten ablation conditions produce unique results."""
        runner = AblationRunner(seed=7)
        configs = _short_configs()
        results = [runner.run(cfg) for cfg in configs]
        assert len(results) == 10
        names = {r.config_name for r in results}
        assert len(names) == 10

    def test_full_model_differs_from_at_least_one_ablation(self) -> None:
        """Behavioral signature: full model should differ from some ablation."""
        runner = AblationRunner(seed=42)
        configs = _short_configs()
        results = {r.config_name: r for r in (runner.run(c) for c in configs)}
        full = results["full_model"]
        diffs = [
            abs(full.mean_entropy - r.mean_entropy) + abs(full.mean_mood - r.mean_mood)
            for name, r in results.items()
            if name != "full_model"
        ]
        assert max(diffs) > 0.0, "Full model indistinguishable from all ablations"
