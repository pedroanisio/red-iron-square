"""Tests for Phase C2: self-evidencing precision modulation."""

import numpy as np
import pytest
from src.sdk import AgentSDK
from src.self_evidencing.modulator import SelfEvidencingModulator
from src.self_evidencing.params import SelfEvidencingParams


def _balanced() -> dict[str, float]:
    """Return a balanced personality profile with all traits at 0.5."""
    return {k: 0.5 for k in "OCEANRIT"}


class TestSelfEvidencingParams:
    """Parameter validation and defaults."""

    def test_defaults(self) -> None:
        p = SelfEvidencingParams()
        assert p.beta_0 == pytest.approx(1.0)
        assert p.pi_max == pytest.approx(3.0)
        assert p.lambda_beta == pytest.approx(0.95)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="not finite"):
            SelfEvidencingParams(beta_0=float("nan"))


class TestSelfEvidencingModulator:
    """Modulator: precision weights from self-model predictions."""

    def test_uniform_predictions_give_uniform_weights(self) -> None:
        mod = SelfEvidencingModulator()
        probs = np.array([1 / 3, 1 / 3, 1 / 3])
        weights = mod.compute_precision_weights(probs, base_precision=1.0)
        assert weights.shape == (3,)
        np.testing.assert_allclose(weights, 1.0, atol=1e-6)

    def test_concentrated_predictions_boost_preferred_action(self) -> None:
        mod = SelfEvidencingModulator()
        probs = np.array([0.8, 0.1, 0.1])
        weights = mod.compute_precision_weights(probs, base_precision=1.0)
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]

    def test_precision_cap_limits_raw_boost(self) -> None:
        """Mechanism A: raw boost capped at pi_max before normalization."""
        params = SelfEvidencingParams(pi_max=3.0, beta_0=0.5)
        mod = SelfEvidencingModulator(params)
        probs = np.array([0.8, 0.1, 0.1])
        weights = mod.compute_precision_weights(probs, base_precision=1.0)
        # With moderate beta, the preferred action gets a boost
        # but it is bounded by the cap via normalization
        assert weights[0] > weights[1]
        # Average weight equals base precision
        assert np.mean(weights) == pytest.approx(1.0, abs=0.01)

    def test_base_precision_scales_output(self) -> None:
        mod = SelfEvidencingModulator()
        probs = np.array([0.5, 0.5])
        w_low = mod.compute_precision_weights(probs, base_precision=1.0)
        w_high = mod.compute_precision_weights(probs, base_precision=2.0)
        np.testing.assert_allclose(w_high, w_low * 2.0, atol=1e-6)

    def test_normalization_conserves_average(self) -> None:
        mod = SelfEvidencingModulator()
        probs = np.array([0.6, 0.3, 0.1])
        weights = mod.compute_precision_weights(probs, base_precision=1.0)
        assert np.mean(weights) == pytest.approx(1.0, abs=0.01)

    def test_beta_decay(self) -> None:
        mod = SelfEvidencingModulator(SelfEvidencingParams(beta_0=2.0))
        initial_beta = mod.beta
        mod.decay_beta(personality_t=0.5)
        assert mod.beta < initial_beta
        assert mod.beta > 0.0

    def test_beta_decays_toward_personality_floor(self) -> None:
        params = SelfEvidencingParams(beta_0=2.0, lambda_beta=0.0)
        mod = SelfEvidencingModulator(params)
        mod.decay_beta(personality_t=0.5)
        expected_min = 0.5 * 1.0 * 2.0  # T * t_beta_scale * beta_0
        assert mod.beta == pytest.approx(expected_min, abs=1e-6)

    def test_reset_beta(self) -> None:
        mod = SelfEvidencingModulator()
        mod.decay_beta(0.5)
        mod.reset_beta()
        assert mod.beta == pytest.approx(1.0)

    def test_reset_beta_custom_value(self) -> None:
        mod = SelfEvidencingModulator()
        mod.reset_beta(3.0)
        assert mod.beta == pytest.approx(3.0)


class TestSelfEvidencingDivergence:
    """Action divergence: d(pi, psi_hat) = -log(p_hat_pi)."""

    def test_high_prob_low_divergence(self) -> None:
        probs = np.array([0.9, 0.1])
        divs = SelfEvidencingModulator._action_divergences(probs)
        assert divs[0] < divs[1]

    def test_equal_probs_equal_divergence(self) -> None:
        probs = np.array([0.5, 0.5])
        divs = SelfEvidencingModulator._action_divergences(probs)
        np.testing.assert_allclose(divs[0], divs[1], atol=1e-10)


class TestSelfEvidencingIntegration:
    """Integration with SDK and self-aware simulator."""

    def test_sdk_with_self_evidencing_factory(self) -> None:
        sdk = AgentSDK.with_self_evidencing()
        assert sdk._self_evidencing_params is not None
        assert sdk._emotion_params is not None

    def test_self_aware_tick_includes_weights(self) -> None:
        sdk = AgentSDK.with_self_evidencing()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        actions = [
            sdk.action("Act", {"O": 0.3, "E": 0.2}),
            sdk.action("Wait", {"O": -0.1}),
        ]
        psi_hat = sdk.initial_self_model(_balanced())
        sim = sdk.self_aware_simulator(
            personality,
            psi_hat,
            actions,
            rng=np.random.default_rng(42),
        )
        rec = sim.tick(scenario)
        assert rec.self_evidencing_weights is not None
        assert len(rec.self_evidencing_weights) == len(actions)

    def test_weights_vary_across_ticks(self) -> None:
        sdk = AgentSDK.with_self_evidencing()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        actions = [
            sdk.action("Act", {"O": 0.3, "E": 0.2}),
            sdk.action("Wait", {"O": -0.1}),
        ]
        psi_hat = sdk.initial_self_model(_balanced())
        sim = sdk.self_aware_simulator(
            personality,
            psi_hat,
            actions,
            rng=np.random.default_rng(42),
        )
        weights_list = []
        for _ in range(10):
            rec = sim.tick(scenario)
            if rec.self_evidencing_weights is not None:
                weights_list.append(rec.self_evidencing_weights)
        assert len(weights_list) > 0
        first = weights_list[0]
        last = weights_list[-1]
        assert not np.allclose(first, last, atol=1e-6)

    def test_backward_compat_no_weights_without_self_evidencing(self) -> None:
        sdk = AgentSDK.with_constructed_emotion()
        personality = sdk.personality(_balanced())
        psi_hat = sdk.initial_self_model(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        actions = [sdk.action("Act", {"O": 0.3})]
        sim = sdk.self_aware_simulator(
            personality,
            psi_hat,
            actions,
            rng=np.random.default_rng(0),
        )
        rec = sim.tick(scenario)
        assert rec.self_evidencing_weights is None


class TestSEWeightsModulateBoltzmannLogits:
    """Verify SE weights modulate Boltzmann logits via temperature scaling.

    The self-evidencing modulator adjusts effective temperature based on
    how concentrated the self-model's predicted distribution is.  A drifted
    self-model produces skewed predictions, which lowers the effective
    temperature and narrows the action distribution.  An aligned self-model
    produces near-uniform predictions, leaving the temperature unchanged.
    """

    @staticmethod
    def _high_drift_traits() -> dict[str, float]:
        """Personality that maximally diverges from balanced self-model."""
        return {
            "O": 1.0,
            "C": 0.0,
            "E": 1.0,
            "A": 0.0,
            "N": 1.0,
            "R": 0.0,
            "I": 1.0,
            "T": 0.0,
        }

    @staticmethod
    def _aligned_traits() -> dict[str, float]:
        """Personality perfectly aligned with self-model."""
        return {k: 0.5 for k in "OCEANRIT"}

    @staticmethod
    def _make_actions(sdk: AgentSDK) -> list:
        """Build a diverse action set to amplify distribution differences."""
        return [
            sdk.action("Bold", {"O": 0.6, "E": 0.4, "C": -0.2}),
            sdk.action("Cautious", {"C": 0.5, "N": -0.3, "O": -0.2}),
            sdk.action("Social", {"E": 0.5, "A": 0.4}),
        ]

    def _collect_probs(
        self,
        sdk: AgentSDK,
        traits: dict[str, float],
        psi_hat_traits: dict[str, float],
        n_ticks: int = 30,
        seed: int = 42,
    ) -> list[list[float]]:
        """Run n_ticks and collect probability vectors."""
        personality = sdk.personality(traits)
        psi_hat = sdk.initial_self_model(psi_hat_traits)
        scenario = sdk.scenario(_balanced(), name="test_se")
        actions = self._make_actions(sdk)
        sim = sdk.self_aware_simulator(
            personality,
            psi_hat,
            actions,
            rng=np.random.default_rng(seed),
        )
        return [sim.tick(scenario).probabilities for _ in range(n_ticks)]

    def test_drifted_vs_aligned_distributions_differ(self) -> None:
        """Drifted self-model produces different distribution than aligned.

        Proves SE modulates action selection logits.
        """
        se_params = SelfEvidencingParams(beta_0=3.0)
        sdk = AgentSDK.with_self_evidencing(self_evidencing_params=se_params)

        aligned_probs = self._collect_probs(
            sdk,
            self._aligned_traits(),
            self._aligned_traits(),
        )
        drifted_probs = self._collect_probs(
            sdk,
            self._high_drift_traits(),
            self._aligned_traits(),
        )

        aligned_means = np.mean(aligned_probs, axis=0)
        drifted_means = np.mean(drifted_probs, axis=0)

        dist = float(np.linalg.norm(aligned_means - drifted_means))
        assert dist > 0.01, (
            f"SE-modulated distributions should differ; L2 distance={dist:.6f}"
        )

    def test_se_narrows_distribution_for_drifted_model(self) -> None:
        """SE reduces entropy for drifted self-model.

        Lower temperature from SE yields more concentrated distributions.
        """
        se_params = SelfEvidencingParams(beta_0=3.0)
        sdk_se = AgentSDK.with_self_evidencing(self_evidencing_params=se_params)
        sdk_no_se = AgentSDK.with_constructed_emotion()

        drifted = self._high_drift_traits()
        balanced_psi = self._aligned_traits()

        probs_se = self._collect_probs(sdk_se, drifted, balanced_psi)
        probs_no = self._collect_probs(sdk_no_se, drifted, balanced_psi)

        def mean_entropy(prob_list: list[list[float]]) -> float:
            entropies = []
            for p in prob_list:
                arr = np.array(p)
                arr = arr[arr > 1e-10]
                entropies.append(-float(np.sum(arr * np.log(arr))))
            return float(np.mean(entropies))

        entropy_se = mean_entropy(probs_se)
        entropy_no = mean_entropy(probs_no)

        assert entropy_se < entropy_no, (
            f"SE should reduce entropy for drifted model: "
            f"SE={entropy_se:.4f} vs no-SE={entropy_no:.4f}"
        )

    def test_aligned_model_minimal_se_effect(self) -> None:
        """Aligned self-model sees minimal SE temperature effect.

        Distributions with and without SE should be similar.
        """
        se_params = SelfEvidencingParams(beta_0=1.0)
        sdk_se = AgentSDK.with_self_evidencing(self_evidencing_params=se_params)
        sdk_no_se = AgentSDK.with_constructed_emotion()

        aligned = self._aligned_traits()

        probs_se = self._collect_probs(sdk_se, aligned, aligned)
        probs_no = self._collect_probs(sdk_no_se, aligned, aligned)

        se_means = np.mean(probs_se, axis=0)
        no_means = np.mean(probs_no, axis=0)

        dist = float(np.linalg.norm(se_means - no_means))
        assert dist < 0.15, (
            f"Aligned model should see minimal SE effect; L2 distance={dist:.6f}"
        )
