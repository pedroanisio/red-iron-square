"""Tests for the self-model bounded context."""

import numpy as np
import pytest
from src.personality.decision import DecisionEngine
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action, PersonalityVector, Scenario
from src.self_model.emotions import (
    SelfEmotionDetector,
    SelfEmotionLabel,
)
from src.self_model.model import SelfModel
from src.self_model.params import SelfModelParams
from src.self_model.simulator import SelfAwareSimulator, SelfAwareTickResult
from src.temporal.state import AgentState


class TestSelfModelParams:
    """Tests for SelfModelParams."""

    def test_defaults(self) -> None:
        p = SelfModelParams()
        assert p.learning_rate == pytest.approx(0.08)
        assert p.shame_scaling == pytest.approx(3.0)

    def test_inf_raises(self) -> None:
        with pytest.raises(Exception):
            SelfModelParams(learning_rate=float("inf"))


class TestSelfModel:
    """Tests for SelfModel."""

    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.init_psi = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def test_construction(self) -> None:
        sm = SelfModel(self.init_psi, self.reg)
        np.testing.assert_array_almost_equal(sm.psi_hat, self.init_psi)
        np.testing.assert_array_almost_equal(sm.anchor, self.init_psi)

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            SelfModel(np.zeros(3), self.reg)

    def test_out_of_range_raises(self) -> None:
        bad = np.array([1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            SelfModel(bad, self.reg)

    def test_psi_hat_returns_copy(self) -> None:
        sm = SelfModel(self.init_psi, self.reg)
        h = sm.psi_hat
        h[0] = 999.0
        assert sm.psi_hat[0] == pytest.approx(0.5)

    def test_initial_coherence_gap_is_zero(self) -> None:
        sm = SelfModel(self.init_psi, self.reg)
        assert sm.current_coherence_gap() == pytest.approx(0.0, abs=1e-10)

    def test_initial_identity_drift_is_zero(self) -> None:
        sm = SelfModel(self.init_psi, self.reg)
        assert sm.current_identity_drift() == pytest.approx(0.0, abs=1e-10)

    def test_update_returns_metrics(self) -> None:
        sm = SelfModel(self.init_psi, self.reg)
        probs = np.array([0.7, 0.3])
        mods = [np.random.default_rng(0).random(8), np.random.default_rng(1).random(8)]
        metrics = sm.update(probs, mods)
        assert "self_coherence" in metrics
        assert "identity_drift" in metrics
        assert "update_magnitude" in metrics

    def test_update_moves_psi_hat(self) -> None:
        sm = SelfModel(self.init_psi, self.reg)
        probs = np.array([0.9, 0.1])
        mod_high = np.ones(8) * 0.8
        mod_low = np.ones(8) * -0.8
        for _ in range(20):
            sm.update(probs, [mod_high, mod_low])
        assert not np.allclose(sm.psi_hat, self.init_psi)

    def test_compute_self_accuracy(self) -> None:
        sm = SelfModel(self.init_psi, self.reg)
        true_psi = np.ones(8) * 0.9
        acc = sm.compute_self_accuracy(true_psi)
        assert acc > 0.0

    def test_sustained_coherence_threat_initially_false(self) -> None:
        sm = SelfModel(self.init_psi, self.reg)
        assert not sm.sustained_coherence_threat()


class TestSelfModelPrediction:
    """Tests for self-model prediction and prediction error."""

    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.engine = DecisionEngine(registry=self.reg)
        self.sm = SelfModel(np.full(8, 0.5), self.reg)
        self.scenario = Scenario(values={"O": 0.5}, registry=self.reg, name="test")
        self.actions = [
            Action("a", "a", modifiers=np.ones(8) * 0.5, registry=self.reg),
            Action("b", "b", modifiers=np.ones(8) * -0.5, registry=self.reg),
        ]

    def test_predict_returns_distribution(self) -> None:
        probs = self.sm.predict_action_distribution(
            self.scenario,
            self.actions,
            self.engine,
        )
        assert probs.shape == (2,)
        assert probs.sum() == pytest.approx(1.0)

    def test_state_aware_prediction(self) -> None:
        state = AgentState(mood=-0.5, energy=0.3)
        probs = self.sm.predict_action_distribution(
            self.scenario,
            self.actions,
            self.engine,
            state=state,
        )
        assert probs.shape == (2,)
        assert probs.sum() == pytest.approx(1.0)

    def test_prediction_error_identical_is_maximal_for_uniform(self) -> None:
        """Uniform actual probs yield H(q,p)/log(K) = 1.0 (max entropy)."""
        probs = np.array([0.5, 0.5])
        err = self.sm.compute_prediction_error(probs, probs)
        assert err == pytest.approx(1.0, abs=0.01)

    def test_prediction_error_peaked_on_peaked_is_low(self) -> None:
        """Peaked actual matching peaked predicted yields low normalized error."""
        actual = np.array([0.99, 0.01])
        predicted = np.array([0.99, 0.01])
        err = self.sm.compute_prediction_error(actual, predicted)
        assert err < 0.15

    def test_prediction_error_in_unit_range(self) -> None:
        actual = np.array([0.9, 0.1])
        predicted = np.array([0.1, 0.9])
        err = self.sm.compute_prediction_error(actual, predicted)
        assert 0.0 <= err <= 1.0


class TestSelfEmotionDetector:
    """Tests for SelfEmotionDetector."""

    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.psi = PersonalityVector(
            values={
                "O": 0.5,
                "C": 0.5,
                "E": 0.5,
                "A": 0.5,
                "N": 0.7,
                "R": 0.5,
                "I": 0.5,
                "T": 0.5,
            },
            registry=self.reg,
        )
        self.sm = SelfModel(np.full(8, 0.5), self.reg)
        self.detector = SelfEmotionDetector()

    def test_detect_all_returns_list(self) -> None:
        result = self.detector.detect_all(self.sm, 0.0, 0.5, self.psi, self.reg)
        assert isinstance(result, list)

    def test_pride_on_good_outcome_low_error(self) -> None:
        result = self.detector.detect_all(self.sm, 0.05, 0.9, self.psi, self.reg)
        labels = {r.label for r in result}
        assert SelfEmotionLabel.PRIDE in labels

    def test_authenticity_when_coherent(self) -> None:
        result = self.detector.detect_all(self.sm, 0.0, 0.0, self.psi, self.reg)
        labels = {r.label for r in result}
        assert SelfEmotionLabel.AUTHENTICITY in labels


class TestSelfAwareSimulator:
    """Tests for SelfAwareSimulator."""

    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.engine = DecisionEngine(registry=self.reg)
        self.psi = PersonalityVector(
            values={
                "O": 0.8,
                "C": 0.5,
                "E": 0.3,
                "A": 0.7,
                "N": 0.5,
                "R": 0.9,
                "I": 0.7,
                "T": 0.3,
            },
            registry=self.reg,
        )
        self.actions = [
            Action(
                "bold",
                "bold",
                modifiers=np.array([1, -0.5, 0.5, 0.3, -0.3, 0.8, 0.7, -0.5]),
                registry=self.reg,
            ),
            Action(
                "safe",
                "safe",
                modifiers=np.array([0.2, 0.9, 0.1, 0.5, 0.5, 0.1, 0.2, 0.8]),
                registry=self.reg,
            ),
        ]
        self.init_sm = np.array([0.7, 0.5, 0.4, 0.6, 0.5, 0.8, 0.6, 0.3])

    def test_tick_returns_self_aware_result(self) -> None:
        sim = SelfAwareSimulator(
            self.psi,
            self.init_sm,
            self.actions,
            self.engine,
            rng=np.random.default_rng(42),
        )
        scenario = Scenario(values={"O": 0.5}, registry=self.reg, name="t")
        result = sim.tick(scenario, outcome=0.5)
        assert isinstance(result, SelfAwareTickResult)
        assert result.psi_hat.shape == (8,)
        assert 0.0 <= result.prediction_error <= 1.0
        assert result.self_accuracy >= 0.0

    def test_multi_tick_self_model_evolves(self) -> None:
        sim = SelfAwareSimulator(
            self.psi,
            self.init_sm,
            self.actions,
            self.engine,
            rng=np.random.default_rng(42),
        )
        scenario = Scenario(values={"O": 0.9, "N": 0.8}, registry=self.reg, name="s")
        initial_psi_hat = sim.self_model.psi_hat.copy()
        for _ in range(20):
            sim.tick(scenario, outcome=0.3)
        assert not np.allclose(sim.self_model.psi_hat, initial_psi_hat)

    def test_self_emotions_are_detected(self) -> None:
        sim = SelfAwareSimulator(
            self.psi,
            self.init_sm,
            self.actions,
            self.engine,
            rng=np.random.default_rng(42),
        )
        scenario = Scenario(values={"O": 0.5}, registry=self.reg, name="t")
        result = sim.tick(scenario, outcome=0.8)
        assert isinstance(result.self_emotions, list)
