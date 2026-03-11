"""Tests for the personality bounded context."""

import numpy as np
import pytest
from src.personality.activations import DEFAULT_ACTIVATION_REGISTRY, ActivationFunctions
from src.personality.decision import DecisionEngine, compute_activation_batch
from src.personality.dimensions import DEFAULT_DIMENSIONS, Dimension, DimensionRegistry
from src.personality.hyperparameters import HyperParameters, ResilienceMode
from src.personality.vectors import Action, PersonalityVector, Scenario


class TestDimension:
    """Tests for Dimension and DimensionRegistry."""

    def test_dimension_is_frozen(self) -> None:
        dim = Dimension(key="X", name="Test", description="Desc")
        with pytest.raises(Exception):
            dim.key = "Y"

    def test_default_has_eight_dims(self) -> None:
        assert len(DEFAULT_DIMENSIONS) == 8

    def test_registry_size(self) -> None:
        reg = DimensionRegistry()
        assert reg.size == 8
        assert len(reg) == 8

    def test_registry_index_lookup(self) -> None:
        reg = DimensionRegistry()
        assert reg.index("O") == 0
        assert reg.index("T") == 7

    def test_registry_duplicate_keys_raises(self) -> None:
        dup = [
            Dimension(key="X", name="A", description=""),
            Dimension(key="X", name="B", description=""),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            DimensionRegistry(dup)

    def test_custom_registry(self) -> None:
        custom = [Dimension(key="A", name="Alpha", description="First")]
        reg = DimensionRegistry(custom)
        assert reg.size == 1
        assert reg.keys == ("A",)


class TestPersonalityVector:
    """Tests for PersonalityVector."""

    def test_repr(self) -> None:
        reg = DimensionRegistry()
        pv = PersonalityVector(values={"O": 0.5}, registry=reg)
        assert "psi(" in repr(pv)

    def test_getitem(self) -> None:
        reg = DimensionRegistry()
        pv = PersonalityVector(values={"O": 0.9, "N": 0.3}, registry=reg)
        assert pv["O"] == pytest.approx(0.9)
        assert pv["N"] == pytest.approx(0.3)


class TestScenario:
    """Tests for Scenario."""

    def test_name_and_description(self) -> None:
        reg = DimensionRegistry()
        s = Scenario(values={"O": 0.5}, registry=reg, name="test", description="desc")
        assert s.name == "test"
        assert "test" in repr(s)


class TestAction:
    """Tests for Action."""

    def test_valid_action(self) -> None:
        reg = DimensionRegistry()
        a = Action("bold", "desc", modifiers=np.ones(8), registry=reg)
        assert a.name == "bold"
        np.testing.assert_array_equal(a.modifiers, np.ones(8))

    def test_wrong_shape_raises(self) -> None:
        reg = DimensionRegistry()
        with pytest.raises(ValueError, match="shape"):
            Action("bad", "desc", modifiers=np.ones(3), registry=reg)

    def test_modifiers_returns_copy(self) -> None:
        reg = DimensionRegistry()
        a = Action("a", "d", modifiers=np.ones(8), registry=reg)
        m = a.modifiers
        m[0] = 999.0
        assert a.modifiers[0] == pytest.approx(1.0)


class TestHyperParameters:
    """Tests for HyperParameters."""

    def test_defaults(self) -> None:
        hp = HyperParameters()
        assert hp.alpha == pytest.approx(3.0)

    def test_inf_raises(self) -> None:
        with pytest.raises(Exception):
            HyperParameters(alpha=float("inf"))


class TestActivationFunctions:
    """Tests for activation functions."""

    def test_openness_output_range(self) -> None:
        hp = HyperParameters()
        for s in [0.0, 0.5, 1.0]:
            for t in [0.0, 0.5, 1.0]:
                val = ActivationFunctions.f_openness(s, t, hp)
                assert 0.0 <= val <= 1.0

    def test_linear_interpolation_shared(self) -> None:
        """Agreeableness, idealism, and tradition use the same function."""
        hp = HyperParameters()
        for s, t in [(0.3, 0.7), (0.8, 0.2), (0.5, 0.5)]:
            a = ActivationFunctions.f_agreeableness(s, t, hp)
            i = ActivationFunctions.f_idealism(s, t, hp)
            tr = ActivationFunctions.f_tradition(s, t, hp)
            assert a == pytest.approx(i)
            assert a == pytest.approx(tr)

    def test_neuroticism_output_range(self) -> None:
        hp = HyperParameters()
        for s in [0.0, 0.5, 1.0]:
            for t in [0.0, 0.5, 1.0]:
                val = ActivationFunctions.f_neuroticism(s, t, hp)
                assert 0.0 <= val <= 1.0

    def test_resilience_activation_mode(self) -> None:
        hp = HyperParameters()
        val = ActivationFunctions.f_resilience(
            0.9, 0.9, hp, mode=ResilienceMode.ACTIVATION
        )
        assert 0.0 <= val <= 1.0

    def test_resilience_buffer_mode(self) -> None:
        hp = HyperParameters()
        val = ActivationFunctions.f_resilience(0.9, 0.9, hp, mode=ResilienceMode.BUFFER)
        assert 0.0 <= val <= 1.0

    def test_all_default_keys_registered(self) -> None:
        reg = DimensionRegistry()
        for key in reg.keys:
            assert key in DEFAULT_ACTIVATION_REGISTRY


class TestDecisionEngine:
    """Tests for DecisionEngine."""

    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.engine = DecisionEngine(registry=self.reg)
        self.psi = PersonalityVector(
            values={
                "O": 0.8,
                "C": 0.5,
                "E": 0.3,
                "A": 0.7,
                "N": 0.6,
                "R": 0.9,
                "I": 0.7,
                "T": 0.3,
            },
            registry=self.reg,
        )
        self.scenario = Scenario(
            values={
                "O": 0.9,
                "C": 0.2,
                "E": 0.7,
                "A": 0.5,
                "N": 0.8,
                "R": 0.9,
                "I": 0.6,
                "T": 0.3,
            },
            registry=self.reg,
            name="test",
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

    def test_activations_in_unit_range(self) -> None:
        acts = self.engine.compute_activations(self.psi, self.scenario)
        assert acts.shape == (8,)
        assert np.all(acts >= 0.0) and np.all(acts <= 1.0)

    def test_utility_returns_scalar(self) -> None:
        u = self.engine.utility(self.psi, self.scenario, self.actions[0])
        assert isinstance(u, float)

    def test_utility_with_override(self) -> None:
        overrides = np.full(8, 0.5)
        u = self.engine.utility(
            self.psi, self.scenario, self.actions[0], activations_override=overrides
        )
        expected = float(np.dot(overrides, self.actions[0].modifiers))
        assert u == pytest.approx(expected)

    def test_decide_returns_action_and_probs(self) -> None:
        rng = np.random.default_rng(42)
        action, probs = self.engine.decide(
            self.psi,
            self.scenario,
            self.actions,
            temperature=1.0,
            rng=rng,
        )
        assert action in self.actions
        assert probs.shape == (2,)
        assert probs.sum() == pytest.approx(1.0)

    def test_decide_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            self.engine.decide(self.psi, self.scenario, self.actions, temperature=0)

    def test_decide_no_actions_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            self.engine.decide(self.psi, self.scenario, [], temperature=1.0)

    def test_missing_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="No activation"):
            DecisionEngine(
                registry=self.reg, activation_registry={"O": lambda s, t, h: 0}
            )


class TestComputeActivationBatch:
    """Tests for compute_activation_batch."""

    def test_batch_shape(self) -> None:
        psi = np.random.default_rng(0).random((10, 8))
        s = np.random.default_rng(1).random((10, 8))
        result = compute_activation_batch(psi, s)
        assert result.shape == (10, 8)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_activation_batch(np.zeros((5, 8)), np.zeros((5, 6)))

    def test_five_dim_batch(self) -> None:
        psi = np.random.default_rng(0).random((3, 5))
        s = np.random.default_rng(1).random((3, 5))
        result = compute_activation_batch(psi, s)
        assert result.shape == (3, 5)
