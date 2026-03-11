"""Tests for the precision bounded context."""

import numpy as np
import pytest
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import PersonalityVector, Scenario
from src.precision.engine import PrecisionEngine, PredictionErrorEngine
from src.precision.params import PrecisionParams
from src.precision.setpoints import INTEROCEPTIVE_KEYS, AllostaticSetPoints
from src.precision.state import (
    PrecisionSnapshot,
    PrecisionState,
    PredictionErrors,
    PredictionErrorSnapshot,
)
from src.sdk import AgentSDK
from src.temporal.state import AgentState


def _make_personality(
    registry: DimensionRegistry, overrides: dict[str, float] | None = None
) -> PersonalityVector:
    vals = {k: 0.5 for k in registry.keys}
    if overrides:
        vals.update(overrides)
    return PersonalityVector(values=vals, registry=registry)


def _make_scenario(
    registry: DimensionRegistry, overrides: dict[str, float] | None = None
) -> Scenario:
    vals = {k: 0.5 for k in registry.keys}
    if overrides:
        vals.update(overrides)
    return Scenario(values=vals, registry=registry, name="test")


class TestPrecisionParams:
    def test_defaults(self) -> None:
        p = PrecisionParams()
        assert p.n_personality == 8
        assert p.n_state == 5
        assert p.default_bias == pytest.approx(0.54)

    def test_inf_raises(self) -> None:
        with pytest.raises(Exception):
            PrecisionParams(default_bias=float("inf"))

    def test_nan_raises(self) -> None:
        with pytest.raises(Exception):
            PrecisionParams(n_mood_precision_weight=float("nan"))


class TestAllostaticSetPoints:
    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.sp = AllostaticSetPoints()

    def test_output_shape(self) -> None:
        p = _make_personality(self.reg)
        result = self.sp.compute(p)
        assert result.shape == (5,)

    def test_arousal_depends_on_E(self) -> None:
        low_e = _make_personality(self.reg, {"E": 0.0})
        high_e = _make_personality(self.reg, {"E": 1.0})
        sp_low = self.sp.compute(low_e)
        sp_high = self.sp.compute(high_e)
        assert sp_high[1] > sp_low[1]

    def test_fixed_setpoints_unchanged_by_personality(self) -> None:
        p1 = _make_personality(self.reg, {"N": 0.1})
        p2 = _make_personality(self.reg, {"N": 0.9})
        sp1 = self.sp.compute(p1)
        sp2 = self.sp.compute(p2)
        for idx in (0, 2, 3, 4):
            assert sp1[idx] == pytest.approx(sp2[idx])

    def test_default_values_match_state_params(self) -> None:
        p = _make_personality(self.reg)
        sp = self.sp.compute(p)
        assert sp[0] == pytest.approx(0.0)
        assert sp[2] == pytest.approx(0.80)
        assert sp[3] == pytest.approx(0.5)
        assert sp[4] == pytest.approx(0.0)


class TestPrecisionState:
    def test_construction(self) -> None:
        ps = PrecisionState(level_0=np.ones(5), level_1=1.0, level_2=1.0)
        assert ps.level_0.shape == (5,)
        assert ps.level_1 == 1.0

    def test_negative_level_0_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PrecisionState(
                level_0=np.array([1.0, -0.1, 1.0, 1.0, 1.0]),
                level_1=1.0,
                level_2=1.0,
            )

    def test_negative_level_1_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PrecisionState(level_0=np.ones(5), level_1=-1.0, level_2=1.0)

    def test_negative_level_2_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PrecisionState(level_0=np.ones(5), level_1=1.0, level_2=0.0)

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            PrecisionState(level_0=np.ones(3), level_1=1.0, level_2=1.0)


class TestPredictionErrors:
    def test_construction(self) -> None:
        pe = PredictionErrors(level_0=np.zeros(5))
        assert pe.level_0.shape == (5,)

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            PredictionErrors(level_0=np.zeros(7))

    def test_allows_negative(self) -> None:
        pe = PredictionErrors(level_0=np.array([-0.5, 0.1, -0.2, 0.0, 0.3]))
        assert pe.level_0[0] == pytest.approx(-0.5)


class TestSnapshots:
    def test_precision_snapshot_from_state(self) -> None:
        ps = PrecisionState(
            level_0=np.array([1.1, 1.2, 1.3, 1.4, 1.5]),
            level_1=2.0,
            level_2=3.0,
        )
        snap = PrecisionSnapshot.from_state(ps)
        assert set(snap.level_0.keys()) == set(INTEROCEPTIVE_KEYS)
        assert snap.level_0["mood"] == pytest.approx(1.1)
        assert snap.level_1 == pytest.approx(2.0)

    def test_prediction_error_snapshot_from_errors(self) -> None:
        pe = PredictionErrors(level_0=np.array([0.1, -0.2, 0.0, 0.3, -0.1]))
        snap = PredictionErrorSnapshot.from_errors(pe)
        assert snap.level_0["arousal"] == pytest.approx(-0.2)


class TestPredictionErrorEngine:
    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.engine = PredictionErrorEngine()

    def test_zero_error_at_setpoint(self) -> None:
        p = _make_personality(self.reg, {"E": 0.5})
        sp = AllostaticSetPoints().compute(p)
        state = AgentState(
            mood=sp[0],
            arousal=float(sp[1]),
            energy=float(sp[2]),
            satisfaction=float(sp[3]),
            frustration=float(sp[4]),
        )
        errors = self.engine.compute(state, p)
        np.testing.assert_array_almost_equal(errors.level_0, np.zeros(5))

    def test_positive_error_above_setpoint(self) -> None:
        p = _make_personality(self.reg)
        state = AgentState(
            mood=0.5,
            arousal=0.9,
            energy=1.0,
            satisfaction=0.8,
            frustration=0.3,
        )
        errors = self.engine.compute(state, p)
        assert errors.level_0[0] > 0
        assert errors.level_0[4] > 0

    def test_output_shape(self) -> None:
        p = _make_personality(self.reg)
        errors = self.engine.compute(AgentState(), p)
        assert errors.level_0.shape == (5,)


class TestPrecisionEngine:
    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.engine = PrecisionEngine()
        self.scenario = _make_scenario(self.reg)

    def test_softplus_always_positive(self) -> None:
        x = np.array([-100, -1, 0, 1, 100])
        result = PrecisionEngine._softplus(x)
        assert np.all(result > 0)

    def test_sigmoid_bounds(self) -> None:
        x = np.array([-100, -1, 0, 1, 100])
        result = PrecisionEngine._sigmoid(x)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_default_produces_positive_precision(self) -> None:
        p = _make_personality(self.reg)
        ps = self.engine.compute(p, AgentState(), self.scenario)
        assert np.all(ps.level_0 > 0)
        assert ps.level_1 > 0
        assert ps.level_2 > 0

    def test_high_N_increases_interoceptive_precision(self) -> None:
        low_n = _make_personality(self.reg, {"N": 0.1})
        high_n = _make_personality(self.reg, {"N": 0.9})
        state = AgentState()
        ps_low = self.engine.compute(low_n, state, self.scenario)
        ps_high = self.engine.compute(high_n, state, self.scenario)
        assert ps_high.level_0[0] > ps_low.level_0[0]

    def test_high_E_increases_policy_precision(self) -> None:
        low_e = _make_personality(self.reg, {"E": 0.1})
        high_e = _make_personality(self.reg, {"E": 0.9})
        state = AgentState()
        ps_low = self.engine.compute(low_e, state, self.scenario)
        ps_high = self.engine.compute(high_e, state, self.scenario)
        assert ps_high.level_1 > ps_low.level_1

    def test_high_T_increases_narrative_precision(self) -> None:
        low_t = _make_personality(self.reg, {"T": 0.1})
        high_t = _make_personality(self.reg, {"T": 0.9})
        state = AgentState()
        ps_low = self.engine.compute(low_t, state, self.scenario)
        ps_high = self.engine.compute(high_t, state, self.scenario)
        assert ps_high.level_2 > ps_low.level_2

    def test_state_modulates_precision(self) -> None:
        p = _make_personality(self.reg)
        s1 = AgentState(
            mood=-0.5,
            arousal=0.2,
            energy=0.3,
            satisfaction=0.1,
            frustration=0.8,
        )
        s2 = AgentState(
            mood=0.5,
            arousal=0.8,
            energy=0.9,
            satisfaction=0.9,
            frustration=0.0,
        )
        ps1 = self.engine.compute(p, s1, self.scenario)
        ps2 = self.engine.compute(p, s2, self.scenario)
        assert not np.allclose(ps1.level_0, ps2.level_0)

    def test_scenario_modulates_precision(self) -> None:
        p = _make_personality(self.reg)
        state = AgentState()
        s1 = _make_scenario(self.reg, {"O": 0.1, "N": 0.1})
        s2 = _make_scenario(self.reg, {"O": 0.9, "N": 0.9})
        ps1 = self.engine.compute(p, state, s1)
        ps2 = self.engine.compute(p, state, s2)
        assert not np.allclose(ps1.level_0, ps2.level_0)

    def test_compute_errors_delegates(self) -> None:
        p = _make_personality(self.reg)
        errors = self.engine.compute_errors(AgentState(), p)
        assert errors.level_0.shape == (5,)


class TestPrecisionIntegration:
    def setup_method(self) -> None:
        self.sdk = AgentSDK.default()
        self.sdk_prec = AgentSDK.with_precision()
        self.personality = self.sdk.personality({k: 0.5 for k in "OCEANRIT"})
        self.scenario = self.sdk.scenario(
            {k: 0.5 for k in "OCEANRIT"},
            name="test",
        )
        self.actions = [
            self.sdk.action("Act", {"O": 0.5, "C": 0.3}),
            self.sdk.action("Wait", {"O": -0.2, "C": 0.1}),
        ]

    def test_tick_without_precision_engine_has_none(self) -> None:
        sim = self.sdk.simulator(self.personality, self.actions)
        rec = sim.tick(self.scenario, outcome=0.5)
        assert rec.precision is None
        assert rec.prediction_errors is None

    def test_tick_with_precision_engine_has_values(self) -> None:
        sim = self.sdk_prec.simulator(self.personality, self.actions)
        rec = sim.tick(self.scenario, outcome=0.5)
        assert rec.precision is not None
        assert rec.prediction_errors is not None
        assert "level_0" in rec.precision
        assert "level_1" in rec.precision
        assert "level_2" in rec.precision

    def test_sdk_with_precision_factory(self) -> None:
        sdk = AgentSDK.with_precision()
        assert sdk._precision_engine is not None
