"""Tests for precision-weighted state transitions (§9)."""

import numpy as np
import pytest
from src.precision.state import PrecisionState, PredictionErrors
from src.sdk import AgentSDK
from src.temporal.precision_state import PrecisionStateParams, update_state_precision
from src.temporal.state import AgentState


def _balanced() -> dict[str, float]:
    return {k: 0.5 for k in "OCEANRIT"}


def _make_precision(
    level_0: np.ndarray | None = None,
    level_1: float = 1.0,
    level_2: float = 1.0,
) -> PrecisionState:
    return PrecisionState(
        level_0=level_0 if level_0 is not None else np.ones(5),
        level_1=level_1,
        level_2=level_2,
    )


def _make_errors(level_0: np.ndarray | None = None) -> PredictionErrors:
    return PredictionErrors(
        level_0=level_0 if level_0 is not None else np.zeros(5),
    )


class TestPrecisionStateParams:
    """PrecisionStateParams validation."""

    def test_defaults(self) -> None:
        p = PrecisionStateParams()
        assert p.decay.shape == (5,)
        assert p.setpoint.shape == (5,)
        assert p.precision_gain == pytest.approx(0.05)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            PrecisionStateParams(decay=np.ones(3))


class TestUpdateStatePrecision:
    """Precision-weighted state transition function."""

    def test_zero_errors_decays_toward_setpoint(self) -> None:
        """With no prediction errors or outcome, state decays to setpoint."""
        state = AgentState(
            mood=0.5,
            arousal=0.8,
            energy=0.9,
            satisfaction=0.7,
            frustration=0.3,
        )
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        pi = _make_precision()
        eps = _make_errors()

        new_state = update_state_precision(state, 0.0, personality, scenario, pi, eps)
        # Should decay toward setpoints
        assert abs(new_state.mood) < abs(state.mood)
        assert new_state.frustration < state.frustration

    def test_positive_outcome_improves_mood(self) -> None:
        state = AgentState()
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        pi = _make_precision()
        eps = _make_errors()

        new_state = update_state_precision(state, 0.8, personality, scenario, pi, eps)
        assert new_state.mood > state.mood

    def test_negative_outcome_increases_frustration(self) -> None:
        state = AgentState()
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        pi = _make_precision()
        eps = _make_errors()

        new_state = update_state_precision(state, -0.8, personality, scenario, pi, eps)
        assert new_state.frustration > state.frustration

    def test_precision_amplifies_error_correction(self) -> None:
        """Higher precision drives stronger state correction."""
        state = AgentState(
            mood=0.0,
            arousal=0.5,
            energy=0.8,
            satisfaction=0.5,
            frustration=0.0,
        )
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        eps = _make_errors(level_0=np.array([0.3, 0.2, 0.0, 0.0, 0.0]))

        low_pi = _make_precision(level_0=np.ones(5))
        high_pi = _make_precision(level_0=np.ones(5) * 5.0)

        state_low = update_state_precision(
            state, 0.0, personality, scenario, low_pi, eps
        )
        state_high = update_state_precision(
            state, 0.0, personality, scenario, high_pi, eps
        )
        # Higher precision -> larger mood correction
        # (more negative due to positive error)
        assert state_high.mood < state_low.mood

    def test_state_stays_bounded(self) -> None:
        """State values remain in valid ranges even with extreme inputs."""
        state = AgentState(
            mood=0.9,
            arousal=0.95,
            energy=0.1,
            satisfaction=0.9,
            frustration=0.9,
        )
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        pi = _make_precision(level_0=np.ones(5) * 10.0)
        eps = _make_errors(level_0=np.ones(5) * 2.0)

        new_state = update_state_precision(state, 1.0, personality, scenario, pi, eps)
        assert -1.0 <= new_state.mood <= 1.0
        assert 0.0 <= new_state.arousal <= 1.0
        assert 0.0 <= new_state.energy <= 1.0
        assert 0.0 <= new_state.satisfaction <= 1.0
        assert 0.0 <= new_state.frustration <= 1.0

    def test_action_effort_costs_energy(self) -> None:
        state = AgentState(energy=0.5)
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        pi = _make_precision()
        eps = _make_errors()

        no_effort = update_state_precision(
            state, 0.0, personality, scenario, pi, eps, action_effort=0.0
        )
        with_effort = update_state_precision(
            state, 0.0, personality, scenario, pi, eps, action_effort=5.0
        )
        assert with_effort.energy < no_effort.energy


class TestPrecisionStateIntegration:
    """Integration: personality-dependent state evolution via precision."""

    def test_high_precision_amplifies_state_correction_over_time(self) -> None:
        """Higher precision drives larger cumulative state changes."""
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")

        state = AgentState(mood=0.0)
        # Persistent positive mood error
        eps = _make_errors(level_0=np.array([0.3, 0.0, 0.0, 0.0, 0.0]))

        low_pi = _make_precision(level_0=np.ones(5))
        high_pi = _make_precision(level_0=np.ones(5) * 5.0)

        s_low = state
        s_high = state
        for _ in range(10):
            s_low = update_state_precision(
                s_low, 0.0, personality, scenario, low_pi, eps
            )
            s_high = update_state_precision(
                s_high, 0.0, personality, scenario, high_pi, eps
            )

        # High precision drives mood more negative (correcting the positive error)
        assert s_high.mood < s_low.mood

    def test_simulator_uses_precision_state_when_flagged(self) -> None:
        """TemporalSimulator uses precision-weighted transitions when enabled."""
        from src.temporal.simulator import TemporalSimulator

        sdk = AgentSDK.with_precision()
        profile = _balanced()
        personality = sdk.personality(profile)
        scenario = sdk.scenario(profile, name="test")
        actions = [sdk.action("Act", {"O": 0.3})]

        sim_prec = TemporalSimulator(
            personality,
            actions,
            sdk._resolve_engine(personality),
            precision_engine=sdk._precision_engine,
            use_precision_state=True,
            rng=np.random.default_rng(42),
        )
        for _ in range(10):
            rec = sim_prec.tick(scenario)
        # Should complete without error and produce valid state
        assert -1.0 <= rec.state_after.mood <= 1.0
        assert 0.0 <= rec.state_after.energy <= 1.0
