"""Tests for Phase C1: constructed emotion subsystem."""

import numpy as np
import pytest
from src.constructed_emotion.affect import AffectSignal, ConstructedAffectiveEngine
from src.constructed_emotion.free_energy import (
    compute_arousal_signal,
    compute_free_energy,
    compute_valence,
)
from src.constructed_emotion.params import ConstructedEmotionParams
from src.constructed_emotion.surprise import SurpriseSpikeDetector
from src.precision.state import PrecisionState, PredictionErrors
from src.sdk import AgentSDK


def _make_precision(
    level_0: np.ndarray | None = None,
    level_1: float = 1.0,
    level_2: float = 1.0,
) -> PrecisionState:
    """Build a PrecisionState."""
    return PrecisionState(
        level_0=level_0 if level_0 is not None else np.ones(5),
        level_1=level_1,
        level_2=level_2,
    )


def _make_errors(level_0: np.ndarray | None = None) -> PredictionErrors:
    """Build PredictionErrors."""
    return PredictionErrors(
        level_0=level_0 if level_0 is not None else np.zeros(5),
    )


def _balanced() -> dict[str, float]:
    return {k: 0.5 for k in "OCEANRIT"}


class TestConstructedEmotionParams:
    """ConstructedEmotionParams validation and defaults."""

    def test_defaults(self) -> None:
        p = ConstructedEmotionParams()
        assert p.mood_ema_alpha == pytest.approx(0.90)
        assert p.surprise_window == 50
        assert p.surprise_sigma_min == pytest.approx(0.05)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="not finite"):
            ConstructedEmotionParams(mood_ema_alpha=float("nan"))


class TestFreeEnergy:
    """Free energy computation: F = sum(Pi * eps^2 - ln Pi)."""

    def test_zero_errors_gives_negative_log_precision(self) -> None:
        pi = _make_precision(level_0=np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
        eps = _make_errors(level_0=np.zeros(5))
        f = compute_free_energy(pi, eps)
        expected = -5.0 * np.log(2.0)
        assert f == pytest.approx(expected, abs=1e-8)

    def test_nonzero_errors_increase_free_energy(self) -> None:
        pi = _make_precision()
        eps_zero = _make_errors()
        eps_big = _make_errors(level_0=np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
        f_zero = compute_free_energy(pi, eps_zero)
        f_big = compute_free_energy(pi, eps_big)
        assert f_big > f_zero

    def test_higher_precision_amplifies_large_error(self) -> None:
        """With large errors, higher precision increases total free energy."""
        big_eps = _make_errors(level_0=np.array([2.0, 0.0, 0.0, 0.0, 0.0]))
        pi_low = _make_precision(level_0=np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        pi_high = _make_precision(level_0=np.array([5.0, 1.0, 1.0, 1.0, 1.0]))
        f_low = compute_free_energy(pi_low, big_eps)
        f_high = compute_free_energy(pi_high, big_eps)
        assert f_high > f_low


class TestValence:
    """Valence = F(t-1) - F(t)."""

    def test_decreasing_free_energy_gives_positive_valence(self) -> None:
        assert compute_valence(5.0, 3.0) == pytest.approx(2.0)

    def test_increasing_free_energy_gives_negative_valence(self) -> None:
        assert compute_valence(3.0, 5.0) == pytest.approx(-2.0)

    def test_stable_free_energy_gives_zero_valence(self) -> None:
        assert compute_valence(4.0, 4.0) == pytest.approx(0.0)


class TestArousalSignal:
    """Arousal = ||Pi * eps||."""

    def test_zero_errors_gives_zero_arousal(self) -> None:
        pi = _make_precision()
        eps = _make_errors()
        assert compute_arousal_signal(pi, eps) == pytest.approx(0.0)

    def test_nonzero_errors_gives_positive_arousal(self) -> None:
        pi = _make_precision()
        eps = _make_errors(level_0=np.array([0.5, 0.0, 0.0, 0.0, 0.0]))
        assert compute_arousal_signal(pi, eps) > 0.0

    def test_precision_scales_arousal(self) -> None:
        eps = _make_errors(level_0=np.array([0.5, 0.0, 0.0, 0.0, 0.0]))
        low = compute_arousal_signal(
            _make_precision(level_0=np.ones(5)),
            eps,
        )
        high = compute_arousal_signal(
            _make_precision(level_0=np.array([3.0, 1.0, 1.0, 1.0, 1.0])),
            eps,
        )
        assert high > low


class TestSurpriseSpikeDetector:
    """Adaptive surprise threshold with warmup fallback."""

    def test_warmup_uses_fixed_threshold(self) -> None:
        det = SurpriseSpikeDetector()
        assert not det.is_warmed_up
        assert det.current_threshold() == pytest.approx(0.20)

    def test_below_warmup_threshold_not_spike(self) -> None:
        det = SurpriseSpikeDetector()
        assert det.observe(0.10) is False

    def test_above_warmup_threshold_is_spike(self) -> None:
        det = SurpriseSpikeDetector()
        assert det.observe(0.30) is True

    def test_adaptive_threshold_after_warmup(self) -> None:
        params = ConstructedEmotionParams(surprise_window=5)
        det = SurpriseSpikeDetector(params)
        for _ in range(5):
            det.observe(1.0)
        assert det.is_warmed_up
        threshold = det.current_threshold()
        assert threshold == pytest.approx(1.0 + 2.0 * 0.05, abs=0.01)

    def test_high_variance_raises_threshold(self) -> None:
        params = ConstructedEmotionParams(surprise_window=5)
        det = SurpriseSpikeDetector(params)
        for v in [0.1, 0.9, 0.1, 0.9, 0.1]:
            det.observe(v)
        assert det.current_threshold() > 0.5

    def test_reset_clears_history(self) -> None:
        det = SurpriseSpikeDetector()
        det.observe(1.0)
        det.reset()
        assert det.history == []


class TestConstructedAffectiveEngine:
    """ConstructedAffectiveEngine: System 1 + System 2."""

    def _make_sdk_personality(self) -> tuple[object, object]:
        sdk = AgentSDK.default()
        return sdk, sdk.personality(_balanced())

    def test_first_tick_valence_is_zero(self) -> None:
        engine = ConstructedAffectiveEngine()
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        pi = _make_precision()
        eps = _make_errors(level_0=np.array([0.1, 0.0, 0.0, 0.0, 0.0]))
        sig = engine.process_tick(pi, eps, personality)
        assert sig.valence == pytest.approx(0.0)

    def test_second_tick_has_nonzero_valence(self) -> None:
        engine = ConstructedAffectiveEngine()
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        pi = _make_precision()
        eps1 = _make_errors(level_0=np.array([0.5, 0.0, 0.0, 0.0, 0.0]))
        eps2 = _make_errors(level_0=np.array([0.1, 0.0, 0.0, 0.0, 0.0]))
        engine.process_tick(pi, eps1, personality)
        sig2 = engine.process_tick(pi, eps2, personality)
        assert sig2.valence > 0.0  # free energy decreased

    def test_mood_tracks_valence(self) -> None:
        engine = ConstructedAffectiveEngine()
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        pi = _make_precision()
        eps_big = _make_errors(level_0=np.ones(5) * 0.5)
        eps_small = _make_errors(level_0=np.ones(5) * 0.01)
        engine.process_tick(pi, eps_big, personality)
        sig = engine.process_tick(pi, eps_small, personality)
        assert sig.mood > 0.0  # positive valence -> positive mood

    def test_high_r_faster_mood_recovery(self) -> None:
        sdk = AgentSDK.default()
        low_r = sdk.personality({**_balanced(), "R": 0.1})
        high_r = sdk.personality({**_balanced(), "R": 0.9})
        pi = _make_precision()
        eps = _make_errors(level_0=np.ones(5) * 0.3)

        eng_low = ConstructedAffectiveEngine()
        eng_high = ConstructedAffectiveEngine()

        for _ in range(5):
            eng_low.process_tick(pi, eps, low_r)
            eng_high.process_tick(pi, eps, high_r)

        eps_zero = _make_errors()
        for _ in range(10):
            sig_low = eng_low.process_tick(pi, eps_zero, low_r)
            sig_high = eng_high.process_tick(pi, eps_zero, high_r)

        assert abs(sig_high.mood) < abs(sig_low.mood)

    def test_surprise_spike_produces_emotions(self) -> None:
        params = ConstructedEmotionParams(surprise_warmup_threshold=0.01)
        engine = ConstructedAffectiveEngine(params)
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        pi = _make_precision(level_0=np.ones(5) * 5.0)
        eps_big = _make_errors(level_0=np.ones(5) * 0.5)
        eps_small = _make_errors(level_0=np.ones(5) * 0.01)
        # First tick establishes baseline free energy
        engine.process_tick(pi, eps_big, personality)
        # Second tick: free energy drops -> positive valence, high arousal
        sig = engine.process_tick(pi, eps_small, personality)
        assert sig.is_surprise_spike
        assert sig.valence > 0.0
        assert len(sig.constructed_emotions) > 0

    def test_no_spike_no_emotions(self) -> None:
        engine = ConstructedAffectiveEngine()
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        pi = _make_precision()
        eps = _make_errors()
        sig = engine.process_tick(pi, eps, personality)
        assert not sig.is_surprise_spike
        assert sig.constructed_emotions == []

    def test_affect_signal_fields(self) -> None:
        engine = ConstructedAffectiveEngine()
        sdk = AgentSDK.default()
        personality = sdk.personality(_balanced())
        pi = _make_precision()
        eps = _make_errors(level_0=np.array([0.1, 0.0, 0.0, 0.0, 0.0]))
        sig = engine.process_tick(pi, eps, personality)
        assert isinstance(sig, AffectSignal)
        assert isinstance(sig.free_energy, float)
        assert isinstance(sig.arousal_signal, float)
        assert isinstance(sig.mood, float)


class TestConstructedEmotionIntegration:
    """Integration with SDK and simulator."""

    def test_sdk_with_constructed_emotion_factory(self) -> None:
        sdk = AgentSDK.with_constructed_emotion()
        assert sdk._emotion_params is not None

    def test_simulator_tick_includes_affect_signal(self) -> None:
        sdk = AgentSDK.with_constructed_emotion()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        actions = [
            sdk.action("Act", {"O": 0.3, "E": 0.2}),
            sdk.action("Wait", {"O": -0.1}),
        ]
        sim = sdk.simulator(personality, actions, rng=np.random.default_rng(42))
        rec = sim.tick(scenario)
        assert rec.affect_signal is not None

    def test_multi_tick_valence_varies(self) -> None:
        sdk = AgentSDK.with_constructed_emotion()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        actions = [
            sdk.action("Act", {"O": 0.3, "E": 0.2}),
            sdk.action("Wait", {"O": -0.1}),
        ]
        sim = sdk.simulator(personality, actions, rng=np.random.default_rng(42))
        valences = []
        for _ in range(20):
            rec = sim.tick(scenario)
            if rec.affect_signal is not None:
                valences.append(rec.affect_signal["valence"])
        assert len(set(round(v, 6) for v in valences)) > 1

    def test_backward_compat_no_affect_without_engine(self) -> None:
        sdk = AgentSDK.with_precision()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        actions = [sdk.action("Act", {"O": 0.3})]
        sim = sdk.simulator(personality, actions, rng=np.random.default_rng(0))
        rec = sim.tick(scenario)
        assert rec.affect_signal is None
