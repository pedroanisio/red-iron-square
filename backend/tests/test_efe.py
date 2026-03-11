"""Tests for the EFE bounded context."""

import numpy as np
import pytest
from src.efe.c_vector import N_INTEROCEPTIVE, CVector
from src.efe.engine import EFEEngine
from src.efe.epistemic import compute_epistemic_value
from src.efe.params import EFEParams
from src.personality.decision import DecisionEngine
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import PersonalityVector, Scenario
from src.sdk import AgentSDK
from src.temporal.memory import MemoryBank, MemoryEntry
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


def _make_memory_bank(
    action_outcomes: dict[str, list[float]],
) -> MemoryBank:
    """Build a MemoryBank from action -> outcomes mapping."""
    bank = MemoryBank()
    tick = 0
    for action_name, outcomes in action_outcomes.items():
        for outcome in outcomes:
            bank.store(
                MemoryEntry(
                    tick=tick,
                    scenario_name="test",
                    action_name=action_name,
                    outcome=outcome,
                    counterfactual=0.0,
                    state_snapshot=AgentState(),
                    valence=outcome * 0.5,
                )
            )
            tick += 1
    return bank


class TestEFEParams:
    def test_defaults(self) -> None:
        p = EFEParams()
        assert p.w_base == pytest.approx(0.5)
        assert p.n_bins == 5
        assert p.memory_window == 50

    def test_inf_raises(self) -> None:
        with pytest.raises(Exception):
            EFEParams(w_base=float("inf"))

    def test_nan_raises(self) -> None:
        with pytest.raises(Exception):
            EFEParams(kappa_mood=float("nan"))


class TestCVector:
    def setup_method(self) -> None:
        self.reg = DimensionRegistry()

    def test_output_shape(self) -> None:
        psi = _make_personality(self.reg)
        cv = CVector(psi)
        assert cv.log_preferences.shape == (N_INTEROCEPTIVE, 5)
        assert cv.preferences.shape == (N_INTEROCEPTIVE, 5)

    def test_preferences_are_normalized(self) -> None:
        psi = _make_personality(self.reg)
        cv = CVector(psi)
        row_sums = cv.preferences.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(N_INTEROCEPTIVE))

    def test_high_N_skews_mood_negative(self) -> None:
        """High-N agents have steeper aversion to negative mood bins."""
        low_n = CVector(_make_personality(self.reg, {"N": 0.1}))
        high_n = CVector(_make_personality(self.reg, {"N": 0.9}))
        low_ratio = low_n.preferences[0, 4] / low_n.preferences[0, 0]
        high_ratio = high_n.preferences[0, 4] / high_n.preferences[0, 0]
        assert high_ratio > low_ratio

    def test_high_E_prefers_higher_arousal(self) -> None:
        """High-E agents prefer higher arousal bins."""
        low_e = CVector(_make_personality(self.reg, {"E": 0.1}))
        high_e = CVector(_make_personality(self.reg, {"E": 0.9}))
        low_peak = np.argmax(low_e.preferences[1])
        high_peak = np.argmax(high_e.preferences[1])
        assert high_peak > low_peak

    def test_high_C_prefers_stable_energy(self) -> None:
        """High-C agents have a bonus for the stable energy bin."""
        low_c = CVector(_make_personality(self.reg, {"C": 0.1}))
        high_c = CVector(_make_personality(self.reg, {"C": 0.9}))
        mid = 2
        assert high_c.preferences[2, mid] > low_c.preferences[2, mid]

    def test_high_A_stronger_satisfaction_gradient(self) -> None:
        """High-A agents have stronger preference for positive satisfaction."""
        low_a = CVector(_make_personality(self.reg, {"A": 0.1}))
        high_a = CVector(_make_personality(self.reg, {"A": 0.9}))
        low_ratio = low_a.preferences[3, 4] / low_a.preferences[3, 0]
        high_ratio = high_a.preferences[3, 4] / high_a.preferences[3, 0]
        assert high_ratio > low_ratio

    def test_high_R_tolerates_frustration(self) -> None:
        """High-R agents have flatter frustration preference distributions."""
        low_r = CVector(_make_personality(self.reg, {"R": 0.1}))
        high_r = CVector(_make_personality(self.reg, {"R": 0.9}))
        low_std = np.std(low_r.preferences[4])
        high_std = np.std(high_r.preferences[4])
        assert high_std < low_std

    def test_copy_semantics(self) -> None:
        psi = _make_personality(self.reg)
        cv = CVector(psi)
        lp1 = cv.log_preferences
        lp2 = cv.log_preferences
        assert lp1 is not lp2


class TestEpistemicValue:
    def test_default_when_no_memory(self) -> None:
        bank = MemoryBank()
        val = compute_epistemic_value("Act", bank, default=1.5)
        assert val == pytest.approx(1.5)

    def test_default_when_single_entry(self) -> None:
        bank = _make_memory_bank({"Act": [0.5]})
        val = compute_epistemic_value("Act", bank, default=1.0)
        assert val == pytest.approx(1.0)

    def test_zero_variance_for_constant_outcomes(self) -> None:
        bank = _make_memory_bank({"Act": [0.5, 0.5, 0.5]})
        val = compute_epistemic_value("Act", bank)
        assert val == pytest.approx(0.0)

    def test_positive_variance_for_varied_outcomes(self) -> None:
        bank = _make_memory_bank({"Act": [0.0, 1.0, 0.0, 1.0]})
        val = compute_epistemic_value("Act", bank)
        assert val > 0.0

    def test_filters_by_action(self) -> None:
        bank = _make_memory_bank({"Act": [0.5, 0.5], "Wait": [0.0, 1.0]})
        val_act = compute_epistemic_value("Act", bank)
        val_wait = compute_epistemic_value("Wait", bank)
        assert val_act < val_wait

    def test_unknown_action_returns_default(self) -> None:
        bank = _make_memory_bank({"Act": [0.5, 0.5]})
        val = compute_epistemic_value("Unknown", bank, default=2.0)
        assert val == pytest.approx(2.0)


class TestMemoryBankActionVariance:
    def test_returns_none_when_insufficient(self) -> None:
        bank = _make_memory_bank({"Act": [0.5]})
        assert bank.action_outcome_variance("Act") is None

    def test_returns_none_for_unknown(self) -> None:
        bank = _make_memory_bank({"Act": [0.5, 0.5]})
        assert bank.action_outcome_variance("Unknown") is None

    def test_correct_variance(self) -> None:
        bank = _make_memory_bank({"Act": [0.0, 1.0]})
        var = bank.action_outcome_variance("Act")
        assert var is not None
        assert var == pytest.approx(0.25)


class TestEFEEngine:
    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.base_engine = DecisionEngine(registry=self.reg)
        self.personality = _make_personality(self.reg)
        self.scenario = _make_scenario(self.reg)
        self.engine = EFEEngine(self.base_engine, self.personality)

    def test_compute_activations_delegates(self) -> None:
        act_base = self.base_engine.compute_activations(self.personality, self.scenario)
        act_efe = self.engine.compute_activations(self.personality, self.scenario)
        np.testing.assert_array_almost_equal(act_base, act_efe)

    def test_utility_returns_finite(self) -> None:
        action = AgentSDK().action("Act", {"O": 0.5, "C": 0.3})
        u = self.engine.utility(self.personality, self.scenario, action)
        assert np.isfinite(u)

    def test_decide_returns_action_and_probs(self) -> None:
        actions = [
            AgentSDK().action("Act", {"O": 0.5, "C": 0.3}),
            AgentSDK().action("Wait", {"O": -0.2, "C": 0.1}),
        ]
        chosen, probs = self.engine.decide(
            self.personality,
            self.scenario,
            actions,
            rng=np.random.default_rng(42),
        )
        assert chosen.name in {"Act", "Wait"}
        assert probs.shape == (2,)
        assert pytest.approx(probs.sum()) == 1.0

    def test_high_O_increases_exploration(self) -> None:
        """High-O agents weight epistemic value more, flatter dists."""
        actions = [
            AgentSDK().action("Explore", {"O": 0.8, "C": 0.1}),
            AgentSDK().action("Exploit", {"O": 0.1, "C": 0.8}),
        ]
        memory = _make_memory_bank(
            {
                "Explore": [0.0, 1.0, -0.5, 0.8, -0.3],
                "Exploit": [0.5, 0.5, 0.5, 0.5, 0.5],
            }
        )
        low_o = EFEEngine(
            self.base_engine,
            _make_personality(self.reg, {"O": 0.1}),
        )
        high_o = EFEEngine(
            self.base_engine,
            _make_personality(self.reg, {"O": 0.9}),
        )
        low_o.bind_memory(memory)
        high_o.bind_memory(memory)

        psi_lo = _make_personality(self.reg, {"O": 0.1})
        psi_hi = _make_personality(self.reg, {"O": 0.9})
        u_lo_explore = low_o.utility(psi_lo, self.scenario, actions[0])
        u_lo_exploit = low_o.utility(psi_lo, self.scenario, actions[1])
        u_hi_explore = high_o.utility(psi_hi, self.scenario, actions[0])
        u_hi_exploit = high_o.utility(psi_hi, self.scenario, actions[1])

        diff_lo = abs(u_lo_explore - u_lo_exploit)
        diff_hi = abs(u_hi_explore - u_hi_exploit)
        assert diff_hi != pytest.approx(diff_lo, abs=1e-6)

    def test_memory_binding(self) -> None:
        """Epistemic value changes after binding memory."""
        action = AgentSDK().action("Act", {"O": 0.5, "C": 0.3})
        u_no_mem = self.engine.utility(self.personality, self.scenario, action)

        memory = _make_memory_bank({"Act": [0.5, 0.5, 0.5]})
        self.engine.bind_memory(memory)
        u_with_mem = self.engine.utility(self.personality, self.scenario, action)
        assert u_no_mem != pytest.approx(u_with_mem)

    def test_registry_exposed(self) -> None:
        assert self.engine.registry is self.reg

    def test_zero_temperature_raises(self) -> None:
        actions = [AgentSDK().action("Act", {"O": 0.5})]
        with pytest.raises(ValueError, match="temperature"):
            self.engine.decide(
                self.personality,
                self.scenario,
                actions,
                temperature=0.0,
            )

    def test_empty_actions_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            self.engine.decide(self.personality, self.scenario, [])


class TestEFEIntegration:
    def setup_method(self) -> None:
        self.sdk = AgentSDK.with_efe()
        self.personality = self.sdk.personality({k: 0.5 for k in "OCEANRIT"})
        self.scenario = self.sdk.scenario(
            {k: 0.5 for k in "OCEANRIT"},
            name="test",
        )
        self.actions = [
            self.sdk.action("Act", {"O": 0.5, "C": 0.3}),
            self.sdk.action("Wait", {"O": -0.2, "C": 0.1}),
        ]

    def test_sdk_with_efe_factory(self) -> None:
        sdk = AgentSDK.with_efe()
        assert sdk._efe_params is not None
        assert sdk._precision_engine is not None

    def test_tick_produces_precision(self) -> None:
        sim = self.sdk.simulator(self.personality, self.actions)
        rec = sim.tick(self.scenario, outcome=0.5)
        assert rec.precision is not None
        assert rec.prediction_errors is not None

    def test_multi_tick_runs(self) -> None:
        sim = self.sdk.simulator(
            self.personality,
            self.actions,
            rng=np.random.default_rng(42),
        )
        for _ in range(20):
            rec = sim.tick(self.scenario)
            assert rec.precision is not None
            assert isinstance(rec.action, str)

    def test_efe_uses_memory(self) -> None:
        """After several ticks, epistemic value reflects accumulated memory."""
        sim = self.sdk.simulator(
            self.personality,
            self.actions,
            rng=np.random.default_rng(0),
        )
        for _ in range(30):
            sim.tick(self.scenario)
        assert len(sim.simulator.memory) == 30

    def test_existing_tests_unaffected(self) -> None:
        """Default SDK still uses DecisionEngine."""
        sdk = AgentSDK.default()
        assert sdk._efe_params is None
        sim = sdk.simulator(self.personality, self.actions)
        rec = sim.tick(self.scenario, outcome=0.5)
        assert rec.precision is None
