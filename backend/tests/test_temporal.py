"""Tests for the temporal bounded context."""

import numpy as np
import pytest
from src.personality.decision import DecisionEngine
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action, PersonalityVector, Scenario
from src.temporal.affective_engine import AffectiveEngine
from src.temporal.emotions import EmotionLabel, EmotionReading
from src.temporal.generators import (
    generate_outcome_sequence,
    generate_scenario_sequence,
)
from src.temporal.memory import MemoryBank, MemoryEntry
from src.temporal.simulator import TemporalSimulator, TickResult
from src.temporal.state import AgentState, StateTransitionParams, update_state


class TestAgentState:
    """Tests for AgentState."""

    def test_default_state(self) -> None:
        s = AgentState()
        assert s.mood == 0.0
        assert s.energy == 1.0

    def test_mood_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            AgentState(mood=1.5)

    def test_energy_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            AgentState(energy=-0.1)

    def test_to_array(self) -> None:
        s = AgentState(mood=0.5, arousal=0.3)
        arr = s.to_array()
        assert arr[0] == pytest.approx(0.5)
        assert arr[1] == pytest.approx(0.3)

    def test_snapshot_is_independent(self) -> None:
        s = AgentState(mood=0.5)
        s2 = s.snapshot()
        assert s2.mood == pytest.approx(0.5)


class TestStateTransitionParams:
    """Tests for StateTransitionParams."""

    def test_defaults_are_valid(self) -> None:
        p = StateTransitionParams()
        assert p.mood_decay == pytest.approx(0.92)

    def test_inf_raises(self) -> None:
        with pytest.raises(Exception):
            StateTransitionParams(mood_decay=float("inf"))


class TestUpdateState:
    """Tests for update_state."""

    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.psi = PersonalityVector(
            values={
                "O": 0.5,
                "C": 0.5,
                "E": 0.5,
                "A": 0.5,
                "N": 0.5,
                "R": 0.5,
                "I": 0.5,
                "T": 0.5,
            },
            registry=self.reg,
        )
        self.scenario = Scenario(
            values={"N": 0.5},
            registry=self.reg,
            name="test",
        )

    def test_positive_outcome_improves_mood(self) -> None:
        state = AgentState(mood=0.0)
        new = update_state(
            state, outcome=0.8, personality=self.psi, scenario=self.scenario
        )
        assert new.mood > 0.0

    def test_negative_outcome_worsens_mood(self) -> None:
        state = AgentState(mood=0.0)
        new = update_state(
            state, outcome=-0.8, personality=self.psi, scenario=self.scenario
        )
        assert new.mood < 0.0

    def test_negative_outcome_increases_frustration(self) -> None:
        state = AgentState(frustration=0.0)
        new = update_state(
            state, outcome=-0.8, personality=self.psi, scenario=self.scenario
        )
        assert new.frustration > 0.0

    def test_state_stays_in_bounds(self) -> None:
        state = AgentState(mood=-0.9, energy=0.1, frustration=0.9)
        new = update_state(
            state, outcome=-1.0, personality=self.psi, scenario=self.scenario
        )
        assert -1.0 <= new.mood <= 1.0
        assert 0.0 <= new.energy <= 1.0
        assert 0.0 <= new.frustration <= 1.0


class TestMemoryBank:
    """Tests for MemoryBank."""

    def _make_entry(
        self, tick: int, outcome: float, valence: float = 0.0
    ) -> MemoryEntry:
        return MemoryEntry(
            tick=tick,
            scenario_name="s",
            action_name="a",
            outcome=outcome,
            counterfactual=0.0,
            state_snapshot=AgentState(),
            valence=valence,
        )

    def test_store_and_len(self) -> None:
        bank = MemoryBank()
        bank.store(self._make_entry(0, 0.5))
        assert len(bank) == 1

    def test_recent_order(self) -> None:
        bank = MemoryBank()
        for i in range(5):
            bank.store(self._make_entry(i, float(i)))
        recent = bank.recent(3)
        assert recent[0].tick == 4
        assert recent[2].tick == 2

    def test_mean_outcome(self) -> None:
        bank = MemoryBank()
        for i in range(5):
            bank.store(self._make_entry(i, 0.5))
        assert bank.mean_outcome(5) == pytest.approx(0.5)

    def test_consecutive_failures(self) -> None:
        bank = MemoryBank()
        bank.store(self._make_entry(0, 0.5))
        bank.store(self._make_entry(1, -0.3))
        bank.store(self._make_entry(2, -0.5))
        assert bank.consecutive_failures() == 2

    def test_total_regret(self) -> None:
        bank = MemoryBank()
        entry = MemoryEntry(
            tick=0,
            scenario_name="s",
            action_name="a",
            outcome=0.0,
            counterfactual=0.5,
            state_snapshot=AgentState(),
            valence=0.0,
        )
        bank.store(entry)
        assert bank.total_regret(10) == pytest.approx(0.5)

    def test_peak_valence(self) -> None:
        bank = MemoryBank()
        bank.store(self._make_entry(0, 0.0, valence=0.3))
        bank.store(self._make_entry(1, 0.0, valence=0.8))
        bank.store(self._make_entry(2, 0.0, valence=0.1))
        assert bank.peak_valence(10) == pytest.approx(0.8)

    def test_max_size_eviction(self) -> None:
        bank = MemoryBank(max_size=3)
        for i in range(5):
            bank.store(self._make_entry(i, 0.0))
        assert len(bank) == 3


class TestAffectiveEngine:
    """Tests for AffectiveEngine."""

    def setup_method(self) -> None:
        self.reg = DimensionRegistry()
        self.psi = PersonalityVector(
            values={
                "O": 0.8,
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
        self.engine = AffectiveEngine()

    def test_detect_all_returns_list(self) -> None:
        state = AgentState(mood=0.5, arousal=0.8)
        acts = np.full(8, 0.5)
        memory = MemoryBank()
        result = self.engine.detect_all(acts, state, self.psi, memory, self.reg)
        assert isinstance(result, list)
        for r in result:
            assert isinstance(r, EmotionReading)
            assert r.intensity >= 0.15

    def test_excitement_positive_mood_high_arousal(self) -> None:
        state = AgentState(mood=0.8, arousal=0.9)
        acts = np.full(8, 0.5)
        result = self.engine.detect_all(acts, state, self.psi, MemoryBank(), self.reg)
        labels = {r.label for r in result}
        assert EmotionLabel.EXCITEMENT in labels

    def test_no_emotions_in_neutral_state(self) -> None:
        state = AgentState(mood=0.0, arousal=0.4, energy=0.5)
        acts = np.full(8, 0.3)
        result = self.engine.detect_all(acts, state, self.psi, MemoryBank(), self.reg)
        intensities = [r.intensity for r in result]
        assert all(i < 0.5 for i in intensities)


class TestTemporalSimulator:
    """Tests for TemporalSimulator."""

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

    def test_single_tick(self) -> None:
        sim = TemporalSimulator(
            self.psi,
            self.actions,
            self.engine,
            rng=np.random.default_rng(42),
        )
        scenario = Scenario(
            values={"O": 0.5, "N": 0.3},
            registry=self.reg,
            name="t0",
        )
        result = sim.tick(scenario, outcome=0.5)
        assert isinstance(result, TickResult)
        assert result.tick == 0
        assert result.action in self.actions
        assert result.outcome == pytest.approx(0.5)

    def test_tick_counter_increments(self) -> None:
        sim = TemporalSimulator(
            self.psi,
            self.actions,
            self.engine,
            rng=np.random.default_rng(42),
        )
        scenario = Scenario(values={"O": 0.5}, registry=self.reg, name="t")
        sim.tick(scenario, outcome=0.3)
        sim.tick(scenario, outcome=0.3)
        assert sim.tick_count == 2

    def test_state_evolves(self) -> None:
        sim = TemporalSimulator(
            self.psi,
            self.actions,
            self.engine,
            rng=np.random.default_rng(42),
        )
        initial_energy = sim.current_state.energy
        scenario = Scenario(
            values={"O": 0.9, "N": 0.9},
            registry=self.reg,
            name="stress",
        )
        for _ in range(10):
            sim.tick(scenario, outcome=-0.5)
        assert sim.current_state.energy < initial_energy

    def test_memory_grows(self) -> None:
        sim = TemporalSimulator(
            self.psi,
            self.actions,
            self.engine,
            rng=np.random.default_rng(42),
        )
        scenario = Scenario(values={"O": 0.5}, registry=self.reg, name="t")
        for _ in range(5):
            sim.tick(scenario, outcome=0.3)
        assert len(sim.memory) == 5

    def test_stochastic_outcome(self) -> None:
        sim = TemporalSimulator(
            self.psi,
            self.actions,
            self.engine,
            rng=np.random.default_rng(42),
        )
        scenario = Scenario(values={"O": 0.5}, registry=self.reg, name="t")
        result = sim.tick(scenario)
        assert -1.0 <= result.outcome <= 1.0


class TestGenerators:
    """Tests for scenario and outcome generators."""

    def test_scenario_sequence_length(self) -> None:
        reg = DimensionRegistry()
        scenarios = generate_scenario_sequence(reg, 20, pattern="crisis_recovery")
        assert len(scenarios) == 20

    def test_outcome_sequence_length(self) -> None:
        outcomes = generate_outcome_sequence(20, pattern="stable")
        assert len(outcomes) == 20

    def test_outcomes_in_range(self) -> None:
        outcomes = generate_outcome_sequence(100, pattern="random")
        assert all(-1.0 <= o <= 1.0 for o in outcomes)

    def test_all_patterns_work(self) -> None:
        reg = DimensionRegistry()
        for pattern in ("stable", "crisis_recovery", "monotony", "loss", "random"):
            scenarios = generate_scenario_sequence(reg, 10, pattern=pattern)
            assert len(scenarios) == 10
            outcomes = generate_outcome_sequence(10, pattern=pattern)
            assert len(outcomes) == 10
