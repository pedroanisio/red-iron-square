"""Unit tests for demo state_mapper helpers.

Covers persona-specific seeds, outcome modulation, and snapshot updates.
"""

from __future__ import annotations

import pytest
from src.demo.models import DemoPersona
from src.demo.personas import DEFAULT_PERSONAS
from src.demo.state_mapper import (
    build_initial_agents,
    build_run_config,
    derive_initial_affect,
    modulate_outcome,
)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _persona(key: str, **trait_overrides: float) -> DemoPersona:
    """Build a test persona with optional trait overrides."""
    base_traits = {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5, "R": 0.5}
    base_traits.update(trait_overrides)
    return DemoPersona(key=key, name=key.title(), summary="test", traits=base_traits)


# ---------------------------------------------------------------------------
# build_run_config
# ---------------------------------------------------------------------------


class TestBuildRunConfig:
    """Seed must differ per persona key so RNG diverges."""

    def test_different_personas_get_different_seeds(self) -> None:
        luna_cfg = build_run_config(DEFAULT_PERSONAS["luna"])
        marco_cfg = build_run_config(DEFAULT_PERSONAS["marco"])
        assert luna_cfg["seed"] != marco_cfg["seed"]

    def test_same_persona_gets_deterministic_seed(self) -> None:
        a = build_run_config(DEFAULT_PERSONAS["luna"])
        b = build_run_config(DEFAULT_PERSONAS["luna"])
        assert a["seed"] == b["seed"]

    def test_seed_is_non_negative_int(self) -> None:
        cfg = build_run_config(DEFAULT_PERSONAS["luna"])
        assert isinstance(cfg["seed"], int)
        assert cfg["seed"] >= 0


# ---------------------------------------------------------------------------
# modulate_outcome
# ---------------------------------------------------------------------------


class TestModulateOutcome:
    """Personality-driven outcome modulation."""

    def test_high_resilience_shifts_outcome_positively(self) -> None:
        resilient = _persona("r", R=0.9, N=0.2)
        neurotic = _persona("n", R=0.2, N=0.9)
        scenario_values = {"O": 0.5, "C": 0.5, "E": 0.5, "N": 0.5, "R": 0.5}
        r_out = modulate_outcome(-0.3, resilient, scenario_values)
        n_out = modulate_outcome(-0.3, neurotic, scenario_values)
        assert r_out > n_out

    def test_aligned_persona_benefits_from_scenario(self) -> None:
        aligned = _persona("a", O=0.9, E=0.9)
        misaligned = _persona("m", O=0.1, E=0.1)
        scenario_values = {"O": 0.9, "E": 0.9}
        a_out = modulate_outcome(0.3, aligned, scenario_values)
        m_out = modulate_outcome(0.3, misaligned, scenario_values)
        assert a_out > m_out

    def test_output_clamped_to_valid_range(self) -> None:
        persona = _persona("extreme", R=1.0, N=0.0, O=1.0, E=1.0)
        scenario_values = {"O": 1.0, "E": 1.0, "R": 1.0, "N": 1.0}
        result = modulate_outcome(0.95, persona, scenario_values)
        assert -1.0 <= result <= 1.0

    def test_negative_clamping(self) -> None:
        persona = _persona("fragile", R=0.0, N=1.0, O=0.0, E=0.0)
        scenario_values = {"O": 0.0, "E": 0.0, "R": 0.0, "N": 0.0}
        result = modulate_outcome(-0.95, persona, scenario_values)
        assert -1.0 <= result <= 1.0

    def test_no_shared_keys_returns_base(self) -> None:
        persona = _persona("iso")
        result = modulate_outcome(0.5, persona, {"X": 0.9, "Y": 0.1})
        assert result == pytest.approx(0.5)

    def test_luna_marco_diverge_on_promotion(self) -> None:
        """The exact scenario from the screenshot: both got 0.3."""
        scenario_values = {
            "O": 0.8,
            "C": 0.5,
            "E": 0.6,
            "A": 0.4,
            "N": 0.6,
            "R": 0.5,
            "I": 0.7,
            "T": 0.6,
        }
        luna_out = modulate_outcome(0.3, DEFAULT_PERSONAS["luna"], scenario_values)
        marco_out = modulate_outcome(0.3, DEFAULT_PERSONAS["marco"], scenario_values)
        assert luna_out != pytest.approx(marco_out, abs=1e-6)


# ---------------------------------------------------------------------------
# derive_initial_affect / build_initial_agents
# ---------------------------------------------------------------------------


class TestDeriveInitialAffect:
    """Initial affect must differ between distinct personality profiles."""

    def test_luna_and_marco_start_different(self) -> None:
        luna = derive_initial_affect(DEFAULT_PERSONAS["luna"].traits)
        marco = derive_initial_affect(DEFAULT_PERSONAS["marco"].traits)
        assert luna != marco

    def test_high_extraversion_raises_energy(self) -> None:
        low_e = derive_initial_affect({"E": 0.1, "N": 0.5, "R": 0.5})
        high_e = derive_initial_affect({"E": 0.9, "N": 0.5, "R": 0.5})
        assert high_e["energy"] > low_e["energy"]

    def test_high_neuroticism_lowers_mood(self) -> None:
        low_n = derive_initial_affect({"E": 0.5, "N": 0.1, "R": 0.5})
        high_n = derive_initial_affect({"E": 0.5, "N": 0.9, "R": 0.5})
        assert low_n["mood"] > high_n["mood"]

    def test_high_resilience_raises_calm(self) -> None:
        low_r = derive_initial_affect({"E": 0.5, "N": 0.5, "R": 0.1})
        high_r = derive_initial_affect({"E": 0.5, "N": 0.5, "R": 0.9})
        assert high_r["calm"] > low_r["calm"]

    def test_build_initial_agents_applies_affect(self) -> None:
        agents = build_initial_agents(DEFAULT_PERSONAS)
        assert agents["luna"].mood != agents["marco"].mood
        assert agents["luna"].energy != agents["marco"].energy
