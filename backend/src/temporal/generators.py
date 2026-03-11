"""Scenario and outcome sequence generators for simulation patterns."""


import numpy as np

from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Scenario


def generate_scenario_sequence(
    registry: DimensionRegistry,
    n_ticks: int,
    pattern: str = "crisis_recovery",
    rng: np.random.Generator | None = None,
) -> list[Scenario]:
    """
    Generate a sequence of scenarios for simulation.

    Patterns: 'stable', 'crisis_recovery', 'monotony', 'random', 'loss'.
    """
    rng = rng or np.random.default_rng()
    n = registry.size
    stress_idx = registry.index("N") if "N" in set(registry.keys) else None
    scenarios = []

    for t in range(n_ticks):
        frac = t / max(1, n_ticks - 1)
        base = _compute_base_stimuli(pattern, frac, n, stress_idx, rng)
        base = np.clip(base, 0, 1)
        scenarios.append(Scenario(
            array=base, registry=registry, name=f"tick_{t}_{pattern}",
        ))

    return scenarios


def _compute_base_stimuli(
    pattern: str, frac: float, n: int,
    stress_idx: int | None, rng: np.random.Generator,
) -> np.ndarray:
    """Compute raw stimulus array for a given pattern and time fraction."""
    if pattern == "stable":
        return 0.4 + 0.1 * rng.random(n)

    if pattern == "crisis_recovery":
        if frac < 0.3:
            return 0.3 + 0.1 * rng.random(n)
        if frac < 0.5:
            base = 0.7 + 0.2 * rng.random(n)
            if stress_idx is not None:
                base[stress_idx] = 0.9
            return base
        if frac < 0.7:
            return 0.5 + 0.15 * rng.random(n)
        return 0.3 + 0.1 * rng.random(n)

    if pattern == "monotony":
        decay = max(0.05, 1.0 - frac * 0.9)
        return decay * (0.3 + 0.1 * rng.random(n))

    if pattern == "loss":
        if frac < 0.4:
            return 0.5 + 0.2 * rng.random(n)
        if frac < 0.5:
            base = 0.8 + 0.15 * rng.random(n)
            if stress_idx is not None:
                base[stress_idx] = 0.95
            return base
        return 0.25 + 0.1 * rng.random(n)

    return rng.random(n)


def generate_outcome_sequence(
    n_ticks: int,
    pattern: str = "crisis_recovery",
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Generate outcome values in [-1, 1] matching a scenario pattern."""
    rng = rng or np.random.default_rng()
    outcomes = []

    for t in range(n_ticks):
        frac = t / max(1, n_ticks - 1)
        o = _compute_outcome(pattern, frac, rng)
        outcomes.append(float(np.clip(o, -1, 1)))

    return outcomes


def _compute_outcome(pattern: str, frac: float, rng: np.random.Generator) -> float:
    """Compute a single outcome value for a given pattern and time fraction."""
    if pattern == "stable":
        return rng.normal(0.2, 0.2)
    if pattern == "crisis_recovery":
        if frac < 0.3:
            return rng.normal(0.3, 0.15)
        if frac < 0.5:
            return rng.normal(-0.5, 0.25)
        if frac < 0.7:
            return rng.normal(-0.1, 0.3)
        return rng.normal(0.3, 0.15)
    if pattern == "monotony":
        return rng.normal(0.1, 0.05)
    if pattern == "loss":
        if frac < 0.4:
            return rng.normal(0.4, 0.15)
        if frac < 0.5:
            return rng.normal(-0.7, 0.15)
        return rng.normal(-0.1, 0.2)
    return rng.normal(0, 0.4)
