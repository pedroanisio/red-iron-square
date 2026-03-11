"""SDK builders for constructing domain objects from plain Python inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.personality.dimensions import DEFAULT_DIMENSIONS, Dimension, DimensionRegistry
from src.personality.vectors import Action, PersonalityVector, Scenario
from src.shared.validators import validate_unit_interval


def build_registry(
    dimensions: Sequence[Dimension] = DEFAULT_DIMENSIONS,
) -> DimensionRegistry:
    """Create a registry for SDK consumers."""
    return DimensionRegistry(dimensions)


def build_personality(
    values: Mapping[str, float],
    registry: DimensionRegistry,
) -> PersonalityVector:
    """Construct a personality vector from sparse dimension values."""
    return PersonalityVector(values=dict(values), registry=registry)


def build_scenario(
    values: Mapping[str, float],
    registry: DimensionRegistry,
    *,
    name: str = "",
    description: str = "",
) -> Scenario:
    """Construct a scenario vector from sparse dimension values."""
    return Scenario(
        values=dict(values),
        registry=registry,
        name=name,
        description=description,
    )


def build_action(
    name: str,
    modifiers: Mapping[str, float],
    registry: DimensionRegistry,
    *,
    description: str = "",
) -> Action:
    """Construct an action from sparse modifiers in [-1, 1]."""
    dense = np.zeros(registry.size)
    for key, value in modifiers.items():
        if not (-1.0 <= value <= 1.0):
            raise ValueError(f"{key}={value} is outside the required [-1, 1] interval.")
        dense[registry.index(key)] = value
    return Action(
        name=name,
        description=description,
        modifiers=dense,
        registry=registry,
    )


def build_initial_self_model(
    values: Mapping[str, float],
    registry: DimensionRegistry,
) -> np.ndarray:
    """Construct an initial self-model vector from sparse values in [0, 1]."""
    dense = np.zeros(registry.size)
    for key, value in values.items():
        validate_unit_interval(key, value)
        dense[registry.index(key)] = value
    return dense
