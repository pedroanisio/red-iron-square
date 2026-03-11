"""Domain vectors: PersonalityVector, Scenario, and Action."""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.shared.types import DimensionVector
from src.personality.dimensions import DimensionRegistry


class PersonalityVector(DimensionVector):
    """
    N-dimensional personality vector psi in [0, 1]^N.

    Immutable once constructed.
    """

    def __repr__(self) -> str:
        return f"psi({self._format_pairs()})"


class Scenario(DimensionVector):
    """
    N-dimensional scenario stimulus vector in [0, 1]^N.

    Each component is the stimulus intensity for the corresponding
    personality dimension.
    """

    def __init__(
        self,
        *,
        values: Optional[dict[str, float]] = None,
        array: Optional[np.ndarray] = None,
        registry: DimensionRegistry = DimensionRegistry(),
        name: str = "",
        description: str = "",
    ) -> None:
        super().__init__(values=values, array=array, registry=registry)
        self.name = name
        self.description = description

    def __repr__(self) -> str:
        return f"S({self._format_pairs()}; '{self.name}')"


class Action:
    """
    A possible action in response to a scenario.

    `modifiers` is an N-dimensional vector in [-1, 1] representing how
    well-suited this action is along each dimension.
    """

    def __init__(
        self,
        name: str,
        description: str,
        modifiers: np.ndarray,
        registry: DimensionRegistry = DimensionRegistry(),
    ) -> None:
        self.name = name
        self.description = description
        self._registry = registry

        m = np.asarray(modifiers, dtype=float)
        if m.shape != (registry.size,):
            raise ValueError(
                f"modifiers must have shape ({registry.size},), got {m.shape}"
            )
        self._modifiers = m

    @property
    def modifiers(self) -> np.ndarray:
        """Copy of the action modifier vector."""
        return self._modifiers.copy()

    def __repr__(self) -> str:
        return f"Action('{self.name}')"
