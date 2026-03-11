"""Base dimension vector — shared init logic for PersonalityVector, Scenario, etc."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.validators import validate_unit_interval

if TYPE_CHECKING:
    from src.personality.dimensions import DimensionRegistry


class DimensionVector:
    """Base class for N-dimensional vectors aligned to a DimensionRegistry.

    Provides the shared dual-constructor pattern (values dict OR numpy array)
    with [0, 1] validation.  Subclasses add domain-specific semantics.
    """

    def __init__(
        self,
        *,
        values: dict[str, float] | None = None,
        array: np.ndarray | None = None,
        registry: DimensionRegistry,
    ) -> None:
        if values is not None and array is not None:
            raise ValueError("Provide either `values` or `array`, not both.")
        if values is None and array is None:
            raise ValueError("Must provide either `values` or `array`.")

        if values is not None:
            arr = np.zeros(registry.size)
            for key, val in values.items():
                validate_unit_interval(key, val)
                arr[registry.index(key)] = val
        else:
            arr = np.asarray(array, dtype=float)
            if arr.shape != (registry.size,):
                raise ValueError(
                    f"Expected array of shape ({registry.size},), got {arr.shape}"
                )
            for i, key in enumerate(registry.keys):
                validate_unit_interval(key, arr[i])

        self._array = arr
        self._registry = registry

    def __getitem__(self, key: str) -> float:
        """Look up a dimension value by its key."""
        return float(self._array[self._registry.index(key)])

    def to_array(self) -> np.ndarray:
        """Return a copy of the underlying numpy array."""
        return self._array.copy()

    @property
    def registry(self) -> DimensionRegistry:
        """The dimension registry this vector is aligned to."""
        return self._registry

    def _format_pairs(self) -> str:
        """Format key=value pairs for repr."""
        return ", ".join(
            f"{k}={self._array[i]:.2f}" for i, k in enumerate(self._registry.keys)
        )
