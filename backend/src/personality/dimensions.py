"""Dimension definitions and registry for personality/scenario vectors."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class Dimension(BaseModel):
    """A single personality/scenario dimension."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    description: str


DEFAULT_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        key="O",
        name="Openness",
        description="Receptivity to novel experience and creative stimuli.",
    ),
    Dimension(
        key="C",
        name="Conscientiousness",
        description="Preference for structure, planning, and order.",
    ),
    Dimension(
        key="E",
        name="Extraversion",
        description="Energy gain/drain from social interaction.",
    ),
    Dimension(
        key="A",
        name="Agreeableness",
        description="Preference for cooperation over competition.",
    ),
    Dimension(
        key="N",
        name="Neuroticism",
        description="Sensitivity to stress (high N = fragile under stress).",
    ),
    Dimension(
        key="R",
        name="Resilience",
        description="Capacity to mobilize or endure under adversity.",
    ),
    Dimension(
        key="I",
        name="Idealism",
        description="Priority of ideal outcomes over pragmatic ones.",
    ),
    Dimension(
        key="T",
        name="Tradition",
        description="Preference for tradition-aligned over novel approaches.",
    ),
)


class DimensionRegistry:
    """Ordered set of dimensions with index lookups.

    Decouples the framework from a hardcoded dimension count: register
    5 (pure OCEAN), 8 (OCEAN + RIT), or any custom set.
    """

    def __init__(self, dimensions: Sequence[Dimension] = DEFAULT_DIMENSIONS) -> None:
        self._dims = tuple(dimensions)
        self._index = {d.key: i for i, d in enumerate(self._dims)}
        if len(self._index) != len(self._dims):
            dupes = [d.key for d in self._dims]
            raise ValueError(f"Duplicate dimension keys: {dupes}")

    @property
    def size(self) -> int:
        """Number of registered dimensions."""
        return len(self._dims)

    @property
    def keys(self) -> tuple[str, ...]:
        """Ordered tuple of dimension keys."""
        return tuple(d.key for d in self._dims)

    def index(self, key: str) -> int:
        """Return the positional index for a dimension key."""
        return self._index[key]

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"DimensionRegistry({self.keys})"
