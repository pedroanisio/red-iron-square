"""Tests for the shared kernel: validators and DimensionVector base type."""

import numpy as np
import pytest
from src.personality.dimensions import DimensionRegistry
from src.shared.types import DimensionVector
from src.shared.validators import validate_real, validate_unit_interval


class TestValidateUnitInterval:
    """Tests for validate_unit_interval."""

    def test_valid_boundaries(self) -> None:
        validate_unit_interval("x", 0.0)
        validate_unit_interval("x", 1.0)
        validate_unit_interval("x", 0.5)

    def test_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="outside the required"):
            validate_unit_interval("x", -0.01)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="outside the required"):
            validate_unit_interval("x", 1.01)


class TestValidateReal:
    """Tests for validate_real."""

    def test_finite_passes(self) -> None:
        validate_real("x", 0.0)
        validate_real("x", -100.0)

    def test_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="not finite"):
            validate_real("x", float("inf"))

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="not finite"):
            validate_real("x", float("nan"))


class TestDimensionVector:
    """Tests for the DimensionVector base class."""

    def setup_method(self) -> None:
        self.registry = DimensionRegistry()

    def test_construct_from_values(self) -> None:
        vec = DimensionVector(values={"O": 0.8, "C": 0.6}, registry=self.registry)
        assert vec["O"] == pytest.approx(0.8)
        assert vec["C"] == pytest.approx(0.6)
        assert vec["E"] == pytest.approx(0.0)

    def test_construct_from_array(self) -> None:
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        vec = DimensionVector(array=arr, registry=self.registry)
        assert vec["O"] == pytest.approx(0.1)
        assert vec["T"] == pytest.approx(0.8)

    def test_both_values_and_array_raises(self) -> None:
        with pytest.raises(ValueError, match="either"):
            DimensionVector(
                values={"O": 0.5},
                array=np.zeros(8),
                registry=self.registry,
            )

    def test_neither_values_nor_array_raises(self) -> None:
        with pytest.raises(ValueError, match="Must provide"):
            DimensionVector(registry=self.registry)

    def test_out_of_range_value_raises(self) -> None:
        with pytest.raises(ValueError):
            DimensionVector(values={"O": 1.5}, registry=self.registry)

    def test_wrong_shape_array_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            DimensionVector(array=np.zeros(3), registry=self.registry)

    def test_to_array_returns_copy(self) -> None:
        vec = DimensionVector(values={"O": 0.5}, registry=self.registry)
        arr = vec.to_array()
        arr[0] = 999.0
        assert vec["O"] == pytest.approx(0.5)

    def test_format_pairs(self) -> None:
        vec = DimensionVector(values={"O": 0.5}, registry=self.registry)
        pairs = vec._format_pairs()
        assert "O=0.50" in pairs
