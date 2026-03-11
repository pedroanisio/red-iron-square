"""Shared kernel: cross-cutting validators, base types, and logging."""

from src.shared.logging import configure_logging as configure_logging
from src.shared.logging import get_logger as get_logger
from src.shared.types import DimensionVector as DimensionVector
from src.shared.validators import validate_real as validate_real
from src.shared.validators import (
    validate_unit_interval as validate_unit_interval,
)
