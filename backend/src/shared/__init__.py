"""Shared kernel: cross-cutting validators, base types, and logging."""

from src.shared.logging import configure_logging, get_logger
from src.shared.types import DimensionVector
from src.shared.validators import validate_real, validate_unit_interval
