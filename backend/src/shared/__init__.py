"""Shared kernel: cross-cutting validators, base types, and logging."""

from src.shared.validators import validate_unit_interval, validate_real
from src.shared.types import DimensionVector
from src.shared.logging import get_logger, configure_logging
