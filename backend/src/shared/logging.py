"""
Structured logging configuration using structlog.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import os
import sys

import structlog
from dotenv import load_dotenv

load_dotenv()

_CONFIGURED = False


def configure_logging(*, json_output: bool | None = None) -> None:
    """
    Configure structlog processors and output format.

    Idempotent — safe to call multiple times.
    Uses LOG_LEVEL env var (default: INFO) and LOG_FORMAT env var
    ('json' for JSON, anything else for console).
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    use_json = json_output if json_output is not None else os.getenv("LOG_FORMAT") == "json"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if use_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.processors.NAME_TO_LEVEL[log_level.lower()],
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(**initial_binds: object) -> structlog.BoundLogger:
    """Return a bound structlog logger with optional initial context."""
    configure_logging()
    logger: structlog.BoundLogger = structlog.get_logger(**initial_binds)
    return logger
