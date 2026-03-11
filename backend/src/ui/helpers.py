"""Shared helpers for the Flask UI layer.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import functools
import json
import urllib.error
from collections.abc import Callable
from typing import Any, cast

try:
    from flask import flash, redirect, url_for
except ModuleNotFoundError:
    flash = redirect = url_for = cast(Any, None)

_KNOWN_ERRORS = (
    json.JSONDecodeError,
    ConnectionError,
    urllib.error.URLError,
    TimeoutError,
    ValueError,
    KeyError,
)

_ERROR_MAP: dict[type, str] = {
    json.JSONDecodeError: "Invalid JSON — check syntax near the reported position.",
    ConnectionError: "Cannot reach the API server. Is it running?",
    urllib.error.URLError: "Cannot reach the API server. Is it running?",
    TimeoutError: "The API request timed out. Try again or check the server.",
    ValueError: "Invalid input — please check the values you entered.",
    KeyError: "Unexpected response from the API — a required field was missing.",
}


def _friendly_error(exc: Exception, action: str) -> str:
    """Map an exception to a user-friendly flash message."""
    for exc_type, message in _ERROR_MAP.items():
        if isinstance(exc, exc_type):
            return f"{action}: {message}"
    raw = str(exc)
    if len(raw) > 200:
        raw = raw[:200] + "..."
    return f"{action}: {raw}"


def _flash_on_error(action: str) -> Callable[..., Any]:
    """Handle known errors by flashing a friendly message and redirecting."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except _KNOWN_ERRORS as exc:
                flash(_friendly_error(exc, action), "error")
                run_id = kwargs.get("run_id") or args[0] if args else None
                if run_id:
                    return redirect(url_for("index", run_id=run_id))
                return redirect(url_for("index"))

        return wrapper

    return decorator


DEFAULT_RUN_CONFIG = {
    "personality": {
        "O": 0.8,
        "C": 0.5,
        "E": 0.3,
        "A": 0.7,
        "N": 0.4,
        "R": 0.9,
        "I": 0.6,
        "T": 0.2,
    },
    "actions": [
        {"name": "bold", "modifiers": {"O": 1.0, "R": 0.8, "N": -0.3}},
        {"name": "safe", "modifiers": {"C": 0.9, "T": 0.8}},
    ],
    "temperature": 1.0,
    "seed": 42,
}


def _parse_lines(raw_value: str) -> list[str]:
    """Parse one textarea into non-empty lines."""
    return [line.strip() for line in raw_value.splitlines() if line.strip()]


def _parse_optional_float(raw_value: str) -> float | None:
    """Parse an optional float form value."""
    value = raw_value.strip()
    return None if value == "" else float(value)
