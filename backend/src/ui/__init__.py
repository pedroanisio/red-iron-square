"""Flask UI package.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ui.api_client import ApiClient

if TYPE_CHECKING:
    from flask import Flask


def create_ui_app(api_client: ApiClient | None = None) -> Flask:
    """Create the Flask UI app lazily."""
    from src.ui.app import create_ui_app as _create_ui_app

    return _create_ui_app(api_client)


__all__ = ["create_ui_app"]
