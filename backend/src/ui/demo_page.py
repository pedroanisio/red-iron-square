"""Demo routes and context for the Flask UI.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.ui.api_client import ApiClient
from src.ui.helpers import _KNOWN_ERRORS, _flash_on_error
from src.ui.models import DemoSession

if TYPE_CHECKING:
    from flask import Flask


def register_demo_routes(app: Flask, client: ApiClient) -> None:
    """Attach demo-control routes to the Flask UI app."""
    from flask import flash, redirect, render_template, request, session, url_for

    @app.get("/demo")
    def demo() -> str:
        """Render the Two Minds demo control room."""
        demo_session_id = request.args.get("session_id") or session.get(
            "demo_session_id"
        )
        context = build_demo_context(client, demo_session_id)
        result: str = render_template("ui/demo.html", **context)
        return result

    @app.post("/demo/create")
    @_flash_on_error("Demo session creation failed")
    def create_demo_session_ui() -> Any:
        """Create one demo session from the Flask UI."""
        act_number = int(request.form.get("act_number", "1"))
        created = client.create_demo_session({"act_number": act_number})
        session["demo_session_id"] = created["session_id"]
        flash(f"Demo session {created['session_id']} created.", "success")
        return redirect(url_for("demo", session_id=created["session_id"]))

    @app.post("/demo/<session_id>/scripted")
    @_flash_on_error("Scripted demo scenario failed")
    def run_demo_scripted_ui(session_id: str) -> Any:
        """Run one scripted demo scene."""
        scenario_key = request.form["scenario_key"]
        client.run_demo_scripted(session_id, scenario_key)
        session["demo_session_id"] = session_id
        flash(f"Ran scripted beat: {scenario_key}.", "success")
        return redirect(url_for("demo", session_id=session_id))

    @app.post("/demo/<session_id>/custom")
    @_flash_on_error("Custom demo scenario failed")
    def run_demo_custom_ui(session_id: str) -> Any:
        """Run one custom demo scene."""
        text = request.form["text"].strip()
        client.run_demo_custom(session_id, {"text": text})
        session["demo_session_id"] = session_id
        flash("Custom scenario sent to the room.", "success")
        return redirect(url_for("demo", session_id=session_id))

    @app.post("/demo/<session_id>/swap")
    @_flash_on_error("Demo personality swap failed")
    def swap_demo_ui(session_id: str) -> Any:
        """Swap demo personalities."""
        client.swap_demo_personalities(session_id)
        session["demo_session_id"] = session_id
        flash("Personalities swapped for the active demo session.", "success")
        return redirect(url_for("demo", session_id=session_id))


def build_demo_context(
    client: ApiClient,
    session_id: str | None,
) -> dict[str, Any]:
    """Gather template context for the demo control room."""
    from flask import session

    api_ok = False
    demo_sessions: list[DemoSession] = []
    demo_session: DemoSession | None = None
    if session_id:
        session["demo_session_id"] = session_id
    try:
        api_ok = client.health()["status"] == "ok"
    except _KNOWN_ERRORS:
        api_ok = False
    if api_ok:
        try:
            demo_sessions = client.list_demo_sessions()
        except _KNOWN_ERRORS:
            demo_sessions = []
        if session_id:
            active_session_ids = {
                item["session_id"] for item in demo_sessions if "session_id" in item
            }
            if session_id in active_session_ids:
                try:
                    demo_session = client.get_demo_session(session_id)
                except RuntimeError:
                    demo_session = None
                    session.pop("demo_session_id", None)
                    session_id = None
                except _KNOWN_ERRORS:
                    demo_session = None
                    session.pop("demo_session_id", None)
                    session_id = None
            else:
                session.pop("demo_session_id", None)
                session_id = None
    return {
        "api_ok": api_ok,
        "demo_session_id": session_id,
        "demo_sessions": demo_sessions,
        "demo_session": demo_session,
        "scripted_scenarios": [
            {"key": "promotion", "label": "The Promotion"},
            {"key": "phone_call", "label": "The Phone Call"},
            {"key": "three_months", "label": "Three Months Later"},
        ],
    }
