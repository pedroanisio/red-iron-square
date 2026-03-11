"""Flask UI for the simulation API.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from flask import Flask

try:
    import flask as flask_module
    from flask import flash, redirect, render_template, request, session, url_for
except ModuleNotFoundError as exc:
    _FLASK_IMPORT_ERROR: ModuleNotFoundError | None = exc
    flask_module = cast(Any, None)
    flash = redirect = render_template = request = session = url_for = cast(Any, None)
else:
    _FLASK_IMPORT_ERROR = None

from src.ui.api_client import ApiClient

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


def create_ui_app(api_client: ApiClient | None = None) -> Flask:
    """Create the Flask UI app."""
    if _FLASK_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Flask is required for the UI. Install the `ui` extra or dev dependencies."
        ) from _FLASK_IMPORT_ERROR
    app = flask_module.Flask(__name__, template_folder="templates")
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "red-iron-square-dev")
    client = api_client or ApiClient(
        os.getenv("RED_IRON_SQUARE_API_URL", "http://127.0.0.1:8000")
    )

    @app.get("/")
    def index() -> str:
        """Render the run dashboard."""
        run_id = request.args.get("run_id") or session.get("run_id")
        context = _build_context(client, run_id)
        result: str = render_template("ui/index.html", **context)
        return result

    @app.post("/runs/create")
    def create_run() -> Any:
        """Create a run from JSON config."""
        try:
            payload = json.loads(request.form["config_json"])
            run = client.create_run(payload)
            session["run_id"] = run["run_id"]
            flash(f"Created run {run['run_id']}.", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"Run creation failed: {exc}", "error")
        return redirect(url_for("index"))

    @app.post("/runs/<run_id>/assist-step")
    def assist_step(run_id: str) -> Any:
        """Trigger one assisted step."""
        try:
            client.assist_step(
                run_id,
                {
                    "goals": _parse_lines(request.form.get("goals", "")),
                    "window": int(request.form.get("window", "5")),
                },
            )
            flash("Assisted step completed.", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"Assisted step failed: {exc}", "error")
        return redirect(url_for("index", run_id=run_id))

    @app.post("/runs/<run_id>/intervention")
    def intervention(run_id: str) -> Any:
        """Trigger one intervention recommendation."""
        try:
            client.intervention(
                run_id,
                {
                    "goals": _parse_lines(request.form.get("goals", "")),
                    "window": int(request.form.get("window", "10")),
                    "apply_patch": request.form.get("apply_patch") == "on",
                },
            )
            flash("Intervention call completed.", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"Intervention failed: {exc}", "error")
        return redirect(url_for("index", run_id=run_id))

    @app.post("/runs/<run_id>/tick")
    def tick(run_id: str) -> Any:
        """Trigger one manual tick from JSON scenario input."""
        try:
            payload = {
                "scenario": json.loads(request.form["scenario_json"]),
                "outcome": _parse_optional_float(request.form.get("outcome", "")),
            }
            client.tick(run_id, payload)
            flash("Manual tick completed.", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"Manual tick failed: {exc}", "error")
        return redirect(url_for("index", run_id=run_id))

    return app


def _build_context(client: ApiClient, run_id: str | None) -> dict[str, Any]:
    api_ok = False
    run = None
    trajectory = None
    recent_runs: list[dict[str, Any]] = []
    if run_id:
        session["run_id"] = run_id
    try:
        api_ok = client.health()["status"] == "ok"
    except Exception:  # noqa: BLE001
        api_ok = False
    if api_ok:
        try:
            recent_runs = client.list_runs()
        except Exception:  # noqa: BLE001
            recent_runs = []
        if run_id:
            try:
                run = client.get_run(run_id)
                trajectory = client.get_trajectory(run_id)
            except Exception:  # noqa: BLE001
                run = None
                trajectory = None
    return {
        "api_ok": api_ok,
        "run_id": run_id,
        "run": run,
        "trajectory": trajectory,
        "recent_runs": recent_runs,
        "default_run_config": json.dumps(DEFAULT_RUN_CONFIG, indent=2),
        "default_scenario": json.dumps(
            {"name": "manual_probe", "values": {"O": 0.8, "N": 0.3}},
            indent=2,
        ),
    }


def _parse_lines(raw_value: str) -> list[str]:
    """Parse one textarea into non-empty lines."""
    return [line.strip() for line in raw_value.splitlines() if line.strip()]


def _parse_optional_float(raw_value: str) -> float | None:
    """Parse an optional float form value."""
    value = raw_value.strip()
    return None if value == "" else float(value)


def main() -> None:
    """Run the Flask development server."""
    app = create_ui_app()
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=5001, debug=debug)


if __name__ == "__main__":
    main()
