"""Flask UI for the simulation API.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, cast

from src.ui.models import RunListItem

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
from src.ui.helpers import (
    _KNOWN_ERRORS,
    DEFAULT_RUN_CONFIG,
    _flash_on_error,
    _parse_lines,
    _parse_optional_float,
)


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
    @_flash_on_error("Run creation failed")
    def create_run() -> Any:
        """Create a run from JSON config."""
        payload = json.loads(request.form["config_json"])
        run = client.create_run(payload)
        session["run_id"] = run["run_id"]
        flash(f"Created run {run['run_id']}.", "success")
        return redirect(url_for("index"))

    @app.post("/runs/<run_id>/assist-step")
    @_flash_on_error("Assisted step failed")
    def assist_step(run_id: str) -> Any:
        """Trigger one assisted step."""
        client.assist_step(
            run_id,
            {
                "goals": _parse_lines(request.form.get("goals", "")),
                "window": int(request.form.get("window", "5")),
            },
        )
        flash("Assisted step completed.", "success")
        return redirect(url_for("index", run_id=run_id))

    @app.post("/runs/<run_id>/intervention")
    @_flash_on_error("Intervention failed")
    def intervention(run_id: str) -> Any:
        """Trigger one intervention recommendation."""
        client.intervention(
            run_id,
            {
                "goals": _parse_lines(request.form.get("goals", "")),
                "window": int(request.form.get("window", "10")),
                "apply_patch": request.form.get("apply_patch") == "on",
            },
        )
        flash("Intervention call completed.", "success")
        return redirect(url_for("index", run_id=run_id))

    @app.post("/runs/<run_id>/tick")
    @_flash_on_error("Manual step failed")
    def tick(run_id: str) -> Any:
        """Trigger one manual tick from JSON scenario input."""
        payload = {
            "scenario": json.loads(request.form["scenario_json"]),
            "outcome": _parse_optional_float(
                request.form.get("outcome", ""),
            ),
        }
        client.tick(run_id, payload)
        flash("Manual tick completed.", "success")
        return redirect(url_for("index", run_id=run_id))

    @app.post("/runs/<run_id>/replay")
    @_flash_on_error("Replay failed")
    def replay_run(run_id: str) -> Any:
        """Create a deterministic replay clone."""
        result = client.replay_run(run_id)
        new_id = result["run"]["run_id"]
        session["run_id"] = new_id
        flash(f"Replay created: {new_id}.", "success")
        return redirect(url_for("index", run_id=new_id))

    @app.post("/runs/<run_id>/branch")
    @_flash_on_error("Branch failed")
    def branch_run(run_id: str) -> Any:
        """Create a branch from an existing run."""
        payload: dict[str, Any] = {}
        tick_val = request.form.get("parent_tick", "").strip()
        if tick_val:
            payload["parent_tick"] = int(tick_val)
        temp_val = request.form.get("temperature", "").strip()
        if temp_val:
            payload["temperature"] = float(temp_val)
        result = client.branch_run(run_id, payload)
        new_id = result["run"]["run_id"]
        session["run_id"] = new_id
        flash(f"Branch created: {new_id}.", "success")
        return redirect(url_for("index", run_id=new_id))

    @app.get("/runs/<run_id>/export")
    @_flash_on_error("Export failed")
    def export_trajectory(run_id: str) -> Any:
        """Download trajectory as JSON."""
        trajectory = client.get_trajectory(run_id)
        response = app.response_class(
            json.dumps(trajectory, indent=2),
            mimetype="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={run_id}.json",
            },
        )
        return response

    @app.post("/runs/<run_id>/orchestrate")
    @_flash_on_error("Orchestration failed")
    def orchestrate_run(run_id: str) -> Any:
        """Run orchestration cycles."""
        cycles = int(request.form.get("cycles", "1"))
        goals = _parse_lines(request.form.get("goals", ""))
        client.orchestrate(run_id, {"cycles": cycles, "goals": goals})
        flash(f"Ran {cycles} orchestration cycle(s).", "success")
        return redirect(url_for("index", run_id=run_id))

    @app.post("/runs/<run_id>/resume")
    @_flash_on_error("Resume failed")
    def resume_run(run_id: str) -> Any:
        """Resume a paused run."""
        goals = _parse_lines(request.form.get("goals", ""))
        client.resume_run(run_id, {"goals": goals})
        flash("Run resumed.", "success")
        return redirect(url_for("index", run_id=run_id))

    @app.get("/campaigns")
    def campaigns() -> str:
        """Render campaign management page."""
        campaign_id = request.args.get("campaign_id")
        campaign_list: list[dict[str, Any]] = []
        campaign = None
        try:
            campaign_list = client.list_campaigns()
        except _KNOWN_ERRORS:
            pass
        if campaign_id:
            try:
                campaign = client.get_campaign_summary(campaign_id)
            except _KNOWN_ERRORS:
                pass
        return render_template(
            "ui/campaigns.html",
            campaigns=campaign_list,
            campaign=campaign,
            campaign_id=campaign_id,
        )

    @app.post("/campaigns/create")
    @_flash_on_error("Campaign creation failed")
    def create_campaign_ui() -> Any:
        """Create a campaign from the UI."""
        payload = {
            "name": request.form["name"],
            "goals": _parse_lines(request.form.get("goals", "")),
            "config_template": json.loads(request.form.get("config_json", "{}")),
        }
        result = client.create_campaign(payload)
        flash(f"Created campaign {result['campaign_id']}.", "success")
        return redirect(url_for("campaigns", campaign_id=result["campaign_id"]))

    @app.get("/compare")
    def compare() -> str:
        """Side-by-side trajectory comparison."""
        left_id = request.args.get("left", "")
        right_id = request.args.get("right", "")
        pairs = _fetch_compare_pairs(client, left_id, right_id)
        return render_template(
            "ui/compare.html",
            left_id=left_id,
            right_id=right_id,
            **pairs,
        )

    return app


def _fetch_compare_pairs(
    client: ApiClient,
    left_id: str,
    right_id: str,
) -> dict[str, Any]:
    """Fetch run and trajectory data for compare view."""
    result: dict[str, Any] = {
        "left_run": None,
        "left_traj": None,
        "right_run": None,
        "right_traj": None,
    }
    for side, run_id in [("left", left_id), ("right", right_id)]:
        if not run_id:
            continue
        try:
            result[f"{side}_run"] = client.get_run(run_id)
            result[f"{side}_traj"] = client.get_trajectory(run_id)
        except _KNOWN_ERRORS:
            pass
    return result


def _build_context(client: ApiClient, run_id: str | None) -> dict[str, Any]:
    """Gather template context, tolerating API failures gracefully."""
    api_ok = False
    run = None
    trajectory = None
    orchestrator_log: list[dict[str, Any]] = []
    recent_runs: list[RunListItem] = []
    if run_id:
        session["run_id"] = run_id
    try:
        api_ok = client.health()["status"] == "ok"
    except _KNOWN_ERRORS:
        api_ok = False
    if api_ok:
        try:
            recent_runs = client.list_runs()
        except _KNOWN_ERRORS:
            recent_runs = []
        if run_id:
            try:
                run = client.get_run(run_id)
                trajectory = client.get_trajectory(run_id)
                orchestrator_log = client.orchestrator_log(run_id)
            except _KNOWN_ERRORS:
                run = None
                trajectory = None
    return {
        "api_ok": api_ok,
        "run_id": run_id,
        "run": run,
        "trajectory": trajectory,
        "orchestrator_log": orchestrator_log,
        "recent_runs": recent_runs,
        "default_run_config": json.dumps(DEFAULT_RUN_CONFIG, indent=2),
        "default_scenario": json.dumps(
            {"name": "manual_probe", "values": {"O": 0.8, "N": 0.3}},
            indent=2,
        ),
    }


def main() -> None:
    """Run the Flask development server."""
    app = create_ui_app()
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=5001, debug=debug)


if __name__ == "__main__":
    main()
