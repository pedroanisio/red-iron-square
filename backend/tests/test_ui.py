"""Tests for the Flask UI.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("flask")

from src.ui.app import create_ui_app
from src.ui.helpers import _friendly_error
from src.ui.models import (
    BranchResult,
    DemoScriptedResult,
    DemoSession,
    DemoSwapResult,
    ReplayResult,
    RunListItem,
    RunSummary,
    TrajectoryData,
)


class FakeUiClient:
    """Fake API client for UI tests."""

    def __init__(self) -> None:
        self.last_demo_custom_payload: dict[str, Any] | None = None
        self.last_demo_scripted: tuple[str, str] | None = None
        self.last_demo_swap_session_id: str | None = None

    def health(self) -> dict[str, str]:
        """Return a healthy status."""
        return {"status": "ok"}

    def list_runs(self) -> list[RunListItem]:
        """Return sample run list items."""
        return [
            RunListItem(
                run_id="run-123",
                mode="temporal",
                status="active",
                tick_count=3,
                updated_at="2026-03-11T10:00:00+00:00",
            ),
            RunListItem(
                run_id="run-456",
                mode="self_aware",
                status="active",
                tick_count=0,
                updated_at="2026-03-11T09:00:00+00:00",
            ),
        ]

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a stub create-run result."""
        return {"run_id": "run-123"}

    def list_demo_sessions(self) -> list[DemoSession]:
        """Return sample demo sessions."""
        return [
            DemoSession(
                session_id="demo-123",
                act_number=1,
                turn_count=2,
                agents=[
                    {
                        "key": "luna",
                        "name": "Luna",
                        "summary": "Warm, fast-reacting, and idealistic.",
                        "mood": 0.4,
                        "energy": 0.8,
                        "calm": 0.3,
                        "emotion_label": "Excited",
                    },
                    {
                        "key": "marco",
                        "name": "Marco",
                        "summary": "Measured, guarded, and practical.",
                        "mood": -0.1,
                        "energy": 0.5,
                        "calm": 0.7,
                        "emotion_label": "Reserved",
                    },
                ],
            )
        ]

    def create_demo_session(self, payload: dict[str, Any]) -> DemoSession:
        """Return a stub create-demo result."""
        return self.get_demo_session("demo-123")

    def get_demo_session(self, session_id: str) -> DemoSession:
        """Return one sample demo session."""
        return DemoSession(
            session_id=session_id,
            act_number=1,
            turn_count=2,
            agents=[
                {
                    "key": "luna",
                    "name": "Luna",
                    "summary": "Warm, fast-reacting, and idealistic.",
                    "mood": 0.4,
                    "energy": 0.8,
                    "calm": 0.3,
                    "emotion_label": "Excited",
                },
                {
                    "key": "marco",
                    "name": "Marco",
                    "summary": "Measured, guarded, and practical.",
                    "mood": -0.1,
                    "energy": 0.5,
                    "calm": 0.7,
                    "emotion_label": "Reserved",
                },
            ],
        )

    def run_demo_scripted(
        self,
        session_id: str,
        scenario_key: str,
    ) -> DemoScriptedResult:
        """Return a stub scripted-demo result."""
        self.last_demo_scripted = (session_id, scenario_key)
        return DemoScriptedResult(
            session_id=session_id,
            scenario_key=scenario_key,
            turn_count=3,
        )

    def run_demo_custom(self, session_id: str, payload: dict[str, Any]) -> DemoSession:
        """Return a stub custom-demo result."""
        self.last_demo_custom_payload = payload
        return self.get_demo_session(session_id)

    def swap_demo_personalities(self, session_id: str) -> DemoSwapResult:
        """Return a stub swap-demo result."""
        self.last_demo_swap_session_id = session_id
        return DemoSwapResult(session_id=session_id, act_number=1, swapped=True)

    def get_run(self, run_id: str) -> RunSummary:
        """Return a sample run summary."""
        return RunSummary(
            run_id=run_id,
            mode="temporal",
            status="active",
            tick_count=1,
            config={},  # type: ignore[typeddict-item]
            parent_run_id=None,
            parent_tick=None,
            latest_tick={"tick": 0, "action": "safe", "outcome": 0.6},
            phases=[],
            agent_invocation_count=2,
            intervention_count=1,
            created_at="now",
            updated_at="now",
        )

    def get_trajectory(self, run_id: str) -> TrajectoryData:
        """Return a sample trajectory."""
        return TrajectoryData(
            run_id=run_id,
            tick_count=2,
            ticks=[
                {
                    "tick": 0,
                    "action": "safe",
                    "outcome": 0.6,
                    "emotions": [],
                    "state_after": {
                        "mood": 0.2,
                        "arousal": 0.5,
                        "energy": 0.8,
                        "satisfaction": 0.4,
                        "frustration": 0.1,
                    },
                },
                {
                    "tick": 1,
                    "action": "bold",
                    "outcome": -0.3,
                    "emotions": [],
                    "state_after": {
                        "mood": -0.1,
                        "arousal": 0.7,
                        "energy": 0.6,
                        "satisfaction": 0.3,
                        "frustration": 0.3,
                    },
                },
            ],
            phases=[
                {"start_tick": 0, "end_tick": 1, "label": "warmup", "notes": ""},
            ],
            agent_invocations=[
                {
                    "agent_name": "observer_agent",
                    "purpose": "summarize_window",
                    "metadata": {"model": "fake"},
                    "output": {"summary": "stable"},
                }
            ],
            interventions=[
                {
                    "action": "patch_params",
                    "reason": "reduce randomness",
                    "applied": True,
                    "payload": {"temperature": 0.5},
                }
            ],
        )

    def assist_step(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a stub assist-step result."""
        return {}

    def intervention(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a stub intervention result."""
        return {}

    def tick(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a stub tick result."""
        return {}

    def replay_run(self, run_id: str) -> ReplayResult:
        """Return a sample replay result."""
        return ReplayResult(
            run=RunListItem(
                run_id="run-replay-1",
                mode="temporal",
                status="active",
                tick_count=0,
                updated_at="now",
            ),
        )

    def branch_run(
        self,
        run_id: str,
        payload: dict[str, Any],
    ) -> BranchResult:
        """Return a sample branch result."""
        return BranchResult(
            run=RunListItem(
                run_id="run-branch-1",
                mode="temporal",
                status="active",
                tick_count=0,
                updated_at="now",
            ),
        )

    def list_campaigns(self) -> list[dict[str, object]]:
        """Return sample campaign list."""
        return [
            {
                "campaign_id": "camp-1",
                "name": "Test Campaign",
                "status": "active",
                "goals": ["explore"],
                "config_template": {},
                "created_at": "now",
                "updated_at": "now",
            },
        ]

    def create_campaign(self, payload: dict[str, object]) -> dict[str, object]:
        """Return a stub create-campaign result."""
        return {"campaign_id": "camp-1"}

    def get_campaign(self, campaign_id: str) -> dict[str, object]:
        """Return a sample campaign."""
        return {
            "campaign_id": campaign_id,
            "name": "Test",
            "status": "active",
            "goals": ["explore"],
            "config_template": {},
            "runs": [],
            "created_at": "now",
            "updated_at": "now",
        }

    def get_campaign_summary(self, campaign_id: str) -> dict[str, object]:
        """Return a sample campaign summary."""
        return {
            "campaign_id": campaign_id,
            "name": "Test",
            "status": "active",
            "goals": ["explore"],
            "config_template": {},
            "runs": [],
            "run_summaries": [],
            "total_ticks": 5,
            "run_count": 1,
            "created_at": "now",
            "updated_at": "now",
        }

    def orchestrate(self, run_id: str, payload: dict[str, object]) -> dict[str, object]:
        """Return a stub orchestration result."""
        return {"cycle": 0, "action_type": "scenario", "result": {}}

    def orchestrator_log(self, run_id: str) -> list[dict[str, object]]:
        """Return sample orchestrator decisions."""
        return [
            {
                "cycle": 0,
                "action_type": "scenario",
                "rationale": "Auto-step",
                "created_at": "now",
            },
        ]

    def resume_run(self, run_id: str, payload: dict[str, object]) -> dict[str, object]:
        """Return a stub resume result."""
        return {}


def test_friendly_error_maps_json_error() -> None:
    """Known exception types produce user-friendly messages."""
    import json

    exc = json.JSONDecodeError("Expecting value", "", 0)
    result = _friendly_error(exc, "Parse failed")

    assert "Invalid JSON" in result
    assert "Parse failed" in result


def test_friendly_error_truncates_long_message() -> None:
    """Unknown exceptions are truncated to 200 characters."""
    exc = RuntimeError("x" * 300)
    result = _friendly_error(exc, "Oops")

    assert len(result) < 220
    assert result.endswith("...")


def test_friendly_error_passes_through_short_unknown() -> None:
    """Short unknown exceptions are shown as-is."""
    exc = RuntimeError("bad value")
    result = _friendly_error(exc, "Step failed")

    assert result == "Step failed: bad value"


def test_index_renders() -> None:
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Red Iron Square" in response.data
    assert b"How This Dashboard Works" in response.data
    assert b"Start here if this is your first simulation." in response.data


def test_index_loads_run_view() -> None:
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/?run_id=run-123")

    assert response.status_code == 200
    assert b"run-123" in response.data
    assert b"AI Calls" in response.data


def test_index_shows_run_browser() -> None:
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"run-123" in response.data
    assert b"run-456" in response.data
    assert b"Recent Runs" in response.data


def test_index_has_accessibility_landmarks() -> None:
    """Page includes skip link, ARIA roles, and landmarks."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/")
    html = response.data

    assert b'class="skip-link"' in html
    assert b'role="banner"' in html
    assert b'role="main"' in html
    assert b'role="complementary"' in html


def test_index_tabs_have_aria_attributes() -> None:
    """Tab buttons use WAI-ARIA tab pattern."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/?run_id=run-123")
    html = response.data

    assert b'role="tablist"' in html
    assert b'role="tab"' in html
    assert b'role="tabpanel"' in html
    assert b'aria-selected="true"' in html


def test_index_json_textareas_have_validation_attr() -> None:
    """JSON textareas include data-validate-json for client-side validation."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/")

    assert b"data-validate-json" in response.data


def test_index_shows_sparkline_with_trajectory() -> None:
    """Sparkline SVG renders with accessible title and hover targets."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/?run_id=run-123")

    assert response.status_code == 200
    assert b"<svg" in response.data
    assert b"Trajectory" in response.data
    assert b"<title>" in response.data
    assert b"<circle" in response.data


def test_run_view_has_action_toolbar() -> None:
    """Run view shows replay, branch, and export buttons."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/?run_id=run-123")
    html = response.data

    assert b"Replay" in html
    assert b"Branch" in html
    assert b"Export JSON" in html


def test_replay_redirects_to_new_run() -> None:
    """Replay route creates a clone and redirects."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post("/runs/run-123/replay")

    assert response.status_code == 302
    assert "run-replay-1" in response.headers["Location"]


def test_branch_redirects_to_new_run() -> None:
    """Branch route creates a fork and redirects."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/runs/run-123/branch",
        data={"parent_tick": "1", "temperature": "0.5"},
    )

    assert response.status_code == 302
    assert "run-branch-1" in response.headers["Location"]


def test_export_returns_json_download() -> None:
    """Export route returns a JSON attachment."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/runs/run-123/export")

    assert response.status_code == 200
    assert response.content_type == "application/json"
    assert "attachment" in response.headers["Content-Disposition"]
    assert b"run-123" in response.data


def test_self_aware_run_shows_identity_metrics() -> None:
    """Self-aware runs display identity drift, coherence, accuracy."""

    class SelfAwareFakeClient(FakeUiClient):
        """Fake client that returns self-aware run data."""

        def get_run(self, run_id: str) -> RunSummary:
            """Return a self-aware run."""
            run = super().get_run(run_id)
            run["mode"] = "self_aware"
            return run

        def get_trajectory(self, run_id: str) -> TrajectoryData:
            """Return trajectory with identity metrics on each tick."""
            traj = super().get_trajectory(run_id)
            for tick in traj["ticks"]:
                tick["identity_drift"] = 0.15
                tick["self_coherence"] = 0.82
                tick["self_accuracy"] = 0.71
            return traj

    app = create_ui_app(api_client=SelfAwareFakeClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/?run_id=run-123")
    html = response.data

    assert b"Identity Drift" in html
    assert b"Self-Coherence" in html
    assert b"Self-Accuracy" in html


def test_htmx_boost_is_enabled() -> None:
    """Body has hx-boost attribute for smoother navigation."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/")

    assert b'hx-boost="true"' in response.data
    assert b"htmx-indicator" in response.data


def test_sparkline_shows_phase_markers() -> None:
    """Phase annotations render as markers on the sparkline."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/?run_id=run-123")

    assert response.status_code == 200
    assert b"phase-marker" in response.data


def test_compare_route_renders_two_runs() -> None:
    """Compare route shows side-by-side run data."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/compare?left=run-123&right=run-456")

    assert response.status_code == 200
    assert b"run-123" in response.data
    assert b"run-456" in response.data
    assert b"Compare" in response.data


def test_campaigns_page_renders() -> None:
    """Campaign page shows campaign list."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/campaigns")

    assert response.status_code == 200
    assert b"Campaigns" in response.data
    assert b"camp-1" in response.data


def test_campaign_detail_renders() -> None:
    """Campaign detail shows summary."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/campaigns?campaign_id=camp-1")

    assert response.status_code == 200
    assert b"camp-1" in response.data


def test_demo_page_renders() -> None:
    """Demo page shows active room controls."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/demo?session_id=demo-123")

    assert response.status_code == 200
    assert b"Two Minds Demo" in response.data
    assert b"demo-123" in response.data
    assert b"Luna" in response.data
    assert b"Marco" in response.data
    assert b"Swap Personalities" in response.data


def test_demo_create_redirects_to_session() -> None:
    """Create-demo route redirects to the active room."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post("/demo/create", data={"act_number": "1"})

    assert response.status_code == 302
    assert "session_id=demo-123" in response.headers["Location"]


def test_demo_scripted_post_redirects() -> None:
    """Scripted-demo route redirects back to the room."""
    fake_client = FakeUiClient()
    app = create_ui_app(api_client=fake_client)
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/demo/demo-123/scripted",
        data={"scenario_key": "promotion"},
    )

    assert response.status_code == 302
    assert "session_id=demo-123" in response.headers["Location"]
    assert fake_client.last_demo_scripted == ("demo-123", "promotion")


def test_demo_custom_post_redirects() -> None:
    """Custom-demo route forwards the prompt and redirects."""
    fake_client = FakeUiClient()
    app = create_ui_app(api_client=fake_client)
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/demo/demo-123/custom",
        data={"text": "What if the dinner goes silent?"},
    )

    assert response.status_code == 302
    assert "session_id=demo-123" in response.headers["Location"]
    assert fake_client.last_demo_custom_payload == {
        "text": "What if the dinner goes silent?"
    }


def test_demo_swap_post_redirects() -> None:
    """Swap-demo route redirects back to the room."""
    fake_client = FakeUiClient()
    app = create_ui_app(api_client=fake_client)
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post("/demo/demo-123/swap")

    assert response.status_code == 302
    assert "session_id=demo-123" in response.headers["Location"]
    assert fake_client.last_demo_swap_session_id == "demo-123"


def test_demo_page_ignores_stale_session_id() -> None:
    """Demo page falls back to empty state for a stale stored session id."""

    class MissingDemoClient(FakeUiClient):
        """Fake client with no active demo sessions."""

        def list_demo_sessions(self) -> list[DemoSession]:
            """Return no demo sessions."""
            return []

        def get_demo_session(self, session_id: str) -> DemoSession:
            """Mimic API 404 behavior from ApiClient."""
            raise RuntimeError('{"detail":"Demo session not found."}')

    app = create_ui_app(api_client=MissingDemoClient())
    app.config["TESTING"] = True
    client = app.test_client()

    with client.session_transaction() as flask_session:
        flask_session["demo_session_id"] = "demo-missing"

    response = client.get("/demo")

    assert response.status_code == 200
    assert b"No room selected" in response.data
    assert b"demo-missing" not in response.data


def test_ui_models_are_importable() -> None:
    """Typed UI models can be imported and have expected annotations."""
    from src.ui.models import (
        AgentInvocation,
        InterventionDecision,
        TickData,
    )

    assert "run_id" in RunListItem.__annotations__
    assert "ticks" in TrajectoryData.__annotations__
    assert "tick" in TickData.__annotations__
    assert "agent_name" in AgentInvocation.__annotations__
    assert "action" in InterventionDecision.__annotations__


def test_orchestrator_controls_visible() -> None:
    """Orchestrator controls render for active runs."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/?run_id=run-123")

    assert b"Orchestrator" in response.data
    assert b"Run Orchestrator" in response.data


def test_orchestrator_log_tab_visible() -> None:
    """Orchestrator log tab renders."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/?run_id=run-123")

    assert b"tab-orch" in response.data


def test_orchestrate_post_redirects() -> None:
    """Orchestrate route runs cycles and redirects."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/runs/run-123/orchestrate",
        data={"cycles": "3", "goals": "explore\ntest"},
    )

    assert response.status_code == 302


def test_resume_post_redirects() -> None:
    """Resume route resumes a paused run and redirects."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/runs/run-123/resume",
        data={"goals": "continue exploring"},
    )

    assert response.status_code == 302


def test_theme_toggle_present() -> None:
    """Theme toggle button renders with light/dark CSS variables."""
    app = create_ui_app(api_client=FakeUiClient())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/")
    html = response.data

    assert b'id="theme-toggle"' in html
    assert b'data-theme="light"' in html
    assert b'data-theme="dark"' in html
    assert b"ris-theme" in html
