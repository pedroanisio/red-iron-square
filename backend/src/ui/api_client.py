"""HTTP client for the Flask UI.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

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


class ApiClient:
    """Call the FastAPI service from the Flask UI."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        """Fetch the API health endpoint."""
        return self._request("GET", "/health")

    def list_runs(self) -> list[RunListItem]:
        """Fetch all runs."""
        runs: list[RunListItem] = self._request("GET", "/runs")["data"]
        return runs

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create one run."""
        data: dict[str, Any] = self._request("POST", "/runs", payload)["data"]
        return data

    def list_demo_sessions(self) -> list[DemoSession]:
        """Fetch all active demo sessions."""
        data: list[DemoSession] = self._request("GET", "/demo/sessions")["data"]
        return data

    def create_demo_session(self, payload: dict[str, Any]) -> DemoSession:
        """Create one demo session."""
        data: DemoSession = self._request("POST", "/demo/sessions", payload)["data"]
        return data

    def get_demo_session(self, session_id: str) -> DemoSession:
        """Fetch one demo session."""
        data: DemoSession = self._request("GET", f"/demo/sessions/{session_id}")["data"]
        return data

    def run_demo_scripted(
        self,
        session_id: str,
        scenario_key: str,
    ) -> DemoScriptedResult:
        """Run one scripted demo scenario."""
        data: DemoScriptedResult = self._request(
            "POST",
            f"/demo/sessions/{session_id}/scripted/{scenario_key}",
        )["data"]
        return data

    def run_demo_custom(self, session_id: str, payload: dict[str, Any]) -> DemoSession:
        """Run one custom demo scenario."""
        data: DemoSession = self._request(
            "POST",
            f"/demo/sessions/{session_id}/scenarios",
            payload,
        )["data"]
        return data

    def swap_demo_personalities(self, session_id: str) -> DemoSwapResult:
        """Swap demo personalities and reset the room."""
        data: DemoSwapResult = self._request(
            "POST",
            f"/demo/sessions/{session_id}/swap",
        )["data"]
        return data

    def get_run(self, run_id: str) -> RunSummary:
        """Fetch one run summary."""
        data: RunSummary = self._request("GET", f"/runs/{run_id}")["data"]
        return data

    def get_trajectory(self, run_id: str) -> TrajectoryData:
        """Fetch one run trajectory."""
        data: TrajectoryData = self._request("GET", f"/runs/{run_id}/trajectory")[
            "data"
        ]
        return data

    def assist_step(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Trigger one assisted step."""
        data: dict[str, Any] = self._request(
            "POST", f"/runs/{run_id}/assist/step", payload
        )["data"]
        return data

    def intervention(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Trigger one intervention recommendation."""
        data: dict[str, Any] = self._request(
            "POST", f"/runs/{run_id}/intervention", payload
        )["data"]
        return data

    def tick(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Trigger one manual tick."""
        data: dict[str, Any] = self._request("POST", f"/runs/{run_id}/tick", payload)[
            "data"
        ]
        return data

    def replay_run(self, run_id: str) -> ReplayResult:
        """Create a deterministic replay clone."""
        data: ReplayResult = self._request("POST", f"/runs/{run_id}/replay")["data"]
        return data

    def branch_run(
        self,
        run_id: str,
        payload: dict[str, Any],
    ) -> BranchResult:
        """Create a branch from an existing run."""
        data: BranchResult = self._request("POST", f"/runs/{run_id}/branches", payload)[
            "data"
        ]
        return data

    def list_campaigns(self) -> list[dict[str, Any]]:
        """Fetch all campaigns."""
        data: list[dict[str, Any]] = self._request("GET", "/campaigns")["data"]
        return data

    def create_campaign(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a campaign."""
        data: dict[str, Any] = self._request("POST", "/campaigns", payload)["data"]
        return data

    def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        """Fetch one campaign with runs."""
        data: dict[str, Any] = self._request("GET", f"/campaigns/{campaign_id}")["data"]
        return data

    def get_campaign_summary(self, campaign_id: str) -> dict[str, Any]:
        """Fetch campaign summary with aggregated stats."""
        data: dict[str, Any] = self._request(
            "GET", f"/campaigns/{campaign_id}/summary"
        )["data"]
        return data

    def orchestrate(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run orchestration cycles."""
        data: dict[str, Any] = self._request(
            "POST",
            f"/runs/{run_id}/orchestrate",
            payload,
        )["data"]
        return data

    def orchestrator_log(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch orchestrator decision log."""
        data: list[dict[str, Any]] = self._request(
            "GET",
            f"/runs/{run_id}/orchestrator-log",
        )["data"]
        return data

    def resume_run(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Resume a paused run."""
        data: dict[str, Any] = self._request(
            "POST",
            f"/runs/{run_id}/resume",
            payload,
        )["data"]
        return data

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            method=method,
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result: dict[str, Any] = json.loads(response.read().decode())
                return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode() or exc.reason
            raise RuntimeError(detail) from exc
