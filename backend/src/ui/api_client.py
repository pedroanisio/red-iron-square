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


class ApiClient:
    """Call the FastAPI service from the Flask UI."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        """Fetch the API health endpoint."""
        return self._request("GET", "/health")

    def list_runs(self) -> list[dict[str, Any]]:
        """Fetch all runs."""
        runs: list[dict[str, Any]] = self._request("GET", "/runs")["data"]
        return runs

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create one run."""
        data: dict[str, Any] = self._request("POST", "/runs", payload)["data"]
        return data

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Fetch one run summary."""
        data: dict[str, Any] = self._request("GET", f"/runs/{run_id}")["data"]
        return data

    def get_trajectory(self, run_id: str) -> dict[str, Any]:
        """Fetch one run trajectory."""
        data: dict[str, Any] = self._request("GET", f"/runs/{run_id}/trajectory")[
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
            with urllib.request.urlopen(req) as response:
                result: dict[str, Any] = json.loads(response.read().decode())
                return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode() or exc.reason
            raise RuntimeError(detail) from exc
