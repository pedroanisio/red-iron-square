"""Tests for open-ended decision API endpoints."""

from __future__ import annotations

from tests.api_support import make_client


class TestOpenEndedDecideEndpoint:
    """POST /decide/open with action proposals."""

    def test_classic_proposals(self) -> None:
        """Accept classic proposals and return a chosen action."""
        client = make_client("open_classic")
        resp = client.post(
            "/decide/open",
            json={
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
                "scenario": {"name": "test", "values": {"O": 0.9}},
                "proposals": [
                    {
                        "kind": "classic",
                        "name": "bold",
                        "description": "bold move",
                        "modifiers": {"O": 1.0},
                    },
                    {
                        "kind": "classic",
                        "name": "safe",
                        "description": "safe move",
                        "modifiers": {"C": 0.9},
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["chosen_action"] in ("bold", "safe")
        assert "proposals" in data
        assert len(data["proposals"]) == 2

    def test_mixed_proposals(self) -> None:
        """Accept mixed classic + text proposals."""
        client = make_client("open_mixed")
        resp = client.post(
            "/decide/open",
            json={
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
                "scenario": {"name": "test", "values": {"O": 0.9}},
                "proposals": [
                    {
                        "kind": "classic",
                        "name": "bold",
                        "description": "bold move",
                        "modifiers": {"O": 1.0},
                    },
                    {
                        "kind": "text",
                        "name": "explain",
                        "description": "explain it",
                        "intent": "explain",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["chosen_action"] in ("bold", "explain")

    def test_empty_proposals_returns_error(self) -> None:
        """Reject empty proposal list."""
        client = make_client("open_empty")
        resp = client.post(
            "/decide/open",
            json={
                "personality": {"O": 0.5},
                "scenario": {"name": "test", "values": {"O": 0.5}},
                "proposals": [],
            },
        )
        assert resp.status_code == 422
