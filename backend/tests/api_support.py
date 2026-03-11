"""Shared API test helpers.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from src.api import create_app
from src.llm.schemas import LLMInvocationMetadata, LLMInvocationResult


class FakeAgentRuntime:
    """Deterministic fake runtime for API tests."""

    def propose_scenario(self, **kwargs: object) -> tuple[object, LLMInvocationResult]:
        from src.llm.schemas import ScenarioProposal

        return (
            ScenarioProposal(
                name="llm_probe",
                description="LLM-generated social probe.",
                values={"O": 0.7, "N": 0.2},
                rationale="Probe novelty with low stress.",
            ),
            LLMInvocationResult(
                raw_text='{"name":"llm_probe"}',
                metadata=LLMInvocationMetadata(model="fake-model"),
            ),
        )

    def summarize_window(self, **kwargs: object) -> tuple[object, LLMInvocationResult]:
        from src.llm.schemas import NarrativeChunk

        return (
            NarrativeChunk(
                summary="The run remains exploratory but stable.",
                tick_start=0,
                tick_end=0,
                evidence=["Outcome remained bounded."],
            ),
            LLMInvocationResult(
                raw_text='{"summary":"The run remains exploratory but stable."}',
                metadata=LLMInvocationMetadata(model="fake-model"),
            ),
        )

    def recommend_intervention(
        self,
        **kwargs: object,
    ) -> tuple[object, LLMInvocationResult]:
        from src.llm.schemas import InterventionRecommendation

        return (
            InterventionRecommendation(
                action="patch_params",
                reason="Lower temperature to stabilize choices.",
                temperature=0.55,
            ),
            LLMInvocationResult(
                raw_text='{"action":"patch_params","temperature":0.55}',
                metadata=LLMInvocationMetadata(model="fake-model"),
            ),
        )


def make_client(test_name: str) -> TestClient:
    """Build an isolated API client for one test."""
    return TestClient(
        create_app(make_database_path(test_name), agent_runtime=FakeAgentRuntime())
    )


def make_database_path(test_name: str) -> str:
    """Reserve one unique SQLite path for a test case."""
    fd, raw_path = tempfile.mkstemp(suffix=".sqlite3", prefix=f"{test_name}-")
    os.close(fd)
    database_path = Path(raw_path)
    database_path.unlink(missing_ok=True)
    return str(database_path)


def create_base_run(client: TestClient) -> str:
    """Create a standard temporal run for API tests."""
    response = client.post(
        "/runs",
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
            "actions": [
                {"name": "bold", "modifiers": {"O": 1.0, "R": 0.8, "N": -0.3}},
                {"name": "safe", "modifiers": {"C": 0.9, "T": 0.8}},
            ],
            "temperature": 1.0,
            "seed": 42,
        },
    )
    return response.json()["data"]["run_id"]
