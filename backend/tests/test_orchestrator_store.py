"""Tests for orchestrator store."""

from __future__ import annotations

from pathlib import Path

from src.orchestrator.models import OrchestratorDecision
from src.orchestrator.store import OrchestratorStore


def _store(tmp_path: Path) -> OrchestratorStore:
    """Create a test store backed by a temporary database."""
    return OrchestratorStore(str(tmp_path / "test.db"))


def test_record_and_list_decisions(tmp_path: Path) -> None:
    """Record two decisions and verify list returns them in order."""
    store = _store(tmp_path)
    store.record_decision(
        OrchestratorDecision(
            run_id="run-1",
            campaign_id=None,
            cycle=0,
            action_type="scenario",
            input_json={"goal": "explore"},
            output_json={"scenario": "bold_probe"},
            rationale="Initial exploration",
        )
    )
    store.record_decision(
        OrchestratorDecision(
            run_id="run-1",
            campaign_id=None,
            cycle=1,
            action_type="observe",
            input_json={},
            output_json={"summary": "stable"},
            rationale="Post-tick observation",
        )
    )
    decisions = store.list_decisions("run-1")
    assert len(decisions) == 2
    assert decisions[0]["action_type"] == "scenario"
    assert decisions[1]["action_type"] == "observe"
    assert decisions[0]["input"]["goal"] == "explore"


def test_latest_cycle(tmp_path: Path) -> None:
    """Verify latest_cycle tracks the maximum cycle number."""
    store = _store(tmp_path)
    assert store.latest_cycle("run-1") == -1
    store.record_decision(
        OrchestratorDecision(
            run_id="run-1",
            campaign_id=None,
            cycle=0,
            action_type="scenario",
            input_json={},
            output_json={},
            rationale="",
        )
    )
    assert store.latest_cycle("run-1") == 0
    store.record_decision(
        OrchestratorDecision(
            run_id="run-1",
            campaign_id=None,
            cycle=3,
            action_type="terminate",
            input_json={},
            output_json={},
            rationale="",
        )
    )
    assert store.latest_cycle("run-1") == 3


def test_decisions_isolated_by_run(tmp_path: Path) -> None:
    """Decisions for different runs do not leak across queries."""
    store = _store(tmp_path)
    store.record_decision(
        OrchestratorDecision(
            run_id="run-1",
            campaign_id=None,
            cycle=0,
            action_type="scenario",
            input_json={},
            output_json={},
            rationale="",
        )
    )
    store.record_decision(
        OrchestratorDecision(
            run_id="run-2",
            campaign_id=None,
            cycle=0,
            action_type="observe",
            input_json={},
            output_json={},
            rationale="",
        )
    )
    assert len(store.list_decisions("run-1")) == 1
    assert len(store.list_decisions("run-2")) == 1


def test_decision_with_campaign(tmp_path: Path) -> None:
    """Campaign ID is persisted and returned correctly."""
    store = _store(tmp_path)
    store.record_decision(
        OrchestratorDecision(
            run_id="run-1",
            campaign_id="camp-1",
            cycle=0,
            action_type="analyze",
            input_json={},
            output_json={},
            rationale="Campaign checkpoint",
        )
    )
    decisions = store.list_decisions("run-1")
    assert decisions[0]["campaign_id"] == "camp-1"
