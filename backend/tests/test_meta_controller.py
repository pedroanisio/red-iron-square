"""Tests for MetaController orchestration loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from src.api.run_service import RunService
from src.api.run_store import RunStore
from src.orchestrator.controller import MetaController
from src.orchestrator.store import OrchestratorStore

if TYPE_CHECKING:
    from pathlib import Path

_Fixture = tuple[MetaController, RunService, OrchestratorStore, str]


@pytest.fixture()
def setup(tmp_path: Path) -> _Fixture:
    """Create a controller with an active run for testing."""
    db = str(tmp_path / "test.db")
    run_store = RunStore(db)
    run_service = RunService(run_store)
    orch_store = OrchestratorStore(db)
    controller = MetaController(run_service, orch_store)
    config = {
        "personality": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
        "actions": [
            {"name": "bold", "modifiers": {"O": 1.0}},
            {"name": "safe", "modifiers": {"C": 0.9}},
        ],
        "temperature": 1.0,
        "seed": 42,
    }
    run = run_service.create_run(config)
    return controller, run_service, orch_store, run["run_id"]


def test_single_cycle(
    setup: _Fixture,
) -> None:
    """One cycle persists a decision and returns expected structure."""
    controller, _, orch_store, run_id = setup
    result = controller.run_cycle(run_id)
    assert result["cycle"] == 0
    assert result["action_type"] == "scenario"
    decisions = orch_store.list_decisions(run_id)
    assert len(decisions) == 1


def test_multiple_cycles(
    setup: _Fixture,
) -> None:
    """Successive cycles increment the cycle counter."""
    controller, _, orch_store, run_id = setup
    for i in range(3):
        result = controller.run_cycle(run_id)
        assert result["cycle"] == i
    decisions = orch_store.list_decisions(run_id)
    assert len(decisions) == 3


def test_auto_run(
    setup: _Fixture,
) -> None:
    """run_auto executes multiple cycles up to the limit."""
    controller, _, orch_store, run_id = setup
    results = controller.run_auto(run_id, max_cycles=5)
    assert len(results) == 5
    decisions = orch_store.list_decisions(run_id)
    assert len(decisions) == 5


def test_resume_paused_run(
    setup: _Fixture,
) -> None:
    """Resuming a paused run transitions back to active and runs a cycle."""
    controller, run_service, _, run_id = setup
    controller.run_cycle(run_id)
    run_service._store.update_run_status(run_id, "paused")
    run = run_service.get_run(run_id)
    assert run["status"] == "paused"
    result = controller.resume(run_id)
    assert result["action_type"] == "scenario"


def test_cannot_orchestrate_paused_run(
    setup: _Fixture,
) -> None:
    """run_cycle raises when the run is paused."""
    controller, run_service, _, run_id = setup
    run_service._store.update_run_status(run_id, "paused")
    with pytest.raises(ValueError, match="paused"):
        controller.run_cycle(run_id)


def test_cannot_resume_active_run(
    setup: _Fixture,
) -> None:
    """Resume raises when the run is not paused."""
    controller, _, _, run_id = setup
    with pytest.raises(ValueError, match="not paused"):
        controller.resume(run_id)
