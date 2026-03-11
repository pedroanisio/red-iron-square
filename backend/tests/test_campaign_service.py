"""Tests for campaign service."""

from __future__ import annotations

import pytest
from src.api.campaign_service import CampaignService
from src.api.campaign_store import CampaignStore
from src.api.run_service import RunService
from src.api.run_store import RunStore


@pytest.fixture()
def services(tmp_path):
    """Build CampaignService with real stores backed by a temp database."""
    db = str(tmp_path / "test.db")
    run_store = RunStore(db)
    run_service = RunService(run_store)
    campaign_store = CampaignStore(db)
    return CampaignService(campaign_store, run_service), run_service


def _default_config():
    """Return a minimal valid run config for testing."""
    return {
        "personality": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
        "actions": [
            {"name": "bold", "modifiers": {"O": 1.0}},
            {"name": "safe", "modifiers": {"C": 0.9}},
        ],
        "temperature": 1.0,
        "seed": 42,
    }


def test_create_campaign_creates_first_run(services) -> None:
    """Creating a campaign should also create the first primary run."""
    svc, _ = services
    result = svc.create_campaign("Test Campaign", ["goal1"], _default_config())
    assert result["name"] == "Test Campaign"
    assert result["status"] == "active"
    runs = svc._store.list_campaign_runs(result["campaign_id"])
    assert len(runs) == 1
    assert runs[0]["role"] == "primary"


def test_list_campaigns(services) -> None:
    """Listing campaigns returns all created campaigns."""
    svc, _ = services
    svc.create_campaign("C1", [], _default_config())
    svc.create_campaign("C2", [], _default_config())
    assert len(svc.list_campaigns()) == 2


def test_get_campaign_includes_runs(services) -> None:
    """Getting a campaign should embed its run links."""
    svc, _ = services
    created = svc.create_campaign("Test", ["g1"], _default_config())
    detail = svc.get_campaign(created["campaign_id"])
    assert "runs" in detail
    assert len(detail["runs"]) == 1


def test_get_campaign_not_found(services) -> None:
    """Fetching a nonexistent campaign raises ValueError."""
    svc, _ = services
    with pytest.raises(ValueError, match="not found"):
        svc.get_campaign("nonexistent")


def test_add_branch_within_campaign(services) -> None:
    """Branching a run inside a campaign links the new run."""
    svc, run_svc = services
    created = svc.create_campaign("Test", [], _default_config())
    cid = created["campaign_id"]
    runs = svc._store.list_campaign_runs(cid)
    run_id = runs[0]["run_id"]
    # Step the run once so branching has something to branch from
    run_svc.step_run(
        run_id,
        {"name": "test", "values": {"O": 0.5}},
        None,
    )
    branch = svc.add_branch(cid, run_id, parent_tick=0)
    assert "run_id" in branch
    all_runs = svc._store.list_campaign_runs(cid)
    assert len(all_runs) == 2
    assert all_runs[1]["role"] == "branch"


def test_campaign_summary_aggregates(services) -> None:
    """Summary should aggregate tick counts across campaign runs."""
    svc, run_svc = services
    created = svc.create_campaign("Test", [], _default_config())
    cid = created["campaign_id"]
    runs = svc._store.list_campaign_runs(cid)
    run_id = runs[0]["run_id"]
    run_svc.step_run(
        run_id,
        {"name": "t", "values": {"O": 0.5}},
        None,
    )
    summary = svc.get_campaign_summary(cid)
    assert summary["total_ticks"] >= 1
    assert summary["run_count"] == 1


def test_checkpoint_rule_every_n_ticks(services) -> None:
    """An every_n_ticks rule should fire at multiples of n."""
    svc, _ = services
    created = svc.create_campaign("Test", [], _default_config())
    cid = created["campaign_id"]
    svc.add_checkpoint_rule(cid, "every_n_ticks", {"n": 5})
    assert len(svc.check_triggers(cid, 3)) == 0
    fired = svc.check_triggers(cid, 5)
    assert len(fired) == 1
    assert fired[0]["trigger_type"] == "every_n_ticks"


def test_checkpoint_threshold_fires_once(services) -> None:
    """A threshold rule should fire once, then not again."""
    svc, _ = services
    created = svc.create_campaign("Test", [], _default_config())
    cid = created["campaign_id"]
    svc.add_checkpoint_rule(cid, "threshold", {"metric": "mood", "value": 0.5})
    assert len(svc.check_triggers(cid, 1)) == 1
    assert len(svc.check_triggers(cid, 2)) == 0


def test_update_campaign_status(services) -> None:
    """Updating status should persist the change."""
    svc, _ = services
    created = svc.create_campaign("Test", [], _default_config())
    svc.update_status(created["campaign_id"], "paused")
    detail = svc.get_campaign(created["campaign_id"])
    assert detail["status"] == "paused"
