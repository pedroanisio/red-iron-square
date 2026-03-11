"""Tests for campaign store."""

from __future__ import annotations

import uuid
from pathlib import Path

from src.api.campaign_models import CampaignRecord, CampaignRunLink, CheckpointRule
from src.api.campaign_store import CampaignStore


def _store(tmp_path: Path) -> CampaignStore:
    """Create a CampaignStore backed by a temporary database."""
    return CampaignStore(str(tmp_path / "test.db"))


def test_create_and_get_campaign(tmp_path: Path) -> None:
    """Round-trip a campaign through create and get."""
    store = _store(tmp_path)
    cid = f"camp-{uuid.uuid4().hex[:8]}"
    record = CampaignRecord(
        campaign_id=cid,
        name="Test",
        status="active",
        goals=["goal1"],
        config_template={"temperature": 1.0},
    )
    store.create_campaign(record)
    result = store.get_campaign(cid)
    assert result is not None
    assert result["campaign_id"] == cid
    assert result["goals"] == ["goal1"]


def test_list_campaigns(tmp_path: Path) -> None:
    """Listing returns all created campaigns."""
    store = _store(tmp_path)
    for i in range(3):
        store.create_campaign(
            CampaignRecord(
                campaign_id=f"camp-{i}",
                name=f"C{i}",
                status="active",
                goals=[],
                config_template={},
            ),
        )
    assert len(store.list_campaigns()) == 3


def test_update_campaign_status(tmp_path: Path) -> None:
    """Status update is persisted and retrievable."""
    store = _store(tmp_path)
    store.create_campaign(
        CampaignRecord(
            campaign_id="camp-1",
            name="C1",
            status="active",
            goals=[],
            config_template={},
        ),
    )
    store.update_campaign_status("camp-1", "complete")
    result = store.get_campaign("camp-1")
    assert result is not None
    assert result["status"] == "complete"


def test_add_and_list_runs(tmp_path: Path) -> None:
    """Run links are persisted in creation order."""
    store = _store(tmp_path)
    store.create_campaign(
        CampaignRecord(
            campaign_id="camp-1",
            name="C1",
            status="active",
            goals=[],
            config_template={},
        ),
    )
    store.add_run(
        CampaignRunLink(
            campaign_id="camp-1",
            run_id="run-1",
            role="primary",
        ),
    )
    store.add_run(
        CampaignRunLink(
            campaign_id="camp-1",
            run_id="run-2",
            role="branch",
        ),
    )
    runs = store.list_campaign_runs("camp-1")
    assert len(runs) == 2
    assert runs[0]["role"] == "primary"


def test_checkpoint_rules(tmp_path: Path) -> None:
    """Checkpoint rules round-trip correctly."""
    store = _store(tmp_path)
    store.create_campaign(
        CampaignRecord(
            campaign_id="camp-1",
            name="C1",
            status="active",
            goals=[],
            config_template={},
        ),
    )
    store.add_checkpoint_rule(
        CheckpointRule(
            campaign_id="camp-1",
            trigger_type="every_n_ticks",
            trigger_config={"n": 5},
        ),
    )
    rules = store.list_checkpoint_rules("camp-1")
    assert len(rules) == 1
    assert rules[0]["trigger_type"] == "every_n_ticks"
    assert rules[0]["trigger_config"]["n"] == 5


def test_update_rule_fired(tmp_path: Path) -> None:
    """Firing a rule sets last_fired_at timestamp."""
    store = _store(tmp_path)
    store.create_campaign(
        CampaignRecord(
            campaign_id="camp-1",
            name="C1",
            status="active",
            goals=[],
            config_template={},
        ),
    )
    store.add_checkpoint_rule(
        CheckpointRule(
            campaign_id="camp-1",
            trigger_type="manual",
            trigger_config={},
        ),
    )
    rules = store.list_checkpoint_rules("camp-1")
    store.update_rule_fired(rules[0]["rule_id"])
    updated = store.list_checkpoint_rules("camp-1")
    assert updated[0]["last_fired_at"] is not None


def test_get_nonexistent_campaign(tmp_path: Path) -> None:
    """Getting a missing campaign returns None."""
    store = _store(tmp_path)
    assert store.get_campaign("does-not-exist") is None
