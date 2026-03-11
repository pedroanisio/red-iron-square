"""Campaign orchestration service."""

from __future__ import annotations

import uuid
from typing import Any

from src.api.campaign_models import CampaignRecord, CampaignRunLink, CheckpointRule
from src.api.campaign_store import CampaignStore
from src.api.run_service import RunService


class CampaignService:
    """Orchestrate multi-run research campaigns."""

    def __init__(
        self,
        campaign_store: CampaignStore,
        run_service: RunService,
    ) -> None:
        """Initialize with persistence and run orchestration dependencies."""
        self._store = campaign_store
        self._runs = run_service

    def create_campaign(
        self,
        name: str,
        goals: list[str],
        config_template: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a campaign and its first run."""
        campaign_id = f"camp-{uuid.uuid4().hex[:12]}"
        record = CampaignRecord(
            campaign_id=campaign_id,
            name=name,
            status="active",
            goals=goals,
            config_template=config_template,
        )
        self._store.create_campaign(record)
        run = self._runs.create_run(config_template)
        self._store.add_run(
            CampaignRunLink(
                campaign_id=campaign_id,
                run_id=run["run_id"],
                role="primary",
            )
        )
        result = self._store.get_campaign(campaign_id)
        if result is None:  # pragma: no cover
            msg = f"Campaign just created but not found: {campaign_id}"
            raise RuntimeError(msg)
        return result

    def list_campaigns(self) -> list[dict[str, Any]]:
        """List all campaigns."""
        return self._store.list_campaigns()

    def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        """Get campaign summary with run list."""
        campaign = self._store.get_campaign(campaign_id)
        if campaign is None:
            msg = f"Campaign not found: {campaign_id}"
            raise ValueError(msg)
        campaign["runs"] = self._store.list_campaign_runs(campaign_id)
        return campaign

    def add_branch(
        self,
        campaign_id: str,
        source_run_id: str,
        parent_tick: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Branch a run within campaign context."""
        campaign = self._store.get_campaign(campaign_id)
        if campaign is None:
            msg = f"Campaign not found: {campaign_id}"
            raise ValueError(msg)
        patch: dict[str, Any] = {}
        if temperature is not None:
            patch["temperature"] = temperature
        branch = self._runs.branch_run(
            source_run_id,
            parent_tick=parent_tick,
            patch=patch or None,
        )
        self._store.add_run(
            CampaignRunLink(
                campaign_id=campaign_id,
                run_id=branch["run_id"],
                role="branch",
            )
        )
        return branch

    def get_campaign_summary(self, campaign_id: str) -> dict[str, Any]:
        """Aggregate stats across all campaign runs."""
        campaign = self.get_campaign(campaign_id)
        run_links = campaign["runs"]
        total_ticks = 0
        run_summaries: list[dict[str, Any]] = []
        for link in run_links:
            try:
                run = self._runs.get_run(link["run_id"])
                total_ticks += run.get("tick_count", 0)
                run_summaries.append(run)
            except (ValueError, KeyError):
                continue
        campaign["run_summaries"] = run_summaries
        campaign["total_ticks"] = total_ticks
        campaign["run_count"] = len(run_links)
        return campaign

    def add_checkpoint_rule(
        self,
        campaign_id: str,
        trigger_type: str,
        trigger_config: dict[str, Any],
    ) -> None:
        """Add a checkpoint trigger rule to a campaign."""
        campaign = self._store.get_campaign(campaign_id)
        if campaign is None:
            msg = f"Campaign not found: {campaign_id}"
            raise ValueError(msg)
        self._store.add_checkpoint_rule(
            CheckpointRule(
                campaign_id=campaign_id,
                trigger_type=trigger_type,
                trigger_config=trigger_config,
            )
        )

    def check_triggers(
        self,
        campaign_id: str,
        current_tick: int,
    ) -> list[dict[str, Any]]:
        """Evaluate checkpoint rules and return those that fired."""
        rules = self._store.list_checkpoint_rules(campaign_id)
        fired: list[dict[str, Any]] = []
        for rule in rules:
            if self._should_fire(rule, current_tick):
                self._store.update_rule_fired(rule["rule_id"])
                fired.append(rule)
        return fired

    def _should_fire(self, rule: dict[str, Any], current_tick: int) -> bool:
        """Determine whether a checkpoint rule should fire."""
        if rule["trigger_type"] == "every_n_ticks":
            n: int = rule["trigger_config"].get("n", 10)
            return current_tick > 0 and current_tick % n == 0
        if rule["trigger_type"] == "threshold":
            return rule.get("last_fired_at") is None
        if rule["trigger_type"] == "manual":
            return False
        return False

    def update_status(self, campaign_id: str, status: str) -> None:
        """Update campaign status."""
        self._store.update_campaign_status(campaign_id, status)
