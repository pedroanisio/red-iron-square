"""Pydantic schemas for campaign API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CampaignCreateRequest(BaseModel):
    """Create a new campaign."""

    name: str
    goals: list[str] = []
    config_template: dict[str, Any] = {}


class CampaignBranchRequest(BaseModel):
    """Branch a run within a campaign."""

    source_run_id: str
    parent_tick: int | None = None
    temperature: float | None = None


class CheckpointRuleRequest(BaseModel):
    """Add a checkpoint trigger rule."""

    trigger_type: str
    trigger_config: dict[str, Any] = {}


class CheckpointTriggerRequest(BaseModel):
    """Trigger a manual checkpoint."""


class CampaignSummary(BaseModel):
    """Campaign summary response."""

    campaign_id: str
    name: str
    status: str
    goals: list[str]
    config_template: dict[str, Any]
    created_at: str
    updated_at: str
    runs: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    total_ticks: int = 0
    run_count: int = 0
