"""Domain models for campaign orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.api.run_models import utc_now


@dataclass(frozen=True)
class CampaignRecord:
    """One research campaign coordinating multiple runs."""

    campaign_id: str
    name: str
    status: str  # active, paused, complete
    goals: list[str]
    config_template: dict[str, object]
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class CampaignRunLink:
    """Associates a run with a campaign."""

    campaign_id: str
    run_id: str
    role: str  # primary, branch, replay
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class CheckpointRule:
    """Trigger rule for automated campaign checkpoints."""

    campaign_id: str
    trigger_type: str  # every_n_ticks, threshold, manual
    trigger_config: dict[str, object]
    last_fired_at: str | None = None
