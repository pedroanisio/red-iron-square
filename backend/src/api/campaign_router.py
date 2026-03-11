"""FastAPI routes for campaign management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.campaign_schemas import (
    CampaignBranchRequest,
    CampaignCreateRequest,
    CheckpointRuleRequest,
)
from src.api.campaign_service import CampaignService


def create_campaign_router(campaign_service: CampaignService) -> APIRouter:
    """Build campaign API routes."""
    router = APIRouter(prefix="/campaigns", tags=["campaigns"])

    @router.post("")
    def create_campaign(body: CampaignCreateRequest) -> dict[str, Any]:
        """Create a new research campaign."""
        result = campaign_service.create_campaign(
            name=body.name,
            goals=body.goals,
            config_template=body.config_template,
        )
        return {"data": result}

    @router.get("")
    def list_campaigns() -> dict[str, Any]:
        """List all campaigns."""
        return {"data": campaign_service.list_campaigns()}

    @router.get("/{campaign_id}")
    def get_campaign(campaign_id: str) -> dict[str, Any]:
        """Get campaign details with run list."""
        try:
            return {"data": campaign_service.get_campaign(campaign_id)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/{campaign_id}/summary")
    def get_campaign_summary(campaign_id: str) -> dict[str, Any]:
        """Get aggregated campaign statistics."""
        try:
            return {"data": campaign_service.get_campaign_summary(campaign_id)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/{campaign_id}/branch")
    def branch_in_campaign(
        campaign_id: str,
        body: CampaignBranchRequest,
    ) -> dict[str, Any]:
        """Branch a run within campaign context."""
        try:
            result = campaign_service.add_branch(
                campaign_id=campaign_id,
                source_run_id=body.source_run_id,
                parent_tick=body.parent_tick,
                temperature=body.temperature,
            )
            return {"data": result}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/{campaign_id}/rules")
    def add_checkpoint_rule(
        campaign_id: str,
        body: CheckpointRuleRequest,
    ) -> dict[str, Any]:
        """Add a checkpoint trigger rule."""
        try:
            campaign_service.add_checkpoint_rule(
                campaign_id=campaign_id,
                trigger_type=body.trigger_type,
                trigger_config=body.trigger_config,
            )
            return {"data": {"status": "created"}}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/{campaign_id}/checkpoint")
    def trigger_checkpoint(campaign_id: str) -> dict[str, Any]:
        """Trigger manual checkpoint evaluation."""
        try:
            fired = campaign_service.check_triggers(campaign_id, current_tick=0)
            return {"data": {"fired": fired}}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
