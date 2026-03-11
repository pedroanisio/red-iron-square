"""FastAPI routes for orchestration."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.orchestrator.controller import MetaController
from src.orchestrator.schemas import OrchestrateCycleRequest, ResumeRequest


def create_orchestrator_router(controller: MetaController) -> APIRouter:
    """Build orchestration API routes."""
    router = APIRouter(tags=["orchestrator"])

    @router.post("/runs/{run_id}/orchestrate")
    def orchestrate(run_id: str, body: OrchestrateCycleRequest) -> dict[str, Any]:
        """Run N orchestration cycles."""
        try:
            if body.cycles <= 1:
                result = controller.run_cycle(run_id, body.goals, body.campaign_id)
                return {"data": result}
            results = controller.run_auto(
                run_id,
                max_cycles=body.cycles,
                goals=body.goals,
                campaign_id=body.campaign_id,
            )
            return {"data": results}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/orchestrator-log")
    def orchestrator_log(run_id: str) -> dict[str, Any]:
        """List all orchestrator decisions for a run."""
        decisions = controller._store.list_decisions(run_id)
        return {"data": decisions}

    @router.post("/runs/{run_id}/resume")
    def resume(run_id: str, body: ResumeRequest) -> dict[str, Any]:
        """Resume a paused run with optional new goals."""
        try:
            result = controller.resume(run_id, body.goals)
            return {"data": result}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
