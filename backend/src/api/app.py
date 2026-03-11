"""FastAPI app factory.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.api.campaign_router import create_campaign_router
from src.api.campaign_service import CampaignService
from src.api.campaign_store import CampaignStore
from src.api.run_router import create_run_router
from src.api.run_service import RunService
from src.api.run_store import RunStore
from src.api.simulation_router import create_simulation_router
from src.llm import AgentRuntime
from src.orchestrator.controller import MetaController
from src.orchestrator.router import create_orchestrator_router
from src.orchestrator.store import OrchestratorStore


def create_app(
    database_path: str | None = None,
    agent_runtime: AgentRuntime | None = None,
) -> Any:
    """Create the FastAPI app."""
    from fastapi import FastAPI

    resolved_db_path = database_path or str(
        Path.cwd().joinpath(".data", "red_iron_square.sqlite3")
    )
    app = FastAPI(
        title="Red Iron Square API",
        version="0.1.0",
        summary="HTTP transport for the personality-driven simulation SDK.",
    )
    run_service = RunService(RunStore(resolved_db_path))
    campaign_service = CampaignService(
        CampaignStore(resolved_db_path),
        run_service,
    )
    orchestrator_store = OrchestratorStore(resolved_db_path)
    controller = MetaController(run_service, orchestrator_store, agent_runtime)
    app.include_router(create_simulation_router())
    app.include_router(create_run_router(run_service, agent_runtime))
    app.include_router(create_campaign_router(campaign_service))
    app.include_router(create_orchestrator_router(controller))
    return app
