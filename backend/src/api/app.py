"""FastAPI application exposing the SDK as HTTP endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.api.schemas import ActionInput, DecisionRequest, SimulationRequest
from src.sdk import AgentSDK


def _build_actions(sdk: AgentSDK, actions: list[ActionInput]) -> list[Any]:
    """Build SDK actions from API payloads."""
    return [
        sdk.action(
            action.name,
            action.modifiers,
            description=action.description,
        )
        for action in actions
    ]


def create_app() -> Any:
    """Create the FastAPI app lazily so the dependency stays optional."""
    from fastapi import FastAPI

    app = FastAPI(
        title="Red Iron Square API",
        version="0.1.0",
        summary="HTTP transport for the personality-driven simulation SDK.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Basic liveness endpoint."""
        return {"status": "ok"}

    @app.post("/decide")
    def decide(request: DecisionRequest) -> dict[str, Any]:
        """Run a one-shot SDK decision."""
        sdk = AgentSDK.default()
        personality = sdk.personality(request.personality)
        scenario = sdk.scenario(
            request.scenario.values,
            name=request.scenario.name,
            description=request.scenario.description,
        )
        actions = _build_actions(sdk, request.actions)
        result = sdk.decide(
            personality,
            scenario,
            actions,
            temperature=request.temperature,
            bias=request.bias,
        )
        return {"data": result.model_dump()}

    @app.post("/simulate")
    def simulate(request: SimulationRequest) -> dict[str, Any]:
        """Run a temporal or self-aware simulation trace."""
        sdk = AgentSDK.default()
        personality = sdk.personality(request.personality)
        actions = _build_actions(sdk, request.actions)
        scenarios = [
            sdk.scenario(
                scenario.values,
                name=scenario.name,
                description=scenario.description,
            )
            for scenario in request.scenarios
        ]
        if request.self_model is None:
            client = sdk.simulator(
                personality,
                actions,
                temperature=request.temperature,
            )
            trace: BaseModel = client.run(
                scenarios,
                outcomes=request.outcomes,
            )
        else:
            sa_client = sdk.self_aware_simulator(
                personality,
                sdk.initial_self_model(request.self_model),
                actions,
                temperature=request.temperature,
            )
            trace = sa_client.run(
                scenarios,
                outcomes=request.outcomes,
            )
        return {"data": trace.model_dump()}

    return app
