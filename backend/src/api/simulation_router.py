"""One-shot simulation routes.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.schemas import ActionInput, DecisionRequest, SimulationRequest
from src.sdk import AgentSDK


def create_simulation_router() -> APIRouter:
    """Create the one-shot simulation router."""
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        """Check basic liveness."""
        return {"status": "ok"}

    @router.post("/decide")
    def decide(request: DecisionRequest) -> dict[str, Any]:
        """Run a one-shot SDK decision."""
        sdk = AgentSDK.default()
        result = sdk.decide(
            sdk.personality(request.personality),
            sdk.scenario(
                request.scenario.values,
                name=request.scenario.name,
                description=request.scenario.description,
            ),
            _build_actions(sdk, request.actions),
            temperature=request.temperature,
            bias=request.bias,
        )
        return {"data": result.model_dump()}

    @router.post("/simulate")
    def simulate(request: SimulationRequest) -> dict[str, Any]:
        """Run a temporal or self-aware trace."""
        sdk = AgentSDK.default()
        personality = sdk.personality(request.personality)
        actions = _build_actions(sdk, request.actions)
        scenarios = [
            sdk.scenario(s.values, name=s.name, description=s.description)
            for s in request.scenarios
        ]
        trace: BaseModel
        if request.self_model is None:
            trace = sdk.simulator(
                personality, actions, temperature=request.temperature
            ).run(
                scenarios,
                outcomes=request.outcomes,
            )
        else:
            trace = sdk.self_aware_simulator(
                personality,
                sdk.initial_self_model(request.self_model),
                actions,
                temperature=request.temperature,
            ).run(scenarios, outcomes=request.outcomes)
        return {"data": trace.model_dump()}

    return router


def _build_actions(sdk: AgentSDK, actions: list[ActionInput]) -> list[Any]:
    """Build SDK actions from API payloads."""
    return [
        sdk.action(action.name, action.modifiers, description=action.description)
        for action in actions
    ]
