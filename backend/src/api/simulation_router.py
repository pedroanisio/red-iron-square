"""One-shot simulation routes.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.schemas import (
    ActionInput,
    DecisionRequest,
    OpenEndedDecisionRequest,
    SimulationRequest,
)
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

    @router.post("/decide/open")
    def decide_open(request: OpenEndedDecisionRequest) -> dict[str, Any]:
        """Run an open-ended decision with action proposals."""
        if not request.proposals:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=422,
                detail="At least one proposal required.",
            )
        proposals = _build_proposals(request.proposals)
        sdk = AgentSDK.with_open_actions()
        personality = sdk.personality(request.personality)
        scenario = sdk.scenario(
            request.scenario.values,
            name=request.scenario.name,
            description=request.scenario.description,
        )
        from src.action_space.encoder import ActionEncoder, HeuristicEncoderBackend

        encoder = ActionEncoder(
            dimension_registry=sdk.registry,
            backend=HeuristicEncoderBackend(),
        )
        actions = encoder.encode_batch(proposals)
        result = sdk.decide(
            personality,
            scenario,
            actions,
            temperature=request.temperature,
        )
        return {
            "data": {
                **result.model_dump(),
                "proposals": [p.model_dump() for p in proposals],
            }
        }

    return router


def _build_proposals(inputs: list[Any]) -> list[Any]:
    """Convert API proposal inputs to domain proposal objects."""
    from src.action_space.proposal import (
        ClassicActionProposal,
        TextActionProposal,
        ToolActionProposal,
        _ProposalBase,
    )

    proposals: list[_ProposalBase] = []
    for inp in inputs:
        if inp.kind == "classic":
            proposals.append(
                ClassicActionProposal(
                    name=inp.name,
                    description=inp.description,
                    modifiers=inp.modifiers or {},
                )
            )
        elif inp.kind == "text":
            proposals.append(
                TextActionProposal(
                    name=inp.name,
                    description=inp.description,
                    intent=inp.intent or "",
                )
            )
        elif inp.kind == "tool":
            proposals.append(
                ToolActionProposal(
                    name=inp.name,
                    description=inp.description,
                    tool_name=inp.tool_name or inp.name,
                    tool_args=inp.tool_args or {},
                )
            )
        else:
            proposals.append(
                ClassicActionProposal(
                    name=inp.name,
                    description=inp.description,
                    modifiers=inp.modifiers or {},
                )
            )
    return proposals


def _build_actions(sdk: AgentSDK, actions: list[ActionInput]) -> list[Any]:
    """Build SDK actions from API payloads."""
    return [
        sdk.action(action.name, action.modifiers, description=action.description)
        for action in actions
    ]
