"""Stateful run routes.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.run_models import PhaseRecord
from src.api.run_schemas import (
    AssistedStepRequest,
    AssistedStepResponse,
    InterventionRequest,
    InterventionResponse,
    PhaseCreateRequest,
    RunBranchRequest,
    RunCreateRequest,
    RunPatchRequest,
    RunReplayResponse,
    RunSummary,
    RunTickRequest,
    TrajectoryResponse,
)
from src.api.run_service import RunService
from src.llm import AgentRuntime, AnthropicAdapter


def create_run_router(
    run_service: RunService,
    agent_runtime: AgentRuntime | None,
) -> APIRouter:
    """Create the stateful run router."""
    router = APIRouter()

    @router.post("/runs")
    def create_run(request: RunCreateRequest) -> dict[str, Any]:
        """Create a persisted simulation run."""
        return {
            "data": RunSummary(
                **run_service.create_run(request.model_dump(mode="json"))
            ).model_dump()
        }

    @router.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Fetch run summary and latest tick."""
        return {"data": _load_run_summary(run_service, run_id).model_dump()}

    @router.post("/runs/{run_id}/tick")
    def tick_run(run_id: str, request: RunTickRequest) -> dict[str, Any]:
        """Advance a run by one persisted tick."""
        try:
            return {
                "data": run_service.step_run(
                    run_id, request.scenario.model_dump(mode="json"), request.outcome
                )
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc

    @router.get("/runs/{run_id}/trajectory")
    def get_trajectory(run_id: str) -> dict[str, Any]:
        """Fetch the full persisted run trajectory."""
        try:
            trajectory = TrajectoryResponse(**run_service.get_trajectory(run_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return {"data": trajectory.model_dump()}

    @router.patch("/runs/{run_id}/params")
    def patch_run(run_id: str, request: RunPatchRequest) -> dict[str, Any]:
        """Patch mutable run configuration."""
        try:
            updated = RunSummary(
                **run_service.patch_run_params(run_id, request.model_dump(mode="json"))
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return {"data": updated.model_dump()}

    @router.post("/runs/{run_id}/phases")
    def create_phase(run_id: str, request: PhaseCreateRequest) -> dict[str, Any]:
        """Create a phase annotation for a run."""
        try:
            phase = run_service.create_phase(
                run_id,
                PhaseRecord(
                    start_tick=request.start_tick,
                    end_tick=request.end_tick,
                    label=request.label,
                    notes=request.notes,
                ),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return {"data": phase}

    @router.post("/runs/{run_id}/replay")
    def replay_run(run_id: str) -> dict[str, Any]:
        """Create a deterministic replay clone of a run."""
        try:
            replay = RunReplayResponse(run=run_service.replay_run(run_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        return {"data": replay.model_dump()}

    @router.post("/runs/{run_id}/branches")
    def branch_run(run_id: str, request: RunBranchRequest) -> dict[str, Any]:
        """Create a branch from an existing run."""
        try:
            branch = RunReplayResponse(
                run=run_service.branch_run(
                    run_id,
                    parent_tick=request.parent_tick,
                    patch={"temperature": request.temperature},
                )
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"data": branch.model_dump()}

    @router.post("/runs/{run_id}/assist/step")
    def assist_step(run_id: str, request: AssistedStepRequest) -> dict[str, Any]:
        """Use the agent runtime to propose a scenario, execute it, and summarize it."""
        runtime = agent_runtime or AgentRuntime(AnthropicAdapter())
        try:
            run = run_service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        recent_ticks = run_service.get_trajectory(run_id)["ticks"][-request.window :]
        current_state = run["latest_tick"] or {"tick": -1, "config": run["config"]}
        proposal_input = {
            "current_state": current_state,
            "trajectory_window": recent_ticks,
            "goals": request.goals,
        }
        proposal, proposal_invocation = runtime.propose_scenario(**proposal_input)
        invocation_rows = [
            run_service.record_agent_invocation(
                run_id,
                agent_name="scenario_agent",
                purpose="propose_scenario",
                input_payload=proposal_input,
                output_payload=proposal.model_dump(),
                invocation=proposal_invocation,
            )
        ]
        tick = run_service.step_run(
            run_id,
            proposal.model_dump(mode="json", include={"name", "description", "values"}),
            requested_outcome=None,
        )
        summary_input = {"ticks": (recent_ticks + [tick])[-request.window :]}
        narrative, narrative_invocation = runtime.summarize_window(**summary_input)
        invocation_rows.append(
            run_service.record_agent_invocation(
                run_id,
                agent_name="observer_agent",
                purpose="summarize_window",
                input_payload=summary_input,
                output_payload=narrative.model_dump(),
                invocation=narrative_invocation,
            )
        )
        response = AssistedStepResponse(
            scenario=proposal.model_dump(),
            tick=tick,
            narrative=narrative.model_dump(),
            invocations=invocation_rows,
        )
        return {"data": response.model_dump()}

    @router.post("/runs/{run_id}/intervention")
    def recommend_intervention(
        run_id: str, request: InterventionRequest
    ) -> dict[str, Any]:
        """Recommend an intervention and apply patches."""
        runtime = agent_runtime or AgentRuntime(AnthropicAdapter())
        try:
            run = run_service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        recent_ticks = run_service.get_trajectory(run_id)["ticks"][-request.window :]
        current_state = run["latest_tick"] or {"tick": -1, "config": run["config"]}
        model_input = {
            "current_state": current_state,
            "ticks": recent_ticks,
            "goals": request.goals,
        }
        recommendation, invocation = runtime.recommend_intervention(**model_input)
        persisted_invocation = run_service.record_agent_invocation(
            run_id,
            agent_name="intervention_agent",
            purpose="recommend_intervention",
            input_payload=model_input,
            output_payload=recommendation.model_dump(),
            invocation=invocation,
        )
        updated_run = None
        applied = False
        if request.apply_patch and recommendation.action == "patch_params":
            updated_run = run_service.apply_intervention_patch(
                run_id, temperature=recommendation.temperature
            )
            applied = True
        decision = run_service.record_intervention_decision(
            run_id,
            action=recommendation.action,
            reason=recommendation.reason,
            payload=recommendation.model_dump(),
            applied=applied,
        )
        response = InterventionResponse(
            recommendation=recommendation.model_dump(),
            invocation=persisted_invocation,
            decision=decision,
            updated_run=updated_run,
        )
        return {"data": response.model_dump()}

    return router


def _load_run_summary(run_service: RunService, run_id: str) -> RunSummary:
    """Load one run summary or raise a 404."""
    try:
        return RunSummary(**run_service.get_run(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
