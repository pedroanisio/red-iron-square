"""MetaController for orchestrated simulation runs.

Implements the decide-act-observe loop that drives automated
orchestration of simulation runs, with human checkpoint support
for pausing and resuming.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.api.run_service import RunService
from src.orchestrator.agents import AgentRuntimeProtocol
from src.orchestrator.models import OrchestrationContext, OrchestratorDecision
from src.orchestrator.store import OrchestratorStore


class MetaController:
    """Decide-act-observe loop for automated run orchestration."""

    def __init__(
        self,
        run_service: RunService,
        orchestrator_store: OrchestratorStore,
        agent_runtime: AgentRuntimeProtocol | None = None,
    ) -> None:
        self._runs = run_service
        self._store = orchestrator_store
        self._runtime = agent_runtime

    def run_cycle(
        self,
        run_id: str,
        goals: list[str] | None = None,
        campaign_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute one decide-act-observe cycle.

        Raises:
            ValueError: If the run is paused.
        """
        run = self._runs.get_run(run_id)
        if run["status"] == "paused":
            raise ValueError(f"Run {run_id} is paused. Resume before orchestrating.")

        cycle = self._store.latest_cycle(run_id) + 1
        goals = goals or []
        context = self._build_context(run_id, cycle, goals, campaign_id, run)
        action_type = self._decide(context, run)
        result = self._act(action_type, context, run)
        self._persist_decision(
            run_id,
            campaign_id,
            cycle,
            action_type,
            goals,
            run,
            result,
        )
        self._apply_gate(action_type, run_id)

        return {
            "cycle": cycle,
            "action_type": action_type,
            "result": result,
            "run_status": run["status"],
        }

    def run_auto(
        self,
        run_id: str,
        max_cycles: int = 10,
        goals: list[str] | None = None,
        campaign_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run multiple cycles until termination or max_cycles."""
        results: list[dict[str, Any]] = []
        for _ in range(max_cycles):
            run = self._runs.get_run(run_id)
            if run["status"] in ("paused", "complete"):
                break
            result = self.run_cycle(run_id, goals, campaign_id)
            results.append(result)
            if result["action_type"] in ("pause", "terminate"):
                break
        return results

    def resume(
        self,
        run_id: str,
        goals: list[str] | None = None,
    ) -> dict[str, Any]:
        """Resume a paused run and execute one cycle.

        Raises:
            ValueError: If the run is not paused.
        """
        run = self._runs.get_run(run_id)
        if run["status"] != "paused":
            raise ValueError(f"Run {run_id} is not paused (status: {run['status']})")
        self._runs._store.update_run_status(run_id, "active")
        return self.run_cycle(run_id, goals)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        run_id: str,
        cycle: int,
        goals: list[str],
        campaign_id: str | None,
        run: dict[str, Any],
    ) -> OrchestrationContext:
        """Assemble the context snapshot for one cycle."""
        trajectory = self._runs.get_trajectory(run_id)
        recent = trajectory.get("ticks", [])[-10:]
        latest_state = recent[-1].get("state_after") if recent else None
        return OrchestrationContext(
            run_id=run_id,
            campaign_id=campaign_id,
            cycle=cycle,
            goals=goals,
            recent_ticks=recent,
            latest_state=latest_state,
            run_status=run["status"],
        )

    def _decide(self, context: OrchestrationContext, run: dict[str, Any]) -> str:
        """Decide the next action type based on context."""
        tick_count = run.get("tick_count", 0)
        if tick_count == 0:
            return "scenario"
        if tick_count % 5 == 0:
            return "analyze"
        if self._runtime and tick_count > 2 and tick_count % 3 == 0:
            return "intervene"
        return "scenario"

    def _act(
        self,
        action_type: str,
        context: OrchestrationContext,
        run: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the decided action via dispatch."""
        _Handler = Callable[
            [OrchestrationContext, dict[str, Any]],
            dict[str, Any],
        ]
        dispatch: dict[str, _Handler] = {
            "scenario": self._act_scenario,
            "observe": self._act_observe,
            "analyze": self._act_analyze,
            "intervene": self._act_intervene,
        }
        handler = dispatch.get(action_type)
        if handler:
            result: dict[str, Any] = handler(context, run)
            return result
        if action_type in ("pause", "terminate"):
            return _static_result(action_type, f"Orchestrator decided to {action_type}")
        return _static_result(action_type, "Unknown action")

    def _act_scenario(
        self,
        context: OrchestrationContext,
        run: dict[str, Any],
    ) -> dict[str, Any]:
        """Propose and execute a scenario tick."""
        config = run.get("config", {})
        if self._runtime:
            from src.orchestrator.agents import run_scenario_agent

            result = run_scenario_agent(
                self._runtime,
                current_state=context.latest_state or {},
                trajectory_window=context.recent_ticks,
                goals=context.goals,
            )
            output = result.get("output", {})
            payload = {
                "name": output.get("name", "orchestrated"),
                "values": output.get("values", {}),
            }
            self._runs.step_run(run["run_id"], payload, None)
            return result
        payload = {"name": "auto_step", "values": config.get("personality", {})}
        tick_result = self._runs.step_run(run["run_id"], payload, None)
        return {
            "action_type": "scenario",
            "output": tick_result,
            "rationale": "Auto-step without agent runtime",
        }

    def _act_observe(
        self,
        context: OrchestrationContext,
        run: dict[str, Any],
    ) -> dict[str, Any]:
        """Summarize recent trajectory."""
        if self._runtime:
            from src.orchestrator.agents import run_observer_agent

            return run_observer_agent(self._runtime, ticks=context.recent_ticks)
        return _static_result("observe", "No agent runtime available")

    def _act_analyze(
        self,
        context: OrchestrationContext,
        run: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze recent trajectory."""
        if self._runtime:
            from src.orchestrator.agents import run_analyst_agent

            return run_analyst_agent(self._runtime, ticks=context.recent_ticks)
        return _static_result("analyze", "No agent runtime available")

    def _act_intervene(
        self,
        context: OrchestrationContext,
        run: dict[str, Any],
    ) -> dict[str, Any]:
        """Recommend an intervention."""
        if self._runtime:
            from src.orchestrator.agents import run_intervention_agent

            return run_intervention_agent(
                self._runtime,
                current_state=context.latest_state or {},
                ticks=context.recent_ticks,
                goals=context.goals,
            )
        return _static_result("intervene", "No agent runtime available")

    def _persist_decision(
        self,
        run_id: str,
        campaign_id: str | None,
        cycle: int,
        action_type: str,
        goals: list[str],
        run: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Record the cycle decision in the orchestrator store."""
        decision = OrchestratorDecision(
            run_id=run_id,
            campaign_id=campaign_id,
            cycle=cycle,
            action_type=action_type,
            input_json={"goals": goals, "tick_count": run.get("tick_count", 0)},
            output_json=result,
            rationale=result.get("rationale", ""),
        )
        self._store.record_decision(decision)

    def _apply_gate(self, action_type: str, run_id: str) -> None:
        """Apply status transitions for pause/terminate actions."""
        status_map = {"pause": "paused", "terminate": "complete"}
        new_status = status_map.get(action_type)
        if new_status:
            self._runs._store.update_run_status(run_id, new_status)


def _static_result(action_type: str, rationale: str) -> dict[str, Any]:
    """Build a result dict when no agent runtime is available."""
    return {"action_type": action_type, "output": {}, "rationale": rationale}
