"""Run application service.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Any, cast

from src.api.run_client_builder import RunClientBuilder
from src.api.run_models import (
    AgentInvocationRecord,
    InterventionDecisionRecord,
    PhaseRecord,
    TickEventRecord,
)
from src.api.run_store import RunStore
from src.llm.schemas import LLMInvocationResult
from src.sdk.builders import build_registry, build_scenario

SimClient = Any  # TemporalSimulationClient | SelfModelSimulationClient

_REGISTRY = build_registry()


class RunService:
    """Coordinate persisted runs with simulator reconstruction."""

    def __init__(
        self,
        store: RunStore,
        client_builder: RunClientBuilder | None = None,
    ) -> None:
        self._store = store
        self._builder = client_builder or RunClientBuilder()
        self._client_cache: dict[str, SimClient] = {}

    def list_runs(self) -> list[dict[str, Any]]:
        """Return all runs with tick counts, most recent first."""
        runs = self._store.list_runs()
        for run in runs:
            ticks = self._store.list_ticks(run["run_id"])
            run["tick_count"] = len(ticks)
        return runs

    def create_run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Create one run and return its summary."""
        mode = "self_aware" if config.get("self_model") is not None else "temporal"
        return self.get_run(self._store.create_run(mode=mode, config=config))

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Return one run summary."""
        run = self._require_run(run_id)
        ticks = self._store.list_ticks(run_id)
        return {
            **run,
            "tick_count": len(ticks),
            "latest_tick": ticks[-1].result if ticks else None,
            "phases": self._store.list_phases(run_id),
            "agent_invocation_count": len(self._store.list_agent_invocations(run_id)),
            "intervention_count": len(self._store.list_intervention_decisions(run_id)),
        }

    def _get_or_build_client(self, run_id: str) -> SimClient:
        """Return a cached simulator client, rebuilding only on cache miss."""
        if run_id not in self._client_cache:
            run = self._require_run(run_id)
            ticks = self._store.list_ticks(run_id)
            self._client_cache[run_id] = self._builder.build(run["config"], ticks)
        return self._client_cache[run_id]

    def _evict_client(self, run_id: str) -> None:
        """Remove a cached client so the next access rebuilds it."""
        self._client_cache.pop(run_id, None)

    def step_run(
        self,
        run_id: str,
        scenario_payload: dict[str, Any],
        requested_outcome: float | None,
    ) -> dict[str, Any]:
        """Execute and persist one tick."""
        client = self._get_or_build_client(run_id)
        scenario = build_scenario(
            scenario_payload["values"],
            _REGISTRY,
            name=scenario_payload.get("name", ""),
            description=scenario_payload.get("description", ""),
        )
        result = cast(
            dict[str, Any],
            client.tick(scenario, outcome=requested_outcome).model_dump(),
        )
        self._store.append_tick(
            run_id,
            TickEventRecord(
                tick=result["tick"],
                scenario=scenario_payload,
                requested_outcome=requested_outcome,
                result=result,
            ),
        )
        return result

    def get_trajectory(self, run_id: str) -> dict[str, Any]:
        """Return the persisted trajectory for one run."""
        run = self._require_run(run_id)
        return {
            "run_id": run["run_id"],
            "tick_count": len(self._store.list_ticks(run_id)),
            "ticks": [tick.result for tick in self._store.list_ticks(run_id)],
            "phases": self._store.list_phases(run_id),
            "agent_invocations": self._store.list_agent_invocations(run_id),
            "interventions": self._store.list_intervention_decisions(run_id),
        }

    def patch_run_params(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Patch mutable run config and return the updated summary."""
        run = self._require_run(run_id)
        updated_config = {
            **run["config"],
            **{k: v for k, v in patch.items() if v is not None},
        }
        self._store.update_run_config(run_id, updated_config)
        self._evict_client(run_id)
        return self.get_run(run_id)

    def create_phase(self, run_id: str, phase: PhaseRecord) -> dict[str, Any]:
        """Persist one phase annotation."""
        self._require_run(run_id)
        self._store.create_phase(run_id, phase)
        return phase.__dict__

    def record_agent_invocation(
        self,
        run_id: str,
        *,
        agent_name: str,
        purpose: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        invocation: LLMInvocationResult,
    ) -> dict[str, Any]:
        """Persist one agent invocation and return its stored form."""
        self._require_run(run_id)
        record = AgentInvocationRecord(
            agent_name=agent_name,
            purpose=purpose,
            input_json=input_payload,
            output_json=output_payload,
            raw_text=invocation.raw_text,
            metadata_json=invocation.metadata.model_dump(),
        )
        self._store.append_agent_invocation(run_id, record)
        return {
            "agent_name": record.agent_name,
            "purpose": record.purpose,
            "input": record.input_json,
            "output": record.output_json,
            "raw_text": record.raw_text,
            "metadata": record.metadata_json,
            "created_at": record.created_at,
        }

    def record_intervention_decision(
        self,
        run_id: str,
        *,
        action: str,
        reason: str,
        payload: dict[str, Any],
        applied: bool,
    ) -> dict[str, Any]:
        """Persist one intervention decision."""
        self._require_run(run_id)
        record = InterventionDecisionRecord(
            action=action,
            reason=reason,
            payload_json=payload,
            applied=applied,
        )
        self._store.append_intervention_decision(run_id, record)
        return {
            "action": record.action,
            "reason": record.reason,
            "payload": record.payload_json,
            "applied": record.applied,
            "created_at": record.created_at,
        }

    def apply_intervention_patch(
        self, run_id: str, *, temperature: float | None
    ) -> dict[str, Any]:
        """Apply a supported intervention patch explicitly."""
        return self.patch_run_params(run_id, {"temperature": temperature})

    def replay_run(self, run_id: str) -> dict[str, Any]:
        """Create a deterministic replay clone."""
        return self._clone_run(run_id, cutoff=None, patch=None)

    def branch_run(
        self,
        run_id: str,
        *,
        parent_tick: int | None,
        patch: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create a branch from a parent run up to a cutoff tick."""
        return self._clone_run(run_id, cutoff=parent_tick, patch=patch)

    def _clone_run(
        self,
        run_id: str,
        *,
        cutoff: int | None,
        patch: dict[str, Any] | None,
    ) -> dict[str, Any]:
        run = self._require_run(run_id)
        ticks = self._store.list_ticks(run_id)
        if cutoff is not None and (cutoff < 0 or cutoff >= len(ticks)):
            raise ValueError(f"parent_tick must be between 0 and {len(ticks) - 1}")
        config = {
            **run["config"],
            **{k: v for k, v in (patch or {}).items() if v is not None},
        }
        clone_id = self._store.create_run(
            mode=run["mode"],
            config=config,
            parent_run_id=run_id,
            parent_tick=cutoff
            if cutoff is not None
            else (ticks[-1].tick if ticks else None),
        )
        copied_ticks = ticks if cutoff is None else ticks[: cutoff + 1]
        for phase in self._store.list_phases(run_id):
            if cutoff is not None and phase["start_tick"] > cutoff:
                continue
            end_tick = phase["end_tick"]
            if cutoff is not None and end_tick is not None:
                end_tick = min(end_tick, cutoff)
            self._store.create_phase(
                clone_id,
                PhaseRecord(
                    start_tick=phase["start_tick"],
                    end_tick=end_tick,
                    label=phase["label"],
                    notes=phase["notes"],
                ),
            )
        for tick in copied_ticks:
            self.step_run(clone_id, tick.scenario, tick.requested_outcome)
        return self.get_run(clone_id)

    def _require_run(self, run_id: str) -> dict[str, Any]:
        run = self._store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        return run
