"""Service layer for Two Minds demo sessions.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from src.api.run_service import RunService
from src.constructed_emotion.affect import AffectSignal
from src.demo.llm_service import DemoLLMService
from src.demo.models import DemoEvent, DemoScenario, DemoSessionState
from src.demo.personas import SCRIPTED_SCENARIOS
from src.demo.schemas import DemoScriptedScenarioResponse, DemoSwapResponse
from src.demo.session_store import DemoSessionRecord, DemoSessionStore
from src.demo.state_mapper import (
    build_demo_personas,
    build_initial_agents,
    build_run_config,
    modulate_outcome,
    session_to_response,
    update_snapshot_from_tick,
)
from src.self_model.simulator import SelfAwareTickResult
from src.temporal.tick_result import TickResult

TickResult.model_rebuild(_types_namespace={"AffectSignal": AffectSignal})
SelfAwareTickResult.model_rebuild(_types_namespace={"AffectSignal": AffectSignal})


class DemoSessionService:
    """Coordinate demo sessions, simulation runs, and websocket events."""

    def __init__(
        self,
        runs: RunService,
        store: DemoSessionStore | None = None,
        llm_service: DemoLLMService | None = None,
    ) -> None:
        self._runs = runs
        self._store = store or DemoSessionStore()
        self._llm = llm_service or DemoLLMService()

    def create_session(self, act_number: int) -> dict[str, object]:
        """Create a fresh demo session and broadcast initialization."""
        personas = build_demo_personas()
        session_id = f"demo-{uuid4().hex[:12]}"
        run_ids = {
            key: self._runs.create_run(build_run_config(persona))["run_id"]
            for key, persona in personas.items()
        }
        session = DemoSessionState(
            session_id=session_id,
            act_number=act_number,
            turn_count=0,
            agents=build_initial_agents(personas),
        )
        self._store.create(DemoSessionRecord(session=session, run_ids=run_ids))
        self._emit(
            session_id,
            "session_initialized",
            {"act_number": act_number, "agent_keys": list(session.agents)},
        )
        return session_to_response(session).model_dump()

    def list_sessions(self) -> list[dict[str, object]]:
        """Return all active demo sessions."""
        return [
            session_to_response(record.session).model_dump()
            for record in self._store.list()
        ]

    def get_session(self, session_id: str) -> dict[str, object]:
        """Return one frontend-facing session payload."""
        return session_to_response(self._store.get(session_id).session).model_dump()

    def run_scripted(
        self,
        session_id: str,
        scenario_key: str,
    ) -> dict[str, object]:
        """Advance both agents through one scripted scenario."""
        scenario = SCRIPTED_SCENARIOS[scenario_key]
        record = self._store.get(session_id)
        self._run_turn(record, scenario)
        return DemoScriptedScenarioResponse(
            session_id=session_id,
            scenario_key=scenario_key,
            turn_count=record.session.turn_count,
        ).model_dump()

    def run_custom(self, session_id: str, text: str) -> dict[str, object]:
        """Advance both agents through one neutral custom scenario."""
        record = self._store.get(session_id)
        scenario, invocation = self._llm.enrich_scenario(
            text,
            current_state=self.get_session(session_id),
            trajectory_window=self._recent_ticks(record),
        )
        self._emit(
            session_id,
            "scenario_enriched",
            {
                "scenario_name": scenario.name,
                "description": scenario.description,
                "llm_enabled": invocation is not None,
            },
        )
        self._run_turn(record, scenario)
        return session_to_response(record.session).model_dump()

    def swap_personalities(self, session_id: str) -> dict[str, object]:
        """Swap Luna/Marco trait vectors and reset the session."""
        record = self._store.get(session_id)
        record.swapped = not record.swapped
        personas = build_demo_personas(swapped=record.swapped)
        record.run_ids = {
            key: self._runs.create_run(build_run_config(persona))["run_id"]
            for key, persona in personas.items()
        }
        record.session.reset_for_swap(build_initial_agents(personas))
        self._emit(session_id, "swap_completed", {"swapped": record.swapped})
        return DemoSwapResponse(
            session_id=session_id,
            act_number=record.session.act_number,
            swapped=True,
        ).model_dump()

    def subscribe(self, session_id: str) -> asyncio.Queue[DemoEvent]:
        """Register a websocket subscriber for one session."""
        return self._store.register(session_id)

    def unsubscribe(
        self,
        session_id: str,
        queue: asyncio.Queue[DemoEvent],
    ) -> None:
        """Remove a websocket subscriber."""
        self._store.unregister(session_id, queue)

    def _run_turn(self, record: DemoSessionRecord, scenario: DemoScenario) -> None:
        session = record.session
        self._emit(
            session.session_id,
            "scenario_received",
            {"scenario_key": scenario.key, "description": scenario.description},
        )
        personas = build_demo_personas(swapped=record.swapped)
        for key in ("luna", "marco"):
            agent_outcome = (
                modulate_outcome(
                    scenario.forced_outcome,
                    personas[key],
                    scenario.values,
                )
                if scenario.forced_outcome is not None
                else None
            )
            tick = self._runs.step_run(
                record.run_ids[key],
                {
                    "name": scenario.name,
                    "description": scenario.description,
                    "values": scenario.values,
                },
                agent_outcome,
            )
            narrative = self._llm.build_narrative(session.agents[key], scenario, tick)
            update_snapshot_from_tick(
                session.agents[key],
                tick,
                narrative.text,
                emotion_label=narrative.emotion_label,
            )
            self._emit(
                session.session_id,
                "agent_state_updated",
                {
                    "agent_key": key,
                    "snapshot": next(
                        agent
                        for agent in session_to_response(session).model_dump()["agents"]
                        if agent["key"] == key
                    ),
                },
            )
            self._emit(
                session.session_id,
                "agent_text_started",
                {"agent_key": key, "text": narrative.text},
            )
            self._emit(
                session.session_id,
                "agent_text_completed",
                {"agent_key": key, "text": narrative.text},
            )
            self._emit(
                session.session_id,
                "audio_unavailable",
                {"agent_key": key, "reason": "Audio streaming not configured yet."},
            )
        session.turn_count += 1
        self._emit(
            session.session_id,
            "turn_completed",
            {"turn_count": session.turn_count, "scenario_key": scenario.key},
        )

    def _recent_ticks(self, record: DemoSessionRecord) -> list[dict[str, object]]:
        ticks: list[dict[str, object]] = []
        for run_id in record.run_ids.values():
            trajectory = self._runs.get_trajectory(run_id)
            ticks.extend(trajectory["ticks"][-2:])
        return ticks[-4:]

    def _emit(
        self,
        session_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        self._store.add_event(
            session_id,
            DemoEvent(event_type=event_type, session_id=session_id, payload=payload),
        )
