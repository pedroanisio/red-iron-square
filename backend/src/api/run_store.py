"""SQLite store for persisted simulation runs.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from src.api.run_models import (
    AgentInvocationRecord,
    InterventionDecisionRecord,
    PhaseRecord,
    TickEventRecord,
    utc_now,
)


class RunStore:
    """Persist runs, ticks, annotations, and agent artifacts in SQLite."""

    def __init__(self, database_path: str) -> None:
        db_parent = Path(database_path).expanduser().resolve().parent
        db_parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(database_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS simulation_run (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    parent_run_id TEXT NULL,
                    parent_tick INTEGER NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tick_event (
                    run_id TEXT NOT NULL,
                    tick INTEGER NOT NULL,
                    scenario_json TEXT NOT NULL,
                    requested_outcome REAL NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, tick)
                );

                CREATE TABLE IF NOT EXISTS phase_annotation (
                    run_id TEXT NOT NULL,
                    start_tick INTEGER NOT NULL,
                    end_tick INTEGER NULL,
                    label TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_invocation (
                    invocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS intervention_decision (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    applied INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def create_run(
        self,
        *,
        mode: str,
        config: dict[str, Any],
        parent_run_id: str | None = None,
        parent_tick: int | None = None,
    ) -> str:
        """Create one persisted run and return its id."""
        run_id = str(uuid4())
        now = utc_now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO simulation_run (
                    run_id, mode, status, config_json,
                    parent_run_id, parent_tick,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    mode,
                    "active",
                    json.dumps(config),
                    parent_run_id,
                    parent_tick,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run record or `None`."""
        row = self._conn.execute(
            """
            SELECT run_id, mode, status, config_json,
                parent_run_id, parent_tick,
                created_at, updated_at
            FROM simulation_run
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "mode": row["mode"],
            "status": row["status"],
            "config": json.loads(row["config_json"]),
            "parent_run_id": row["parent_run_id"],
            "parent_tick": row["parent_tick"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_run_config(self, run_id: str, config: dict[str, Any]) -> None:
        """Replace the stored run configuration."""
        self._update_run(run_id, config_json=json.dumps(config))

    def append_tick(self, run_id: str, event: TickEventRecord) -> None:
        """Persist one tick event."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tick_event (
                    run_id, tick, scenario_json,
                    requested_outcome, result_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.tick,
                    json.dumps(event.scenario),
                    event.requested_outcome,
                    json.dumps(event.result),
                    event.created_at,
                ),
            )
            self._touch_run(run_id)
            self._conn.commit()

    def list_ticks(self, run_id: str) -> list[TickEventRecord]:
        """Return all ticks ordered by tick index."""
        rows = self._conn.execute(
            """
            SELECT tick, scenario_json, requested_outcome, result_json, created_at
            FROM tick_event
            WHERE run_id = ?
            ORDER BY tick ASC
            """,
            (run_id,),
        ).fetchall()
        return [
            TickEventRecord(
                tick=row["tick"],
                scenario=json.loads(row["scenario_json"]),
                requested_outcome=row["requested_outcome"],
                result=json.loads(row["result_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def create_phase(self, run_id: str, phase: PhaseRecord) -> None:
        """Persist one phase annotation."""
        self._insert_simple(
            """
            INSERT INTO phase_annotation (
                run_id, start_tick, end_tick, label, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                phase.start_tick,
                phase.end_tick,
                phase.label,
                phase.notes,
                phase.created_at,
            ),
        )

    def list_phases(self, run_id: str) -> list[dict[str, Any]]:
        """Return all phase annotations for a run."""
        return self._select_json_rows(
            """
            SELECT start_tick, end_tick, label, notes, created_at
            FROM phase_annotation
            WHERE run_id = ?
            ORDER BY start_tick ASC, created_at ASC
            """,
            (run_id,),
        )

    def append_agent_invocation(
        self, run_id: str, invocation: AgentInvocationRecord
    ) -> None:
        """Persist one LLM invocation."""
        self._insert_simple(
            """
            INSERT INTO agent_invocation (
                run_id, agent_name, purpose,
                input_json, output_json, raw_text,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                invocation.agent_name,
                invocation.purpose,
                json.dumps(invocation.input_json),
                json.dumps(invocation.output_json),
                invocation.raw_text,
                json.dumps(invocation.metadata_json),
                invocation.created_at,
            ),
        )

    def list_agent_invocations(self, run_id: str) -> list[dict[str, Any]]:
        """Return persisted LLM invocations."""
        rows = self._conn.execute(
            """
            SELECT agent_name, purpose, input_json,
                output_json, raw_text,
                metadata_json, created_at
            FROM agent_invocation
            WHERE run_id = ?
            ORDER BY invocation_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [
            {
                "agent_name": row["agent_name"],
                "purpose": row["purpose"],
                "input": json.loads(row["input_json"]),
                "output": json.loads(row["output_json"]),
                "raw_text": row["raw_text"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def append_intervention_decision(
        self,
        run_id: str,
        decision: InterventionDecisionRecord,
    ) -> None:
        """Persist one intervention decision."""
        self._insert_simple(
            """
            INSERT INTO intervention_decision (
                run_id, action, reason, payload_json, applied, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                decision.action,
                decision.reason,
                json.dumps(decision.payload_json),
                1 if decision.applied else 0,
                decision.created_at,
            ),
        )

    def list_intervention_decisions(self, run_id: str) -> list[dict[str, Any]]:
        """Return persisted intervention decisions."""
        return self._select_json_rows(
            """
            SELECT action, reason, payload_json AS payload, applied, created_at
            FROM intervention_decision
            WHERE run_id = ?
            ORDER BY decision_id ASC
            """,
            (run_id,),
        )

    def _insert_simple(self, query: str, params: tuple[object, ...]) -> None:
        with self._lock:
            self._conn.execute(query, params)
            self._touch_run(str(params[0]))
            self._conn.commit()

    def _update_run(self, run_id: str, **fields: object) -> None:
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = [*fields.values(), utc_now(), run_id]
        with self._lock:
            self._conn.execute(
                "UPDATE simulation_run "
                f"SET {assignments}, updated_at = ? "
                "WHERE run_id = ?",
                values,
            )
            self._conn.commit()

    def _touch_run(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE simulation_run SET updated_at = ? WHERE run_id = ?",
            (utc_now(), run_id),
        )

    def _select_json_rows(
        self,
        query: str,
        params: tuple[object, ...],
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(query, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key, value in list(item.items()):
                if isinstance(value, str) and value.startswith(("{", "[")):
                    item[key] = json.loads(value)
                if key == "applied":
                    item[key] = bool(value)
            results.append(item)
        return results
