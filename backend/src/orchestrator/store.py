"""SQLite persistence for orchestrator decisions."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from src.api.run_models import utc_now  # noqa: F401
from src.orchestrator.models import OrchestratorDecision

ORCHESTRATOR_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS orchestrator_decision (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    campaign_id TEXT,
    cycle INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    input_json TEXT NOT NULL DEFAULT '{}',
    output_json TEXT NOT NULL DEFAULT '{}',
    rationale TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


class OrchestratorStore:
    """SQLite persistence for orchestrator decisions."""

    def __init__(self, db_path: str = ".data/red_iron_square.sqlite3") -> None:
        """Initialize store with database path."""
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if they do not exist."""
        with self._lock, self._connect() as conn:
            conn.executescript(ORCHESTRATOR_SCHEMA_SQL)

    def record_decision(self, decision: OrchestratorDecision) -> None:
        """Persist one orchestrator decision."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO orchestrator_decision "
                "(run_id, campaign_id, cycle, action_type, "
                "input_json, output_json, rationale, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.run_id,
                    decision.campaign_id,
                    decision.cycle,
                    decision.action_type,
                    json.dumps(decision.input_json),
                    json.dumps(decision.output_json),
                    decision.rationale,
                    decision.created_at,
                ),
            )

    def list_decisions(self, run_id: str) -> list[dict[str, Any]]:
        """List all decisions for a run, ordered by cycle."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM orchestrator_decision "
                "WHERE run_id = ? ORDER BY cycle, decision_id",
                (run_id,),
            ).fetchall()
        return [_decision_row_to_dict(dict(r)) for r in rows]

    def latest_cycle(self, run_id: str) -> int:
        """Return the highest cycle number for a run, or -1 if none."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(cycle) as max_cycle "
                "FROM orchestrator_decision WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None or row["max_cycle"] is None:
            return -1
        result: int = row["max_cycle"]
        return result


def _decision_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a decision SQLite row to a plain dict."""
    row["input"] = json.loads(row.pop("input_json", "{}"))
    row["output"] = json.loads(row.pop("output_json", "{}"))
    return row
