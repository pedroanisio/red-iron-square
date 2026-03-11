"""Helpers for the SQLite run store.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA_SQL = """
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


def parse_json_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode JSON-like text fields in row dictionaries."""
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
