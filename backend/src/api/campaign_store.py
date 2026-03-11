"""SQLite persistence for campaigns."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from src.api.campaign_models import CampaignRecord, CampaignRunLink, CheckpointRule
from src.api.run_models import utc_now

CAMPAIGN_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS campaign (
    campaign_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    goals_json TEXT NOT NULL DEFAULT '[]',
    config_template_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS campaign_run (
    campaign_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'primary',
    created_at TEXT NOT NULL,
    PRIMARY KEY (campaign_id, run_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_rule (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    trigger_config_json TEXT NOT NULL DEFAULT '{}',
    last_fired_at TEXT,
    FOREIGN KEY (campaign_id) REFERENCES campaign(campaign_id)
);
"""


class CampaignStore:
    """SQLite persistence for campaigns."""

    def __init__(self, db_path: str = ".data/red_iron_square.sqlite3") -> None:
        """Initialize store with database path."""
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection with row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if they do not exist."""
        with self._lock, self._connect() as conn:
            conn.executescript(CAMPAIGN_SCHEMA_SQL)

    def create_campaign(self, record: CampaignRecord) -> None:
        """Persist a new campaign."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO campaign "
                "(campaign_id, name, status, goals_json,"
                " config_template_json, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.campaign_id,
                    record.name,
                    record.status,
                    json.dumps(record.goals),
                    json.dumps(record.config_template),
                    record.created_at,
                    record.updated_at,
                ),
            )

    def get_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        """Fetch one campaign by ID."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM campaign WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        if row is None:
            return None
        return _campaign_row_to_dict(dict(row))

    def list_campaigns(self) -> list[dict[str, Any]]:
        """Fetch all campaigns ordered by creation time descending."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM campaign ORDER BY created_at DESC",
            ).fetchall()
        return [_campaign_row_to_dict(dict(r)) for r in rows]

    def update_campaign_status(self, campaign_id: str, status: str) -> None:
        """Update campaign status and refresh updated_at."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE campaign SET status = ?, updated_at = ? WHERE campaign_id = ?",
                (status, utc_now(), campaign_id),
            )

    def add_run(self, link: CampaignRunLink) -> None:
        """Associate a run with a campaign."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO campaign_run"
                " (campaign_id, run_id, role, created_at)"
                " VALUES (?, ?, ?, ?)",
                (link.campaign_id, link.run_id, link.role, link.created_at),
            )

    def list_campaign_runs(self, campaign_id: str) -> list[dict[str, Any]]:
        """List all runs belonging to a campaign."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM campaign_run WHERE campaign_id = ? ORDER BY created_at",
                (campaign_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_checkpoint_rule(self, rule: CheckpointRule) -> None:
        """Add a checkpoint trigger rule."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO checkpoint_rule"
                " (campaign_id, trigger_type, trigger_config_json,"
                " last_fired_at) VALUES (?, ?, ?, ?)",
                (
                    rule.campaign_id,
                    rule.trigger_type,
                    json.dumps(rule.trigger_config),
                    rule.last_fired_at,
                ),
            )

    def list_checkpoint_rules(self, campaign_id: str) -> list[dict[str, Any]]:
        """List all checkpoint rules for a campaign."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM checkpoint_rule WHERE campaign_id = ? ORDER BY rule_id",
                (campaign_id,),
            ).fetchall()
        return [_rule_row_to_dict(dict(r)) for r in rows]

    def update_rule_fired(self, rule_id: int) -> None:
        """Update last_fired_at timestamp on a checkpoint rule."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE checkpoint_rule SET last_fired_at = ? WHERE rule_id = ?",
                (utc_now(), rule_id),
            )


def _campaign_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a campaign SQLite row to a plain dict."""
    row["goals"] = json.loads(row.pop("goals_json", "[]"))
    row["config_template"] = json.loads(
        row.pop("config_template_json", "{}"),
    )
    return row


def _rule_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a checkpoint_rule row to a plain dict."""
    row["trigger_config"] = json.loads(
        row.pop("trigger_config_json", "{}"),
    )
    return row
