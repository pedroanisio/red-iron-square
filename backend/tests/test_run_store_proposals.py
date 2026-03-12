"""Tests for proposal persistence in RunStore."""

from __future__ import annotations

import tempfile

from src.api.run_models import TickEventRecord
from src.api.run_store import RunStore


def _make_store() -> RunStore:
    """Create an in-memory store for testing."""
    fd_path = tempfile.mktemp(suffix=".sqlite3")
    return RunStore(fd_path)


class TestRunStoreProposals:
    """Verify proposal metadata round-trips through persistence."""

    def test_tick_without_proposals(self) -> None:
        """Tick events without proposals remain unchanged."""
        store = _make_store()
        run_id = store.create_run(mode="temporal", config={"test": True})
        event = TickEventRecord(
            tick=0,
            scenario={"name": "s1", "values": {}},
            requested_outcome=None,
            result={"action": "bold", "outcome": 0.5},
        )
        store.append_tick(run_id, event)
        ticks = store.list_ticks(run_id)
        assert len(ticks) == 1
        assert ticks[0].proposals == []
        assert ticks[0].result["action"] == "bold"

    def test_tick_with_proposals(self) -> None:
        """Proposal metadata is persisted and retrieved."""
        store = _make_store()
        run_id = store.create_run(mode="temporal", config={"test": True})
        proposals = [
            {"kind": "classic", "name": "bold", "modifiers": {"O": 1.0}},
            {"kind": "text", "name": "explain", "intent": "explain"},
        ]
        event = TickEventRecord(
            tick=0,
            scenario={"name": "s1", "values": {}},
            requested_outcome=None,
            result={"action": "bold", "outcome": 0.5},
            proposals=proposals,
        )
        store.append_tick(run_id, event)
        ticks = store.list_ticks(run_id)
        assert len(ticks) == 1
        assert len(ticks[0].proposals) == 2
        assert ticks[0].proposals[0]["kind"] == "classic"
        assert ticks[0].proposals[1]["intent"] == "explain"
        # Proposals should not leak into result dict
        assert "proposals" not in ticks[0].result

    def test_multiple_ticks_mixed(self) -> None:
        """Mix of ticks with and without proposals."""
        store = _make_store()
        run_id = store.create_run(mode="temporal", config={})
        store.append_tick(
            run_id,
            TickEventRecord(
                tick=0,
                scenario={"name": "s1", "values": {}},
                requested_outcome=None,
                result={"action": "a"},
            ),
        )
        store.append_tick(
            run_id,
            TickEventRecord(
                tick=1,
                scenario={"name": "s2", "values": {}},
                requested_outcome=0.5,
                result={"action": "b"},
                proposals=[{"kind": "tool", "name": "search"}],
            ),
        )
        ticks = store.list_ticks(run_id)
        assert ticks[0].proposals == []
        assert len(ticks[1].proposals) == 1
