# ruff: noqa: E501
from pathlib import Path

import pytest

from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.operations import TombstoneOperator, TombstoneOutbox


def projection():
    return project_tombstone_identities([{"message_fullname": "t3_x", "source_revision_id": None}])


def test_submit_is_idempotent_and_applied_replay_does_not_call_backends(tmp_path: Path):
    calls = []
    operator = TombstoneOperator(TombstoneOutbox(tmp_path / "state.db"),
        sqlite_delete=lambda p, **k: (calls.append("sqlite") or {"deleted_count": 1}),
        qdrant_delete=lambda p, **k: (calls.append("qdrant") or {"deleted_count": 1}))
    first = operator.submit(projection(), scope={"snapshot_id": "s"})
    second = operator.submit(projection(), scope={"snapshot_id": "s"})
    assert first["identity"] == second["identity"]
    assert calls == ["sqlite", "qdrant"]


def test_qdrant_outage_persists_failure_and_retry_replays(tmp_path: Path):
    outbox = TombstoneOutbox(tmp_path / "state.db")
    calls = []
    def sqlite_delete(p, **k):
        calls.append("sqlite")
        return {"deleted_count": 1}
    def qdrant_delete(p, **k):
        calls.append("qdrant")
        if calls.count("qdrant") == 1:
            raise RuntimeError("outage")
        return {"deleted_count": 1}
    operator = TombstoneOperator(outbox, sqlite_delete=sqlite_delete, qdrant_delete=qdrant_delete)
    row = outbox.register(projection(), scope={"snapshot_id": "s"})
    with pytest.raises(RuntimeError):
        operator.replay(row["identity"])
    assert outbox.get(row["identity"])["status"] == "failed"
    assert operator.replay(row["identity"])["status"] == "applied"


def test_zero_match_completion_and_stale_recovery(tmp_path: Path):
    outbox = TombstoneOutbox(tmp_path / "state.db")
    row = outbox.register(projection(), scope={"snapshot_id": "s"})
    with __import__("sqlite3").connect(outbox.path) as db:
        db.execute("UPDATE tombstone_outbox SET status='in_progress', claimed_at='2000-01-01T00:00:00+00:00' WHERE identity=?", (row["identity"],))
    assert outbox.recover_stale(before="2020-01-01T00:00:00+00:00") == 1
    operator = TombstoneOperator(outbox, sqlite_delete=lambda p, **k: {"deleted_count": 0}, qdrant_delete=lambda p, **k: {"deleted_count": 0})
    result = operator.replay(row["identity"])
    assert result["status"] == "applied" and result["deleted_count"] == 0

def test_reconciliation_error_persists_and_retry_succeeds(tmp_path: Path):
    outbox = TombstoneOutbox(tmp_path / "state.db")
    calls = []

    def sqlite_delete(p, **k):
        calls.append("sqlite")
        return {"observed_count": 1, "deleted_count": 1}

    def qdrant_delete(p, **k):
        calls.append("qdrant")
        if calls.count("qdrant") == 1:
            return {"observed_count": 0, "deleted_count": 1}
        return {"observed_count": 1, "deleted_count": 1}

    operator = TombstoneOperator(
        outbox, sqlite_delete=sqlite_delete, qdrant_delete=qdrant_delete
    )
    row = outbox.register(projection(), scope={"snapshot_id": "s"})
    with pytest.raises(ValueError, match="reconciliation"):
        operator.replay(row["identity"])
    failed = outbox.get(row["identity"])
    assert failed["status"] == "failed"
    assert "reconciliation" in failed["last_error"]
    assert calls == ["sqlite", "qdrant"]
    assert operator.replay(row["identity"])["status"] == "applied"
