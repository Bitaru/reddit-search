# ruff: noqa: E501
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.operations import (
    TombstoneOperator,
    TombstoneOutbox,
    build_purge_inventory,
    write_purge_inventory,
)
from reddit_search.operations.purge import _project_row


def projection_records():
    return [{"message_fullname": "t3_x", "source_revision_id": None}]


def make_outbox(tmp_path: Path) -> TombstoneOutbox:
    return TombstoneOutbox(tmp_path / "state.db")


def test_inventory_is_deterministic_and_binds_source_hash(tmp_path: Path):
    outbox = make_outbox(tmp_path)
    outbox.register(
        project_tombstone_identities(projection_records()),
        scope={"snapshot_id": "s"},
    )
    first = build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv1.json")
    second = build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv2.json")
    assert first == second
    assert first["kind"] == "tombstone_outbox_purge_inventory"
    assert first["executed"] is False
    assert first["not_a_deletion_claim"] is True
    assert first["outbox"]["sha256"] == hashlib.sha256(outbox.path.read_bytes()).hexdigest()
    assert first["outbox"]["row_count"] == 1
    assert first["outbox"]["status_counts"] == {"pending": 1, "in_progress": 0, "applied": 0, "failed": 0}
    assert first["rows"][0]["status"] == "pending"
    assert first["rows"][0]["backend_results"] is None
    assert first["rows"][0]["scope"] == {"snapshot_id": "s"}


def test_inventory_does_not_mutate_outbox(tmp_path: Path):
    outbox = make_outbox(tmp_path)

    outbox.register(project_tombstone_identities(projection_records()), scope={"snapshot_id": "s"})
    before = outbox.path.read_bytes()
    build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv.json")
    assert outbox.path.read_bytes() == before


def test_pending_and_failed_rows_keep_truthful_fields(tmp_path: Path):
    outbox = make_outbox(tmp_path)

    identity = outbox.register(
        project_tombstone_identities(projection_records()), scope={"snapshot_id": "s"}
    )["identity"]
    outbox.fail(identity, RuntimeError("outage"))
    inventory = build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv.json")
    row = inventory["rows"][0]
    assert row["status"] == "failed"
    assert "outage" in row["last_error"]
    assert row["backend_results"] is None
    assert row["attempts"] == 0  # fail() does not claim; only replay's _claim does


def test_applied_rows_expose_persisted_backend_results_verbatim(tmp_path: Path):
    outbox = make_outbox(tmp_path)
    operator = TombstoneOperator(
        outbox,
        sqlite_delete=lambda p, **k: {"observed_count": 1, "deleted_count": 1},
        qdrant_delete=lambda p, **k: {"observed_count": 1, "deleted_count": 1},
    )
    row = outbox.register(
        project_tombstone_identities(projection_records()), scope={"snapshot_id": "s"}
    )
    operator.replay(row["identity"])
    inventory = build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv.json")
    projected = inventory["rows"][0]
    assert projected["status"] == "applied"
    persisted = outbox.get(row["identity"])["backend_results"]
    assert projected["backend_results"] == persisted
    assert projected["deleted_count"] == 2


def test_stale_in_progress_row_is_reported_without_time_judgment(tmp_path: Path):
    outbox = make_outbox(tmp_path)

    identity = outbox.register(
        project_tombstone_identities(projection_records()), scope={"snapshot_id": "s"}
    )["identity"]
    with sqlite3.connect(outbox.path) as db:
        db.execute(
            "UPDATE tombstone_outbox SET status='in_progress', claimed_at='2000-01-01T00:00:00+00:00' WHERE identity=?",
            (identity,),
        )
    inventory = build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv.json")
    row = inventory["rows"][0]
    assert row["status"] == "in_progress"
    assert row["claimed_at"] == "2000-01-01T00:00:00+00:00"
    assert row["backend_results"] is None


def test_writer_is_atomic_and_byte_stable(tmp_path: Path):
    outbox = make_outbox(tmp_path)

    outbox.register(project_tombstone_identities(projection_records()), scope={"snapshot_id": "s"})
    result = build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv.json")
    first = write_purge_inventory(tmp_path / "a.json", result)
    second = write_purge_inventory(tmp_path / "b.json", result)
    assert first.read_bytes() == second.read_bytes()


def test_writer_rejects_tampered_result(tmp_path: Path):
    outbox = make_outbox(tmp_path)
    result = build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv.json")
    tampered = dict(result)
    tampered["executed"] = True
    with pytest.raises(ValueError, match="not_a_deletion_claim"):
        write_purge_inventory(tmp_path / "inv.json", tampered)


def test_malformed_scope_without_snapshot_id_fails_closed(tmp_path: Path):
    outbox = make_outbox(tmp_path)
    with sqlite3.connect(outbox.path) as db:
        db.execute(
            """INSERT INTO tombstone_outbox
            (identity, scope, records, status, requested_count, requested_at)
            VALUES ('ident-1', '{}', '[]', 'pending', 0, '2026-01-01T00:00:00+00:00')"""
        )
    output = tmp_path / "inv.json"
    with pytest.raises(ValueError, match="snapshot_id"):
        build_purge_inventory(outbox=outbox.path, output=output)
    assert not output.exists()


def test_archive_coverage_scope_is_rejected(tmp_path: Path):
    outbox = make_outbox(tmp_path)
    scope = json.dumps({"snapshot_id": "s", "coverage_status": "matched"})
    with sqlite3.connect(outbox.path) as db:
        db.execute(
            """INSERT INTO tombstone_outbox
            (identity, scope, records, status, requested_count, requested_at)
            VALUES ('ident-1', ?, '[]', 'pending', 0, '2026-01-01T00:00:00+00:00')""",
            (scope,),
        )
    with pytest.raises(ValueError, match="archive coverage"):
        build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv.json")


def test_schema_mismatch_fails_before_output(tmp_path: Path):
    path = tmp_path / "foreign.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE other (x INTEGER)")
    with pytest.raises(ValueError, match="schema mismatch"):
        build_purge_inventory(outbox=path, output=tmp_path / "inv.json")
    assert not (tmp_path / "inv.json").exists()


def test_malformed_records_fail_closed(tmp_path: Path):
    outbox = make_outbox(tmp_path)
    with sqlite3.connect(outbox.path) as db:
        db.execute(
            """INSERT INTO tombstone_outbox
            (identity, scope, records, status, requested_count, requested_at)
            VALUES ('ident-1', '{"snapshot_id": "s"}', 'not-json', 'pending', 0, '2026-01-01T00:00:00+00:00')"""
        )
    output = tmp_path / "inv.json"
    with pytest.raises(ValueError, match="malformed records"):
        build_purge_inventory(outbox=outbox.path, output=output)
    assert not output.exists()


def test_row_counter_reconciliation_rejects_wrong_requested_count(tmp_path: Path):
    outbox = make_outbox(tmp_path)
    records = json.dumps(projection_records(), sort_keys=True)
    with sqlite3.connect(outbox.path) as db:
        db.execute(
            """INSERT INTO tombstone_outbox
            (identity, scope, records, status, requested_count, requested_at)
            VALUES ('ident-1', '{"snapshot_id": "s"}', ?, 'pending', 5, '2026-01-01T00:00:00+00:00')""",
            (records,),
        )
    with pytest.raises(ValueError, match="requested_count"):
        build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv.json")


def test_counter_mismatch_between_deleted_and_observed_is_rejected(tmp_path: Path):
    outbox = make_outbox(tmp_path)
    records = json.dumps(projection_records(), sort_keys=True)
    with sqlite3.connect(outbox.path) as db:
        db.execute(
            """INSERT INTO tombstone_outbox
            (identity, scope, records, status, requested_count, requested_at, observed_count, deleted_count)
            VALUES ('ident-1', '{"snapshot_id": "s"}', ?, 'failed', 1, '2026-01-01T00:00:00+00:00', 1, 2)""",
            (records,),
        )
    with pytest.raises(ValueError, match="deletions than observations"):
        build_purge_inventory(outbox=outbox.path, output=tmp_path / "inv.json")


def test_missing_outbox_raises_before_output(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        build_purge_inventory(outbox=tmp_path / "missing.db", output=tmp_path / "inv.json")


def test_project_row_rejects_unknown_status():
    names = [
        "identity",
        "scope",
        "records",
        "status",
        "attempts",
        "last_error",
        "requested_count",
        "observed_count",
        "deleted_count",
        "requested_at",
        "claimed_at",
        "observed_at",
        "backend_results",
    ]
    row = tuple(
        {
            "identity": "i",
            "scope": '{"snapshot_id": "s"}',
            "records": json.dumps(projection_records(), sort_keys=True),
            "status": "weird",
            "attempts": 0,
            "last_error": None,
            "requested_count": 1,
            "observed_count": 0,
            "deleted_count": 0,
            "requested_at": "2026-01-01T00:00:00+00:00",
            "claimed_at": None,
            "observed_at": None,
            "backend_results": None,
        }[name]
        for name in names
    )
    with pytest.raises(ValueError, match="unsupported status"):
        _project_row(row)


def test_writer_rejects_non_inventory_payload(tmp_path: Path):
    with pytest.raises(ValueError, match="tombstone_outbox_purge_inventory"):
        write_purge_inventory(tmp_path / "inv.json", {"kind": "other"})


def test_inventory_is_order_safe_against_different_physical_column_order(tmp_path: Path):
    canonical_outbox = make_outbox(tmp_path)
    identity = canonical_outbox.register(
        project_tombstone_identities(projection_records()), scope={"snapshot_id": "s"}
    )["identity"]
    reordered = tmp_path / "reordered.db"
    with sqlite3.connect(reordered) as db:
        db.execute(
            """CREATE TABLE tombstone_outbox (
            backend_results TEXT, observed_at TEXT, claimed_at TEXT, requested_at TEXT NOT NULL,
            deleted_count INTEGER NOT NULL DEFAULT 0, observed_count INTEGER NOT NULL DEFAULT 0,
            requested_count INTEGER NOT NULL DEFAULT 0, last_error TEXT,
            attempts INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,
            records TEXT NOT NULL, scope TEXT NOT NULL, identity TEXT PRIMARY KEY
        )"""
        )
        db.execute(
            """INSERT INTO tombstone_outbox
            (identity, scope, records, status, requested_count, requested_at)
            VALUES (?, '{"snapshot_id": "s"}', ?, 'pending', 1, '2026-01-01T00:00:00+00:00')""",
            (identity, json.dumps(projection_records(), sort_keys=True)),
        )
    reordered_inventory = build_purge_inventory(
        outbox=reordered, output=tmp_path / "reordered.json"
    )
    canonical_inventory = build_purge_inventory(
        outbox=canonical_outbox.path, output=tmp_path / "canonical.json"
    )
    # requested_at is the real outbox's wall-clock registration time; every
    # other projected field must be identical despite the physical order.
    def normalized(inventory):
        return [{k: v for k, v in row.items() if k != "requested_at"} for row in inventory["rows"]]

    assert normalized(reordered_inventory) == normalized(canonical_inventory)
