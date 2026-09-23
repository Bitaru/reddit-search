# ruff: noqa: E501
import json
import sqlite3
from pathlib import Path

import pytest

from reddit_search.corpus.sqlite_store import LexicalStore
from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.ingest.state import file_sha256
from reddit_search.operations import TombstoneOutbox, local_sqlite_operator, validate_replay_targets
from reddit_search.operations.outbox import TombstoneOperator
from reddit_search.operations.wiring import reject_live_qdrant


def _unit(unit_id: str, snapshot: str, fullname: str, revision: str):
    from reddit_search.corpus.units import SearchUnit

    return SearchUnit(
        unit_id=unit_id,
        snapshot_id=snapshot,
        message_fullname=fullname,
        source_revision_id=revision,
        thread_fullname="t3_thread",
        focus_field="body",
        focus_start=0,
        focus_end=4,
        focus_text="text",
        context_only_text="",
        context_text="text",
        missing_context_ids=(),
        permalink="/x",
        subreddit="test",
        created_utc=1,
        synthetic=True,
    )


@pytest.fixture()
def source_db(tmp_path: Path) -> Path:
    source = tmp_path / "source.db"
    with LexicalStore(source) as store:
        store.index_units([_unit("drop", "snap", "t1_x", "rev")])
    return source


@pytest.fixture()
def outbox_row(tmp_path: Path, source_db: Path):
    store = TombstoneOutbox(tmp_path / "outbox.db")
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_x", "source_revision_id": "rev"}]
    )
    row = store.register(projection, scope={"snapshot_id": "snap"})
    return store, row


def test_local_replay_writes_derivative_keeps_source_and_persists_refusal(
    tmp_path: Path, source_db: Path, outbox_row
):
    store, row = outbox_row
    source_hash = file_sha256(source_db)
    output = tmp_path / "output.db"
    operator = local_sqlite_operator(store, sqlite_input=source_db, sqlite_output=output)
    with pytest.raises(PermissionError, match="not permitted"):
        operator.replay(row["identity"])
    assert file_sha256(source_db) == source_hash
    with LexicalStore(output) as derived:
        assert derived.units(snapshot_id="snap") == []
    persisted = store.get(row["identity"])
    assert persisted["status"] == "failed"
    assert "not permitted" in persisted["last_error"]


def test_qdrant_refusal_never_marks_applied_and_side_effect_precedes_error(
    tmp_path: Path, source_db: Path, outbox_row
):
    store, row = outbox_row
    calls: list[str] = []

    def sqlite_ok(projection, **kwargs):
        calls.append("sqlite")
        return {"observed_count": 1, "deleted_count": 1}

    operator = TombstoneOperator(store, sqlite_delete=sqlite_ok, qdrant_delete=reject_live_qdrant)
    with pytest.raises(PermissionError):
        operator.replay(row["identity"])
    assert calls == ["sqlite"]
    assert store.get(row["identity"])["status"] == "failed"


def test_validate_replay_targets_rejects_scope_and_coverage_mismatch(
    tmp_path: Path, source_db: Path, outbox_row
):
    store, row = outbox_row
    output = tmp_path / "out.db"
    with pytest.raises(ValueError, match="snapshot"):
        validate_replay_targets(
            store,
            row["identity"],
            expected_snapshot_id="other-snap",
            sqlite_input=source_db,
            sqlite_output=output,
        )
    coverage_row = store.register(
        project_tombstone_identities([{"message_fullname": "t1_y", "source_revision_id": "rev2"}]),
        scope={"snapshot_id": "snap"},
    )
    with sqlite3.connect(store.path) as db:
        db.execute(
            "UPDATE tombstone_outbox SET records=? WHERE identity=?",
            (
                json.dumps(
                    [
                        {
                            "message_fullname": "t1_y",
                            "source_revision_id": "rev2",
                            "status": "matched",
                        }
                    ]
                ),
                coverage_row["identity"],
            ),
        )
    with pytest.raises(ValueError, match="archive coverage"):
        validate_replay_targets(
            store,
            coverage_row["identity"],
            expected_snapshot_id="snap",
            sqlite_input=source_db,
            sqlite_output=output,
        )


def test_validate_replay_targets_rejects_missing_identity_and_alias(
    tmp_path: Path, source_db: Path, outbox_row
):
    store, row = outbox_row
    output = tmp_path / "out.db"
    with pytest.raises(KeyError):
        validate_replay_targets(
            store,
            "missing-identity",
            expected_snapshot_id="snap",
            sqlite_input=source_db,
            sqlite_output=output,
        )
    with pytest.raises(ValueError, match="differ"):
        validate_replay_targets(
            store,
            row["identity"],
            expected_snapshot_id="snap",
            sqlite_input=source_db,
            sqlite_output=source_db,
        )
    with pytest.raises(ValueError, match="differ"):
        validate_replay_targets(
            store,
            row["identity"],
            expected_snapshot_id="snap",
            sqlite_input=source_db,
            sqlite_output=tmp_path / "out2.db",
            sqlite_manifest=source_db,
        )


def test_validate_replay_targets_rejects_malformed_records(
    tmp_path: Path, source_db: Path, outbox_row
):
    store, row = outbox_row
    output = tmp_path / "out.db"
    malformed_payloads = [
        "not-a-list",
        "x",
        json.dumps([]),
        json.dumps([1, 2]),
        json.dumps({"a": 1}),
    ]
    for payload in malformed_payloads:
        with sqlite3.connect(store.path) as db:
            db.execute(
                "UPDATE tombstone_outbox SET records=? WHERE identity=?",
                (payload, row["identity"]),
            )
        with pytest.raises(ValueError):
            validate_replay_targets(
                store,
                row["identity"],
                expected_snapshot_id="snap",
                sqlite_input=source_db,
                sqlite_output=output,
            )


def test_validate_replay_targets_rejects_outbox_output_and_manifest_aliases(
    tmp_path: Path, source_db: Path, outbox_row
):
    store, row = outbox_row
    with pytest.raises(ValueError, match="outbox"):
        validate_replay_targets(
            store,
            row["identity"],
            expected_snapshot_id="snap",
            sqlite_input=source_db,
            sqlite_output=store.path,
        )
    with pytest.raises(ValueError, match="outbox"):
        validate_replay_targets(
            store,
            row["identity"],
            expected_snapshot_id="snap",
            sqlite_input=source_db,
            sqlite_output=tmp_path / "out.db",
            sqlite_manifest=store.path,
        )


def test_applied_replay_is_idempotent_without_repeated_boundaries(
    tmp_path: Path, source_db: Path, outbox_row
):
    store, row = outbox_row
    calls: list[str] = []

    def sqlite_ok(projection, **kwargs):
        calls.append("sqlite")
        return {"observed_count": 1, "deleted_count": 1}

    store.complete(
        row["identity"],
        observed_count=1,
        deleted_count=1,
        backend_results=[{"backend": "sqlite", "observed_count": 1, "deleted_count": 1}],
    )
    operator = TombstoneOperator(
        store,
        sqlite_delete=sqlite_ok,
        qdrant_delete=reject_live_qdrant,
    )
    result = operator.replay(row["identity"])
    assert result["status"] == "applied"
    assert calls == []
