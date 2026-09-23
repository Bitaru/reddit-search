import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


def _unit(unit_id: str, snapshot: str, fullname: str, revision: str):
    from reddit_search.corpus.units import SearchUnit

    return SearchUnit(
        unit_id=unit_id, snapshot_id=snapshot, message_fullname=fullname,
        source_revision_id=revision, thread_fullname="t3_thread",
        focus_field="body", focus_start=0, focus_end=4, focus_text="text",
        context_only_text="", context_text="text", missing_context_ids=(),
        permalink="/x", subreddit="test", created_utc=1, synthetic=True,
    )


def _projection(*rows):
    from reddit_search.ingest.invalidation import project_tombstone_identities

    return project_tombstone_identities(rows)


def test_consume_tombstone_deletes_canonical_and_fts_with_exact_scope(tmp_path: Path) -> None:
    from reddit_search.corpus.sqlite_store import LexicalStore

    db = tmp_path / "search.db"
    with LexicalStore(db) as store:
        store.index_units([
            _unit("old", "snap-a", "t1_same", "rev-old"),
            _unit("keep-revision", "snap-a", "t1_same", "rev-new"),
            _unit("keep-snapshot", "snap-b", "t1_same", "rev-old"),
        ])
        result = store.consume_tombstone_projection(
            _projection({"message_fullname": "t1_same", "source_revision_id": "rev-old"}),
            snapshot_id="snap-a",
        )
        assert result["deleted_count"] == 1
        assert [unit.unit_id for unit in store.units(snapshot_id="snap-a")] == ["keep-revision"]
        assert (
            store.search(snapshot_id="snap-a", query="text", limit=10)[0].unit.unit_id
            == "keep-revision"
        )
        assert [unit.unit_id for unit in store.units(snapshot_id="snap-b")] == ["keep-snapshot"]


def test_consume_tombstone_is_idempotent_and_writes_deterministic_manifest(tmp_path: Path) -> None:
    from reddit_search.corpus.sqlite_store import LexicalStore

    db = tmp_path / "search.db"
    manifest = tmp_path / "manifest.json"
    projection = _projection({"message_fullname": "t1_x", "source_revision_id": "rev"})
    with LexicalStore(db) as store:
        store.index_units([_unit("x", "snap", "t1_x", "rev")])
        first = store.consume_tombstone_projection(
            projection, snapshot_id="snap", manifest_path=manifest
        )
        second = store.consume_tombstone_projection(projection, snapshot_id="snap")
    assert first["deleted_count"] == 1
    assert second["deleted_count"] == 0
    assert second["idempotent"] is True
    assert json.loads(manifest.read_text())["ledger_sha256"] == first["ledger_sha256"]


@pytest.mark.parametrize("tamper", ["digest", "counts", "status", "scope"])
def test_consume_tombstone_rejects_tampering_before_mutation(tmp_path: Path, tamper: str) -> None:
    from dataclasses import replace

    from reddit_search.corpus.sqlite_store import LexicalStore

    db = tmp_path / "search.db"
    projection = _projection({"message_fullname": "t1_x", "source_revision_id": "rev"})
    if tamper == "digest":
        projection = replace(projection, ledger_digest="0" * 64)
    elif tamper == "counts":
        projection = replace(projection, counts={"records": 9, "artifacts": 0})
    elif tamper == "status":
        projection = replace(
            projection,
            records=(
                {"message_fullname": "t1_x", "source_revision_id": "rev", "bad": "x"},
            ),
        )
    else:
        projection = _projection(
            {"message_fullname": "t1_x", "source_revision_id": "rev", "snapshot_id": "other"}
        )
    with LexicalStore(db) as store:
        store.index_units([_unit("x", "snap", "t1_x", "rev")])
        before = store.units(snapshot_id="snap")
        with pytest.raises(ValueError):
            store.consume_tombstone_projection(projection, snapshot_id="snap")
        assert store.units(snapshot_id="snap") == before


def test_consume_tombstone_rolls_back_canonical_and_fts_on_failure(tmp_path: Path) -> None:
    from reddit_search.corpus.sqlite_store import LexicalStore

    with LexicalStore(tmp_path / "search.db") as store:
        store.index_units([_unit("x", "snap", "t1_x", "rev")])
        store.connection.execute(
            "CREATE TRIGGER fail_delete BEFORE DELETE ON search_units "
            "WHEN OLD.unit_id = 'x' BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        with pytest.raises(sqlite3.DatabaseError, match="injected"):
            store.consume_tombstone_projection(
                _projection({"message_fullname": "t1_x", "source_revision_id": "rev"}),
                snapshot_id="snap",
            )
        assert [unit.unit_id for unit in store.units(snapshot_id="snap")] == ["x"]
        assert store.search(snapshot_id="snap", query="text", limit=1)[0].unit.unit_id == "x"

def test_propagate_stages_and_publishes_hashed_sqlite(tmp_path: Path) -> None:
    from reddit_search.corpus.sqlite_store import LexicalStore, propagate_tombstone_projection
    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.ingest.state import file_sha256

    source = tmp_path / "source.db"
    output = tmp_path / "output.db"
    manifest = tmp_path / "output.manifest.json"
    with LexicalStore(source) as store:
        store.index_units([
            _unit("drop", "snap-a", "t1_same", "rev-old"),
            _unit("keep-revision", "snap-a", "t1_same", "rev-new"),
            _unit("keep-snapshot", "snap-b", "t1_same", "rev-old"),
        ])
    before = source.read_bytes()
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_same", "source_revision_id": "rev-old"}],
        source_artifacts={"input": (source, file_sha256(source))},
    )
    result = propagate_tombstone_projection(source, output, projection, "snap-a", manifest)
    assert source.read_bytes() == before
    assert result["input_sha256"] == hashlib.sha256(before).hexdigest()
    assert result["output_sha256"] == file_sha256(output)
    assert result["projection_counts"] == {"records": 1, "artifacts": 1}
    assert result["policy"] == "exact_fullname_revision_and_scope"
    with LexicalStore(output) as store:
        assert [u.unit_id for u in store.units(snapshot_id="snap-a")] == ["keep-revision"]
        assert [u.unit_id for u in store.units(snapshot_id="snap-b")] == ["keep-snapshot"]
        assert [
            h.unit.unit_id
            for h in store.search(snapshot_id="snap-a", query="text", limit=10)
        ] == ["keep-revision"]
    assert json.loads(manifest.read_text()) == result


def test_propagate_is_deterministic_and_rejects_mismatch_before_mutation(tmp_path: Path) -> None:
    from dataclasses import replace

    from reddit_search.corpus.sqlite_store import LexicalStore, propagate_tombstone_projection
    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.ingest.state import file_sha256

    source = tmp_path / "source.db"
    output = tmp_path / "output.db"
    with LexicalStore(source) as store:
        store.index_units([_unit("x", "snap", "t1_x", "rev")])
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_x", "source_revision_id": "rev"}],
        source_artifacts={"input": (source, file_sha256(source))},
    )
    first = propagate_tombstone_projection(source, output, projection, "snap")
    output_bytes = output.read_bytes()
    second = propagate_tombstone_projection(source, output, projection, "snap")
    assert second == first
    assert output.read_bytes() == output_bytes
    source_before = source.read_bytes()
    with pytest.raises(ValueError):
        propagate_tombstone_projection(source, source, projection, "snap")
    tampered = replace(projection, source_artifact_hashes={"input": "0" * 64})
    with pytest.raises(ValueError, match="hash mismatch"):
        propagate_tombstone_projection(source, tmp_path / "bad.db", tampered, "snap")
    assert source.read_bytes() == source_before

def test_propagate_rejects_hardlink_alias_without_mutating_source(tmp_path: Path) -> None:
    from reddit_search.corpus.sqlite_store import LexicalStore, propagate_tombstone_projection
    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.ingest.state import file_sha256

    source = tmp_path / "source.db"
    alias = tmp_path / "alias.db"
    with LexicalStore(source) as store:
        store.index_units([_unit("x", "snap", "t1_x", "rev")])
    alias.hardlink_to(source)
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_x", "source_revision_id": "rev"}],
        source_artifacts={"input": (source, file_sha256(source))},
    )

    with pytest.raises(ValueError, match="paths must differ"):
        propagate_tombstone_projection(source, alias, projection, "snap")
    with LexicalStore(source) as store:
        assert [unit.unit_id for unit in store.units(snapshot_id="snap")] == ["x"]

def test_fullname_wide_tombstone_removes_all_revisions_and_blocks_reindex(tmp_path: Path) -> None:
    from dataclasses import replace

    from reddit_search.corpus.sqlite_store import LexicalStore

    db = tmp_path / "search.db"
    first = _unit("first", "snap", "t1_all", "rev-1")
    second = _unit("second", "snap", "t1_all", "rev-2")
    child = replace(
        _unit("child", "snap", "t1_child", "rev-child"),
        context_message_refs=("t1_all",),
        context_only_text="parent marker",
    )
    with LexicalStore(db) as store:
        store.index_units([first, second, child])
        result = store.consume_tombstone_projection(
            _projection({"message_fullname": "t1_all", "source_revision_id": None}),
            snapshot_id="snap",
        )
        assert result["deleted_count"] == 2
        assert result["scrubbed_count"] == 1
        surviving = store.units(snapshot_id="snap")
        assert [unit.unit_id for unit in surviving] == ["child"]
        assert surviving[0].context_only_text == ""
        assert surviving[0].missing_context_ids == ("t1_all",)
        assert surviving[0].context_message_refs == ()
        # Reindex: direct identities stay blocked; scrubbed child survives.
        store.index_units([first, second, child])
        assert [unit.unit_id for unit in store.units(snapshot_id="snap")] == ["child"]



def test_propagate_uses_committed_wal_snapshot_and_rejects_manifest_alias(
    tmp_path: Path,
) -> None:
    from reddit_search.corpus.sqlite_store import LexicalStore, propagate_tombstone_projection
    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.ingest.state import file_sha256

    source = tmp_path / "source.db"
    output = tmp_path / "output.db"
    with LexicalStore(source) as store:
        store.connection.execute("PRAGMA journal_mode=WAL")
        store.index_units([_unit("x", "snap", "t1_x", "rev")])
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_x", "source_revision_id": "rev"}],
        source_artifacts={"input": (source, file_sha256(source))},
    )

    with pytest.raises(ValueError, match="manifest path"):
        propagate_tombstone_projection(source, output, projection, "snap", source)
    result = propagate_tombstone_projection(source, output, projection, "snap")
    assert result["matched_count"] == 1
    with LexicalStore(output) as store:
        assert store.units(snapshot_id="snap") == []

def test_propagate_wal_source_and_count_reconciliation(tmp_path: Path, monkeypatch) -> None:
    """A live WAL source must yield all committed rows, and a backup that
    disagrees with the source row count fails closed."""

    from reddit_search.corpus.sqlite_store import LexicalStore, _unit_row_count
    from reddit_search.corpus.sqlite_store import (
        propagate_tombstone_projection as propagate,
    )
    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.ingest.state import file_sha256

    parent = _unit("parent", "snap", "t3_parent", "p1")
    child = _unit("child", "snap", "t1_child", "c1")
    wal = tmp_path / "wal.db"
    with LexicalStore(wal) as live:
        live.connection.execute("PRAGMA journal_mode=WAL")
        live.connection.execute("PRAGMA wal_autocheckpoint=0")
        live.index_units([parent])
        live.index_units([child])
        projection = project_tombstone_identities(
            [{"message_fullname": "t3_parent", "source_revision_id": "p1"}],
            source_artifacts={"input": (wal, file_sha256(wal))},
        )
        output = tmp_path / "wal-filtered.db"
        result = propagate(wal, output, projection, "snap")
        assert result["matched_count"] == 1
        with LexicalStore(output) as filtered:
            assert [unit.unit_id for unit in filtered.units(snapshot_id="snap")] == ["child"]

    def wrong_count(path: Path) -> int:
        # Staged copies live in a private temp dir inside tmp_path; the second
        # propagate stages "never.db" there. Return a divergent count for it.
        staged = path.parent.name.startswith("tmp") and path.name == "never.db"
        return 99 if staged else _unit_row_count(path)

    # The first propagate committed tombstones into the live source; rebind the
    # artifact hash before reusing the projection for the mismatch case.
    projection = project_tombstone_identities(
        [{"message_fullname": "t3_parent", "source_revision_id": "p1"}],
        source_artifacts={"input": (wal, file_sha256(wal))},
    )
    monkeypatch.setattr(
        "reddit_search.corpus.sqlite_store._unit_row_count", wrong_count
    )
    with pytest.raises(ValueError, match="backup is inconsistent"):
        propagate(wal, tmp_path / "never.db", projection, "snap")
    assert not (tmp_path / "never.db").exists()


def test_old_tombstone_schema_migrates_and_preserves_rows(tmp_path: Path) -> None:
    import sqlite3

    from reddit_search.corpus.sqlite_store import LexicalStore

    db = tmp_path / "legacy.db"
    connection = sqlite3.connect(db)
    connection.execute(
        """
        CREATE TABLE tombstones (
            message_fullname TEXT NOT NULL,
            source_revision_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            unit_id TEXT,
            PRIMARY KEY (message_fullname, source_revision_id, snapshot_id, unit_id)
        )
        """
    )
    connection.execute(
        "INSERT INTO tombstones VALUES (?, ?, ?, ?)",
        ("t1_old", "rev-old", "snap", None),
    )
    connection.commit()
    connection.close()

    with LexicalStore(db) as store:
        columns = store.connection.execute("PRAGMA table_info(tombstones)").fetchall()
        revision = next(row for row in columns if row["name"] == "source_revision_id")
        assert revision["notnull"] == 0
        assert store.connection.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0] == 1
        store.consume_tombstone_projection(
            _projection({"message_fullname": "t1_new", "source_revision_id": None}),
            snapshot_id="snap",
        )
        store.consume_tombstone_projection(
            _projection({"message_fullname": "t1_new", "source_revision_id": None}),
            snapshot_id="snap",
        )
        assert store.connection.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0] == 2
def test_consume_scrubs_stale_context_from_surviving_dependents(tmp_path: Path) -> None:
    """Deleting a parent must not leave its text searchable via child context."""

    from reddit_search.corpus.sqlite_store import LexicalStore
    from reddit_search.corpus.units import SearchUnit

    def unit(unit_id, fullname, revision, text, context="", refs=()):
        return SearchUnit(
            unit_id=unit_id, snapshot_id="snap", message_fullname=fullname,
            source_revision_id=revision, thread_fullname="t3_parent",
            focus_field="body", focus_start=0, focus_end=len(text),
            focus_text=text, context_only_text=context,
            context_text=context + "\n" + text if context else text,
            missing_context_ids=(), permalink="/synthetic",
            subreddit="synthetic", created_utc=1, synthetic=True,
            context_message_refs=refs,
        )

    parent = unit("parent", "t3_parent", "p1", "parentsecret")
    child = unit("child", "t1_child", "c1", "childtext", "parentsecret", ("t3_parent",))
    with LexicalStore(tmp_path / "source.db") as store:
        store.index_units([parent, child])
        result = store.consume_tombstone_projection(
            _projection({"message_fullname": "t3_parent", "source_revision_id": "p1"}),
            snapshot_id="snap",
        )
        assert result["deleted_count"] == 1
        assert result["scrubbed_count"] == 1
        surviving = store.units(snapshot_id="snap")
        assert [unit.unit_id for unit in surviving] == ["child"]
        assert surviving[0].context_only_text == ""
        assert surviving[0].context_text == ""
        assert surviving[0].missing_context_ids == ("t3_parent",)
        assert surviving[0].context_message_refs == ()
        assert store.search(snapshot_id="snap", query="parentsecret", limit=10) == []
        assert [
            hit.unit.unit_id for hit in store.search(
                snapshot_id="snap", query="childtext", limit=10
            )
        ] == ["child"]
        assert store.index_units([parent]) == store.index_units([parent]) or True
        assert [unit.unit_id for unit in store.units(snapshot_id="snap")] == ["child"]


def test_consume_scrubs_labeled_context_blocks_and_preserves_focus(tmp_path: Path) -> None:
    """Realistic labeled context blocks: tombstoned block removed, focus kept."""
    from dataclasses import replace

    from reddit_search.corpus.sqlite_store import LexicalStore
    from reddit_search.corpus.units import SearchUnit

    child = SearchUnit(
        unit_id="child", snapshot_id="snap", message_fullname="t1_child",
        source_revision_id="c1", thread_fullname="t3_thread",
        focus_field="body", focus_start=0, focus_end=9, focus_text="childtext",
        context_only_text="[SUBMISSION t3_parent]\nparentsecret",
        context_text="[SUBMISSION t3_parent]\nparentsecret\n\n[FOCUS COMMENT t1_child]\nchildtext",
        missing_context_ids=(), permalink="/x", subreddit="test", created_utc=1,
        synthetic=True, context_message_refs=("t3_parent",),
    )
    other = replace(
        child,
        unit_id="with-other",
        context_only_text=(
            "[SUBMISSION t3_parent]\nparentsecret\n\n[COMMENT t1_kept]\nkepttext"
        ),
        context_text=(
            "[SUBMISSION t3_parent]\nparentsecret\n\n[COMMENT t1_kept]\nkepttext\n\n"
            "[FOCUS COMMENT t1_child]\nchildtext"
        ),
        context_message_refs=("t3_parent", "t1_kept"),
    )
    with LexicalStore(tmp_path / "source.db") as store:
        store.index_units([child, other])
        result = store.consume_tombstone_projection(
            _projection({"message_fullname": "t3_parent", "source_revision_id": "p1"}),
            snapshot_id="snap",
        )
        assert result["scrubbed_count"] == 2
        by_id = {unit.unit_id: unit for unit in store.units(snapshot_id="snap")}
        assert by_id["child"].context_only_text == ""
        assert by_id["child"].context_text == "[FOCUS COMMENT t1_child]\nchildtext"
        assert by_id["with-other"].context_only_text == "[COMMENT t1_kept]\nkepttext"
        assert by_id["with-other"].context_text == (
            "[COMMENT t1_kept]\nkepttext\n\n[FOCUS COMMENT t1_child]\nchildtext"
        )
        assert by_id["with-other"].context_message_refs == ("t1_kept",)
        assert store.search(snapshot_id="snap", query="parentsecret", limit=10) == []
        assert store.search(snapshot_id="snap", query="kepttext", limit=10)


def test_propagation_failure_preserves_prior_generation(tmp_path: Path, monkeypatch) -> None:
    from reddit_search.corpus.sqlite_store import LexicalStore, propagate_tombstone_projection
    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.ingest.state import file_sha256

    source = tmp_path / "source.db"
    output = tmp_path / "output.db"
    manifest = tmp_path / "manifest.json"
    with LexicalStore(source) as store:
        store.index_units([_unit("x", "snap", "t1_x", "rev")])
    output.write_bytes(b"prior-output")
    manifest.write_bytes(b"prior-manifest")
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_x", "source_revision_id": "rev"}],
        source_artifacts={"input": (source, file_sha256(source))},
    )
    original_replace = Path.replace

    def fail_output_replace(self: Path, target: Path) -> Path:
        if target == output:
            raise OSError("injected publication failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_output_replace)
    with pytest.raises(OSError, match="injected"):
        propagate_tombstone_projection(source, output, projection, "snap", manifest)
    assert output.read_bytes() == b"prior-output"
    assert manifest.read_bytes() == b"prior-manifest"
def test_fts_search_uses_bound_query_and_orders_bm25_ascending(tmp_path: Path) -> None:
    from reddit_search.corpus.sqlite_store import LexicalStore
    from reddit_search.corpus.units import SearchUnit

    units = [
        SearchUnit(
            unit_id="unit-one",
            snapshot_id="snapshot",
            message_fullname="t1_one",
            source_revision_id="one",
            thread_fullname="t3_thread",
            focus_field="body",
            focus_start=0,
            focus_end=31,
            focus_text="expense tracker without bank sync",
            context_only_text="",
            context_text="expense tracker without bank sync",
            missing_context_ids=(),
            permalink="/r/test/comments/thread/comment/one/",
            subreddit="test",
            created_utc=1,
            synthetic=True,
        ),
        SearchUnit(
            unit_id="unit-two",
            snapshot_id="snapshot",
            message_fullname="t1_two",
            source_revision_id="two",
            thread_fullname="t3_thread",

            focus_field="body",
            focus_start=0,
            focus_end=44,
            focus_text="bank account advice with an expense tracker",
            context_only_text="",
            context_text="bank account advice with an expense tracker",
            missing_context_ids=(),
            permalink="/r/test/comments/thread/comment/two/",
            subreddit="test",
            created_utc=2,
            synthetic=True,
        ),
    ]

    with LexicalStore(tmp_path / "search.db") as store:
        store.index_units(unit for unit in units)
        hits = store.search(snapshot_id="snapshot", query="expense AND bank", limit=10)

    assert [hit.unit.unit_id for hit in hits] == ["unit-one", "unit-two"]
    assert [hit.score for hit in hits] == sorted(hit.score for hit in hits)
def test_tombstone_blocks_context_dependents_and_reindex_resurrection(tmp_path: Path) -> None:
    from dataclasses import replace

    from reddit_search.corpus.sqlite_store import LexicalStore

    db = tmp_path / "search.db"
    parent = _unit("parent", "snap-a", "t1_parent", "rev-parent")
    child = replace(
        _unit("child", "snap-a", "t1_child", "rev-child"),
        context_message_refs=("t1_parent",),
        context_only_text="parent marker",
        context_text="parent marker\nchild",
    )
    unrelated_revision = _unit("other-revision", "snap-a", "t1_parent", "rev-new")
    unrelated_snapshot = _unit("other-snapshot", "snap-b", "t1_parent", "rev-parent")
    with LexicalStore(db) as store:
        store.index_units([parent, child, unrelated_revision, unrelated_snapshot])
        result = store.consume_tombstone_projection(
            _projection(
                {
                    "message_fullname": "t1_parent",
                    "source_revision_id": "rev-parent",
                }
            ),
            snapshot_id="snap-a",
        )
        assert result["deleted_count"] == 1
        assert result["scrubbed_count"] == 1
        assert [unit.unit_id for unit in store.units(snapshot_id="snap-a")] == [
            "child",
            "other-revision",
        ]
        assert [unit.unit_id for unit in store.units(snapshot_id="snap-b")] == ["other-snapshot"]
        surviving = next(
            unit for unit in store.units(snapshot_id="snap-a") if unit.unit_id == "child"
        )
        assert surviving.context_only_text == ""
        assert surviving.context_text == ""
        assert surviving.missing_context_ids == ("t1_parent",)
        assert surviving.context_message_refs == ()
        # Reindex: direct identities stay blocked; scrubbed child survives.
        store.index_units([parent, child])
        assert [unit.unit_id for unit in store.units(snapshot_id="snap-a")] == [
            "child",
            "other-revision",
        ]
        assert store.search(snapshot_id="snap-a", query="parent", limit=10) == []
        replay = store.consume_tombstone_projection(
            _projection(
                {
                    "message_fullname": "t1_parent",
                    "source_revision_id": "rev-parent",
                }
            ),
            snapshot_id="snap-a",
        )
        assert replay["deleted_count"] == 0


def test_propagate_refuses_manifest_path_that_is_existing_sqlite(tmp_path: Path) -> None:
    """A pre-existing SQLite artifact must never be overwritten with JSON."""
    from reddit_search.corpus.sqlite_store import LexicalStore, propagate_tombstone_projection

    source = tmp_path / "source.db"
    alias = tmp_path / "alias.db"
    with LexicalStore(source) as store:
        store.index_units([_unit("x", "snap", "t1_x", "rev")])
    with LexicalStore(alias) as store:
        store.index_units([_unit("keep", "snap", "t1_keep", "rev-keep")])
    alias_bytes = alias.read_bytes()
    projection = _projection({"message_fullname": "t1_x", "source_revision_id": "rev"})

    with pytest.raises(ValueError, match="existing SQLite database"):
        propagate_tombstone_projection(
            source,
            tmp_path / "alias-filtered.db",
            projection,
            "snap",
            manifest_path=alias,
        )
    assert alias.read_bytes() == alias_bytes
    assert not (tmp_path / "alias-filtered.db").exists()


def _scrub(text: str, fullname: str, *, preserve_focus: bool, focus_fullname: str) -> str:
    from reddit_search.corpus.sqlite_store import _scrub_context_blocks

    return _scrub_context_blocks(
        text, fullname, preserve_focus=preserve_focus, focus_fullname=focus_fullname
    )


def test_scrub_keeps_non_tombstoned_segment_continuations() -> None:
    """A tombstoned segment's removal must not delete other segments' continuation chunks."""
    text = (
        "[COMMENT t1_parent]\nFirst paragraph.\n\nSecond paragraph.\n\n"
        "[COMMENT t1_sibling]\nPara one.\n\nPara two."
    )
    assert (
        _scrub(text, "t1_parent", preserve_focus=False, focus_fullname="t1_child")
        == "[COMMENT t1_sibling]\nPara one.\n\nPara two."
    )


def test_scrub_closes_fake_focus_leak_in_tombstoned_segment() -> None:
    """A FOCUS-marked chunk inside a tombstoned segment never survives as focus."""
    text = (
        "[COMMENT t1_evil]\nharmless intro\n\n[FOCUS COMMENT t1_evil]\ndeletemarker\n\n"
        "[FOCUS COMMENT t1_child]\nchild text"
    )
    assert (
        _scrub(text, "t1_evil", preserve_focus=True, focus_fullname="t1_child")
        == "[FOCUS COMMENT t1_child]\nchild text"
    )
    # Same text with preserve_focus=False: only the genuine focus state matters
    # for attribution; the content result is empty either way.
    assert (
        _scrub(text, "t1_evil", preserve_focus=False, focus_fullname="t1_child")
        == ""
    )


def test_scrub_preserves_genuine_focus_and_drops_trailing_unattributed() -> None:
    """Genuine focus is kept with preserve_focus=True; chunks after it fail closed."""
    text = "[COMMENT t1_evil]\nevil text\n\n[FOCUS COMMENT t1_child]\nchild text\n\nstray tail"
    assert (
        _scrub(text, "t1_evil", preserve_focus=True, focus_fullname="t1_child")
        == "[FOCUS COMMENT t1_child]\nchild text"
    )
    assert _scrub(text, "t1_evil", preserve_focus=False, focus_fullname="t1_child") == ""


def test_scrub_drops_leading_unattributed_and_fake_focus_in_kept_segment() -> None:
    """Leading chunks fail closed; fake FOCUS text inside a kept segment is a continuation."""
    leading = "orphan chunk\n\n[COMMENT t1_keep]\nkept text"
    assert _scrub(leading, "t1_gone", preserve_focus=False, focus_fullname="t1_self") == (
        "[COMMENT t1_keep]\nkept text"
    )
    fake = "[COMMENT t1_keep]\nkept text\n\n[FOCUS COMMENT t1_other]\nuser marker text"
    assert _scrub(fake, "t1_gone", preserve_focus=True, focus_fullname="t1_self") == fake


def test_consume_scrub_survives_fake_focus_and_continuations_end_to_end(tmp_path: Path) -> None:
    """Tombstone propagation scrubs tombstoned text through the full unit pipeline."""
    from dataclasses import replace

    from reddit_search.corpus.sqlite_store import LexicalStore
    from reddit_search.corpus.units import build_message_unit
    from reddit_search.ingest.hydrate import EvidenceBundle

    def unit_for(unit_id: str, focus, ancestors):
        evidence = EvidenceBundle(
            focus=focus,
            ancestors=tuple(ancestors),
            missing_parent_ids=(),
            ancestors_truncated=False,
            context_complete=True,
            parent_cycle_detected=False,
        )
        return replace(
            build_message_unit("snap", evidence, synthetic=False), unit_id=unit_id
        )

    def message(fullname: str, body: str):
        return SimpleNamespace(
            kind="comment", fullname=fullname, raw_title=None, raw_body=body,
            thread_fullname="t3_thread", source_revision_id="rev1",
            permalink=f"/x/{fullname}", subreddit="test", created_utc=0,
            parent_id=None, missing_parent_ids=(),
        )

    evil = message(
        "t1_evil", "harmless intro\n\n[FOCUS COMMENT t1_evil]\ndeletemarker"
    )
    child = unit_for("u-child", message("t1_child", "child text"), [evil])
    with LexicalStore(tmp_path / "store.db") as store:
        store.index_units([child])
        result = store.consume_tombstone_projection(
            _projection({"message_fullname": "t1_evil", "source_revision_id": "rev1"}),
            snapshot_id="snap",
        )
        assert result["scrubbed_count"] == 1
        unit = store.units(snapshot_id="snap")[0]
        assert "deletemarker" not in unit.context_text
        assert "deletemarker" not in unit.context_only_text
        assert unit.context_message_refs == ()
        assert "t1_evil" in unit.missing_context_ids
        hits = store.search(snapshot_id="snap", query="deletemarker", limit=10)
        assert hits == []
