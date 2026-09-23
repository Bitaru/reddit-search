# ruff: noqa: E501
import json
import sqlite3
from pathlib import Path

import pytest

from reddit_search.corpus.sqlite_store import LexicalStore, propagate_tombstone_projection
from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.ingest.state import file_sha256
from reddit_search.operations.artifact_purge import (
    ArtifactPurgeError,
    ArtifactPurgeScope,
    run_artifact_purge,
)
from reddit_search.operations.operator import OperatorBoundaryError
from reddit_search.operations.outbox import TombstoneOutbox


def _unit(unit_id: str, snapshot: str, fullname: str, revision: str, text: str = "text"):
    """Build a SearchUnit; pass text explicitly when asserting on stored rows."""
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
        focus_text=text,
        context_only_text="",
        context_text=text,
        missing_context_ids=(),
        permalink="/x",
        subreddit="test",
        created_utc=1,
        synthetic=True,
    )


def _ledger(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    ledger = tmp_path / "ledger.jsonl"
    payload = [{**row, "reason": "unit test purge"} for row in rows]
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload), encoding="utf-8"
    )
    return ledger


def _published(tmp_path: Path):
    """Build source + first published generation holding keep and drop units.

    The published generation is produced by tombstoning an unrelated third
    unit, so both ``t1_keep`` and ``t1_x`` survive into it and a purge of
    ``t1_x`` has real work to do.
    """
    source = tmp_path / "source.db"
    units = [
        _unit("seed", "snap", "t1_seed", "r_seed", "seed text"),
        _unit("keep", "snap", "t1_keep", "r_keep", "keep me"),
        _unit("drop", "snap", "t1_x", "r_x", "drop me"),
    ]
    with LexicalStore(source) as store:
        store.index_units(units)
    published = tmp_path / "artifact" / "published.db"
    manifest = published.with_name(published.name + ".manifest.json")
    projection = project_tombstone_identities(
        [
            {
                "message_fullname": "t1_seed",
                "source_revision_id": "r_seed",
                "reason": "seed",
            }
        ],
        source_artifacts={"sqlite": (source, file_sha256(source))},
    )
    propagate_tombstone_projection(
        source, published, projection, snapshot_id="snap", manifest_path=manifest
    )
    return source, published, manifest


def _scope(tmp_path: Path, ledger: Path, *, artifact_root: Path | None = None) -> ArtifactPurgeScope:
    return ArtifactPurgeScope(
        artifact_root=artifact_root or tmp_path / "artifact",
        outbox_path=tmp_path / "outbox" / "outbox.db",
        ledger_path=ledger,
        snapshot_id="snap",
        backup_dir=tmp_path / "backups",
        reconciliation_path=tmp_path / "reconciliation" / "purge.json",
    )


# --- happy path --------------------------------------------------------------


def test_purge_publishes_scrubbed_generation_and_records_claim(tmp_path: Path):
    source, published, manifest = _published(tmp_path)
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    prior_db_sha = file_sha256(published)
    result = run_artifact_purge(_scope(tmp_path, ledger))
    assert result["status"] == "applied"
    assert result["destructive"] is True
    assert result["archives_touched"] is False
    assert result["executed"] == {
        "backup": True,
        "propagation": True,
        "publication": True,
        "outbox_claim": True,
    }
    assert result["generation"]["changed"] is True
    assert result["generation"]["prior"]["db_sha256"] == prior_db_sha
    assert result["generation"]["new"]["db_sha256"] != prior_db_sha
    # Published generation is scrubbed and manifest binding is fresh.
    with LexicalStore(published) as store:
        assert store.units(snapshot_id="snap") == [_unit("keep", "snap", "t1_keep", "r_keep", "keep me")]
    reloaded = json.loads(manifest.read_text(encoding="utf-8"))
    assert reloaded["output_sha256"] == file_sha256(published)
    # Content-addressed, verified backup of the prior generation exists.
    backup_db = Path(result["backup"]["db"]["name"])
    assert Path(result["backup"]["directory"], backup_db.name).is_file()
    assert Path(result["backup"]["directory"], backup_db.name).read_bytes() is not None
    assert result["backup"]["prior_generation"]["db_sha256"] == prior_db_sha
    # Outbox claim uses the destructive mode label.
    outbox = TombstoneOutbox(Path(result["scope"]["outbox"]))
    row = outbox.get(result["outbox"]["identity"])
    assert row is not None
    assert row["status"] == "applied"
    assert row["scope"]["mode"] == "destructive_artifact_purge"
    assert row["scope"]["published_db_sha256"] == prior_db_sha


# --- idempotency -------------------------------------------------------------


def test_rerun_of_fully_purged_artifact_is_no_op(tmp_path: Path):
    source, published, manifest = _published(tmp_path)
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    first = run_artifact_purge(_scope(tmp_path, ledger))
    assert first["status"] == "applied"
    before_db = published.read_bytes()
    before_outbox = Path(first["scope"]["outbox"]).read_bytes()
    second = run_artifact_purge(_scope(tmp_path, ledger))
    assert second["status"] == "no_op"
    assert second["idempotent"] is True
    assert second["executed"] == {
        "backup": False,
        "propagation": False,
        "publication": False,
        "outbox_claim": False,
    }
    assert second["generation"]["changed"] is False
    # Nothing churned: published db bytes and outbox bytes are identical.
    assert published.read_bytes() == before_db
    assert Path(second["scope"]["outbox"]).read_bytes() == before_outbox
    # The no-op reconciliation still binds the manifest hash.
    assert second["generation"]["new"]["db_sha256"] == file_sha256(published)


# --- validation rails (exit-3 class from run_artifact_purge) ------------------


def test_backup_dir_inside_artifact_root_is_refused(tmp_path: Path):
    source, published, manifest = _published(tmp_path)
    prior = published.read_bytes()
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = ArtifactPurgeScope(
        artifact_root=tmp_path / "artifact",
        outbox_path=tmp_path / "outbox.db",
        ledger_path=ledger,
        snapshot_id="snap",
        backup_dir=tmp_path / "artifact" / "backups",
        reconciliation_path=tmp_path / "reconciliation.json",
    )
    with pytest.raises(ArtifactPurgeError, match="backup directory must be outside"):
        run_artifact_purge(scope)
    assert published.read_bytes() == prior  # nothing mutated


def test_reconciliation_inside_artifact_root_is_refused(tmp_path: Path):
    _published(tmp_path)
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = ArtifactPurgeScope(
        artifact_root=tmp_path / "artifact",
        outbox_path=tmp_path / "outbox.db",
        ledger_path=ledger,
        snapshot_id="snap",
        backup_dir=tmp_path / "backups",
        reconciliation_path=tmp_path / "artifact" / "reconciliation.json",
    )
    with pytest.raises(ArtifactPurgeError, match="reconciliation output must be outside"):
        run_artifact_purge(scope)


def test_tampered_manifest_is_refused(tmp_path: Path):
    source, published, manifest = _published(tmp_path)
    prior = published.read_bytes()
    manifest.write_text(json.dumps({**json.loads(manifest.read_text(encoding="utf-8")), "output_sha256": "0" * 64}), encoding="utf-8")
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    with pytest.raises(ArtifactPurgeError, match="tampered or stale"):
        run_artifact_purge(_scope(tmp_path, ledger))
    assert published.read_bytes() == prior


def test_tampered_published_db_is_refused(tmp_path: Path):
    source, published, manifest = _published(tmp_path)
    with sqlite3.connect(published) as db:
        db.execute("UPDATE search_units SET focus_text = 'tampered' WHERE unit_id = 'keep'")
    tampered = published.read_bytes()
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    with pytest.raises(ArtifactPurgeError, match="tampered or stale"):
        run_artifact_purge(_scope(tmp_path, ledger))
    assert published.read_bytes() == tampered


def test_cross_snapshot_is_refused(tmp_path: Path):
    source, published, manifest = _published(tmp_path)
    prior = published.read_bytes()
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = ArtifactPurgeScope(
        artifact_root=tmp_path / "artifact",
        outbox_path=tmp_path / "outbox.db",
        ledger_path=ledger,
        snapshot_id="other-snap",
        backup_dir=tmp_path / "backups",
        reconciliation_path=tmp_path / "reconciliation.json",
    )
    with pytest.raises(ArtifactPurgeError, match="cross-snapshot|snapshot"):
        run_artifact_purge(scope)
    assert published.read_bytes() == prior


def test_archive_coverage_ledger_row_is_refused(tmp_path: Path):
    _published(tmp_path)
    # The JSONL loader rejects unknown fields; coverage rows are rejected by the
    # projection layer when records carry matched/unresolved status directly.
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    assert run_artifact_purge(_scope(tmp_path, ledger))["status"] == "applied"
    with pytest.raises(ValueError, match="archive coverage"):
        project_tombstone_identities(
            [{"message_fullname": "t1_x", "source_revision_id": "r_x", "status": "matched"}]
        )


def test_outbox_scope_conflict_on_same_records_is_refused(tmp_path: Path):
    _published(tmp_path)
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = _scope(tmp_path, ledger)
    # Register a conflicting claim first: same records bound to another snapshot.
    outbox = TombstoneOutbox(scope.outbox_path)
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_x", "source_revision_id": "r_x"}]
    )
    outbox.register(projection, scope={"snapshot_id": "elsewhere"})
    with pytest.raises(ArtifactPurgeError, match="does not match outbox row scope"):
        run_artifact_purge(scope)


def test_same_records_purge_to_different_artifact_root_is_refused(tmp_path: Path):
    _published(tmp_path)
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = _scope(tmp_path, ledger)
    # First purge records the claim bound to tmp_path/artifact.
    run_artifact_purge(scope)
    # A second purge scope pointing at a different, valid artifact root must be
    # refused: the outbox already binds these tombstones elsewhere.
    other_root = tmp_path / "artifact2"
    other_db = other_root / "published.db"
    other_db.parent.mkdir(parents=True, exist_ok=True)
    other_db.write_bytes((tmp_path / "artifact" / "published.db").read_bytes())
    (other_root / "published.db.manifest.json").write_text(
        (tmp_path / "artifact" / "published.db.manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    conflicting = ArtifactPurgeScope(
        artifact_root=other_root,
        outbox_path=scope.outbox_path,
        ledger_path=ledger,
        snapshot_id="snap",
        backup_dir=tmp_path / "backups2",
        reconciliation_path=tmp_path / "reconciliation2.json",
    )
    with pytest.raises(ArtifactPurgeError, match="different artifact root"):
        run_artifact_purge(conflicting)


def test_empty_ledger_is_refused(tmp_path: Path):
    _published(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    with pytest.raises(ArtifactPurgeError, match="at least one tombstone"):
        run_artifact_purge(_scope(tmp_path, ledger))


def test_missing_manifest_is_refused(tmp_path: Path):
    source, published, manifest = _published(tmp_path)
    manifest.unlink()
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    with pytest.raises(ArtifactPurgeError, match="manifest"):
        run_artifact_purge(_scope(tmp_path, ledger))


def test_multiple_db_artifacts_require_explicit_choice(tmp_path: Path):
    source, published, manifest = _published(tmp_path)
    (tmp_path / "artifact" / "second.db").write_bytes(published.read_bytes())
    (tmp_path / "artifact" / "second.db.manifest.json").write_text("{}", encoding="utf-8")
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    with pytest.raises(ArtifactPurgeError, match="exactly one"):
        run_artifact_purge(_scope(tmp_path, ledger))
    # Explicit selection works.
    scope = ArtifactPurgeScope(
        artifact_root=tmp_path / "artifact",
        outbox_path=tmp_path / "outbox.db",
        ledger_path=ledger,
        snapshot_id="snap",
        backup_dir=tmp_path / "backups",
        reconciliation_path=tmp_path / "reconciliation.json",
        artifact_db=published,
    )
    result = run_artifact_purge(scope)
    assert result["status"] == "applied"


# --- backup verification failure path ----------------------------------------


def test_backup_failure_leaves_generation_unmutated(tmp_path: Path, monkeypatch):
    source, published, manifest = _published(tmp_path)
    prior = published.read_bytes()
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = _scope(tmp_path, ledger)

    import reddit_search.operations.artifact_purge as module

    def explode(source_db: Path, destination: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(module, "_copy_sqlite_backup", explode)
    with pytest.raises(ArtifactPurgeError, match="backup could not be written"):
        run_artifact_purge(scope)
    assert published.read_bytes() == prior
    # The outbox was never created and no reconciliation was written.
    assert not scope.outbox_path.exists()
    assert not scope.reconciliation_path.exists()


def test_backup_integrity_check_failure_is_refused(tmp_path: Path, monkeypatch):
    source, published, manifest = _published(tmp_path)
    prior = published.read_bytes()
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = _scope(tmp_path, ledger)

    import reddit_search.operations.artifact_purge as module

    real_counts = module._table_counts

    def lying_integrity(path: Path) -> str:
        return "corrupt page"

    original_connect = sqlite3.connect

    def patched_connect(target, *args, **kwargs):
        connection = original_connect(target, *args, **kwargs)
        if target.endswith(f"{module._FINGERPRINT_TABLES[0]}") is False and "integrity" in str(kwargs):
            return connection
        return connection

    monkeypatch.setattr(module, "_table_counts", real_counts)
    monkeypatch.setattr(
        module,
        "_verify_backup",
        lambda *a, **k: (_ for _ in ()).throw(ArtifactPurgeError("backup db failed integrity check")),
    )
    with pytest.raises(ArtifactPurgeError, match="integrity"):
        run_artifact_purge(scope)
    assert published.read_bytes() == prior
    assert not scope.outbox_path.exists()


# --- rollback on propagation/publication failure -----------------------------


def test_propagation_failure_keeps_prior_generation_published(tmp_path: Path, monkeypatch):
    source, published, manifest = _published(tmp_path)
    prior = published.read_bytes()
    prior_manifest = manifest.read_text(encoding="utf-8")
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = _scope(tmp_path, ledger)

    import reddit_search.operations.artifact_purge as module

    def failing_boundary(input_path, output_path, manifest_path, projection, *, snapshot_id):
        raise RuntimeError("propagation exploded mid-staging")

    monkeypatch.setattr(module, "_propagate_boundary", failing_boundary)
    with pytest.raises(RuntimeError, match="propagation exploded"):
        run_artifact_purge(scope)
    # Prior generation is still the published one, byte for byte.
    assert published.read_bytes() == prior
    assert manifest.read_text(encoding="utf-8") == prior_manifest
    # The outbox row exists and honestly records the failed attempt.
    outbox = TombstoneOutbox(scope.outbox_path)
    rows = outbox.pending()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "propagation exploded" in rows[0]["last_error"]


def test_publication_failure_keeps_prior_generation_published(tmp_path: Path, monkeypatch):
    source, published, manifest = _published(tmp_path)
    prior = published.read_bytes()
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = _scope(tmp_path, ledger)

    import reddit_search.operations.artifact_purge as module

    def failing_publish(db, target_manifest, staging_db, staging_manifest):
        raise ArtifactPurgeError("atomic swap failed")

    monkeypatch.setattr(module, "_publish_generation", failing_publish)
    with pytest.raises(ArtifactPurgeError, match="atomic swap failed"):
        run_artifact_purge(scope)
    assert published.read_bytes() == prior
    outbox = TombstoneOutbox(scope.outbox_path)
    rows = outbox.pending()
    assert rows and rows[0]["status"] == "failed"


# --- retry semantics ---------------------------------------------------------


def test_failed_claim_is_retryable_to_completion(tmp_path: Path):
    source, published, manifest = _published(tmp_path)
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "r_x"}])
    scope = _scope(tmp_path, ledger)

    import reddit_search.operations.artifact_purge as module

    calls = {"count": 0}
    real_boundary = module._propagate_boundary

    def flaky_boundary(input_path, output_path, manifest_path, projection, *, snapshot_id):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient staging failure")
        return real_boundary(input_path, output_path, manifest_path, projection, snapshot_id=snapshot_id)

    module._propagate_boundary = flaky_boundary
    try:
        with pytest.raises(OperatorBoundaryError, match="transient"):
            run_artifact_purge(scope)
        # Retry the failed claim through the same boundary; the second call
        # passes through to the real propagation and the purge completes.
        result = run_artifact_purge(scope)
    finally:
        module._propagate_boundary = real_boundary
    assert result["status"] == "applied"
    assert result["generation"]["changed"] is True
    assert calls["count"] == 2
    with LexicalStore(published) as store:
        assert store.units(snapshot_id="snap") == [_unit("keep", "snap", "t1_keep", "r_keep", "keep me")]
