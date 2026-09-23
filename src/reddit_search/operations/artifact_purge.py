# ruff: noqa: E501
"""Destructive managed-artifact tombstone purge with backup and atomic publication.

The managed artifacts in this project are published SQLite derivatives plus
their sidecar manifests (the layout :func:`propagate_tombstone_projection`
publishes: ``<db>`` + ``<db>.manifest.json``). Purging a tombstoned identity
from a managed artifact means:

1. Validate an explicit scope, including alias and containment rails.
2. Verify the published generation against its manifest hash binding.
3. Back up the current generation (db via the SQLite backup API, manifest as a
   byte copy) under a content-addressed name and verify the backup readback
   before any mutation.
4. Run the tombstone projection through the configured operator flow
   (register -> claim -> boundaries) with propagation staged to a NEW output
   path, never in place, then atomically publish the new generation into the
   artifact root only after full success.
5. Record the whole operation as a durable, hash-bound, content-free claim in
   the outbox with a mode distinguishing destructive artifact purge from
   replay, plus a reconciliation manifest recording every generation hash.

Original archives are never rewritten: deletion claims apply only to derived
artifacts. The read-only ``tombstones inventory`` command is untouched by and
unrelated to this module.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from reddit_search.corpus.sqlite_store import propagate_tombstone_projection
from reddit_search.ingest.invalidation import (
    TombstoneLedger,
    TombstoneProjection,
    load_tombstone_ledger,
    project_tombstone_identities,
)
from reddit_search.ingest.state import atomic_write_json, canonical_json_bytes, file_sha256

from .operator import (
    OperatorBoundaryError,
    _reject_alias,
    _resolved,
    suppression_precheck,
)
from .outbox import TombstoneOperator, TombstoneOutbox
from .wiring import validate_replay_targets

_SCHEMA_VERSION = 1
_PURGE_MODE = "destructive_artifact_purge"
_FINGERPRINT_TABLES = ("search_units", "tombstones", "unit_fts")


class ArtifactPurgeError(ValueError):
    """Scope validation, backup, or publication refusal; nothing was mutated."""


@dataclass(frozen=True, slots=True)
class ArtifactPurgeScope:
    """Fully explicit destructive-purge configuration; no inference, no defaults."""

    artifact_root: Path
    outbox_path: Path
    ledger_path: Path
    snapshot_id: str
    backup_dir: Path
    reconciliation_path: Path
    artifact_db: Path | None = None


def _discover_artifact(root: Path, explicit: Path | None) -> tuple[Path, Path]:
    """Resolve the published db and its expected sidecar manifest."""
    root_resolved = _resolved(root)
    if not root_resolved.is_dir():
        raise ArtifactPurgeError(f"artifact root is not a directory: {root}")
    if explicit is not None:
        db = _resolved(explicit)
        if not db.is_file():
            raise ArtifactPurgeError(f"artifact db does not exist: {explicit}")
        if db.parent != root_resolved:
            raise ArtifactPurgeError(
                f"artifact db must be a direct child of the artifact root: {explicit}"
            )
    else:
        candidates = sorted(
            path for path in root_resolved.iterdir() if path.is_file() and path.suffix == ".db"
        )
        if len(candidates) != 1:
            raise ArtifactPurgeError(
                "artifact root must contain exactly one published .db file; found "
                f"{len(candidates)}: pass --artifact-db explicitly"
            )
        db = candidates[0]
    manifest = db.with_name(db.name + ".manifest.json")
    if not manifest.is_file():
        raise ArtifactPurgeError(
            f"artifact root is missing the expected sidecar manifest: {manifest}"
        )
    return db, manifest


def _load_published_manifest(manifest: Path, db: Path, snapshot_id: str) -> dict[str, Any]:
    """Verify the published generation against its manifest hash binding."""
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactPurgeError(f"artifact manifest is not readable JSON: {manifest}") from error
    if not isinstance(value, dict):
        raise ArtifactPurgeError(f"artifact manifest must contain a JSON object: {manifest}")
    recorded = value.get("output_sha256")
    actual = file_sha256(db)
    if not isinstance(recorded, str) or recorded != actual:
        raise ArtifactPurgeError(
            "published generation does not match the artifact manifest "
            f"(recorded {recorded!r}, actual {actual!r}); refusing to purge a tampered or stale artifact"
        )
    recorded_snapshot = value.get("snapshot_id")
    if recorded_snapshot != snapshot_id:
        raise ArtifactPurgeError(
            f"artifact manifest snapshot {recorded_snapshot!r} does not match the "
            f"requested snapshot {snapshot_id!r}; refusing cross-snapshot purge"
        )
    return value


def validate_artifact_purge_scope(
    scope: ArtifactPurgeScope,
) -> tuple[Path, Path, Path]:
    """Resolve the artifact and reject aliasing/containment rails before any state is created.

    Backup directory and reconciliation output must be outside the artifact
    root; the root must contain the expected sidecar manifest. The outbox may
    not exist yet; the ledger must.
    """
    if not scope.snapshot_id or not scope.snapshot_id.strip():
        raise ArtifactPurgeError("snapshot_id must be a non-empty string")
    if not scope.ledger_path.is_file():
        raise ArtifactPurgeError(f"tombstone ledger does not exist: {scope.ledger_path}")
    db, manifest = _discover_artifact(scope.artifact_root, scope.artifact_db)
    root_resolved = _resolved(scope.artifact_root)
    backup_resolved = _resolved(scope.backup_dir)
    reconciliation_resolved = _resolved(scope.reconciliation_path)
    if backup_resolved.is_relative_to(root_resolved):
        raise ArtifactPurgeError(
            f"backup directory must be outside the artifact root: {scope.backup_dir}"
        )
    if reconciliation_resolved.is_relative_to(root_resolved):
        raise ArtifactPurgeError(
            f"reconciliation output must be outside the artifact root: {scope.reconciliation_path}"
        )
    named: list[tuple[str, Path]] = [
        ("outbox", scope.outbox_path),
        ("ledger", scope.ledger_path),
        ("artifact_db", db),
        ("artifact_manifest", manifest),
        ("backup_dir", scope.backup_dir),
        ("reconciliation", scope.reconciliation_path),
    ]
    for index, (label, _path) in enumerate(named):
        _reject_alias(label, named[index:])
    return root_resolved, db, manifest


def _claim_identity(projection: TombstoneProjection, claim_scope: dict[str, Any]) -> str:
    """Recompute the outbox identity exactly as TombstoneOutbox.register does."""
    records = [dict(item) for item in projection.records]
    return hashlib.sha256(
        canonical_json_bytes({"records": records, "scope": dict(claim_scope)})
    ).hexdigest()


def _read_outbox_row_status(outbox_path: Path, identity: str) -> str | None:
    """Read one outbox row status read-only without creating the outbox."""
    path = _resolved(outbox_path)
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = connection.execute(
            "SELECT status FROM tombstone_outbox WHERE identity = ?", (identity,)
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    return None if row is None else str(row[0])


def _reject_conflicting_claims(
    outbox_path: Path, records_json: str, claim_scope: dict[str, Any]
) -> None:
    """Refuse claims that bind the same tombstones to a conflicting scope.

    Rows registered by the replay/operate flow (no purge mode) with the same
    snapshot legitimately coexist; a snapshot mismatch or a purge claim on a
    different artifact root is a validation failure.
    """
    path = _resolved(outbox_path)
    if not path.is_file():
        return
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT scope FROM tombstone_outbox WHERE records = ?", (records_json,)
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return
    for (scope_json,) in rows:
        try:
            other = json.loads(scope_json)
        except (TypeError, ValueError):
            continue
        if not isinstance(other, dict):
            continue
        if other.get("snapshot_id") != claim_scope["snapshot_id"]:
            raise ArtifactPurgeError(
                "requested snapshot does not match outbox row scope: "
                f"{other.get('snapshot_id')!r} != {claim_scope['snapshot_id']!r}"
            )
        if other.get("mode") == _PURGE_MODE and other.get("artifact_root") != claim_scope[
            "artifact_root"
        ]:
            raise ArtifactPurgeError(
                "outbox already binds these tombstones to a different artifact root: "
                f"{other.get('artifact_root')!r} != {claim_scope['artifact_root']!r}"
            )


def _table_counts(path: Path) -> dict[str, int]:
    """Count rows of the fingerprint tables via a fresh read-only connection."""
    connection = sqlite3.connect(f"file:{_resolved(path)}?mode=ro", uri=True)
    try:
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        counts: dict[str, int] = {}
        for table in _FINGERPRINT_TABLES:
            if table in tables:
                counts[table] = int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
        return counts
    finally:
        connection.close()


def _semantic_fingerprint(path: Path) -> dict[str, Any]:
    """Hash the canonical content of every managed table, read-only.

    SQLite database files are not byte-stable across backup/staging copies
    (header and freelist churn), so generation equality is judged on semantic
    content: the ordered rows of ``search_units``, ``tombstones``, and
    ``unit_fts``.
    """
    connection = sqlite3.connect(f"file:{_resolved(path)}?mode=ro", uri=True)
    try:
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        digests: dict[str, Any] = {}
        for table in _FINGERPRINT_TABLES:
            if table not in tables:
                digests[table] = None
                continue
            columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
            quoted = ", ".join(f'"{name}"' for name in columns)
            rows = connection.execute(
                f'SELECT {quoted} FROM "{table}" ORDER BY {quoted}'
            ).fetchall()
            digests[table] = {
                "rows": len(rows),
                "sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
            }
    finally:
        connection.close()
    return {"tables": digests, "sha256": hashlib.sha256(canonical_json_bytes(digests)).hexdigest()}


def _copy_sqlite_backup(source: Path, destination: Path) -> None:
    """Copy a SQLite database through the backup API into a staged temp file."""
    staged = destination.with_name(destination.name + ".tmp")
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(staged)
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()
    staged.replace(destination)


def _verify_backup(
    published_db: Path,
    backup_db: Path,
    backup_manifest: Path,
    expected_manifest_bytes: bytes,
) -> None:
    """Verify a written backup by reading it back before anything is mutated."""
    recorded_db_sha = file_sha256(backup_db)
    readback_db_sha = file_sha256(backup_db)
    if recorded_db_sha != readback_db_sha:
        raise ArtifactPurgeError(f"backup db readback hash mismatch: {backup_db}")
    if backup_manifest.read_bytes() != expected_manifest_bytes:
        raise ArtifactPurgeError(f"backup manifest readback mismatch: {backup_manifest}")
    connection = sqlite3.connect(f"file:{_resolved(backup_db)}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()
    if integrity != "ok":
        raise ArtifactPurgeError(f"backup db failed integrity check: {backup_db}: {integrity}")
    published_counts = _table_counts(published_db)
    backup_counts = _table_counts(backup_db)
    if published_counts != backup_counts:
        raise ArtifactPurgeError(
            f"backup row counts diverge from the published generation: "
            f"{published_counts} != {backup_counts}"
        )


def _backup_generation(db: Path, manifest: Path, backup_dir: Path) -> dict[str, Any]:
    """Back up the current generation under a content-addressed name and verify it.

    Fails before any artifact mutation when the backup cannot be written or
    verified. The content-addressed name binds the backup to the published
    generation's exact file hash.
    """
    backup_resolved = _resolved(backup_dir)
    try:
        backup_resolved.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArtifactPurgeError(f"backup directory could not be created: {error}") from error
    published_db_sha = file_sha256(db)
    manifest_bytes = manifest.read_bytes()
    published_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    db_name = f"{published_db_sha}.db"
    manifest_name = f"{published_db_sha}.manifest.json"
    try:
        _copy_sqlite_backup(db, backup_resolved / db_name)
        (backup_resolved / manifest_name).write_bytes(manifest_bytes)
    except OSError as error:
        raise ArtifactPurgeError(f"backup could not be written: {error}") from error
    _verify_backup(db, backup_resolved / db_name, backup_resolved / manifest_name, manifest_bytes)
    return {
        "directory": str(backup_resolved),
        "content_addressing": "sha256 of the published generation's db file",
        "db": {"name": db_name, "sha256": file_sha256(backup_resolved / db_name)},
        "manifest": {
            "name": manifest_name,
            "sha256": file_sha256(backup_resolved / manifest_name),
        },
        "prior_generation": {
            "db_sha256": published_db_sha,
            "manifest_sha256": published_manifest_sha,
        },
    }


def _publish_generation(
    db: Path, manifest: Path, staging_db: Path, staging_manifest: Path
) -> dict[str, Any]:
    """Atomically swap a verified staged generation into the artifact root.

    Publishes only after both staged files copy cleanly into the root's
    filesystem; on any failure the prior generation remains the published one
    (same rollback pattern propagate_tombstone_projection uses).
    """
    staging_db_sha = file_sha256(staging_db)
    staging_manifest_bytes = staging_manifest.read_bytes()
    staging_manifest_sha = hashlib.sha256(staging_manifest_bytes).hexdigest()
    tmp_db = db.with_name(db.name + ".purge-tmp")
    tmp_manifest = manifest.with_name(manifest.name + ".purge-tmp")
    try:
        tmp_db.write_bytes(staging_db.read_bytes())
        tmp_manifest.write_bytes(staging_manifest_bytes)
        if file_sha256(tmp_db) != staging_db_sha:
            raise ArtifactPurgeError("staged db copy into the artifact root failed verification")
        if tmp_manifest.read_bytes() != staging_manifest_bytes:
            raise ArtifactPurgeError("staged manifest copy into the artifact root failed verification")
    except Exception:
        tmp_db.unlink(missing_ok=True)
        tmp_manifest.unlink(missing_ok=True)
        raise
    old_db = db.read_bytes()
    old_manifest = manifest.read_bytes()
    try:
        tmp_manifest.replace(manifest)
        tmp_db.replace(db)
    except Exception:
        manifest.write_bytes(old_manifest)
        db.write_bytes(old_db)
        tmp_db.unlink(missing_ok=True)
        tmp_manifest.unlink(missing_ok=True)
        raise
    if file_sha256(db) != staging_db_sha or manifest.read_bytes() != staging_manifest_bytes:
        db.write_bytes(old_db)
        manifest.write_bytes(old_manifest)
        raise ArtifactPurgeError(
            "published generation failed verification; the previous generation was restored"
        )
    return {"db_sha256": staging_db_sha, "manifest_sha256": staging_manifest_sha}


def _dense_not_applicable(projection: TombstoneProjection, *, scope: dict[str, Any]) -> dict[str, Any]:
    """Honest dense boundary: artifact purge manages SQLite derivatives only."""
    return {
        "observed_count": 0,
        "deleted_count": 0,
        "mode": "not_applicable",
        "reason": (
            "destructive artifact purge covers managed SQLite derivatives only; "
            "dense vector deletion is not part of this operation"
        ),
    }


def _propagate_boundary(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    projection: TombstoneProjection,
    *,
    snapshot_id: str,
) -> dict[str, Any]:
    """Stage a fresh tombstone-filtered generation; never writes in place."""
    return propagate_tombstone_projection(
        input_path,
        output_path,
        projection,
        snapshot_id=snapshot_id,
        manifest_path=manifest_path,
    )


def _outbox_block(row: dict[str, Any] | None, identity: str, *, recorded: bool) -> dict[str, Any]:
    return {
        "identity": identity,
        "mode": _PURGE_MODE,
        "recorded": recorded,
        "status": None if row is None else row.get("status"),
        "attempts": None if row is None else row.get("attempts"),
        "requested_count": None if row is None else row.get("requested_count"),
        "observed_count": None if row is None else row.get("observed_count"),
        "deleted_count": None if row is None else row.get("deleted_count"),
    }


def _build_reconciliation(
    *,
    scope: ArtifactPurgeScope,
    root: Path,
    db: Path,
    manifest: Path,
    ledger: TombstoneLedger,
    projection: TombstoneProjection,
    precheck: dict[str, Any],
    outbox: dict[str, Any],
    backup: dict[str, Any] | None,
    prior_generation: dict[str, str],
    new_generation: dict[str, str],
    staged: dict[str, Any] | None,
    generation_changed: bool,
    publication_retry: bool,
    status: str,
    executed: dict[str, bool],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Deterministic core plus one clearly non-deterministic observation block."""
    manifest_doc: dict[str, Any] = {
        "kind": "tombstone_artifact_purge_reconciliation",
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "destructive": True,
        "archives_touched": False,
        "idempotent": not generation_changed,
        "snapshot_id": scope.snapshot_id,
        "scope": {
            "artifact_root": str(root),
            "artifact_db": str(_resolved(db)),
            "artifact_manifest": str(_resolved(manifest)),
            "outbox": str(_resolved(scope.outbox_path)),
            "ledger": str(_resolved(scope.ledger_path)),
            "backup_dir": str(_resolved(scope.backup_dir)),
            "reconciliation": str(_resolved(scope.reconciliation_path)),
        },
        "ledger": {
            "path": str(ledger.path),
            "file_sha256": file_sha256(scope.ledger_path),
            "digest": ledger.digest,
            "count": ledger.count,
        },
        "projection": {
            "ledger_digest": projection.ledger_digest,
            "counts": dict(projection.counts),
            "source_artifact_hashes": dict(projection.source_artifact_hashes),
        },
        "suppression_precheck": precheck,
        "outbox": outbox,
        "backup": backup,
        "generation": {
            "changed": generation_changed,
            "publication_retry": publication_retry,
            "prior": dict(prior_generation),
            "new": dict(new_generation),
            "staged": staged,
        },
        "executed": dict(executed),
        "limitations": [
            (
                "Deletion claims apply to the managed SQLite derivative only; the "
                "original archives were never read for deletion, never rewritten, "
                "and remain untouched."
            ),
            (
                "Backup databases are produced through the SQLite backup API, so "
                "their bytes differ from the published file (header churn); backup "
                "fidelity is verified by integrity_check, row-count parity, and "
                "recorded hashes instead of byte equality."
            ),
            (
                "Generation equality is judged on semantic fingerprints of "
                "search_units, tombstones, and unit_fts because SQLite file bytes "
                "are not stable across staging copies."
            ),
        ],
    }
    manifest_doc["observation"] = {
        "observed_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
    }
    return manifest_doc


def _write_noop_reconciliation(
    *,
    scope: ArtifactPurgeScope,
    root: Path,
    db: Path,
    manifest: Path,
    ledger: TombstoneLedger,
    projection: TombstoneProjection,
    precheck: dict[str, Any],
    outbox_row_status: str | None,
    identity: str,
    prior_generation: dict[str, str],
    started: float,
) -> dict[str, Any]:
    """Publish a no-op reconciliation for an already-purged artifact."""
    outbox = _outbox_block(
        None if outbox_row_status is None else {"status": outbox_row_status},
        identity,
        recorded=outbox_row_status is not None,
    )
    reconciliation = _build_reconciliation(
        scope=scope,
        root=root,
        db=db,
        manifest=manifest,
        ledger=ledger,
        projection=projection,
        precheck=precheck,
        outbox=outbox,
        backup=None,
        prior_generation=prior_generation,
        new_generation=dict(prior_generation),
        staged=None,
        generation_changed=False,
        publication_retry=False,
        status="no_op",
        executed={"backup": False, "propagation": False, "publication": False, "outbox_claim": False},
        elapsed_seconds=time.monotonic() - started,
    )
    reconciliation["no_op_reason"] = (
        "suppression precheck reports every ledger identity already tombstoned in "
        "the published generation and the manifest hash binding is intact; no "
        "generation was churned and no claim was recorded"
    )
    atomic_write_json(scope.reconciliation_path, reconciliation)
    return reconciliation


def run_artifact_purge(scope: ArtifactPurgeScope) -> dict[str, Any]:
    """Run backup -> claim -> staged propagation -> atomic publication for one ledger.

    Idempotent: when the published generation already suppresses every ledger
    identity (proven by the read-only suppression precheck plus the manifest
    hash binding), a no-op reconciliation is produced and no generation, claim,
    or backup is churned. On any failure the prior generation remains the
    published one.
    """
    started = time.monotonic()
    root, db, manifest = validate_artifact_purge_scope(scope)
    prior_generation = {
        "db_sha256": file_sha256(db),
        "manifest_sha256": file_sha256(manifest),
    }
    _load_published_manifest(manifest, db, scope.snapshot_id)
    ledger = load_tombstone_ledger(scope.ledger_path)
    if ledger.count == 0:
        raise ArtifactPurgeError("tombstone ledger must contain at least one tombstone")
    projection = project_tombstone_identities(
        [record.as_dict() for record in ledger.records],
        source_artifacts={"sqlite": (db, prior_generation["db_sha256"])},
    )
    precheck = suppression_precheck(db, projection, snapshot_id=scope.snapshot_id)
    claim_scope = {
        "mode": _PURGE_MODE,
        "snapshot_id": scope.snapshot_id,
        "artifact_root": str(root),
        "artifact_db": db.name,
        "published_db_sha256": prior_generation["db_sha256"],
    }
    records_json = json.dumps(
        [dict(item) for item in projection.records], sort_keys=True
    )
    _reject_conflicting_claims(scope.outbox_path, records_json, claim_scope)
    identity = _claim_identity(projection, claim_scope)
    row_status = _read_outbox_row_status(scope.outbox_path, identity)
    fully_suppressed = (
        precheck["checked"] > 0 and precheck["already_suppressed"] == precheck["checked"]
    )
    if fully_suppressed and row_status not in {"pending", "in_progress"}:
        return _write_noop_reconciliation(
            scope=scope,
            root=root,
            db=db,
            manifest=manifest,
            ledger=ledger,
            projection=projection,
            precheck=precheck,
            outbox_row_status=row_status,
            identity=identity,
            prior_generation=prior_generation,
            started=started,
        )

    # Backup (and its full readback verification) precedes outbox claim
    # registration: a backup failure leaves nothing written anywhere.
    backup = _backup_generation(db, manifest, scope.backup_dir)
    outbox = TombstoneOutbox(scope.outbox_path)
    row = outbox.register(projection, scope=claim_scope)
    identity = row["identity"]
    staging = Path(tempfile.mkdtemp(prefix="reddit-search-artifact-purge-"))
    try:
        staging_db = staging / db.name
        staging_manifest = staging / manifest.name
        validate_replay_targets(
            outbox,
            identity,
            expected_snapshot_id=scope.snapshot_id,
            sqlite_input=db,
            sqlite_output=staging_db,
            sqlite_manifest=staging_manifest,
        )
        if row["status"] == "in_progress":
            raise ArtifactPurgeError(
                f"outbox row {identity} is in_progress; recover stale claims before purging"
            )
        publication_retry = row["status"] == "applied"
        if publication_retry:
            # The claim was applied but the published generation is not purged
            # (publication interrupted or the artifact was rewound). Propagation
            # is deterministic and idempotent; re-stage and re-publish.
            sqlite_result = _propagate_boundary(
                db, staging_db, staging_manifest, projection, snapshot_id=scope.snapshot_id
            )
            backend_results = [
                TombstoneOperator._normalize("sqlite", sqlite_result),
                TombstoneOperator._normalize(
                    "qdrant", _dense_not_applicable(projection, scope=claim_scope)
                ),
            ]
            row = outbox.complete(
                identity,
                observed_count=sum(item["observed_count"] for item in backend_results),
                deleted_count=sum(item["deleted_count"] for item in backend_results),
                backend_results=backend_results,
            )
        else:
            sqlite_delete = partial(
                _propagate_boundary, db, staging_db, staging_manifest
            )
            operator = TombstoneOperator(
                outbox, sqlite_delete=sqlite_delete, qdrant_delete=_dense_not_applicable
            )
            try:
                row = operator.replay(identity)
            except Exception as error:
                raise OperatorBoundaryError(str(error)) from error
        if row["status"] != "applied":
            raise OperatorBoundaryError(f"outbox row is {row['status']}, not applied")
        backend_results = row.get("backend_results") or []
        sqlite_backend = next(
            (item for item in backend_results if item.get("backend") == "sqlite"), None
        )
        if (
            sqlite_backend is None
            or not staging_db.is_file()
            or file_sha256(staging_db) != sqlite_backend.get("output_sha256")
        ):
            raise OperatorBoundaryError(
                "staged generation hash does not match the persisted propagation result"
            )
        prior_fingerprint = _semantic_fingerprint(db)
        staged_fingerprint = _semantic_fingerprint(staging_db)
        generation_changed = prior_fingerprint["sha256"] != staged_fingerprint["sha256"]
        if generation_changed:
            try:
                new_generation = _publish_generation(db, manifest, staging_db, staging_manifest)
            except Exception as error:
                outbox.fail(identity, error)
                raise
        else:
            new_generation = dict(prior_generation)
        executed = {
            "backup": True,
            "propagation": True,
            "publication": generation_changed,
            "outbox_claim": True,
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    reconciliation = _build_reconciliation(
        scope=scope,
        root=root,
        db=db,
        manifest=manifest,
        ledger=ledger,
        projection=projection,
        precheck=precheck,
        outbox=_outbox_block(row, identity, recorded=True),
        backup=backup,
        prior_generation=prior_generation,
        new_generation=new_generation,
        staged={
            "db_sha256": new_generation["db_sha256"] if generation_changed else None,
            "manifest_sha256": new_generation["manifest_sha256"] if generation_changed else None,
            "semantic_fingerprint": staged_fingerprint["sha256"],
            "published": generation_changed,
        },
        generation_changed=generation_changed,
        publication_retry=publication_retry,
        status="applied",
        executed=executed,
        elapsed_seconds=time.monotonic() - started,
    )
    atomic_write_json(scope.reconciliation_path, reconciliation)
    return reconciliation


__all__ = [
    "ArtifactPurgeError",
    "ArtifactPurgeScope",
    "run_artifact_purge",
]
