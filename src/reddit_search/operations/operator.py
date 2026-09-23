"""Configured end-to-end tombstone operator: one command, honest statuses."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reddit_search.corpus.sqlite_store import propagate_tombstone_projection
from reddit_search.ingest.invalidation import (
    TombstoneLedger,
    TombstoneProjection,
    load_tombstone_ledger,
    project_tombstone_identities,
)
from reddit_search.ingest.state import atomic_write_json, file_sha256
from reddit_search.retrieval.dense import EmbeddingRecipe, QdrantHttpIndex

from .outbox import TombstoneOperator, TombstoneOutbox
from .wiring import reject_live_qdrant, validate_replay_targets


class OperatorBoundaryError(RuntimeError):
    """A replay boundary failed after the outbox row was claimed."""


@dataclass(frozen=True, slots=True)
class DenseScope:
    """Explicit loopback dense deletion configuration."""

    base_url: str
    collection: str
    recipe: EmbeddingRecipe


@dataclass(frozen=True, slots=True)
class OperatorScope:
    """Fully explicit operator configuration; no inference, no defaults."""

    outbox_path: Path
    ledger_path: Path
    snapshot_id: str
    sqlite_input: Path
    sqlite_output: Path
    sqlite_manifest: Path | None
    reconciliation_path: Path
    dense: DenseScope | None = None


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _aliases(a: Path, b: Path) -> bool:
    """Return whether two paths name the same file."""
    resolved_a, resolved_b = _resolved(a), _resolved(b)
    if resolved_a == resolved_b:
        return True
    try:
        return resolved_a.exists() and resolved_b.exists() and resolved_a.samefile(resolved_b)
    except OSError:
        return False


def _reject_alias(label: str, pairs: list[tuple[str, Path]]) -> None:
    target_label, target = pairs[0]
    for other_label, other in pairs[1:]:
        if _aliases(target, other):
            raise ValueError(f"{label} path aliases {other_label} path: {target}")


def validate_operator_scope(
    *,
    outbox: Path,
    ledger: Path,
    sqlite_input: Path,
    sqlite_output: Path,
    sqlite_manifest: Path | None,
    reconciliation: Path,
    dense: DenseScope | None,
) -> None:
    """Reject aliasing among every operator path before any state is created.

    The outbox database may not exist yet; the ledger and SQLite input must
    exist. All other paths are pairwise-distinct by resolved identity.
    """
    if not sqlite_input.is_file():
        raise ValueError(f"SQLite input does not exist: {sqlite_input}")
    if not ledger.is_file():
        raise ValueError(f"tombstone ledger does not exist: {ledger}")
    named: list[tuple[str, Path]] = [
        ("outbox", outbox),
        ("ledger", ledger),
        ("sqlite_input", sqlite_input),
        ("sqlite_output", sqlite_output),
        ("reconciliation", reconciliation),
    ]
    if sqlite_manifest is not None:
        named.append(("sqlite_manifest", sqlite_manifest))
    for index, (label, _path) in enumerate(named):
        _reject_alias(label, named[index:])
    if dense is not None:
        if not dense.base_url.strip():
            raise ValueError("dense base_url must not be empty when dense is enabled")
        if not dense.collection.strip():
            raise ValueError("dense collection must not be empty when dense is enabled")
        # Validates URL/loopback/recipe invariants without any network I/O.
        QdrantHttpIndex(
            dense.base_url, collection=dense.collection, recipe=dense.recipe
        )


def suppression_precheck(
    sqlite_input: Path,
    projection: TombstoneProjection,
    *,
    snapshot_id: str,
) -> dict[str, Any]:
    """Read-only report of ledger identities already tombstoned in the input.

    Mirrors the propagate direct clause: a ``tombstones`` row suppresses a
    ledger identity when the fullname and snapshot match and the stored
    revision is either NULL (fullname-wide) or exactly the ledger revision.
    ``unit_id`` is ignored so precheck never under-reports suppression.
    """
    input_hash = file_sha256(sqlite_input)
    suppressed: list[dict[str, str | None]] = []
    checked = 0
    uri = f"{sqlite_input.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "tombstones" not in tables:
            return {
                "kind": "suppression_precheck",
                "input_sha256": input_hash,
                "snapshot_id": snapshot_id,
                "checked": 0,
                "already_suppressed": 0,
                "suppressed_identities": [],
            }
        for record in projection.records:
            checked += 1
            fullname = record["message_fullname"]
            revision = record.get("source_revision_id")
            query = (
                "SELECT 1 FROM tombstones WHERE message_fullname = ? AND snapshot_id = ? "
                "AND (source_revision_id IS NULL OR source_revision_id = ?) LIMIT 1"
            )
            row = db.execute(query, (fullname, snapshot_id, revision)).fetchone()
            if row is not None:
                suppressed.append(
                    {"message_fullname": fullname, "source_revision_id": revision}
                )
    return {
        "kind": "suppression_precheck",
        "input_sha256": input_hash,
        "snapshot_id": snapshot_id,
        "checked": checked,
        "already_suppressed": len(suppressed),
        "suppressed_identities": sorted(
            suppressed,
            key=lambda item: (
                str(item["message_fullname"]),
                str(item["source_revision_id"] or ""),
            ),
        ),
    }


def _build_dense_delete(dense: DenseScope) -> Any:
    index = QdrantHttpIndex(
        dense.base_url, collection=dense.collection, recipe=dense.recipe
    )

    def dense_delete(projection: TombstoneProjection, *, snapshot_id: str) -> dict[str, Any]:
        return index.delete_tombstone_projection(projection, snapshot_id=snapshot_id)

    return dense_delete


def _sqlite_boundary(
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None,
    projection: TombstoneProjection,
    *,
    snapshot_id: str,
) -> dict[str, Any]:
    return propagate_tombstone_projection(
        input_path,
        output_path,
        projection,
        snapshot_id=snapshot_id,
        manifest_path=manifest_path,
    )


def _build_manifest(
    *,
    scope: OperatorScope,
    ledger: TombstoneLedger,
    projection: TombstoneProjection,
    precheck: dict[str, Any],
    row: dict[str, Any],
    backend_results: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Deterministic core plus one clearly non-deterministic block."""
    dense_result = next(
        (item for item in backend_results if item.get("backend") == "qdrant"), None
    )
    sqlite_result = next(
        (item for item in backend_results if item.get("backend") == "sqlite"), None
    )
    manifest: dict[str, Any] = {
        "kind": "tombstone_operator_reconciliation",
        "schema_version": 1,
        "status": str(row["status"]),
        "applied": row["status"] == "applied",
        "snapshot_id": scope.snapshot_id,
        "scope": {
            "outbox": str(_resolved(scope.outbox_path)),
            "ledger": str(_resolved(scope.ledger_path)),
            "sqlite_input": str(_resolved(scope.sqlite_input)),
            "sqlite_output": str(_resolved(scope.sqlite_output)),
            "sqlite_manifest": (
                str(_resolved(scope.sqlite_manifest)) if scope.sqlite_manifest else None
            ),
            "reconciliation": str(_resolved(scope.reconciliation_path)),
            "dense": (
                None
                if scope.dense is None
                else {
                    "mode": "loopback",
                    "base_url": scope.dense.base_url,
                    "collection": scope.dense.collection,
                    "recipe": {
                        "model_id": scope.dense.recipe.model_id,
                        "revision": scope.dense.recipe.revision,
                        "dimension": scope.dense.recipe.dimension,
                        "query_instruction": scope.dense.recipe.query_instruction,
                        "document_text_field": scope.dense.recipe.document_text_field,
                        "context_recipe_version": scope.dense.recipe.context_recipe_version,
                    },
                }
            ),
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
        "outbox": {
            "identity": row["identity"],
            "status": row["status"],
            "attempts": row["attempts"],
            "requested_count": row["requested_count"],
            "observed_count": row["observed_count"],
            "deleted_count": row["deleted_count"],
            "backend_results": backend_results,
        },
        "sqlite": None
        if sqlite_result is None
        else {
            "input_sha256": sqlite_result.get("input_sha256"),
            "output_sha256": sqlite_result.get("output_sha256"),
            "matched_count": sqlite_result.get("matched_count"),
            "deleted_count": sqlite_result.get("deleted_count"),
            "scrubbed_count": sqlite_result.get("scrubbed_count"),
            "idempotent": sqlite_result.get("idempotent"),
            "manifest_path": sqlite_result.get("manifest_path"),
        },
        "dense": None
        if dense_result is None
        else {"result": dict(dense_result)},
        "executed": {"sqlite": sqlite_result is not None, "dense": dense_result is not None},
    }
    # The only non-deterministic fields. Excluded from byte-stability checks.
    manifest["observation"] = {
        "observed_at": row.get("observed_at"),
        "elapsed_seconds": elapsed_seconds,
    }
    return manifest


def _reject_scope_conflict(
    outbox_path: Path, records: list[dict[str, Any]], snapshot_id: str
) -> None:
    """Refuse a snapshot claim that conflicts with an identical-records row.

    The outbox identity binds records plus scope, so re-registering under a
    different snapshot would silently create a second row for the same
    tombstones. That is a validation failure, not a boundary failure.
    """
    if not outbox_path.is_file():
        return
    records_json = json.dumps(records, sort_keys=True)
    with sqlite3.connect(outbox_path) as db:
        try:
            rows = db.execute(
                "SELECT scope FROM tombstone_outbox WHERE records = ?",
                (records_json,),
            ).fetchall()
        except sqlite3.OperationalError:
            return
    for (scope_json,) in rows:
        scope = json.loads(scope_json)
        if scope.get("snapshot_id") != snapshot_id:
            raise ValueError(
                "requested snapshot does not match outbox row scope: "
                f"{scope.get('snapshot_id')!r} != {snapshot_id!r}"
            )


def run_configured_operation(scope: OperatorScope) -> dict[str, Any]:
    """Run register -> precheck -> replay -> reconciliation for one ledger.

    Idempotent on identical inputs; fail-closed on hash mismatch; writes no
    reconciliation when any boundary fails (the outbox row already records the
    truthful error and remains retryable via this same command).
    """
    started = time.monotonic()
    validate_operator_scope(
        outbox=scope.outbox_path,
        ledger=scope.ledger_path,
        sqlite_input=scope.sqlite_input,
        sqlite_output=scope.sqlite_output,
        sqlite_manifest=scope.sqlite_manifest,
        reconciliation=scope.reconciliation_path,
        dense=scope.dense,
    )
    ledger = load_tombstone_ledger(scope.ledger_path)
    if ledger.count == 0:
        raise ValueError("tombstone ledger must contain at least one tombstone")
    projection = project_tombstone_identities(
        [record.as_dict() for record in ledger.records],
        source_artifacts={
            "sqlite": (scope.sqlite_input, file_sha256(scope.sqlite_input))
        },
    )
    _reject_scope_conflict(
        scope.outbox_path,
        [dict(record) for record in projection.records],
        scope.snapshot_id,
    )
    outbox = TombstoneOutbox(scope.outbox_path)
    row = outbox.register(projection, scope={"snapshot_id": scope.snapshot_id})
    identity = row["identity"]
    validate_replay_targets(
        outbox,
        identity,
        expected_snapshot_id=scope.snapshot_id,
        sqlite_input=scope.sqlite_input,
        sqlite_output=scope.sqlite_output,
        sqlite_manifest=scope.sqlite_manifest,
    )
    precheck = suppression_precheck(
        scope.sqlite_input, projection, snapshot_id=scope.snapshot_id
    )
    def sqlite_delete(
        projection: TombstoneProjection, *, snapshot_id: str
    ) -> dict[str, Any]:
        return _sqlite_boundary(
            scope.sqlite_input,
            scope.sqlite_output,
            scope.sqlite_manifest,
            projection,
            snapshot_id=snapshot_id,
        )

    dense_delete = (
        reject_live_qdrant
        if scope.dense is None
        else _build_dense_delete(scope.dense)
    )
    operator = TombstoneOperator(
        outbox, sqlite_delete=sqlite_delete, qdrant_delete=dense_delete
    )
    try:
        row = operator.replay(identity)
    except Exception as error:
        raise OperatorBoundaryError(str(error)) from error
    if row["status"] != "applied":
        raise OperatorBoundaryError(f"outbox row is {row['status']}, not applied")
    backend_results = row.get("backend_results") or []
    sqlite_result = next(
        (item for item in backend_results if item.get("backend") == "sqlite"), None
    )
    if sqlite_result is None or not isinstance(
        sqlite_result.get("output_sha256"), str
    ):
        raise OperatorBoundaryError("applied row lacks a persisted SQLite result")
    actual_output = _resolved(scope.sqlite_output)
    if not actual_output.is_file() or file_sha256(actual_output) != sqlite_result[
        "output_sha256"
    ]:
        raise OperatorBoundaryError(
            "SQLite derivative output hash does not match the persisted propagation result"
        )
    manifest = _build_manifest(
        scope=scope,
        ledger=ledger,
        projection=projection,
        precheck=precheck,
        row=row,
        backend_results=backend_results,
        elapsed_seconds=time.monotonic() - started,
    )
    atomic_write_json(scope.reconciliation_path, manifest)
    return manifest


def dense_scope_from_flags(
    *,
    dense_mode: str,
    dense_url: str | None,
    dense_collection: str | None,
    dense_model_id: str | None,
    dense_revision: str | None,
    dense_dimension: int | None,
    dense_query_instruction: str | None,
) -> DenseScope | None:
    """Build DenseScope from CLI flags; reject partial or conflicting groups."""
    loopback_flags = {
        "dense_url": dense_url,
        "dense_collection": dense_collection,
        "dense_model_id": dense_model_id,
        "dense_revision": dense_revision,
        "dense_dimension": dense_dimension,
        "dense_query_instruction": dense_query_instruction,
    }
    provided = [name for name, value in loopback_flags.items() if value is not None]
    if dense_mode == "disabled":
        if provided:
            raise ValueError(
                "dense flags require --dense-mode loopback: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in sorted(provided))
            )
        return None
    if dense_mode != "loopback":
        raise ValueError(f"unsupported dense mode: {dense_mode}")
    missing = [name for name, value in loopback_flags.items() if value is None]
    if missing:
        raise ValueError(
            "--dense-mode loopback requires all dense flags; missing: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in sorted(missing))
        )
    assert dense_url is not None
    assert dense_collection is not None
    assert dense_model_id is not None
    assert dense_revision is not None
    assert dense_dimension is not None
    assert dense_query_instruction is not None
    recipe = EmbeddingRecipe(
        model_id=dense_model_id,
        revision=dense_revision,
        dimension=dense_dimension,
        query_instruction=dense_query_instruction,
    )
    return DenseScope(
        base_url=dense_url, collection=dense_collection, recipe=recipe
    )


__all__ = [
    "DenseScope",
    "OperatorBoundaryError",
    "OperatorScope",
    "dense_scope_from_flags",
    "run_configured_operation",
    "suppression_precheck",
    "validate_operator_scope",
]
