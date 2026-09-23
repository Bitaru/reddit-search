"""Local-only SQLite replay wiring for the durable tombstone outbox."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from reddit_search.corpus.sqlite_store import propagate_tombstone_projection
from reddit_search.ingest.invalidation import TombstoneProjection, project_tombstone_identities

from .outbox import TombstoneOperator, TombstoneOutbox


def reject_live_qdrant(projection: TombstoneProjection, *, scope: dict[str, Any]) -> None:
    """Fail closed: live Qdrant deletion is not permitted from local replay."""
    raise PermissionError("live Qdrant deletion is not permitted from local replay")


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


def validate_replay_targets(
    outbox: TombstoneOutbox,
    identity: str,
    *,
    expected_snapshot_id: str,
    sqlite_input: Path,
    sqlite_output: Path,
    sqlite_manifest: Path | None = None,
) -> dict[str, Any]:
    """Validate outbox identity, paths, and scope before any boundary claim."""
    row = outbox.get(identity)
    if row is None:
        raise KeyError(identity)
    outbox_resolved = outbox.path.resolve()
    input_resolved = sqlite_input.resolve()
    output_resolved = sqlite_output.resolve()
    if input_resolved == output_resolved:
        raise ValueError("SQLite input and output paths must differ")
    if sqlite_manifest is not None:
        manifest_resolved = sqlite_manifest.resolve()
        if manifest_resolved in {input_resolved, output_resolved}:
            raise ValueError("manifest path must differ from SQLite input and output")
    if output_resolved == outbox_resolved or (
        sqlite_output.exists() and sqlite_output.samefile(outbox.path)
    ):
        raise ValueError("output path must differ from the outbox database")
    if sqlite_manifest is not None and (
        manifest_resolved == outbox_resolved
        or (sqlite_manifest.exists() and sqlite_manifest.samefile(outbox.path))
    ):
        raise ValueError("manifest path must differ from the outbox database")
    if not input_resolved.is_file():
        raise ValueError(f"SQLite input does not exist: {sqlite_input}")
    scope = row["scope"]
    snapshot_id = scope.get("snapshot_id") if isinstance(scope, dict) else None
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("outbox row has no expected snapshot scope")
    if snapshot_id != expected_snapshot_id:
        raise ValueError("requested snapshot does not match outbox row scope")
    records = row["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("outbox row records must be a non-empty list")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("outbox row records must be objects")
        if record.get("status") in {"matched", "unresolved"} or record.get("coverage_status") in {
            "matched",
            "unresolved",
        }:
            raise ValueError("outbox row contains archive coverage, not a tombstone")
    # Ensure the selected projection still canonicalizes to the persisted identity.
    project_tombstone_identities(records)
    return row


def local_sqlite_operator(
    outbox: TombstoneOutbox,
    *,
    sqlite_input: Path,
    sqlite_output: Path,
    sqlite_manifest: Path | None = None,
) -> TombstoneOperator:
    """Build an operator whose SQLite boundary writes a fresh derivative only."""
    boundary = partial(
        _sqlite_boundary,
        sqlite_input,
        sqlite_output,
        sqlite_manifest,
    )
    return TombstoneOperator(
        outbox,
        sqlite_delete=boundary,
        qdrant_delete=reject_live_qdrant,
    )
