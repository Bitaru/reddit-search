# ruff: noqa: E501
"""Read-only tombstone outbox purge inventory.

This module inventories the durable ``TombstoneOutbox`` state without executing
any deletion. It never contacts Qdrant, never calls a delete boundary, and
never mutates the outbox database: the SQLite connection is opened in
read-only URI mode. The inventory is an operator planning artifact, not a
deletion claim and not archive-coverage evidence.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.ingest.state import atomic_write_json, file_sha256

_SCHEMA_VERSION = 1
_EXPECTED_COLUMN_ORDER: tuple[str, ...] = (
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
)
_KNOWN_STATUSES = frozenset({"pending", "in_progress", "applied", "failed"})
_STATUS_ORDER = ("pending", "in_progress", "applied", "failed")
_COVERAGE_STATUSES = frozenset({"matched", "unresolved"})


def build_purge_inventory(*, outbox: Path, output: Path) -> dict[str, Any]:
    """Project the outbox rows into a deterministic, read-only inventory.

    The outbox database is opened read-only and its SHA-256 is bound into the
    result. Rows are validated and projected content-free; applied rows expose
    their persisted ``backend_results`` verbatim while every other status keeps
    its truthful persisted fields (``null`` where no result exists). No
    deletion, coverage inference, or output write happens here: the caller
    publishes with :func:`write_purge_inventory` only after the whole
    projection succeeded.
    """
    outbox_path = outbox.resolve()
    if not outbox_path.is_file():
        raise ValueError(f"outbox database does not exist: {outbox_path}")
    if outbox_path == output.resolve():
        raise ValueError("output path must differ from the outbox database")
    outbox_sha256 = file_sha256(outbox_path)

    connection = sqlite3.connect(f"file:{outbox_path}?mode=ro", uri=True)
    try:
        connection.execute("BEGIN DEFERRED")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(tombstone_outbox)")]
        if set(columns) != set(_EXPECTED_COLUMN_ORDER):
            raise ValueError(
                "outbox schema mismatch: tombstone_outbox must have exactly "
                f"{sorted(_EXPECTED_COLUMN_ORDER)}, found {sorted(columns)}"
            )
        quoted = ", ".join(f'"{name}"' for name in _EXPECTED_COLUMN_ORDER)
        rows = connection.execute(
            f"SELECT {quoted} FROM tombstone_outbox ORDER BY identity"
        ).fetchall()
        projected_rows = [_project_row(row) for row in rows]
    except sqlite3.Error as error:
        raise ValueError(f"outbox database could not be read: {error}") from error
    finally:
        connection.close()

    status_counts = {status: 0 for status in _STATUS_ORDER}
    for item in projected_rows:
        status_counts[item["status"]] += 1
    return {
        "kind": "tombstone_outbox_purge_inventory",
        "schema_version": _SCHEMA_VERSION,
        "executed": False,
        "not_a_deletion_claim": True,
        "outbox": {
            "path": str(outbox_path),
            "sha256": outbox_sha256,
            "row_count": len(projected_rows),
            "status_counts": status_counts,
        },
        "rows": projected_rows,
        "limitations": [
            (
                "Read-only inventory: the outbox database was opened with SQLite URI "
                "mode=ro and was not modified, and no deletion was requested or executed."
            ),
            (
                "This artifact is not a deletion claim, not archive-coverage evidence, "
                "and not a production purge: it projects persisted outbox rows only."
            ),
            (
                "Rows omitted from the outbox, unregistered artifacts, and archive "
                "coverage are not represented here; absence of a row is not absence "
                "of content."
            ),
            (
                "Applied rows expose their persisted backend_results verbatim; pending, "
                "failed, and in_progress rows carry truthful persisted status and "
                "counts, without any freshness or success judgment."
            ),
        ],
    }


def write_purge_inventory(path: Path, result: dict[str, Any]) -> Path:
    """Atomically publish a previously computed purge inventory."""
    if result.get("kind") != "tombstone_outbox_purge_inventory":
        raise ValueError("purge inventory writer requires a tombstone_outbox_purge_inventory result")
    if result.get("executed") is not False or result.get("not_a_deletion_claim") is not True:
        raise ValueError("purge inventory must remain executed=false and not_a_deletion_claim=true")
    atomic_write_json(path, result)
    return path


def _project_row(row: tuple[Any, ...]) -> dict[str, Any]:
    item = dict(zip(_EXPECTED_COLUMN_ORDER, row, strict=True))
    identity = item["identity"]
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("outbox row identity must be a non-empty string")
    status = item["status"]
    if status not in _KNOWN_STATUSES:
        raise ValueError(f"outbox row {identity} has unsupported status: {status!r}")

    scope = _parse_scope(identity, item["scope"])
    projection = project_tombstone_identities(_parse_records(identity, item["records"]))

    requested_count = item["requested_count"]
    observed_count = item["observed_count"]
    deleted_count = item["deleted_count"]
    attempts = item["attempts"]
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (requested_count, observed_count, deleted_count, attempts)
    ):
        raise ValueError(f"outbox row {identity} has malformed counters")
    if deleted_count > observed_count:
        raise ValueError(f"outbox row {identity} reports more deletions than observations")
    if requested_count != len(projection.records):
        raise ValueError(
            f"outbox row {identity} requested_count {requested_count} does not match "
            f"its {len(projection.records)} persisted records"
        )

    backend_results: Any = None
    if item["backend_results"] is not None:
        try:
            backend_results = json.loads(item["backend_results"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"outbox row {identity} has malformed backend_results") from error
        if not isinstance(backend_results, list):
            raise ValueError(f"outbox row {identity} backend_results must be a list")
    if status == "applied":
        if backend_results is None:
            raise ValueError(
                f"applied outbox row {identity} lacks persisted backend_results"
            )
        if item["observed_at"] is None:
            raise ValueError(f"applied outbox row {identity} lacks observed_at")
    return {
        "identity": identity,
        "status": status,
        "scope": scope,
        "records": [dict(record) for record in projection.records],
        "records_sha256": projection.ledger_digest,
        "requested_count": requested_count,
        "observed_count": observed_count,
        "deleted_count": deleted_count,
        "attempts": attempts,
        "requested_at": item["requested_at"],
        "claimed_at": item["claimed_at"],
        "observed_at": item["observed_at"],
        "last_error": item["last_error"],
        "backend_results": backend_results,
    }


def _parse_scope(identity: str, raw: Any) -> dict[str, Any]:
    try:
        scope = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"outbox row {identity} has malformed scope") from error
    if not isinstance(scope, dict):
        raise ValueError(f"outbox row {identity} scope must be a JSON object")
    if scope.get("coverage_status") in _COVERAGE_STATUSES:
        raise ValueError(
            f"outbox row {identity} scope is archive coverage, not an operator scope"
        )
    snapshot_id = scope.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError(f"outbox row {identity} scope lacks a non-empty snapshot_id")
    return scope


def _parse_records(identity: str, raw: Any) -> list[dict[str, Any]]:
    try:
        records = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"outbox row {identity} has malformed records") from error
    if not isinstance(records, list) or not records:
        raise ValueError(f"outbox row {identity} must persist a non-empty record list")
    return records
