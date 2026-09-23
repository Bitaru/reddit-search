"""Read-only snapshot and dense-collection reconciliation manifests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reddit_search.ingest.invalidation import (
    EMPTY_TOMBSTONE_LEDGER,
    TombstoneLedger,
    tombstone_blocks_row,
)
from reddit_search.ingest.state import atomic_write_json, canonical_json_bytes, file_sha256
from reddit_search.retrieval.dense import DenseIndexJobStore, EmbeddingRecipe

_SCHEMA_VERSION = 1


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _unit_hash(unit: Mapping[str, Any], field: str, fallback: object) -> str:
    value = unit.get(field)
    if isinstance(value, str) and value:
        return value
    return _sha(fallback)


def _unit_key(unit: Mapping[str, Any]) -> str:
    value = unit.get("unit_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reconciliation units require a non-empty unit_id")
    return value


def _unit_identity(unit: Mapping[str, Any]) -> dict[str, str]:
    unit_id = _unit_key(unit)
    content_hash = _unit_hash(unit, "content_hash", unit.get("content", unit.get("focus_text", "")))
    context_hash = _unit_hash(
        unit,
        "context_hash",
        {
            "context_text": unit.get("context_text"),
            "context_message_refs": unit.get("context_message_refs", ()),
        },
    )
    return {"unit_id": unit_id, "content_hash": content_hash, "context_hash": context_hash}


def _read_index_jobs(
    path: Path, *, snapshot_id: str, recipe: EmbeddingRecipe
) -> dict[str, dict[str, str | None]]:
    recipe_hash = DenseIndexJobStore.recipe_hash(recipe)
    with sqlite3.connect(path) as db:
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(dense_index_jobs)")}
        required = {"snapshot_id", "recipe_hash", "unit_id", "content_hash", "status", "error"}
        if not required.issubset(columns):
            raise ValueError("index job database has an unsupported schema")
        rows = db.execute(
            "SELECT unit_id, content_hash, status, error FROM dense_index_jobs "
            "WHERE snapshot_id = ? AND recipe_hash = ? ORDER BY unit_id",
            (snapshot_id, recipe_hash),
        ).fetchall()
    return {
        str(unit_id): {
            "content_hash": str(content_hash),
            "status": str(status),
            "error": None if error is None else str(error),
        }
        for unit_id, content_hash, status, error in rows
    }


def _load_index_manifest(
    value: Path | Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    if value is None:
        return None, {}
    if isinstance(value, Path):
        payload = json.loads(value.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("index manifest must contain an object")
        return payload, {"index_manifest": file_sha256(value)}
    return dict(value), {"index_manifest": _sha(value)}


def reconcile_snapshot_collection(
    *,
    snapshot_id: str,
    recipe: EmbeddingRecipe,
    current_units: Sequence[Mapping[str, Any]],
    previous_units: Sequence[Mapping[str, Any]] = (),
    tombstone_ledger: TombstoneLedger = EMPTY_TOMBSTONE_LEDGER,
    index_jobs_path: Path | None = None,
    index_manifest: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, read-only reconciliation result.

    Unit rows are identity-only inputs. The function never contacts Qdrant and never
    changes a source, SQLite database, job ledger, or index manifest.
    """
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("snapshot_id must not be empty")
    current = {_unit_key(row): row for row in current_units}
    previous = {_unit_key(row): row for row in previous_units}
    if len(current) != len(current_units) or len(previous) != len(previous_units):
        raise ValueError("reconciliation unit IDs must be unique")

    current_identity = {key: _unit_identity(row) for key, row in current.items()}
    previous_identity = {key: _unit_identity(row) for key, row in previous.items()}
    blocked = sorted(
        key for key, row in current.items() if tombstone_blocks_row(tombstone_ledger, row)
    )
    blocked_set = set(blocked)
    active_current = {
        key: value for key, value in current_identity.items() if key not in blocked_set
    }
    added = sorted(set(current) - set(previous))
    removed = sorted(set(previous) - set(current))
    changed = sorted(
        key
        for key in set(current).intersection(previous)
        if current_identity[key] != previous_identity[key]
    )

    jobs = (
        _read_index_jobs(index_jobs_path, snapshot_id=snapshot_id, recipe=recipe)
        if index_jobs_path
        else {}
    )
    pending: list[str] = []
    indexed_ids: list[str] = []
    unresolved: list[dict[str, Any]] = []
    for unit_id in sorted(active_current):
        job = jobs.get(unit_id)
        if job is None:
            pending.append(unit_id)
            continue
        if (
            job["status"] == "indexed"
            and job["content_hash"] == active_current[unit_id]["content_hash"]
        ):
            indexed_ids.append(unit_id)
            continue
        pending.append(unit_id)
        mismatch: dict[str, Any] = {
            "unit_id": unit_id,
            "kind": "index_pending" if job["status"] == "pending" else "index_not_ready",
            "status": job["status"],
        }
        if job["content_hash"] != active_current[unit_id]["content_hash"]:
            mismatch["kind"] = "indexed_content_mismatch"
        if job["error"] is not None:
            mismatch["error"] = job["error"]
        unresolved.append(mismatch)

    manifest, manifest_hashes = _load_index_manifest(index_manifest)
    expected_collection = recipe.collection_name(snapshot_id)
    collection_mismatches: list[dict[str, str]] = []
    if manifest is not None:
        if manifest.get("snapshot_id") != snapshot_id:
            collection_mismatches.append(
                {
                    "field": "snapshot_id",
                    "expected": snapshot_id,
                    "actual": str(manifest.get("snapshot_id")),
                }
            )
        if manifest.get("recipe") != recipe.manifest():
            collection_mismatches.append(
                {
                    "field": "recipe",
                    "expected": _sha(recipe.manifest()),
                    "actual": _sha(manifest.get("recipe")),
                }
            )
        actual_collection = manifest.get("collection")
        if actual_collection is not None and actual_collection != expected_collection:
            collection_mismatches.append(
                {
                    "field": "collection",
                    "expected": expected_collection,
                    "actual": str(actual_collection),
                }
            )
        actual_count = manifest.get("actual_count")
        if isinstance(actual_count, int) and actual_count != len(active_current):
            collection_mismatches.append(
                {
                    "field": "actual_count",
                    "expected": str(len(active_current)),
                    "actual": str(actual_count),
                }
            )

    input_hashes: dict[str, Any] = {
        "current_units_sha256": _sha([current_identity[key] for key in sorted(current_identity)]),
        "previous_units_sha256": _sha(
            [previous_identity[key] for key in sorted(previous_identity)]
        ),
        **manifest_hashes,
    }
    if index_jobs_path is not None:
        input_hashes["index_jobs_sha256"] = file_sha256(index_jobs_path)
    tombstone_identity = {
        "digest": tombstone_ledger.digest,
        "count": tombstone_ledger.count,
        "path": str(tombstone_ledger.path) if tombstone_ledger.path else None,
    }
    status = "ready"
    if collection_mismatches or unresolved or pending or removed or added or changed:
        status = "needs_reconciliation"
    return {
        "kind": "snapshot_collection_reconciliation",
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "snapshot_id": snapshot_id,
        "recipe": recipe.manifest(),
        "recipe_sha256": _sha(recipe.manifest()),
        "collection": {"expected": expected_collection, "mismatches": collection_mismatches},
        "tombstones": tombstone_identity,
        "input_hashes": input_hashes,
        "units": {
            "current_count": len(current),
            "active_count": len(active_current),
            "added": added,
            "removed": removed,
            "changed": changed,
            "tombstone_blocked": blocked,
        },
        "index": {
            "expected_count": len(active_current),
            "indexed_count": len(indexed_ids),
            "pending_reindex": sorted(set(pending)),
            "indexed_unit_ids": indexed_ids,
        },
        "unresolved_mismatches": unresolved,
        "limitations": [
            (
                "Read-only local reconciliation; no Qdrant request or production artifact "
                "purge was performed."
            ),
            "An indexed count is evidence only for the supplied local manifest/job ledger.",
            (
                "Tombstone-blocked rows are excluded from expected index work; archive "
                "coverage is not inferred."
            ),
        ],
    }


def write_reconciliation_manifest(path: Path, result: Mapping[str, Any]) -> Path:
    """Atomically publish a previously computed reconciliation result."""
    atomic_write_json(path, dict(result))
    return path
