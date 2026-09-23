# ruff: noqa: E501
"""Durable, content-free tombstone cleanup outbox and operator workflow."""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reddit_search.ingest.invalidation import TombstoneProjection, project_tombstone_identities
from reddit_search.ingest.state import canonical_json_bytes


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TombstoneOutbox:
    """SQLite-backed idempotency and operator state for tombstone cleanup."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS tombstone_outbox (
                identity TEXT PRIMARY KEY, scope TEXT NOT NULL, records TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, requested_count INTEGER NOT NULL DEFAULT 0,
                observed_count INTEGER NOT NULL DEFAULT 0, deleted_count INTEGER NOT NULL DEFAULT 0,
                requested_at TEXT NOT NULL, claimed_at TEXT, observed_at TEXT,
                backend_results TEXT
            )"""
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(tombstone_outbox)")}
            if "claimed_at" not in columns:
                db.execute("ALTER TABLE tombstone_outbox ADD COLUMN claimed_at TEXT")
            if "backend_results" not in columns:
                db.execute("ALTER TABLE tombstone_outbox ADD COLUMN backend_results TEXT")

    def register(
        self, projection: TombstoneProjection, *, scope: Mapping[str, Any]
    ) -> dict[str, Any]:
        records = [dict(item) for item in projection.records]
        identity = hashlib.sha256(
            canonical_json_bytes({"records": records, "scope": dict(scope)})
        ).hexdigest()
        now = _now()
        with sqlite3.connect(self.path) as db:
            db.execute(
                """INSERT OR IGNORE INTO tombstone_outbox
                (identity, scope, records, status, requested_count, requested_at)
                VALUES (?, ?, ?, 'pending', ?, ?)""",
                (
                    identity,
                    json.dumps(dict(scope), sort_keys=True),
                    json.dumps(records, sort_keys=True),
                    len(records),
                    now,
                ),
            )
            row = db.execute(
                "SELECT * FROM tombstone_outbox WHERE identity = ?", (identity,)
            ).fetchone()
            assert row is not None
            return self._row(db, row)

    def recover_stale(self, *, before: str | None = None) -> int:
        with sqlite3.connect(self.path) as db:
            threshold = before or _now()
            result = db.execute(
                "UPDATE tombstone_outbox SET status='pending', last_error='recovered stale in-progress work' "
                "WHERE status='in_progress' AND claimed_at < ?",
                (threshold,),
            )
            return result.rowcount

    def pending(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            return [
                self._row(db, row)
                for row in db.execute(
                    "SELECT * FROM tombstone_outbox WHERE status IN ('pending','failed') "
                    "ORDER BY requested_at, identity"
                )
            ]

    def get(self, identity: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT * FROM tombstone_outbox WHERE identity = ?", (identity,)
            ).fetchone()
            return None if row is None else self._row(db, row)

    @staticmethod
    def _row(db: sqlite3.Connection, row: tuple[Any, ...]) -> dict[str, Any]:
        columns = [item[1] for item in db.execute("PRAGMA table_info(tombstone_outbox)")]
        result = dict(zip(columns, row, strict=True))
        result["scope"] = json.loads(result["scope"])
        result["records"] = json.loads(result["records"])
        if result.get("backend_results"):
            result["backend_results"] = json.loads(result["backend_results"])
        return result

    def _claim(self, identity: str) -> dict[str, Any]:
        now = _now()
        with sqlite3.connect(self.path) as db:
            result = db.execute(
                "UPDATE tombstone_outbox SET status='in_progress', attempts=attempts+1, "
                "claimed_at=?, observed_at=NULL WHERE identity=? AND status IN ('pending','failed')",
                (now, identity),
            )
            if result.rowcount != 1:
                row = db.execute(
                    "SELECT status FROM tombstone_outbox WHERE identity=?", (identity,)
                ).fetchone()
                if row is None:
                    raise KeyError(identity)
                raise RuntimeError(f"outbox row is already {row[0]}")
            row = db.execute(
                "SELECT * FROM tombstone_outbox WHERE identity=?", (identity,)
            ).fetchone()
            assert row is not None
            return self._row(db, row)

    def complete(
        self,
        identity: str,
        *,
        observed_count: int,
        deleted_count: int,
        backend_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "UPDATE tombstone_outbox SET status='applied', observed_count=?, deleted_count=?, "
                "last_error=NULL, observed_at=?, backend_results=? WHERE identity=?",
                (
                    observed_count,
                    deleted_count,
                    _now(),
                    json.dumps(backend_results, sort_keys=True),
                    identity,
                ),
            )
        return self.get(identity) or {}

    def fail(self, identity: str, error: Exception) -> dict[str, Any]:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "UPDATE tombstone_outbox SET status='failed', last_error=?, observed_at=? WHERE identity=?",
                (str(error), _now(), identity),
            )
        return self.get(identity) or {}


class TombstoneOperator:
    """Replay outbox rows through explicitly injected SQLite and Qdrant boundaries."""

    def __init__(
        self,
        outbox: TombstoneOutbox,
        *,
        sqlite_delete: Callable[..., Any],
        qdrant_delete: Callable[..., Any],
    ) -> None:
        self.outbox = outbox
        self.sqlite_delete = sqlite_delete
        self.qdrant_delete = qdrant_delete

    def _invoke(
        self,
        boundary: Callable[..., Any],
        projection: TombstoneProjection,
        scope: Mapping[str, Any],
    ) -> Any:
        parameters = inspect.signature(boundary).parameters
        if "snapshot_id" in parameters:
            return boundary(projection, snapshot_id=scope["snapshot_id"])
        return boundary(projection, scope=scope)

    def submit(
        self, projection: TombstoneProjection, *, scope: Mapping[str, Any]
    ) -> dict[str, Any]:
        row = self.outbox.register(projection, scope=scope)
        return self.replay(row["identity"])

    @staticmethod
    def _normalize(name: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} boundary result must be a mapping")
        if "observed_count" in value:
            observed = value["observed_count"]
        elif "matched_count" in value:
            observed = value["matched_count"]
        elif "deleted_count" in value:
            observed = value["deleted_count"]
        else:
            raise ValueError(f"{name} boundary result omits counts")
        deleted = value.get("deleted_count")
        if not isinstance(observed, int) or not isinstance(deleted, int):
            raise ValueError(f"{name} boundary counts must be integers")
        if observed < 0 or deleted < 0 or deleted > observed:
            raise ValueError(f"{name} boundary counts fail reconciliation")
        normalized = {"backend": name, "observed_count": observed, "deleted_count": deleted}
        # Preserve remaining JSON-serializable fields verbatim so the persisted
        # outbox row carries the full backend result (byte-stable reconciliation
        # input on idempotent re-runs). Non-serializable values are dropped.
        for key in sorted(set(value) - {"backend", "observed_count", "deleted_count"}):
            candidate = value[key]
            try:
                json.dumps(candidate, sort_keys=True)
            except (TypeError, ValueError):
                continue
            normalized[key] = candidate
        return normalized

    def replay(self, identity: str) -> dict[str, Any]:
        row = self.outbox.get(identity)
        if row is None:
            raise KeyError(identity)
        if row["status"] == "applied":
            return row
        claimed = self.outbox._claim(identity)
        projection = project_tombstone_identities(claimed["records"])
        try:
            sqlite_result = self._normalize(
                "sqlite", self._invoke(self.sqlite_delete, projection, claimed["scope"])
            )
            qdrant_result = self._normalize(
                "qdrant", self._invoke(self.qdrant_delete, projection, claimed["scope"])
            )
            results = [sqlite_result, qdrant_result]
            return self.outbox.complete(
                identity,
                observed_count=sum(item["observed_count"] for item in results),
                deleted_count=sum(item["deleted_count"] for item in results),
                backend_results=results,
            )
        except Exception as error:
            self.outbox.fail(identity, error)
            raise

    def replay_pending(self) -> list[dict[str, Any]]:
        return [self.replay(row["identity"]) for row in self.outbox.pending()]
