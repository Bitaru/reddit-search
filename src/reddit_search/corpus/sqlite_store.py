"""Transactional SQLite FTS5 storage for the network-free lexical baseline."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reddit_search.ingest.invalidation import TombstoneProjection
from reddit_search.ingest.state import atomic_write_json, canonical_json_bytes, file_sha256

from .units import SearchUnit

# Context fields are concatenations of "\n\n"-separated blocks, each optionally
# introduced by a labeled marker naming the contributing message, e.g.
# "[SUBMISSION t3_x]\n…" or "[COMMENT t1_y]\n…". The focus block is
# "[FOCUS …]" and carries the unit's own text; it is never a contributor.
_CONTEXT_CONTRIBUTOR_MARKER = re.compile(r"^\[(?:SUBMISSION|COMMENT) (t[13]_[^ \]\n]+)\]")
_FOCUS_FULLNAME_MARKER = re.compile(
    r"^\[FOCUS (?:SUBMISSION|COMMENT) (t[13]_[^ \]\n]+)(?: \d+:\d+)?\]"
)


def _scrub_context_blocks(
    text: str, fullname: str, *, preserve_focus: bool, focus_fullname: str
) -> str:
    """Remove one tombstoned contributor's text from a context field.

    Serialization joins one message's labeled block with its raw-text
    continuation chunks using ``\\n\\n``, so attribution must follow marker
    contiguity: a contributor marker starts a segment and unlabeled chunks
    continue the same segment. Scrubbing ``fullname`` drops its whole segment;
    other contributors' segments — including their continuation chunks — are
    kept intact. A ``[FOCUS <KIND> <fullname>]`` chunk is the unit's genuine
    focus only when ``<fullname>`` equals ``focus_fullname``; any other
    FOCUS-marked chunk is user text and a continuation of the preceding
    contributor segment (or unattributable). Chunks before any marker and
    unattributable chunks after a genuine focus block are dropped fail-closed.
    """
    if not text:
        return text
    kept: list[str] = []
    active: list[str] | None = None  # current contributor segment, None if unattributable
    focus_seen = False
    for chunk in text.split("\n\n"):
        contributor = _CONTEXT_CONTRIBUTOR_MARKER.match(chunk)
        if contributor is not None:
            _flush_segment(kept, active, fullname)
            active = None
            if not focus_seen and contributor.group(1) != fullname:
                active = [chunk]
            continue
        focus = _FOCUS_FULLNAME_MARKER.match(chunk)
        if focus is not None and not focus_seen and focus.group(1) == focus_fullname:
            _flush_segment(kept, active, fullname)
            active = None
            focus_seen = True
            if preserve_focus:
                kept.append(chunk)
            continue
        if focus_seen:
            # After the genuine focus block, chunks are unattributable.
            continue
        if active is not None:
            # Continuation of the active segment, including fake FOCUS-marked
            # user text and the chunks that follow it.
            active.append(chunk)
        # Unattributable chunk: drop fail-closed.
    _flush_segment(kept, active, fullname)
    return "\n\n".join(kept)


def _flush_segment(kept: list[str], segment: list[str] | None, fullname: str) -> None:
    """Emit a completed contributor segment unless it names the scrubbed fullname."""
    if segment is not None and _CONTEXT_CONTRIBUTOR_MARKER.match(segment[0]).group(1) != fullname:
        kept.extend(segment)


@dataclass(frozen=True, slots=True)
class LexicalHit:
    unit: SearchUnit
    score: float
    rank: int
    snippet: str


def _unit_row_count(path: Path) -> int:
    """Count canonical unit rows via a fresh read-only connection."""
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM search_units").fetchone()[0])
    finally:
        connection.close()


def propagate_tombstone_projection(
    input_path: Path,
    output_path: Path,
    projection: TombstoneProjection,
    snapshot_id: str,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Publish a tombstone-filtered SQLite derivative without changing input."""
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    if input_resolved == output_resolved or (
        output_path.exists() and input_path.samefile(output_path)
    ):
        raise ValueError("input and output paths must differ")
    if manifest_path is not None:
        manifest_resolved = manifest_path.resolve()
        if manifest_resolved in {input_resolved, output_resolved} or (
            manifest_path.exists()
            and (
                (input_path.exists() and manifest_path.samefile(input_path))
                or (output_path.exists() and manifest_path.samefile(output_path))
            )
        ):
            raise ValueError("manifest path must differ from input and output")
        if (
            manifest_path.exists()
            and manifest_path.is_file()
            and manifest_path.read_bytes()[:16] == b"SQLite format 3\x00"
        ):
            raise ValueError(
                "manifest path is an existing SQLite database; refusing to overwrite: "
                f"{manifest_path}"
            )
    if not input_path.is_file():
        raise ValueError(f"SQLite input does not exist: {input_path}")
    source_hash = file_sha256(input_path)
    expected = (
        projection.source_artifact_hashes.get("input")
        or projection.source_artifact_hashes.get("sqlite")
        or projection.source_artifact_hashes.get("corpus_sqlite")
    )
    if expected is not None and source_hash != expected:
        raise ValueError("SQLite input hash mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as directory:
        staged = Path(directory) / output_path.name
        source = sqlite3.connect(input_path)
        destination = sqlite3.connect(staged)
        try:
            source.backup(destination)
            destination.commit()
        finally:
            destination.close()
            source.close()
        staged_units = _unit_row_count(staged)
        original_units = _unit_row_count(input_path)
        if staged_units != original_units:
            raise ValueError(
                f"SQLite backup is inconsistent: source has {original_units} units, "
                f"staged copy has {staged_units}"
            )
        with LexicalStore(staged) as store:
            result = store.consume_tombstone_projection(projection, snapshot_id=snapshot_id)
        output_hash = file_sha256(staged)
        result.update(
            {
                "input_sha256": source_hash,
                "output_sha256": output_hash,
                "projection_counts": dict(projection.counts),
                "policy": "exact_fullname_revision_and_scope",
                "snapshot_id": snapshot_id,
            }
        )
        if manifest_path is None:
            staged.replace(output_path)
        else:
            manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            atomic_write_json(manifest_tmp, result)
            old_output = output_path.read_bytes() if output_path.exists() else None
            old_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
            try:
                manifest_tmp.replace(manifest_path)
                staged.replace(output_path)
            except Exception:
                if old_manifest is None:
                    manifest_path.unlink(missing_ok=True)
                else:
                    manifest_path.write_bytes(old_manifest)
                if old_output is None:
                    output_path.unlink(missing_ok=True)
                else:
                    output_path.write_bytes(old_output)
                raise
    return result


class LexicalStore:
    """Keep canonical unit rows and their FTS index synchronized in one transaction."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def __enter__(self) -> LexicalStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.connection.close()

    def consume_tombstone_projection(
        self,
        projection: TombstoneProjection,
        *,
        snapshot_id: str,
        manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        """Apply an explicitly supplied identity projection, atomically.

        No archive lookup is performed: rows must match both fullname and revision,
        and an optional projection snapshot/unit scope must agree exactly.
        """
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("snapshot_id is required")
        records = [dict(record) for record in projection.records]
        expected_digest = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
        if projection.ledger_digest != expected_digest:
            raise ValueError("tombstone projection ledger identity is invalid")
        if projection.counts != {
            "records": len(records),
            "artifacts": len(projection.source_artifact_hashes),
        }:
            raise ValueError("tombstone projection counts are invalid")
        if projection.as_dict().get("propagation_status") != "not_claimed":
            raise ValueError("tombstone projection status is invalid")
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in projection.source_artifact_hashes.values()
        ):
            raise ValueError("tombstone projection artifact identity is invalid")
        allowed_fields = {
            "message_fullname",
            "source_revision_id",
            "candidate_id",
            "unit_id",
            "snapshot_id",
        }
        for record in records:
            if set(record) - allowed_fields:
                raise ValueError("tombstone projection has unsupported fields")
        for record in records:
            fullname = record.get("message_fullname")
            revision = record.get("source_revision_id")
            scoped_snapshot = record.get("snapshot_id")
            if (
                not isinstance(fullname, str)
                or not fullname.startswith(("t1_", "t3_"))
                or (
                    revision is not None and (not isinstance(revision, str) or not revision.strip())
                )
                or (scoped_snapshot is not None and scoped_snapshot != snapshot_id)
            ):
                raise ValueError("tombstone projection identity is invalid")
        input_hash = hashlib.sha256(canonical_json_bytes(projection.as_dict())).hexdigest()
        matched = deleted = 0
        scrubbed_keys: set[tuple[str, str]] = set()
        try:
            with self.connection:
                for record in records:
                    fullname = record["message_fullname"]
                    revision = record["source_revision_id"]
                    scoped_snapshot = record.get("snapshot_id") or snapshot_id
                    unit_scope = record.get("unit_id")
                    if revision is None:
                        self.connection.execute(
                            """
                            INSERT INTO tombstones
                                (message_fullname, source_revision_id, snapshot_id, unit_id)
                            SELECT ?, NULL, ?, ?
                            WHERE NOT EXISTS (
                                SELECT 1 FROM tombstones
                                WHERE message_fullname = ? AND source_revision_id IS NULL
                                  AND snapshot_id = ? AND unit_id IS ?
                            )
                            """,
                            (
                                fullname,
                                scoped_snapshot,
                                unit_scope,
                                fullname,
                                scoped_snapshot,
                                unit_scope,
                            ),
                        )
                    else:
                        self.connection.execute(
                            """
                            INSERT OR IGNORE INTO tombstones
                                (message_fullname, source_revision_id, snapshot_id, unit_id)
                            VALUES (?, ?, ?, ?)
                            """,
                            (fullname, revision, scoped_snapshot, unit_scope),
                        )
                    direct_clause = (
                        "message_fullname = ?"
                        if revision is None
                        else "(message_fullname = ? AND source_revision_id = ?)"
                    )
                    direct_params = [fullname] if revision is None else [fullname, revision]
                    ref_clause = (
                        "EXISTS (SELECT 1 FROM json_each(search_units.context_message_refs)"
                        " WHERE json_each.value = ?)"
                    )
                    if unit_scope is not None:
                        direct_clause = f"({direct_clause} AND unit_id = ?)"
                        direct_params.append(unit_scope)
                    direct_rows = self.connection.execute(
                        "SELECT unit_id FROM search_units WHERE snapshot_id = ? AND "
                        + direct_clause,
                        [scoped_snapshot, *direct_params],
                    ).fetchall()
                    matched += len(direct_rows)
                    for row in direct_rows:
                        self.connection.execute(
                            "DELETE FROM unit_fts WHERE unit_id = ? AND snapshot_id = ?",
                            (row["unit_id"], scoped_snapshot),
                        )
                        self.connection.execute(
                            "DELETE FROM search_units WHERE unit_id = ? AND snapshot_id = ?",
                            (row["unit_id"], scoped_snapshot),
                        )
                        deleted += 1
                    # Units that only reference the tombstoned message in their
                    # context survive, with the contributor's text scrubbed from
                    # every context field so stale context can no longer match.
                    context_rows = self.connection.execute(
                        "SELECT unit_id, message_fullname, source_revision_id,"
                        " focus_text, context_only_text, context_text,"
                        " missing_context_ids, context_message_refs"
                        " FROM search_units WHERE snapshot_id = ? AND " + ref_clause,
                        [scoped_snapshot, fullname],
                    ).fetchall()
                    matched += len(context_rows)
                    for row in context_rows:
                        refs = [
                            ref
                            for ref in json.loads(row["context_message_refs"])
                            if ref != fullname
                        ]
                        context_only_text = _scrub_context_blocks(
                            row["context_only_text"],
                            fullname,
                            preserve_focus=False,
                            focus_fullname=row["message_fullname"],
                        )
                        context_text = _scrub_context_blocks(
                            row["context_text"],
                            fullname,
                            preserve_focus=True,
                            focus_fullname=row["message_fullname"],
                        )
                        missing = list(json.loads(row["missing_context_ids"]))
                        if fullname not in missing:
                            missing.append(fullname)
                        self.connection.execute(
                            "DELETE FROM unit_fts WHERE unit_id = ? AND snapshot_id = ?",
                            (row["unit_id"], scoped_snapshot),
                        )
                        self.connection.execute(
                            """
                            UPDATE search_units SET context_only_text = ?, context_text = ?,
                                missing_context_ids = ?, context_message_refs = ?
                            WHERE unit_id = ? AND snapshot_id = ?
                            """,
                            (
                                context_only_text,
                                context_text,
                                json.dumps(missing),
                                json.dumps(refs),
                                row["unit_id"],
                                scoped_snapshot,
                            ),
                        )
                        self.connection.execute(
                            "INSERT INTO unit_fts (unit_id, snapshot_id, focus_text,"
                            " context_only_text) VALUES (?, ?, ?, ?)",
                            (
                                row["unit_id"],
                                scoped_snapshot,
                                row["focus_text"],
                                context_only_text,
                            ),
                        )
                        scrubbed_keys.add((scoped_snapshot, row["unit_id"]))
        except Exception:
            raise
        output_hash = file_sha256(self.path)
        result = {
            "kind": "sqlite_tombstone_propagation",
            "snapshot_id": snapshot_id,
            "ledger_digest": projection.ledger_digest,
            "ledger_sha256": projection.ledger_digest,
            "input_sha256": input_hash,
            "output_sha256": output_hash,
            "matched_count": matched,
            "deleted_count": deleted,
            "scrubbed_count": len(scrubbed_keys),
            "idempotent": deleted == 0,
        }
        if manifest_path is not None:
            atomic_write_json(manifest_path, result)
        return result

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS search_units (
                    snapshot_id TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    message_fullname TEXT NOT NULL,
                    source_revision_id TEXT NOT NULL,
                    thread_fullname TEXT NOT NULL,
                    focus_field TEXT NOT NULL,
                    focus_start INTEGER NOT NULL,
                    focus_end INTEGER NOT NULL,
                    focus_text TEXT NOT NULL,
                    context_only_text TEXT NOT NULL,
                    context_text TEXT NOT NULL,
                    missing_context_ids TEXT NOT NULL,
                    permalink TEXT NOT NULL,
                    subreddit TEXT NOT NULL,
                    created_utc INTEGER NOT NULL,
                    synthetic INTEGER NOT NULL,
                    ancestors_truncated INTEGER NOT NULL,
                    context_message_refs TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    context_recipe_version TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, unit_id)
                );
                CREATE TABLE IF NOT EXISTS tombstones (
                    message_fullname TEXT NOT NULL,
                    source_revision_id TEXT,
                    snapshot_id TEXT NOT NULL,
                    unit_id TEXT,
                    PRIMARY KEY (message_fullname, source_revision_id, snapshot_id, unit_id)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS unit_fts USING fts5(
                    unit_id UNINDEXED,
                    snapshot_id UNINDEXED,
                    focus_text,
                    context_only_text
                );
                """
            )
            self._migrate_tombstones()

    def _migrate_tombstones(self) -> None:
        columns = self.connection.execute("PRAGMA table_info(tombstones)").fetchall()
        revision = next((row for row in columns if row["name"] == "source_revision_id"), None)
        if revision is None or revision["notnull"] == 0:
            return
        with self.connection:
            self.connection.execute("ALTER TABLE tombstones RENAME TO tombstones_legacy")
            self.connection.execute(
                """
                CREATE TABLE tombstones (
                    message_fullname TEXT NOT NULL,
                    source_revision_id TEXT,
                    snapshot_id TEXT NOT NULL,
                    unit_id TEXT,
                    PRIMARY KEY (message_fullname, source_revision_id, snapshot_id, unit_id)
                )
                """
            )
            self.connection.execute(
                """
                INSERT INTO tombstones
                    (message_fullname, source_revision_id, snapshot_id, unit_id)
                SELECT message_fullname, source_revision_id, snapshot_id, unit_id
                FROM tombstones_legacy
                """
            )
            self.connection.execute("DROP TABLE tombstones_legacy")

    def index_units(self, units: Iterable[SearchUnit]) -> None:
        """Replace units atomically, skipping identities blocked by tombstones."""
        with self.connection:
            for unit in units:
                if self._is_blocked(unit):
                    continue
                self.connection.execute(
                    "DELETE FROM unit_fts WHERE unit_id = ? AND snapshot_id = ?",
                    (unit.unit_id, unit.snapshot_id),
                )
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO search_units (
                        snapshot_id, unit_id, message_fullname, source_revision_id,
                        thread_fullname, focus_field, focus_start, focus_end, focus_text,
                        context_only_text, context_text, missing_context_ids, permalink,
                        subreddit, created_utc, synthetic, context_message_refs,
                        chunking_version, context_recipe_version, ancestors_truncated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _unit_values(unit),
                )
                self.connection.execute(
                    "INSERT INTO unit_fts (unit_id, snapshot_id, focus_text, context_only_text) "
                    "VALUES (?, ?, ?, ?)",
                    (unit.unit_id, unit.snapshot_id, unit.focus_text, unit.context_only_text),
                )

    def _is_blocked(self, unit: SearchUnit) -> bool:
        refs = set(unit.context_message_refs)
        rows = self.connection.execute(
            "SELECT message_fullname, source_revision_id, unit_id "
            "FROM tombstones WHERE snapshot_id = ?",
            (unit.snapshot_id,),
        ).fetchall()
        return any(
            (row["unit_id"] is None or row["unit_id"] == unit.unit_id)
            and (
                (
                    row["message_fullname"] == unit.message_fullname
                    and (
                        row["source_revision_id"] is None
                        or row["source_revision_id"] == unit.source_revision_id
                    )
                )
                or row["unit_id"] is None
                and row["message_fullname"] in refs
            )
            for row in rows
        )

    def search(self, *, snapshot_id: str, query: str, limit: int) -> list[LexicalHit]:
        """Run a parameterized FTS query; BM25 lower scores rank first in SQLite."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = self.connection.execute(
            """
            SELECT
                search_units.*,
                bm25(unit_fts, 3.0, 1.0) AS lexical_score,
                snippet(unit_fts, 2, '<mark>', '</mark>', '…', 16) AS lexical_snippet
            FROM unit_fts
            JOIN search_units
              ON search_units.unit_id = unit_fts.unit_id
             AND search_units.snapshot_id = unit_fts.snapshot_id
            WHERE unit_fts MATCH ?
              AND search_units.snapshot_id = ?
            ORDER BY lexical_score ASC, search_units.unit_id ASC
            LIMIT ?
            """,
            (query, snapshot_id, limit),
        ).fetchall()
        return [
            LexicalHit(
                unit=_unit_from_row(row),
                score=float(row["lexical_score"]),
                rank=rank,
                snippet=str(row["lexical_snippet"]),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def units(self, *, snapshot_id: str) -> list[SearchUnit]:
        """Return all canonical units for one snapshot in stable ID order."""
        rows = self.connection.execute(
            """
            SELECT *
            FROM search_units
            WHERE snapshot_id = ?
            ORDER BY unit_id ASC
            """,
            (snapshot_id,),
        ).fetchall()
        return [_unit_from_row(row) for row in rows]


def _unit_values(unit: SearchUnit) -> tuple[object, ...]:
    return (
        unit.snapshot_id,
        unit.unit_id,
        unit.message_fullname,
        unit.source_revision_id,
        unit.thread_fullname,
        unit.focus_field,
        unit.focus_start,
        unit.focus_end,
        unit.focus_text,
        unit.context_only_text,
        unit.context_text,
        json.dumps(unit.missing_context_ids),
        unit.permalink,
        unit.subreddit,
        unit.created_utc,
        int(unit.synthetic),
        json.dumps(unit.context_message_refs),
        unit.chunking_version,
        unit.context_recipe_version,
        int(unit.ancestors_truncated),
    )


def _unit_from_row(row: sqlite3.Row) -> SearchUnit:
    return SearchUnit(
        unit_id=str(row["unit_id"]),
        snapshot_id=str(row["snapshot_id"]),
        message_fullname=str(row["message_fullname"]),
        source_revision_id=str(row["source_revision_id"]),
        thread_fullname=str(row["thread_fullname"]),
        focus_field=str(row["focus_field"]),
        focus_start=int(row["focus_start"]),
        focus_end=int(row["focus_end"]),
        focus_text=str(row["focus_text"]),
        context_only_text=str(row["context_only_text"]),
        context_text=str(row["context_text"]),
        missing_context_ids=tuple(json.loads(str(row["missing_context_ids"]))),
        permalink=str(row["permalink"]),
        subreddit=str(row["subreddit"]),
        created_utc=int(row["created_utc"]),
        synthetic=bool(row["synthetic"]),
        context_message_refs=tuple(json.loads(str(row["context_message_refs"]))),
        chunking_version=str(row["chunking_version"]),
        context_recipe_version=str(row["context_recipe_version"]),
        ancestors_truncated=bool(row["ancestors_truncated"]),
    )
