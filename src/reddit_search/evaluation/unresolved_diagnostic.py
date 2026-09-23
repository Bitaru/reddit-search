"""Deterministic, read-only diagnosis of unresolved archive exposure rows."""
from __future__ import annotations

import json
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from reddit_search.ingest.state import atomic_write_json, file_sha256

from .archive_coverage import _validate_audit


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _load_index_identities(index_directory: Path) -> set[tuple[str, str, str, str]]:
    if not index_directory.is_dir():
        raise ValueError("archive index directory is missing")
    buckets = sorted(index_directory.glob("source-*/bucket-*.sqlite"))
    if not buckets:
        raise ValueError("archive index has no bucket databases")
    identities: set[tuple[str, str, str, str]] = set()
    for bucket in buckets:
        connection = sqlite3.connect(bucket)
        try:
            schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='archive'"
            ).fetchone()
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(archive)").fetchall()
            }
            required = {"fullname", "revision", "thread", "text"}
            if not schema or not schema[0] or not required <= columns:
                raise ValueError(f"archive index schema is incomplete: {bucket}")
            identities.update(
                tuple(_norm(str(value)) for value in row)
                for row in connection.execute(
                    "SELECT fullname, revision, thread, text FROM archive"
                )
            )
        finally:
            connection.close()
    return identities


def diagnose_unresolved(
    audit_directory: Path, corpus_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Classify unresolved rows using only a completed audit, corpus, and retained index."""
    audit_directory, corpus_path, output_directory = map(
        Path, (audit_directory, corpus_path, output_directory)
    )
    manifest, rows, inputs = _validate_audit(audit_directory)
    corpus_input = inputs["corpus"]
    if str(corpus_path) != str(corpus_input["path"]):
        raise ValueError("retained corpus path does not match audit identity")
    if file_sha256(corpus_path) != corpus_input["sha256"]:
        raise ValueError("retained corpus sha256 mismatch")
    unresolved = [row for row in rows if row.get("status") == "unresolved"]
    if len(unresolved) != manifest["counts"]["unresolved"]:
        raise ValueError("unresolved row count does not match audit")
    for row in unresolved:
        if row.get("archive_matches"):
            raise ValueError("unresolved row contains archive matches")

    connection = sqlite3.connect(corpus_path)
    connection.row_factory = sqlite3.Row
    try:
        columns = {str(item[1]) for item in connection.execute("PRAGMA table_info(search_units)")}
        required = {"snapshot_id", "message_fullname", "source_revision_id"}
        if not required <= columns:
            raise ValueError("retained corpus search_units schema is incomplete")
        snapshot_id = manifest["snapshot_id"]
        snapshot_query = "SELECT 1 FROM search_units WHERE snapshot_id = ? LIMIT 1"
        if not connection.execute(snapshot_query, (snapshot_id,)).fetchone():
            raise ValueError("retained corpus snapshot is missing")
        diagnostics: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        raw_index_directory = Path(str(manifest["identity"].get("index_directory", "")))
        index_directory = raw_index_directory
        if not index_directory.is_absolute() and not index_directory.exists():
            index_directory = audit_directory / index_directory
        index_identities = _load_index_identities(index_directory)
        index_files = sorted(index_directory.glob("source-*/bucket-*.sqlite"))
        index_observed = [
            {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in index_files
        ]
        for row in unresolved:
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            fullname = row.get("message_fullname", source.get("message_fullname"))
            revision = row.get("source_revision_id", source.get("source_revision_id"))
            if not isinstance(fullname, str) or not isinstance(revision, str):
                raise ValueError("unresolved row lacks canonical identity")
            matches = connection.execute(
                "SELECT unit_id FROM search_units "
                "WHERE snapshot_id = ? AND message_fullname = ? "
                "AND source_revision_id = ? ORDER BY unit_id",
                (snapshot_id, fullname, revision),
            ).fetchall()
            thread = row.get("thread_fullname", source.get("thread_fullname"))
            text = row.get("focus_text", source.get("text"))
            index_key = (
                _norm(fullname),
                _norm(revision),
                _norm(thread) if isinstance(thread, str) else "",
                _norm(text) if isinstance(text, str) else "",
            )
            if index_key in index_identities:
                raise ValueError("unresolved row matches retained archive index")
            if len(matches) > 1:
                raise ValueError("retained corpus identity is duplicated")
            corpus_present = bool(matches)
            normalized_fullname = _norm(fullname)
            kind = (
                "submission"
                if normalized_fullname.startswith("t3_")
                else "comment"
                if normalized_fullname.startswith("t1_")
                else "unknown"
            )
            key = f"{kind}:{'present' if corpus_present else 'absent'}"
            counts[key] += 1
            diagnostics.append({
                "audit_row": row.get("line"), "candidate_id": row.get("candidate_id"),
                "message_fullname": fullname, "source_revision_id": revision,
                "source_kind": kind, "corpus_identity_present": corpus_present,
                "corpus_unit_ids": [item[0] for item in matches],
                "cause": "unknown_exact_identity_or_normalization_mismatch",
            })
    finally:
        connection.close()

    output_directory.mkdir(parents=True, exist_ok=True)
    rows_path = output_directory / "unresolved_diagnostics.jsonl"
    diagnostics.sort(
        key=lambda item: (
            item.get("audit_row") if isinstance(item.get("audit_row"), int) else 0,
            str(item.get("candidate_id", "")),
        )
    )
    temporary_rows_path = rows_path.with_suffix(rows_path.suffix + ".tmp")
    try:
        with temporary_rows_path.open("w", encoding="utf-8", newline="\n") as stream:
            for item in diagnostics:
                stream.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
        temporary_rows_path.replace(rows_path)
    finally:
        temporary_rows_path.unlink(missing_ok=True)
    coverage_path = audit_directory / "archive_index_coverage.jsonl"
    coverage = []
    with coverage_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("archive index coverage row must be an object")
            coverage.append(item)
    report = {
        "kind": "unresolved_archive_diagnostic", "schema_version": 1,
        "snapshot_id": manifest["snapshot_id"], "tombstone_safe": False,
        "cause": "unknown_exact_identity_or_normalization_mismatch",
        "counts": {"unresolved": len(diagnostics), **dict(sorted(counts.items()))},
        "source_index_coverage": coverage,
        "inputs": {
            "audit_manifest_sha256": file_sha256(
                audit_directory / "archive_audit_manifest.json"
            ),
            "corpus": {"path": str(corpus_path), "sha256": file_sha256(corpus_path)},
            "index_observed": index_observed,
        },
        "outputs": {rows_path.name: {"rows": len(diagnostics), "sha256": file_sha256(rows_path)}},
    }
    atomic_write_json(output_directory / "unresolved_diagnostic_manifest.json", report)
    return {
        **report,
        "manifest_path": str(output_directory / "unresolved_diagnostic_manifest.json"),
    }
