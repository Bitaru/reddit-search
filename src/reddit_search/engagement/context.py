from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import zstandard
from pydantic import ValidationError

from reddit_search.engagement.contracts import ShortlistCandidate, ShortlistManifest
from reddit_search.ingest.invalidation import load_tombstone_ledger
from reddit_search.ingest.normalize import NormalizationError, normalize_record
from reddit_search.ingest.reader import ArchiveReader, SourceReadError, SourceSpec
from reddit_search.ingest.sources import load_source_registry
from reddit_search.ingest.state import file_sha256

_ALLOWED_SCOPES = frozenset(
    {"authorized_local_research", "user_provided_local_research_no_external_transmission"}
)
_STATUS_FIELDS = (
    "locked",
    "archived",
    "removed_by_category",
    "removed_by",
    "banned_by",
    "spam",
    "deleted",
    "removed",
    "is_removed",
    "is_archived",
    "is_locked",
    "score",
    "num_comments",
)


def _archived_removal_reason(status: dict[str, Any], message_text: str) -> str | None:
    for field in ("removed_by_category", "removed_by", "banned_by"):
        if status.get(field) not in (None, "", False):
            return f"historical {field}={status[field]!r}"
    if status.get("deleted") or status.get("removed") or status.get("is_removed"):
        return "historical removed/deleted status"
    if message_text.strip().casefold() in {"[removed]", "[deleted]"}:
        return "message text is archived as removed/deleted"
    return None


def _record_thread_issue(
    state: dict[str, Any], code: str, detail: str, *, example_limit: int = 20
) -> None:
    counts = state["counts"]
    counts[code] = counts.get(code, 0) + 1
    if len(state["examples"]) < example_limit:
        state["examples"].append(f"{code}: {detail[:512]}")


def _record_archive_content_note(state: dict[str, Any], fullname: str, detail: str) -> None:
    counts = state["archive_content_counts"]
    counts["historical_removal"] = counts.get("historical_removal", 0) + 1
    if len(state["archive_content_notes"]) < 20:
        state["archive_content_notes"].append(f"{fullname}: {detail[:512]}")

_DEFAULT_MESSAGES = 100_000
_DEFAULT_THREAD_BYTES = 64 * 1024 * 1024
_DEFAULT_STAGING_BYTES = 2 * 1024 * 1024 * 1024


def build_thread_context(
    candidates_path: Path,
    shortlist_manifest_path: Path,
    registry_path: Path,
    output_dir: Path,
    *,
    tombstones_path: Path | None = None,
    max_messages_per_thread: int = _DEFAULT_MESSAGES,
    max_bytes_per_thread: int = _DEFAULT_THREAD_BYTES,
    max_staging_bytes: int = _DEFAULT_STAGING_BYTES,
) -> dict[str, Any]:
    """Scan each authorized registered archive once and stage bounded thread context."""
    if max_messages_per_thread <= 0 or max_bytes_per_thread <= 0:
        raise ValueError("thread limits must be positive")
    if max_staging_bytes < 32_768:
        raise ValueError("max_staging_bytes must be at least 32768 for bounded SQLite staging")
    for path in (candidates_path, shortlist_manifest_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"output directory must be fresh: {output_dir}")
    raw_manifest = json.loads(shortlist_manifest_path.read_text(encoding="utf-8"))
    manifest = ShortlistManifest.model_validate(raw_manifest)
    if file_sha256(candidates_path) != manifest.candidates_sha256:
        raise ValueError("candidate artifact hash does not match shortlist manifest")
    candidates: dict[str, dict[str, Any]] = {}
    candidate_count = 0
    with candidates_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            candidate_count += 1
            try:
                candidate = ShortlistCandidate.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(
                    f"invalid shortlist candidate at line {line_number}: {error}"
                ) from error
            if candidate.app_id != manifest.app_id:
                raise ValueError("candidate app_id does not match shortlist manifest")
            item = candidate.model_dump(mode="json")
            thread = item["thread_fullname"]
            prior = candidates.get(thread)
            if (
                prior is not None
                and prior["reply_target_fullname"] != item["reply_target_fullname"]
            ):
                prior["reply_targets"].append(item["reply_target_fullname"])
            elif prior is None:
                item["reply_targets"] = [item["reply_target_fullname"]]
                candidates[thread] = item
    if manifest.candidate_count != candidate_count:
        raise ValueError("shortlist candidate count does not match manifest")

    registry = load_source_registry(registry_path)
    records = registry.get("sources")
    if not isinstance(records, list):
        raise ValueError("source registry must contain sources list")
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("source_id"), str):
            raise ValueError("registry contains malformed source")
        source_id = record["source_id"]
        if source_id in seen:
            raise ValueError("registry source IDs must be unique")
        seen.add(source_id)
        if record.get("usage_scope") not in _ALLOWED_SCOPES:
            raise ValueError(f"source {source_id} usage_scope is not authorized locally")
        if record.get("status") != "complete":
            raise ValueError(f"source {source_id} is not verified")
        if record.get("invalid_records", 0) != 0:
            raise ValueError(f"source {source_id} has invalid records")
        if not isinstance(record.get("verified_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", record["verified_sha256"]
        ):
            raise ValueError(f"source {source_id} lacks a verified SHA-256")
        path = Path(str(record.get("source_path", "")))
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append(record)
    tombstones = load_tombstone_ledger(tombstones_path)
    expected_tombstone_hash = manifest.inputs.get("tombstone_sha256")
    actual_tombstone_hash = (
        file_sha256(tombstones_path)
        if tombstones_path is not None
        else hashlib.sha256(b"").hexdigest()
    )
    if expected_tombstone_hash != actual_tombstone_hash:
        raise ValueError("tombstone ledger hash does not match shortlist manifest")
    thread_issues: dict[str, dict[str, Any]] = {
        thread: {
            "counts": {},
            "examples": [],
            "archive_content_counts": {},
            "archive_content_notes": [],
        }
        for thread in candidates
    }
    matched_by_thread: dict[str, int] = {thread: 0 for thread in candidates}
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    db_path = output_dir / ".context.sqlite3"
    db_descriptor = os.open(db_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(db_descriptor, 0o600)
    os.close(db_descriptor)
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA page_size=4096")
    db.execute(f"PRAGMA max_page_count={max(1, max_staging_bytes // 4096)}")
    db.execute(
        "CREATE TABLE messages("
        "thread TEXT, fullname TEXT, revision TEXT, payload BLOB, status BLOB, "
        "size INTEGER, PRIMARY KEY(thread,fullname,revision))"
    )
    db.execute(
        "CREATE TABLE thread_usage("
        "thread TEXT PRIMARY KEY, message_count INTEGER, byte_count INTEGER)"
    )
    db.execute("CREATE INDEX by_thread ON messages(thread)")
    reports: list[dict[str, Any]] = []
    omitted_by_thread: dict[str, int] = {thread: 0 for thread in candidates}
    total_staged_bytes = 0
    scan_complete = True
    scanned = 0
    try:
        for record in sources:
            source = SourceSpec(
                source_id=str(record["source_id"]),
                path=Path(str(record["source_path"])),
                source_kind=record["source_kind"],
                declared_month=str(record["declared_month"]),
                source_role=record.get("source_role", "discovery"),
                usage_scope=str(record["usage_scope"]),
            )
            reader = ArchiveReader()
            errors: list[str] = []
            matched = 0
            normalization_error_count = 0
            try:
                for envelope in reader.iter_records(source):
                    scanned += 1
                    raw = envelope.payload
                    if source.source_kind == "submission":
                        identity = raw.get("name") or (
                            "t3_" + raw["id"] if isinstance(raw.get("id"), str) else None
                        )
                        thread = (
                            identity
                            if isinstance(identity, str) and identity in candidates
                            else None
                        )
                    else:
                        thread = raw.get("link_id")
                        if not isinstance(thread, str) or thread not in candidates:
                            continue
                    if thread is None:
                        continue
                    matched += 1
                    matched_by_thread[thread] += 1
                    try:
                        message = normalize_record(envelope)
                    except NormalizationError as error:
                        normalization_error_count += 1
                        if len(errors) < 20:
                            errors.append(
                                f"un-normalizable matching row "
                                f"{source.source_id}:{envelope.line_number}: {str(error)[:512]}"
                            )
                        scan_complete = False
                        continue
                    revision = message.source_revision_id
                    if tombstones.matches(message.fullname, revision):
                        _record_thread_issue(
                            thread_issues[thread],
                            "tombstoned",
                            f"tombstoned message omitted: {message.fullname}",
                        )
                        omitted_by_thread[thread] += 1
                        continue
                    status = {key: raw[key] for key in _STATUS_FIELDS if key in raw}
                    removal_reason = _archived_removal_reason(status, message.focus_text)
                    if removal_reason:
                        if message.fullname == thread:
                            _record_thread_issue(
                                thread_issues[thread],
                                "root_text_removed",
                                f"{message.fullname}: {removal_reason}",
                            )
                        elif message.fullname in candidates[thread]["reply_targets"]:
                            _record_thread_issue(
                                thread_issues[thread],
                                "target_text_removed",
                                f"{message.fullname}: {removal_reason}",
                            )
                        else:
                            _record_archive_content_note(
                                thread_issues[thread], message.fullname, removal_reason
                            )
                    item = {
                        "fullname": message.fullname,
                        "kind": message.kind,
                        "thread_fullname": message.thread_fullname,
                        "parent_fullname": message.parent_fullname,
                        "raw_title": message.raw_title,
                        "raw_body": message.raw_body,
                        "subreddit": message.subreddit,
                        "created_utc": message.created_utc,
                        "source_revision_id": revision,
                        "permalink": message.permalink,
                        "archive_score": message.archive_score,
                        "depth": message.depth,
                        "provenance": [
                            {"source_id": source.source_id, "line_number": envelope.line_number}
                        ],
                        "archived_status": status,
                    }
                    encoded = json.dumps(
                        item, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                    prior = db.execute(
                        "SELECT payload,status FROM messages "
                        "WHERE thread=? AND fullname=? AND revision=?",
                        (thread, message.fullname, revision),
                    ).fetchone()
                    if prior:
                        previous = json.loads(prior[0])
                        previous_status = json.loads(prior[1])
                        if previous_status != status:
                            _record_thread_issue(
                                thread_issues[thread],
                                "status_conflict",
                                f"conflicting archived status for {message.fullname}",
                            )
                        previous["provenance"].append(
                            {"source_id": source.source_id, "line_number": envelope.line_number}
                        )
                        merged = json.dumps(
                            previous, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                        ).encode()
                        delta = len(merged) - len(prior[0])
                        usage = db.execute(
                            "SELECT byte_count FROM thread_usage WHERE thread=?", (thread,)
                        ).fetchone()
                        if (
                            usage[0] + delta > max_bytes_per_thread
                            or total_staged_bytes + delta > max_staging_bytes
                        ):
                            _record_thread_issue(
                                thread_issues[thread],
                                "provenance_omitted",
                                f"duplicate provenance omitted for {message.fullname}",
                            )
                            omitted_by_thread[thread] += 1
                            continue
                        db.execute(
                            "UPDATE messages SET payload=?,size=size+? "
                            "WHERE thread=? AND fullname=? AND revision=?",
                            (merged, delta, thread, message.fullname, revision),
                        )
                        db.execute(
                            "UPDATE thread_usage SET byte_count=byte_count+? WHERE thread=?",
                            (delta, thread),
                        )
                        total_staged_bytes += delta
                        continue
                    revs = db.execute(
                        "SELECT revision FROM messages WHERE thread=? AND fullname=?",
                        (thread, message.fullname),
                    ).fetchall()
                    if revs:
                        _record_thread_issue(
                            thread_issues[thread],
                            "revision_conflict",
                            f"conflicting revisions for {message.fullname}",
                        )
                    usage = db.execute(
                        "SELECT message_count,byte_count FROM thread_usage WHERE thread=?",
                        (thread,),
                    ).fetchone()
                    count, used = usage if usage else (0, 0)
                    if (
                        count >= max_messages_per_thread
                        or used + len(encoded) > max_bytes_per_thread
                    ):
                        _record_thread_issue(
                            thread_issues[thread],
                            "thread_cap",
                            f"thread cap exceeded at {message.fullname}",
                        )
                        omitted_by_thread[thread] += 1
                        continue
                    if total_staged_bytes + len(encoded) > max_staging_bytes:
                        omitted_by_thread[thread] += 1
                        _record_thread_issue(
                            thread_issues[thread], "staging_cap", "global staging byte cap exceeded"
                        )
                        continue
                    db.execute(
                        "INSERT INTO messages VALUES(?,?,?,?,?,?)",
                        (
                            thread,
                            message.fullname,
                            revision,
                            encoded,
                            json.dumps(status, sort_keys=True).encode(),
                            len(encoded),
                        ),
                    )
                    db.execute(
                        "INSERT INTO thread_usage VALUES(?,?,?) "
                        "ON CONFLICT(thread) DO UPDATE SET "
                        "message_count=message_count+1, "
                        "byte_count=byte_count+excluded.byte_count",
                        (thread, 1, len(encoded)),
                    )
                    total_staged_bytes += len(encoded)
                if reader.stats.invalid_records:
                    errors.append(f"invalid JSONL records: {reader.stats.invalid_records}")
                    scan_complete = False
                expected_hashes = [record.get("verified_sha256"), record.get("expected_checksum")]
                if any(
                    expected is not None and expected != reader.stats.verified_sha256
                    for expected in expected_hashes
                ):
                    errors.append("verified source SHA-256 mismatch")
                    scan_complete = False
            except sqlite3.OperationalError as error:
                errors.append(f"SQLite staging limit reached or staging failed: {str(error)[:512]}")
                scan_complete = False
            except SourceReadError as error:
                if len(errors) < 20:
                    errors.append(f"source read failed: {str(error)[:512]}")
                scan_complete = False
            reports.append(
                {
                    "source_id": source.source_id,
                    "declared_month": source.declared_month,
                    "source_kind": source.source_kind,
                    "matched_records": matched,
                    "normalization_error_count": normalization_error_count,
                    "lines_seen": reader.stats.lines_seen,
                    "invalid_records": reader.stats.invalid_records,
                    "verified_sha256": reader.stats.verified_sha256,
                    "complete": reader.stats.complete
                    and reader.stats.verified_sha256 == record.get("verified_sha256")
                    and not errors,
                    "errors": errors,
                }
            )
        db.commit()
        out_path = output_dir / "threads.jsonl.zst"
        compressor = zstandard.ZstdCompressor()
        written = 0
        sqlite_bytes = db_path.stat().st_size
        remaining_staging_bytes = max_staging_bytes - sqlite_bytes
        output_byte_limit = max(
            0, remaining_staging_bytes - min(65_536, remaining_staging_bytes // 64)
        )
        thread_reports: list[dict[str, Any]] = []
        temp_descriptor, temp_name = tempfile.mkstemp(prefix=".context-", dir=output_dir)
        temp_out = Path(temp_name)
        try:
            os.fchmod(temp_descriptor, 0o600)
            with (
                os.fdopen(temp_descriptor, "wb") as target,
                compressor.stream_writer(target, closefd=False) as compressed,
            ):
                for thread, candidate in sorted(candidates.items()):
                    rows = db.execute(
                        "SELECT fullname,revision,payload FROM messages "
                        "WHERE thread=? ORDER BY CASE WHEN fullname=? THEN 0 ELSE 1 END, "
                        "fullname, revision",
                        (thread, thread),
                    ).fetchall()
                    retained = 0
                    published_names: set[str] = set()
                    published_parents: list[str | None] = []
                    for _, _, data in rows:
                        message = json.loads(data)
                        line = data + b"\n"
                        if written + len(line) > output_byte_limit:
                            omitted_by_thread[thread] += 1
                            _record_thread_issue(
                                thread_issues[thread],
                                "output_cap",
                                "output staging byte cap exceeded",
                            )
                            continue
                        compressed.write(line)
                        written += len(line)
                        retained += 1
                        published_names.add(message["fullname"])
                        published_parents.append(message["parent_fullname"])
                    missing_root = thread not in published_names
                    missing_parents = sorted(
                        {
                            parent
                            for parent in published_parents
                            if parent and parent != thread and parent not in published_names
                        }
                    )
                    missing_reply_targets = sorted(
                        set(candidate["reply_targets"]) - published_names
                    )
                    if missing_root:
                        _record_thread_issue(thread_issues[thread], "missing_root", thread)
                    if missing_parents:
                        _record_thread_issue(
                            thread_issues[thread], "missing_parent", ", ".join(missing_parents)
                        )
                    if missing_reply_targets:
                        _record_thread_issue(
                            thread_issues[thread],
                            "missing_reply_target",
                            ", ".join(missing_reply_targets),
                        )
                    issue_state = thread_issues[thread]
                    thread_error = bool(issue_state["counts"]) or bool(omitted_by_thread[thread])
                    thread_reports.append(
                        {
                            "thread_fullname": thread,
                            "matched_records": matched_by_thread[thread],
                            "retained_messages": retained,
                            "omitted_messages": omitted_by_thread[thread],
                            "root_missing": missing_root,
                            "missing_parents": missing_parents,
                            "reply_targets": candidate["reply_targets"],
                            "missing_reply_targets": missing_reply_targets,
                            "coverage_reasons": issue_state["examples"],
                            "coverage_reason_counts": issue_state["counts"],
                            "archive_content_notes": issue_state["archive_content_notes"],
                            "archive_content_counts": issue_state["archive_content_counts"],
                            "thread_context_complete_for_sources": (
                                scan_complete and not thread_error
                            ),
                            "coverage_status": "needs_context"
                            if (not scan_complete or thread_error)
                            else "complete_for_scanned_sources",
                        }
                    )
        except BaseException:
            temp_out.unlink(missing_ok=True)
            raise
        temp_out.replace(out_path)
        registry_hash = file_sha256(registry_path)
        manifest_out = {
            "kind": "engagement_thread_context_manifest",
            "schema_version": 1,
            "inputs": {
                "candidates_sha256": manifest.candidates_sha256,
                "shortlist_manifest_sha256": file_sha256(shortlist_manifest_path),
                "registry_sha256": registry_hash,
                "tombstone_sha256": actual_tombstone_hash,
                "tombstone_canonical_sha256": tombstones.digest,
            },
            "scan_complete": scan_complete,
            "scan_status": "complete" if scan_complete else "incomplete",
            "live_status": "unverified",
            "source_ids": [r["source_id"] for r in sources],
            "declared_months": sorted({str(r["declared_month"]) for r in sources}),
            "sources": reports,
            "threads": thread_reports,
            "scanned_record_count": scanned,
            "output_path": str(out_path),
            "output_sha256": file_sha256(out_path),
            "limits": {
                "max_messages_per_thread": max_messages_per_thread,
                "max_bytes_per_thread": max_bytes_per_thread,
                "max_staging_bytes": max_staging_bytes,
            },
        }
        manifest_path_out = output_dir / "context_manifest.json"
        manifest_descriptor = os.open(
            manifest_path_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        os.fchmod(manifest_descriptor, 0o600)
        with os.fdopen(manifest_descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
        return {
            "scan_complete": scan_complete,
            "thread_count": len(thread_reports),
            "output_path": str(out_path),
            "manifest_path": str(output_dir / "context_manifest.json"),
            "output_sha256": manifest_out["output_sha256"],
        }
    finally:
        db.close()
        db_path.unlink(missing_ok=True)
