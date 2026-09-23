"""Read-only bounded audit of registered archives against retained exposures."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import tempfile
import time
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import zstandard

from reddit_search.ingest.normalize import NormalizationError, normalize_record
from reddit_search.ingest.reader import ArchiveReader, SourceReadError, SourceSpec
from reddit_search.ingest.sources import load_source_registry
from reddit_search.ingest.state import atomic_write_json, file_sha256, json_sha256, load_json
from reddit_search.resources import (
    DEFAULT_MAX_PROCESS_RSS_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    BudgetError,
    ResourceLimits,
    check_output_budget,
    check_rss_budget,
    process_rss_bytes,
)
from reddit_search.telemetry import capture_stage_telemetry, write_stage_telemetry


def _pick_identity(row: dict[str, Any], nested: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    for key in keys:
        if nested.get(key) is not None:
            return nested[key]
    return None


_SCHEMA_VERSION = 2
_CHECKPOINT_KIND = "registered_archive_audit_checkpoint"


def _identity_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", value).casefold() if isinstance(value, str) else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(131072), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_rows(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None, str]]:
    raw = path.open("rb")
    if path.suffix == ".zst":
        stream = zstandard.ZstdDecompressor().stream_reader(raw)
        handle = io.TextIOWrapper(stream, encoding="utf-8")
    else:
        handle = io.TextIOWrapper(raw, encoding="utf-8")
    with handle:
        for number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not a JSON object")
            except (json.JSONDecodeError, ValueError) as error:
                yield number, None, str(error), line.rstrip("\n")
            else:
                yield number, value, None, line.rstrip("\n")


def _canonical(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> tuple[str, int]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("wb") as stream:
        for row in rows:
            payload = _canonical(row)
            stream.write(payload)
            digest.update(payload)
            count += 1
    temporary.replace(path)
    return digest.hexdigest(), count


def _source_path(registry_path: Path, item: dict[str, Any]) -> Path:
    raw = item.get("source_path", item.get("path"))
    if not isinstance(raw, str) or not raw:
        raise ValueError("registered source has no source_path/path")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else registry_path.parent / path


def _source_spec(registry_path: Path, item: dict[str, Any]) -> SourceSpec:
    kind = item.get("kind", item.get("source_kind"))
    if kind not in {"submission", "comment"}:
        raise ValueError(f"invalid source kind: {kind!r}")
    return SourceSpec(
        str(item["source_id"]),
        _source_path(registry_path, item),
        kind,
        str(item.get("declared_month", "")),
        str(item.get("source_role", "discovery")),
        str(item.get("usage_scope", "synthetic")),
    )


def audit_registered_archives(
    registry_path: Path,
    corpus_path: Path,
    exposure_paths: list[Path],
    output_directory: Path,
    *,
    max_records: int,
    snapshot_id: str | None = None,
    max_bytes: int | None = None,
    duplicate_links_path: Path | None = None,
    candidate_splits_path: Path | None = None,
    checkpoint_path: Path | None = None,
    index_path: Path | None = None,
    progress_path: Path | None = None,
    max_process_rss_bytes: int = DEFAULT_MAX_PROCESS_RSS_BYTES,
    min_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES,
    disk_provider: Callable[[Path], int] | None = None,
    rss_provider: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Audit registered archives without mutating any input.

    Checkpoints are identity-bound and written only after a source completes.
    The bounded API remains unchanged unless operational controls are supplied.
    """
    if index_path is not None and checkpoint_path is None:
        raise ValueError("index_path requires checkpoint_path")
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    limits = ResourceLimits(
        minimum_free_disk_bytes=min_free_disk_bytes,
        max_process_rss_bytes=max_process_rss_bytes,
    )
    limits.validate()
    started = time.monotonic()
    operational_error: str | None = None
    registry = load_source_registry(registry_path)
    raw_sources = registry.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("source registry must contain a sources list")
    if any(not isinstance(item, dict) for item in raw_sources):
        raise ValueError("source registry sources must contain only objects")
    sources = sorted(
        (dict(item) for item in raw_sources),
        key=lambda item: (
            str(item.get("source_id", "")),
            str(item.get("source_path", item.get("path", ""))),
        ),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    source_stats: list[dict[str, Any]] = []
    records_used = bytes_used = 0
    exhausted = False
    checkpoint_active_source: str | None = None
    input_paths = [registry_path, corpus_path, *exposure_paths]
    if duplicate_links_path is not None:
        input_paths.append(duplicate_links_path)
    if candidate_splits_path is not None:
        input_paths.append(candidate_splits_path)
    input_hashes = {
        "registry": {"path": str(registry_path), "sha256": file_sha256(registry_path)},
        "corpus": {"path": str(corpus_path), "sha256": file_sha256(corpus_path)},
        "exposure": [{"path": str(path), "sha256": file_sha256(path)} for path in exposure_paths],
        "duplicate_links": (
            {"path": str(duplicate_links_path), "sha256": file_sha256(duplicate_links_path)}
            if duplicate_links_path
            else None
        ),
        "candidate_splits": (
            {"path": str(candidate_splits_path), "sha256": file_sha256(candidate_splits_path)}
            if candidate_splits_path
            else None
        ),
    }
    identity = {
        "schema_version": _SCHEMA_VERSION,
        "algorithm_version": "archive-audit-v1",
        "snapshot_id": snapshot_id,
        "input_hashes": input_hashes,
        "budget": {
            "max_records": max_records,
            "max_bytes": max_bytes,
            "max_process_rss_bytes": max_process_rss_bytes,
            "min_free_disk_bytes": min_free_disk_bytes,
        },
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "progress_path": str(progress_path) if progress_path else None,
    }
    index_path = index_path or (
        checkpoint_path.with_name(checkpoint_path.name + ".sqlite") if checkpoint_path else None
    )
    if index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] | None = None
    completed_ids: set[str] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        previous = load_json(checkpoint_path)
        if (
            previous.get("kind") != _CHECKPOINT_KIND
            or previous.get("schema_version") != _SCHEMA_VERSION
        ):
            raise ValueError("invalid archive audit checkpoint")
        if previous.get("identity") != identity:
            raise ValueError("checkpoint identity does not match inputs, snapshot, or policy")
        if previous.get("index_path") != str(index_path):
            raise ValueError("checkpoint index path does not match checkpoint")
        completed_ids = set(
            previous.get("completed_source_ids", previous.get("completed_sources", []))
        )
        if not all(isinstance(item, str) for item in completed_ids):
            raise ValueError("checkpoint has malformed completed source IDs")
        source_stats_by_id = {
            item["source_id"]: item
            for item in previous.get("source_stats", [])
            if isinstance(item, dict) and isinstance(item.get("source_id"), str)
        }
        valid_completed_ids: set[str] = set()
        for item in sources:
            source_id = str(item.get("source_id", ""))
            if source_id not in completed_ids:
                break
            prior_stats = source_stats_by_id.get(source_id)
            prior_reader_stats = (
                prior_stats.get("reader_stats", {}) if prior_stats is not None else {}
            )
            prior_fingerprint = (
                prior_reader_stats.get("verified_sha256")
                if isinstance(prior_reader_stats, dict)
                else None
            )
            try:
                current_fingerprint = _sha256(_source_path(registry_path, item))
            except OSError:
                break
            if not isinstance(prior_fingerprint, str) or current_fingerprint != prior_fingerprint:
                break
            valid_completed_ids.add(source_id)
        completed_ids = valid_completed_ids
        source_stats = [
            source_stats_by_id[source_id]
            for source_id in sorted(completed_ids)
            if source_id in source_stats_by_id
        ]
        records_used = sum(
            int(item["reader_stats"]["lines_seen"])
            for item in source_stats
        )
        bytes_used = sum(
            int(item["reader_stats"]["compressed_bytes_read"])
            for item in source_stats
        )
    elif index_path is not None and index_path.exists():
        raise ValueError("stale archive audit index exists without checkpoint")
    if previous is not None and (index_path is None or not index_path.exists()):
        raise ValueError("checkpoint index is missing")

    def progress(current: str | None, completed: int) -> None:
        if progress_path:
            atomic_write_json(
                progress_path,
                {
                    "kind": "registered_archive_audit_progress",
                    "current_source": current,
                    "completed_sources": completed,
                    "total_sources": len(sources),
                    "records": records_used,
                    "bytes": bytes_used,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

    def checkpoint(
        current: str | None,
        complete: bool = False,
        *,
        committed_records: int | None = None,
        committed_bytes: int | None = None,
    ) -> None:
        if checkpoint_path:
            committed_stats = [
                item for item in source_stats if item["source_id"] in completed_ids
            ]
            atomic_write_json(
                checkpoint_path,
                {
                    "kind": _CHECKPOINT_KIND,
                    "schema_version": _SCHEMA_VERSION,
                    "identity": identity,
                    "index_path": str(index_path),
                    "completed_source_ids": sorted(completed_ids),
                    "source_stats": committed_stats,
                    "records_used": records_used
                    if committed_records is None
                    else committed_records,
                    "bytes_used": bytes_used if committed_bytes is None else committed_bytes,
                    "active_source": current,
                    "complete": complete,
                },
            )
        progress(current, len(completed_ids))

    db_temp = (
        tempfile.TemporaryDirectory(prefix="archive-audit-")
        if index_path is None
        else nullcontext()
    )
    db = sqlite3.connect(str(index_path or (Path(db_temp.name) / "audit.sqlite")))
    if previous is not None:
        db.execute(
            "DELETE FROM archive"
            if not completed_ids
            else "DELETE FROM archive WHERE source_id NOT IN ({})".format(
                ",".join("?" * len(completed_ids))
            ),
            tuple(completed_ids),
        )
        db.commit()
    else:
        db.executescript(
            "CREATE TABLE corpus(fullname TEXT,revision TEXT,thread TEXT,text TEXT,"
            "unit_id TEXT,candidate_id TEXT); CREATE INDEX corpus_identity ON corpus("
            "fullname,revision,thread,text); CREATE INDEX corpus_ids ON "
            "corpus(unit_id,candidate_id);"
            " CREATE TABLE archive(fullname TEXT,revision TEXT,thread TEXT,text TEXT,"
            "source_id TEXT,line INTEGER); CREATE INDEX archive_identity ON archive("
            "fullname,revision,thread,text);"
        )
    if previous is not None:
        # Persistent index is the source of truth on resume.
        pass
    else:
        corpus = sqlite3.connect(f"file:{corpus_path}?mode=ro", uri=True)
        try:
            columns = {str(row[1]) for row in corpus.execute("PRAGMA table_info(search_units)")}
            required = {"message_fullname", "source_revision_id", "thread_fullname", "focus_text"}
            if missing := required - columns:
                raise ValueError(f"corpus lacks required search_units columns: {sorted(missing)}")
            selected = [
                name
                for name in (
                    "message_fullname",
                    "source_revision_id",
                    "thread_fullname",
                    "focus_text",
                    "unit_id",
                    "candidate_id",
                )
                if name in columns
            ]
            query = f"SELECT {','.join(selected)} FROM search_units"
            params: tuple[Any, ...] = ()
            if snapshot_id is not None:
                if "snapshot_id" not in columns:
                    raise ValueError("corpus lacks snapshot_id for requested snapshot filter")
                query += " WHERE snapshot_id = ?"
                params = (snapshot_id,)
            for values in corpus.execute(query, params):
                data = dict(zip(selected, values, strict=True))
                db.execute(
                    "INSERT INTO corpus VALUES (?,?,?,?,?,?)",
                    (
                        data["message_fullname"],
                        data["source_revision_id"],
                        data["thread_fullname"],
                        _identity_text(data["focus_text"]),
                        data.get("unit_id"),
                        data.get("candidate_id"),
                    ),
                )
        finally:
            corpus.close()
        db.commit()
    # Resolve the requested exposure identities against the selected corpus before
    # touching any archive.  A nonempty exposure set with no corpus identities is
    # almost certainly a snapshot mismatch and must not masquerade as a complete
    # zero-match audit.
    exposure_row_count = 0
    target_identity_count = 0
    for path in sorted(exposure_paths, key=str):
        for _line, row, error, _raw in _json_rows(path):
            if error or row is None:
                continue
            exposure_row_count += 1
            nested = (
                row.get("source")
                if isinstance(row.get("source"), dict)
                else row.get("source_bundle")
                if isinstance(row.get("source_bundle"), dict)
                else {}
            )
            candidate = row.get("candidate_id") or row.get("unit_id")
            identities = (
                db.execute(
                    "SELECT fullname,revision,thread,text FROM corpus "
                    "WHERE unit_id=? OR candidate_id=?",
                    (str(candidate), str(candidate)),
                ).fetchall()
                if candidate is not None
                else []
            )
            if not identities and all(
                isinstance(_pick_identity(row, nested, *key), str)
                for key in (
                    ("message_fullname", "fullname"),
                    ("source_revision_id", "revision"),
                    ("thread_fullname", "thread"),
                    ("focus_text", "text"),
                )
            ):
                identities = [(
                    _pick_identity(row, nested, "message_fullname", "fullname"),
                    _pick_identity(row, nested, "source_revision_id", "revision"),
                    _pick_identity(row, nested, "thread_fullname", "thread"),
                    _identity_text(_pick_identity(row, nested, "focus_text", "text")),
                )]
            target_identity_count += len(identities)
    if exposure_row_count and not target_identity_count:
        raise ValueError(
            f"exposure target resolution produced no corpus identities for snapshot "
            f"{snapshot_id!r}; refusing archive scan"
        )

    for item in sources:
        base_records_used, base_bytes_used = records_used, bytes_used
        spec = _source_spec(registry_path, item)
        if spec.source_id in completed_ids:
            continue
        db.execute("DELETE FROM archive WHERE source_id=?", (spec.source_id,))
        db.commit()
        reader = ArchiveReader()
        norm_errors: list[dict[str, Any]] = []
        read_error = None
        scanned = 0
        old_lines = old_bytes = 0
        try:
            for envelope in reader.iter_records(spec):
                line_delta = reader.stats.lines_seen - old_lines
                byte_delta = reader.stats.compressed_bytes_read - old_bytes
                old_lines, old_bytes = (
                    reader.stats.lines_seen,
                    reader.stats.compressed_bytes_read,
                )
                records_used += line_delta
                bytes_used += byte_delta
                if records_used > max_records or max_bytes is not None and bytes_used > max_bytes:
                    exhausted = True
                    break
                try:
                    message = normalize_record(envelope)
                except NormalizationError as error:
                    norm_errors.append({"line": envelope.line_number, "error": str(error)})
                    continue
                scanned += 1
                db.execute(
                    "INSERT INTO archive VALUES (?,?,?,?,?,?)",
                    (
                        message.fullname,
                        message.source_revision_id,
                        message.thread_fullname,
                        _identity_text(message.focus_text),
                        spec.source_id,
                        envelope.line_number,
                    ),
                )
        except SourceReadError as error:
            read_error = str(error)
        tail_line_delta = reader.stats.lines_seen - old_lines
        tail_byte_delta = reader.stats.compressed_bytes_read - old_bytes
        records_used += tail_line_delta
        bytes_used += tail_byte_delta
        if records_used > max_records or max_bytes is not None and bytes_used > max_bytes:
            exhausted = True
        expected_checksum = item.get("expected_checksum")
        if (
            read_error is None
            and not norm_errors
            and not exhausted
            and reader.stats.complete
            and isinstance(expected_checksum, str)
            and reader.stats.verified_sha256 != expected_checksum
        ):
            read_error = (
                f"expected checksum mismatch: expected {expected_checksum}, "
                f"got {reader.stats.verified_sha256}"
            )
            db.execute("DELETE FROM archive WHERE source_id=?", (spec.source_id,))
            db.commit()
        source_stats.append(
            {
                "source_id": spec.source_id,
                "path": str(spec.path),
                "records_scanned": scanned,
                "read_error": read_error,
                "normalization_errors": norm_errors,
                "reader_stats": {
                    "lines_seen": reader.stats.lines_seen,
                    "valid_records": reader.stats.valid_records,
                    "invalid_records": reader.stats.invalid_records,
                    "invalid_line_numbers": reader.stats.invalid_line_numbers,
                    "compressed_bytes_read": reader.stats.compressed_bytes_read,
                    "decompressed_bytes_read": reader.stats.decompressed_bytes_read,
                    "complete": bool(
                        reader.stats.complete
                        and read_error is None
                        and not norm_errors
                        and not exhausted
                    ),
                    "verified_sha256": reader.stats.verified_sha256
                    if reader.stats.complete
                    and read_error is None
                    and not norm_errors
                    and not exhausted
                    else None,
                },
            }
        )
        try:
            check_rss_budget(
                limits=limits,
                sampler=rss_provider or process_rss_bytes,
                stage="archive-audit",
                reason=f"source:{spec.source_id}",
            )
            check_output_budget(output_directory, estimated_bytes=bytes_used, limits=limits)
        except BudgetError as error:
            operational_error = str(error)
            exhausted = True
            db.execute("DELETE FROM archive WHERE source_id=?", (spec.source_id,))
            db.commit()
            checkpoint_active_source = spec.source_id
            checkpoint(
                spec.source_id,
                committed_records=base_records_used,
                committed_bytes=base_bytes_used,
            )
            break
        complete_source = bool(source_stats[-1]["reader_stats"]["complete"])
        if complete_source:
            completed_ids.add(spec.source_id)
            checkpoint_active_source = None
            db.commit()
            checkpoint(None)
        else:
            checkpoint_active_source = spec.source_id
            if checkpoint_path:
                checkpoint(
                    spec.source_id,
                    committed_records=base_records_used,
                    committed_bytes=base_bytes_used,
                )
            db.commit()
        progress(spec.source_id, len(completed_ids))
        if exhausted or not complete_source:
            break
    db.commit()
    if checkpoint_path:
        checkpoint(
            checkpoint_active_source,
            complete=not exhausted and len(completed_ids) == len(sources),
            committed_records=(
                records_used if checkpoint_active_source is None else base_records_used
            ),
            committed_bytes=bytes_used if checkpoint_active_source is None else base_bytes_used,
        )

    exposure_errors: list[dict[str, Any]] = []
    matched_count = 0
    unresolved_count = 0

    def exposure_rows() -> Iterator[dict[str, Any]]:
        nonlocal matched_count, unresolved_count
        for path in sorted(exposure_paths, key=str):
            try:
                for line, row, error, raw in _json_rows(path):
                    if error:
                        result = {
                            "path": str(path),
                            "line": line,
                            "status": "error",
                            "error": error,
                            "raw": raw,
                        }
                        exposure_errors.append(result)
                        yield result
                        continue
                    assert row is not None
                    nested = (
                        row.get("source")
                        if isinstance(row.get("source"), dict)
                        else row.get("source_bundle")
                        if isinstance(row.get("source_bundle"), dict)
                        else {}
                    )

                    identities = []
                    candidate = row.get("candidate_id") or row.get("unit_id")
                    if candidate is not None:
                        identities = db.execute(
                            "SELECT fullname,revision,thread,text FROM corpus WHERE unit_id=? OR candidate_id=? ORDER BY fullname,revision,thread,text",  # noqa: E501
                            (str(candidate), str(candidate)),
                        ).fetchall()
                    if not identities and all(
                        isinstance(_pick_identity(row, nested, *key), str)
                        for key in (
                            ("message_fullname", "fullname"),
                            ("source_revision_id", "revision"),
                            ("thread_fullname", "thread"),
                            ("focus_text", "text"),
                        )
                    ):
                        identities = [
                            (
                                _pick_identity(row, nested, "message_fullname", "fullname"),
                                _pick_identity(row, nested, "source_revision_id", "revision"),
                                _pick_identity(row, nested, "thread_fullname", "thread"),
                                _identity_text(_pick_identity(row, nested, "focus_text", "text")),
                            )
                        ]
                    matches = [
                        dict(source_id=str(found[0]), line=int(found[1]))
                        for identity in identities
                        for found in db.execute(
                            "SELECT source_id,line FROM archive WHERE fullname=? AND revision=? AND thread=? AND text=? ORDER BY source_id,line",  # noqa: E501
                            identity,
                        )
                    ]
                    if matches:
                        matched_count += 1
                    else:
                        unresolved_count += 1
                    result = dict(row)
                    result.update(
                        {
                            "path": str(path),
                            "line": line,
                            "status": "matched" if matches else "unresolved",
                            "corpus_matches": len(identities),
                            "archive_matches": matches,
                        }
                    )
                    yield result
            except (OSError, UnicodeError, zstandard.ZstdError) as error:
                result = {
                    "path": str(path),
                    "line": None,
                    "status": "error",
                    "error": str(error),
                }
                exposure_errors.append(result)
                yield result

    exposure_hash, exposure_count = _write_jsonl(
        output_directory / "archive_exposure_matches.jsonl", exposure_rows()
    )
    coverage = []
    for item in source_stats:
        matched = db.execute(
            "SELECT COUNT(*) FROM archive WHERE source_id=? AND EXISTS "
            "(SELECT 1 FROM corpus WHERE corpus.fullname=archive.fullname "
            "AND corpus.revision=archive.revision AND corpus.thread=archive.thread "
            "AND corpus.text=archive.text)",
            (item["source_id"],),
        ).fetchone()[0]
        coverage.append({**item, "matched_records": int(matched)})
    coverage_hash, coverage_count = _write_jsonl(
        output_directory / "archive_index_coverage.jsonl", iter(coverage)
    )
    collision_sql = (
        "SELECT fullname,revision,thread,text,COUNT(*) archive_count,"
        "(SELECT COUNT(*) FROM corpus c WHERE c.fullname=a.fullname AND "
        "c.revision=a.revision AND c.thread=a.thread AND c.text=a.text) "
        "corpus_count FROM archive a GROUP BY fullname,revision,thread,text "
        "HAVING COUNT(*)>1 OR corpus_count>1 UNION SELECT fullname,revision,"
        "thread,text,(SELECT COUNT(*) FROM archive a WHERE a.fullname=c.fullname "
        "AND a.revision=c.revision AND a.thread=c.thread AND a.text=c.text),"
        "COUNT(*) FROM corpus c GROUP BY fullname,revision,thread,text "
        "HAVING COUNT(*)>1 ORDER BY fullname,revision,thread,text"
    )
    collision_hash, collision_count = _write_jsonl(
        output_directory / "archive_identity_collisions.jsonl",
        (
            {
                "identity": list(row[:4]),
                "archive_count": int(row[4]),
                "corpus_count": int(row[5]),
            }
            for row in db.execute(collision_sql)
        ),
    )
    db.close()
    if index_path is None:
        db_temp.cleanup()
    unused = [
        {"path": str(path), "kind": label, "reason": "provenance-only; not used"}
        for path, label in (
            (duplicate_links_path, "duplicate_links"),
            (candidate_splits_path, "candidate_splits"),
        )
        if path is not None
    ]
    complete = bool(
        not exhausted
        and not exposure_errors
        and len(completed_ids) == len(sources)
        and all(item["reader_stats"]["complete"] for item in source_stats)
    )
    telemetry_path = output_directory / "stage_telemetry.json"
    clock_calls = iter((started, time.monotonic()))
    telemetry = capture_stage_telemetry(
        stage="archive_audit",
        input_hashes=input_hashes,
        run_identity=json_sha256(identity),
        limits={
            "max_records": max_records,
            "max_bytes": max_bytes,
            "max_process_rss_bytes": max_process_rss_bytes,
            "min_free_disk_bytes": min_free_disk_bytes,
        },
        path=output_directory,
        clock=lambda: next(clock_calls),
        rss_provider=rss_provider or process_rss_bytes,
        disk_provider=disk_provider or (lambda path: shutil.disk_usage(path).free),
        cache_provider=lambda: "not_used",
        cleanup_provider=lambda: "complete" if index_path is None else "not_applicable",
    )
    if not complete:
        telemetry["status"] = "incomplete"
        telemetry["errors"].append(
            {"field": "archive_audit", "error": operational_error or "audit incomplete"}
        )
    write_stage_telemetry(telemetry_path, telemetry)
    telemetry_identity = {
        "path": str(telemetry_path),
        "sha256": file_sha256(telemetry_path),
        "status": telemetry["status"],
    }
    input_paths = [registry_path, corpus_path, *exposure_paths]
    if duplicate_links_path is not None:
        input_paths.append(duplicate_links_path)
    manifest = {
        "kind": "registered_archive_audit",
        "schema_version": _SCHEMA_VERSION,
        "identity": identity,
        "input_hashes": input_hashes,
        "operational": {
            "index_path": str(index_path) if index_path else None,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "progress_path": str(progress_path) if progress_path else None,
            "resume": "source-boundary" if checkpoint_path else "disabled",
            "telemetry": telemetry_identity,
        },
        "operational_error": operational_error,
        "complete": complete,
        "snapshot_id": snapshot_id,
        "optional_inputs_unused": unused,
        "sources": source_stats,
        "counts": {
            "matched": matched_count,
            "unresolved": unresolved_count,
            "collisions": collision_count,
            "exposure_errors": len(exposure_errors),
        },
        "budget": {
            "max_records": max_records,
            "records_used": records_used,
            "max_bytes": max_bytes,
            "bytes_used": bytes_used,
            "exhausted": exhausted,
        },
        "algorithm": {
            "identity": [
                "fullname",
                "source_revision_id",
                "thread_fullname",
                "NFKC-casefold focus_text",
            ],
            "near_text": False,
        },
        "outputs": {
            "archive_exposure_matches.jsonl": {"sha256": exposure_hash, "rows": exposure_count},
            "archive_index_coverage.jsonl": {"sha256": coverage_hash, "rows": coverage_count},
            "archive_identity_collisions.jsonl": {
                "sha256": collision_hash,
                "rows": collision_count,
            },
        },
    }
    temporary = output_directory / "archive_audit_manifest.json.tmp"
    if progress_path:
        atomic_write_json(
            progress_path,
            {
                "kind": "registered_archive_audit_progress",
                "current_source": checkpoint_active_source,
                "completed_sources": len(completed_ids),
                "total_sources": len(sources),
                "records": records_used,
                "bytes": bytes_used,
                "complete": manifest["complete"],
            },
        )
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_directory / "archive_audit_manifest.json")
    return manifest
