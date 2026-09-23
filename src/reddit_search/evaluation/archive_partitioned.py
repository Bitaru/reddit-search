"""Bounded retained exact-match archive indexing for archive audits."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import time
import unicodedata
from collections.abc import Callable, Iterator
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
    check_rss_budget,
    process_rss_bytes,
)
from reddit_search.telemetry import (
    capture_stage_telemetry,
    telemetry_file_identity,
    write_stage_telemetry,
)

SCHEMA_VERSION = 3
PARTITION_COUNT = 64
ALGORITHM_VERSION = "archive-audit-exact-match-v1"


def _text(value: Any) -> str:
    return unicodedata.normalize("NFKC", value).casefold() if isinstance(value, str) else ""


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("message_fullname", row.get("fullname", ""))),
        str(row.get("source_revision_id", row.get("revision", ""))),
        str(row.get("thread_fullname", row.get("thread", ""))),
        _text(row.get("focus_text", row.get("text", ""))),
    )


def _key(value: tuple[str, str, str, str]) -> str:
    return "\x1f".join(value)


def _bucket(value: tuple[str, str, str, str]) -> int:
    return (
        int.from_bytes(hashlib.sha256(_key(value).encode()).digest()[:2], "big") % PARTITION_COUNT
    )


def _source_dir(root: Path, source_id: str) -> Path:
    return root / ("source-" + hashlib.sha256(source_id.encode()).hexdigest()[:24])


def _index_bytes(root: Path) -> int:
    """Count files once, even when accounting roots overlap."""
    return _dedup_bytes((root,))


def _dedup_bytes(roots: tuple[Path, ...]) -> int:
    seen: set[Path] = set()
    total = 0
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root.absolute()
        paths = [resolved] if resolved.is_file() else resolved.rglob("*")
        for path in paths:
            try:
                if path.is_file():
                    key = path.resolve()
                    if key not in seen:
                        seen.add(key)
                        total += path.stat().st_size
            except OSError:
                continue
    return total


def _check_index_budget(
    root: Path,
    max_bytes: int,
    minimum_free: int,
    extra_paths: tuple[Path, ...] = (),
    reserved_bytes: int = 0,
) -> None:
    roots = (root, *(path for path in extra_paths if path is not None))
    size = _dedup_bytes(roots) + reserved_bytes
    if size > max_bytes:
        raise BudgetError(f"archive index budget exceeded: {size} bytes, limit {max_bytes}")
    free = shutil.disk_usage(root).free
    if free < minimum_free + reserved_bytes:
        raise BudgetError(
            f"insufficient free disk: {free} bytes free, need the {minimum_free}-byte reserve"
        )


def _check_combined_budget(
    index_root: Path,
    output_root: Path,
    minimum_free: int,
    max_bytes: int,
) -> None:
    _check_index_budget(index_root, max_bytes, minimum_free, (output_root,))


def _partition_metadata(root: Path) -> dict[str, dict[str, int | str]]:
    metadata: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("bucket-*.sqlite")):
        metadata[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return metadata


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


def _pick(row: dict[str, Any], nested: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    for key in keys:
        if nested.get(key) is not None:
            return nested[key]
    return None


def _resolve_source_path(registry_path: Path, item: dict[str, Any]) -> Path:
    raw = item.get("source_path", item.get("path"))
    if not isinstance(raw, str) or not raw:
        raise ValueError("registered source has no source_path/path")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else registry_path.parent / path


def _canonical(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _write_jsonl(
    path: Path, rows: Iterator[dict[str, Any]], *, publish: bool = True
) -> tuple[str, int]:
    temp = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    try:
        with temp.open("wb") as stream:
            for row in rows:
                payload = _canonical(row)
                stream.write(payload)
                digest.update(payload)
                count += 1
        if publish:
            temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), count


def _load_corpus_identities(
    corpus_path: Path, snapshot_id: str | None
) -> tuple[dict[str, list[tuple[str, str, str, str]]], dict[tuple[str, str, str, str], int]]:
    corpus_ids: dict[str, list[tuple[str, str, str, str]]] = {}
    corpus_counts: dict[tuple[str, str, str, str], int] = {}
    db = sqlite3.connect(f"file:{corpus_path}?mode=ro", uri=True)
    try:
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(search_units)")}
        required = {"message_fullname", "source_revision_id", "thread_fullname", "focus_text"}
        if required - columns:
            raise ValueError("corpus lacks required search_units columns")
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
        for values in db.execute(query, params):
            row = dict(zip(selected, values, strict=True))
            ident = _identity(row)
            corpus_counts[ident] = corpus_counts.get(ident, 0) + 1
            for key in (row.get("unit_id"), row.get("candidate_id")):
                if key is not None:
                    corpus_ids.setdefault(str(key), []).append(ident)
    finally:
        db.close()
    return corpus_ids, corpus_counts


def _load_exposure_targets(
    exposure_paths: list[Path],
    corpus_ids: dict[str, list[tuple[str, str, str, str]]],
) -> tuple[
    set[tuple[str, str, str, str]],
    list[dict[str, Any]],
    list[tuple[Path, int, dict[str, Any], list[tuple[str, str, str, str]]] | dict[str, Any]],
]:
    targets: set[tuple[str, str, str, str]] = set()
    exposure_errors: list[dict[str, Any]] = []
    exposures: list[
        tuple[Path, int, dict[str, Any], list[tuple[str, str, str, str]]] | dict[str, Any]
    ] = []
    for path in sorted(exposure_paths, key=str):
        try:
            for line, row, error, raw_line in _json_rows(path):
                if error:
                    exposure_errors.append(
                        {
                            "path": str(path),
                            "line": line,
                            "status": "error",
                            "error": error,
                            "raw": raw_line,
                        }
                    )
                    exposures.append(exposure_errors[-1])
                    continue
                assert row is not None
                nested = (
                    row.get("source")
                    if isinstance(row.get("source"), dict)
                    else row.get("source_bundle")
                    if isinstance(row.get("source_bundle"), dict)
                    else {}
                )
                candidate = row.get("candidate_id") or row.get("unit_id")
                identities = (
                    list(corpus_ids.get(str(candidate), [])) if candidate is not None else []
                )
                if not identities and all(
                    isinstance(_pick(row, nested, *keys), str)
                    for keys in (
                        ("message_fullname", "fullname"),
                        ("source_revision_id", "revision"),
                        ("thread_fullname", "thread"),
                        ("focus_text", "text"),
                    )
                ):
                    identities = [
                        (
                            _pick(row, nested, "message_fullname", "fullname"),
                            _pick(row, nested, "source_revision_id", "revision"),
                            _pick(row, nested, "thread_fullname", "thread"),
                            _text(_pick(row, nested, "focus_text", "text")),
                        )
                    ]
                targets.update(identities)
                exposures.append((path, line, row, identities))
        except (OSError, UnicodeError, zstandard.ZstdError) as error:
            event = {"path": str(path), "line": None, "status": "error", "error": str(error)}
            exposure_errors.append(event)
            exposures.append(event)
    return targets, exposure_errors, exposures


def _scan_index_identities(
    index_directory: Path, completed: set[str]
) -> tuple[dict[tuple[str, str, str, str], list[dict[str, Any]]], dict[str, int]]:
    collision_matches: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    coverage_by_source: dict[str, int] = {sid: 0 for sid in completed}
    for sid in sorted(completed):
        for path in sorted(_source_dir(index_directory, sid).glob("bucket-*.sqlite")):
            connection = sqlite3.connect(path)
            try:
                for row in connection.execute(
                    "SELECT fullname,revision,thread,text,source_id,line FROM archive"
                ):
                    ident = tuple(row[:4])
                    collision_matches.setdefault(ident, []).append(
                        {"source_id": str(row[4]), "line": int(row[5])}
                    )
                    coverage_by_source[sid] += 1
            finally:
                connection.close()
    return collision_matches, coverage_by_source


def _open_partition_connections(
    index_directory: Path, completed: set[str]
) -> dict[tuple[str, int], sqlite3.Connection]:
    connections: dict[tuple[str, int], sqlite3.Connection] = {}
    for sid in sorted(completed):
        for path in sorted(_source_dir(index_directory, sid).glob("bucket-*.sqlite")):
            bucket = int(path.stem.removeprefix("bucket-"))
            connections[(sid, bucket)] = sqlite3.connect(path)
    return connections


def _matches_for(
    identities: list[tuple[str, str, str, str]],
    partition_connections: dict[tuple[str, int], sqlite3.Connection],
    completed_source_ids: set[str],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for ident in identities:
        bucket = _bucket(tuple(ident))
        for sid in sorted(completed_source_ids):
            connection = partition_connections.get((sid, bucket))
            if connection is None:
                continue
            for found in connection.execute(
                "SELECT source_id,line FROM archive WHERE fullname=? AND revision=? "
                "AND thread=? AND text=? ORDER BY source_id,line",
                ident,
            ):
                matches.append({"source_id": str(found[0]), "line": int(found[1])})
    return sorted(matches, key=lambda item: (item["source_id"], item["line"]))


def audit_registered_archives_partitioned(
    registry_path: Path,
    corpus_path: Path,
    exposure_paths: list[Path],
    output_directory: Path,
    *,
    max_records: int,
    index_directory: Path,
    snapshot_id: str | None = None,
    max_bytes: int | None = None,
    checkpoint_path: Path | None = None,
    progress_path: Path | None = None,
    max_index_bytes: int = 10 * 1024**3,
    max_process_rss_bytes: int = DEFAULT_MAX_PROCESS_RSS_BYTES,
    min_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES,
    clock: Callable[[], float] = time.monotonic,
    rss_provider: Callable[[], int] | None = None,
    disk_provider: Callable[[Path], int] | None = None,
) -> dict[str, Any]:
    started_at = clock()
    if max_records <= 0 or (max_bytes is not None and max_bytes <= 0):
        raise ValueError("audit limits must be positive")
    if max_index_bytes <= 0:
        raise ValueError("max_index_bytes must be positive")
    if checkpoint_path is None:
        raise ValueError("index_directory requires checkpoint_path")
    limits = ResourceLimits(
        minimum_free_disk_bytes=min_free_disk_bytes, max_process_rss_bytes=max_process_rss_bytes
    )
    limits.validate()
    output_directory.mkdir(parents=True, exist_ok=True)
    index_directory.mkdir(parents=True, exist_ok=True)
    registry = load_source_registry(registry_path)
    raw = registry.get("sources")
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError("source registry must contain only source objects")
    source_paths = {}
    for item in raw:
        sid = str(item.get("source_id", ""))
        path = str(_resolve_source_path(registry_path, item))
        prior = source_paths.setdefault(path, sid)
        if prior != sid:
            raise ValueError(f"source path is registered under multiple source IDs: {path}")
    source_ids = [str(item.get("source_id", "")) for item in raw]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source registry contains duplicate source IDs")
    sources = sorted(
        (dict(item) for item in raw),
        key=lambda item: (
            str(item.get("source_id", "")),
            str(item.get("source_path", item.get("path", ""))),
        ),
    )
    input_hashes = {
        "registry": {"path": str(registry_path), "sha256": file_sha256(registry_path)},
        "corpus": {"path": str(corpus_path), "sha256": file_sha256(corpus_path)},
        "exposure": [{"path": str(path), "sha256": file_sha256(path)} for path in exposure_paths],
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "snapshot_id": snapshot_id,
        "input_hashes": input_hashes,
        "index_directory": str(index_directory),
        "partition_count": PARTITION_COUNT,
        "budget": {
            "max_records": max_records,
            "max_bytes": max_bytes,
            "max_index_bytes": max_index_bytes,
            "max_process_rss_bytes": max_process_rss_bytes,
            "min_free_disk_bytes": min_free_disk_bytes,
        },
    }
    previous = load_json(checkpoint_path) if checkpoint_path.exists() else None
    if previous is not None:
        if previous.get("identity") != identity:
            raise ValueError("checkpoint identity or policy does not match")
        expected = previous.get("partition_metadata")
        if expected is not None and expected != _partition_metadata(index_directory):
            raise ValueError("checkpoint partition metadata does not match index")
    elif any(index_directory.iterdir()):
        raise ValueError("stale archive index directory exists without checkpoint")
    completed = set(previous.get("completed_source_ids", [])) if previous else set()
    old_stats = {item["source_id"]: item for item in (previous or {}).get("source_stats", [])}
    valid = set()
    for item in sources:
        sid = str(item.get("source_id", ""))
        if sid not in completed:
            break
        prior = old_stats.get(sid, {}).get("reader_stats", {})
        if file_sha256(_resolve_source_path(registry_path, item)) != prior.get("verified_sha256"):
            break
        if not _source_dir(index_directory, sid).exists():
            break
        valid.add(sid)
    completed = valid
    for item in sources:
        if str(item.get("source_id", "")) not in completed:
            shutil.rmtree(
                _source_dir(index_directory, str(item.get("source_id", ""))), ignore_errors=True
            )
    for path in sorted(index_directory.iterdir()):
        if path.is_dir() and (
            path.name.endswith(".staging")
            or path.name.startswith("source-")
            and path.name not in {_source_dir(index_directory, sid).name for sid in completed}
        ):
            shutil.rmtree(path, ignore_errors=True)
    source_stats = [old_stats[sid] for sid in sorted(completed) if sid in old_stats]
    records_used = sum(int(item["reader_stats"]["lines_seen"]) for item in source_stats)
    bytes_used = sum(int(item["reader_stats"]["compressed_bytes_read"]) for item in source_stats)
    corpus_ids, corpus_counts = _load_corpus_identities(corpus_path, snapshot_id)
    targets, exposure_errors, exposures = _load_exposure_targets(exposure_paths, corpus_ids)
    if exposures and not targets:
        raise ValueError(
            f"exposure target resolution produced no corpus identities for snapshot "
            f"{snapshot_id!r}; refusing archive scan"
        )

    started = time.monotonic()
    exhausted = False
    operational_error: str | None = None
    checkpoint_active: str | None = None

    def write_checkpoint(
        active: str | None,
        complete: bool = False,
        committed_records: int | None = None,
        committed_bytes: int | None = None,
    ) -> None:
        atomic_write_json(
            checkpoint_path,
            {
                "kind": "registered_archive_audit_checkpoint",
                "schema_version": SCHEMA_VERSION,
                "identity": identity,
                "partition_metadata": _partition_metadata(index_directory),
                "index_directory": str(index_directory),
                "completed_source_ids": sorted(completed),
                "source_stats": [item for item in source_stats if item["source_id"] in completed],
                "records_used": records_used if committed_records is None else committed_records,
                "bytes_used": bytes_used if committed_bytes is None else committed_bytes,
                "active_source": active,
                "complete": complete,
            },
        )
        if progress_path:
            atomic_write_json(
                progress_path,
                {
                    "kind": "registered_archive_audit_progress",
                    "current_source": active,
                    "completed_sources": len(completed),
                    "total_sources": len(sources),
                    "records": records_used,
                    "bytes": bytes_used,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

    for item in sources:
        sid = str(item.get("source_id", ""))
        if sid in completed:
            continue
        base_records, base_bytes = records_used, bytes_used
        spec = SourceSpec(
            sid,
            _resolve_source_path(registry_path, item),
            str(item.get("kind", item.get("source_kind", "submission"))),
            str(item.get("declared_month", "")),
            str(item.get("source_role", "discovery")),
            str(item.get("usage_scope", "synthetic")),
        )
        final_dir = _source_dir(index_directory, sid)
        staging = index_directory / (final_dir.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        connections: dict[int, sqlite3.Connection] = {}
        norm_errors: list[dict[str, Any]] = []
        reader = ArchiveReader()
        old_lines = old_bytes = scanned = 0
        read_error = None
        try:
            for envelope in reader.iter_records(spec):
                records_used += reader.stats.lines_seen - old_lines
                bytes_used += reader.stats.compressed_bytes_read - old_bytes
                old_lines, old_bytes = reader.stats.lines_seen, reader.stats.compressed_bytes_read
                if records_used > max_records or (max_bytes is not None and bytes_used > max_bytes):
                    exhausted = True
                    break
                try:
                    message = normalize_record(envelope)
                except NormalizationError as error:
                    norm_errors.append({"line": envelope.line_number, "error": str(error)})
                    continue
                ident = (
                    message.fullname,
                    message.source_revision_id,
                    message.thread_fullname,
                    _text(message.focus_text),
                )
                if ident not in targets:
                    continue
                bucket = _bucket(ident)
                connection = connections.get(bucket)
                if connection is None:
                    path = staging / f"bucket-{bucket:03d}.sqlite"
                    connection = sqlite3.connect(path)
                    connection.execute(
                        "CREATE TABLE archive("
                        "fullname TEXT,revision TEXT,thread TEXT,text TEXT,"
                        "source_id TEXT,line INTEGER)"
                    )
                    connection.execute(
                        "CREATE INDEX archive_identity ON archive(fullname,revision,thread,text)"
                    )
                    connections[bucket] = connection
                connection.execute(
                    "INSERT INTO archive VALUES (?,?,?,?,?,?)", (*ident, sid, envelope.line_number)
                )
                scanned += 1
                if scanned % 128 == 0:
                    for connection in connections.values():
                        connection.commit()
                    _check_index_budget(
                        index_directory,
                        max_index_bytes,
                        min_free_disk_bytes,
                        (output_directory, checkpoint_path, progress_path),
                    )
        except SourceReadError as error:
            read_error = str(error)
        records_used += reader.stats.lines_seen - old_lines
        bytes_used += reader.stats.compressed_bytes_read - old_bytes
        if records_used > max_records or (max_bytes is not None and bytes_used > max_bytes):
            exhausted = True
        for connection in connections.values():
            connection.commit()
            connection.close()
        expected = item.get("expected_checksum")
        if (
            read_error is None
            and not norm_errors
            and not exhausted
            and reader.stats.complete
            and isinstance(expected, str)
            and reader.stats.verified_sha256 != expected
        ):
            read_error = (
                "expected checksum mismatch: "
                f"expected {expected}, got {reader.stats.verified_sha256}"
            )
        complete_source = bool(
            reader.stats.complete and read_error is None and not norm_errors and not exhausted
        )
        report = {
            "source_id": sid,
            "path": str(spec.path),
            "records_scanned": scanned,
            "reader_stats": {
                "lines_seen": reader.stats.lines_seen,
                "valid_records": reader.stats.valid_records,
                "invalid_records": reader.stats.invalid_records,
                "invalid_line_numbers": reader.stats.invalid_line_numbers,
                "compressed_bytes_read": reader.stats.compressed_bytes_read,
                "decompressed_bytes_read": reader.stats.decompressed_bytes_read,
                "complete": complete_source,
                "verified_sha256": reader.stats.verified_sha256 if complete_source else None,
            },
            "normalization_errors": norm_errors,
            "read_error": read_error,
        }
        source_stats.append(report)
        try:
            check_rss_budget(limits=limits, stage="archive-audit", reason=f"source:{sid}")
            _check_index_budget(
                index_directory,
                max_index_bytes,
                min_free_disk_bytes,
                (output_directory, checkpoint_path, progress_path),
            )
        except BudgetError as error:
            operational_error = str(error)
            exhausted = True
            complete_source = False
            for connection in connections.values():
                connection.close()
        if complete_source:
            shutil.rmtree(final_dir, ignore_errors=True)
            staging.replace(final_dir)
            completed.add(sid)
            checkpoint_active = None
            write_checkpoint(None)
        else:
            shutil.rmtree(staging, ignore_errors=True)
            checkpoint_active = sid
            write_checkpoint(sid, committed_records=base_records, committed_bytes=base_bytes)
        if exhausted or not complete_source:
            break
    write_checkpoint(
        checkpoint_active,
        complete=not exhausted and not exposure_errors and len(completed) == len(sources),
        committed_records=records_used
        if checkpoint_active is None
        else sum(
            int(item["reader_stats"]["lines_seen"])
            for item in source_stats
            if item["source_id"] in completed
        ),
        committed_bytes=bytes_used
        if checkpoint_active is None
        else sum(
            int(item["reader_stats"]["compressed_bytes_read"])
            for item in source_stats
            if item["source_id"] in completed
        ),
    )
    collision_matches, coverage_by_source = _scan_index_identities(index_directory, completed)
    partition_connections = _open_partition_connections(index_directory, completed)

    match_count = unresolved_count = 0

    def close_partition_connections() -> None:
        for connection in partition_connections.values():
            connection.close()
        partition_connections.clear()

    def output_rows() -> Iterator[dict[str, Any]]:
        nonlocal match_count, unresolved_count
        for event in exposures:
            if isinstance(event, dict):
                yield event
                continue
            path, line, row, identities = event
            matches = _matches_for(identities, partition_connections, completed)
            if matches:
                match_count += 1
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

    output_paths = [
        output_directory / "archive_exposure_matches.jsonl",
        output_directory / "archive_index_coverage.jsonl",
        output_directory / "archive_identity_collisions.jsonl",
    ]
    try:
        exposure_hash, exposure_count = _write_jsonl(output_paths[0], output_rows(), publish=False)
        coverage = [
            {**item, "matched_records": coverage_by_source.get(item["source_id"], 0)}
            for item in source_stats
        ]
        coverage_hash, coverage_count = _write_jsonl(output_paths[1], iter(coverage), publish=False)
        collisions = (
            {
                "identity": list(ident),
                "archive_count": len(collision_matches[ident]),
                "corpus_count": corpus_counts.get(ident, 0),
            }
            for ident in sorted(collision_matches)
            if len(collision_matches[ident]) > 1 or corpus_counts.get(ident, 0) > 1
        )
        collision_hash, collision_count = _write_jsonl(
            output_paths[2], collisions, publish=False
        )
        reserved = sum(
            path.with_suffix(path.suffix + ".tmp").stat().st_size for path in output_paths
        )
        _check_index_budget(
            index_directory, max_index_bytes, min_free_disk_bytes,
            (output_directory, checkpoint_path, progress_path), reserved_bytes=reserved
        )
        for path in output_paths:
            path.with_suffix(path.suffix + ".tmp").replace(path)
    except BudgetError as error:
        close_partition_connections()
        operational_error = str(error)
        exhausted = True
        for path in output_paths:
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
        exposure_hash = coverage_hash = collision_hash = hashlib.sha256(b"").hexdigest()
        exposure_count = coverage_count = collision_count = 0
    except BaseException:
        close_partition_connections()
        for path in output_paths:
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
        raise
    close_partition_connections()
    complete = bool(
        not exhausted
        and not exposure_errors
        and len(completed) == len(sources)
        and all(item["reader_stats"]["complete"] for item in source_stats)
    )
    telemetry_path = output_directory / "stage_telemetry.json"
    telemetry = capture_stage_telemetry(
        stage="archive_audit",
        input_hashes=input_hashes,
        run_identity=json_sha256(identity),
        limits={
            "max_records": max_records,
            "max_bytes": max_bytes,
            "max_index_bytes": max_index_bytes,
            "max_process_rss_bytes": max_process_rss_bytes,
            "min_free_disk_bytes": min_free_disk_bytes,
        },
        path=output_directory,
        clock=clock,
        started_at=started_at,
        rss_provider=rss_provider or process_rss_bytes,
        disk_provider=disk_provider or (lambda path: shutil.disk_usage(path).free),
        cache_provider=lambda: "not_used",
        cleanup_provider=lambda: "not_applicable",
    )
    if not complete:
        telemetry["status"] = "incomplete"
        telemetry["errors"].append(
            {"field": "archive_audit", "error": operational_error or "audit incomplete"}
        )
    write_stage_telemetry(telemetry_path, telemetry)
    telemetry_identity = telemetry_file_identity(telemetry_path)
    telemetry_identity["status"] = telemetry["status"]
    manifest = {
        "kind": "registered_archive_audit",
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "input_hashes": input_hashes,
        "operational": {
            "telemetry": telemetry_identity,
            "index_directory": str(index_directory),
            "checkpoint_path": str(checkpoint_path),
            "progress_path": str(progress_path) if progress_path else None,
            "resume": "source-boundary",
            "mode": "retained-exact-match",
            "partition_count": PARTITION_COUNT,
        },
        "operational_error": operational_error,
        "complete": complete,
        "snapshot_id": snapshot_id,
        "optional_inputs_unused": [],
        "sources": source_stats,
        "counts": {
            "matched": match_count,
            "unresolved": unresolved_count,
            "collisions": collision_count,
            "exposure_errors": len(exposure_errors),
        },
        "budget": {
            "max_records": max_records,
            "records_used": records_used,
            "max_bytes": max_bytes,
            "bytes_used": bytes_used,
            "max_index_bytes": max_index_bytes,
            "index_bytes": _index_bytes(index_directory),
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
            "retained_exact_match_only": True,
            "partition_count": PARTITION_COUNT,
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
    atomic_write_json(output_directory / "archive_audit_manifest.json", manifest)
    return manifest
