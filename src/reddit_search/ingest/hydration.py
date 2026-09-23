"""Bounded second-pass thread hydration over authorized local sources (execution step 3).

Rereads every registered discovery/context source once, retaining every message
that belongs to a selected thread under an explicit storage budget. The output
separates three roles:

- selected messages (already in the discovery shard; re-emitted unchanged),
- context messages (same thread, not selected; hydration-only context),
- rejected controls (did not match any discovery rule; control-only sample).

Context and control retention both keep the lowest stable-rank N candidates
(deterministic, unbiased by archive order). Byte-budget exhaustion or any read
failure marks the run incomplete: shards are withheld and no consumer may treat
the output as a complete corpus.
"""

from __future__ import annotations

import heapq
import json
import logging
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any

from reddit_search.resources import (
    DEFAULT_MAX_PROCESS_RSS_BYTES,
    DEFAULT_MAX_STAGING_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    MEMORY_POLICY_VERSION,
    BudgetError,
    ResourceLimits,
    check_output_budget,
    check_rss_budget,
    process_rss_bytes,
)
from reddit_search.telemetry import (
    capture_stage_telemetry,
    load_stage_telemetry,
    telemetry_file_identity,
    write_stage_telemetry,
)

from .invalidation import load_tombstone_ledger, tombstone_identity
from .normalize import NormalizationError, NormalizedMessage, normalize_record
from .reader import ArchiveReader, SourceReadError, SourceSpec
from .shard import (
    parse_selected_message,
    read_selected_messages,
    serialize_message,
    stable_rank,
    write_compressed_jsonl,
)
from .sources import (
    load_source_registry,
    persist_source_registry,
    record_validation_failure,
)
from .state import atomic_write_json, file_sha256, json_sha256, load_json

logger = logging.getLogger(__name__)

_PROGRESS_INTERVAL = 10_000_000


@dataclass(frozen=True, slots=True)
class HydrationLimits:
    """Explicit resource and coverage bounds for one hydration run."""

    max_context_messages: int
    seed: int
    control_target: int = 0
    checkpoint_interval_records: int = 5_000_000
    max_staging_bytes: int = DEFAULT_MAX_STAGING_BYTES
    minimum_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES
    max_process_rss_bytes: int = DEFAULT_MAX_PROCESS_RSS_BYTES

def freeze_thread_ids(selection_path: Path) -> dict[str, int]:
    """Read the discovery shard once and freeze its selected thread IDs."""
    thread_ids: set[str] = set()
    selected_count = 0
    for message in read_selected_messages(selection_path):
        thread_ids.add(message.thread_fullname)
        selected_count += 1
    return {"selected_count": selected_count, "thread_count": len(thread_ids)}


class _RankRetainer:
    """Keep the lowest stable-rank N items seen, in bounded memory."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.seed = seed
        self.items: dict[str, NormalizedMessage] = {}
        self.worst_heap: list[tuple[int, str]] = []  # (-rank, fullname); root = worst
        self.candidate_count = 0

    def offer(self, message: NormalizedMessage) -> None:
        self.candidate_count += 1
        if message.fullname in self.items:
            return
        rank = stable_rank(message.fullname, self.seed)
        if len(self.items) < self.capacity:
            self.items[message.fullname] = message
            heapq.heappush(self.worst_heap, (-rank, message.fullname))
            return
        if rank >= -self.worst_heap[0][0]:
            return
        _, evicted_fullname = heapq.heappushpop(self.worst_heap, (-rank, message.fullname))
        del self.items[evicted_fullname]
        self.items[message.fullname] = message

    def ordered(self) -> list[NormalizedMessage]:
        """Retained messages ordered by ascending rank, fullname as tiebreak."""
        return [
            self.items[fullname]
            for _, fullname in sorted(
                (-negated_rank, fullname) for negated_rank, fullname in self.worst_heap
            )
        ]

    def serialized_bytes(self) -> int:
        return sum(
            len(json.dumps(serialize_message(message), ensure_ascii=False, sort_keys=True).encode())
            + 1
            for message in self.items.values()
        )


_HYDRATION_STATE_SCHEMA_VERSION = 4


def _hydration_identity(
    registry_path: Path,
    registry: dict[str, Any],
    selection_path: Path,
    limits: HydrationLimits,
    max_records: int | None,
) -> dict[str, Any]:
    source_records: list[dict[str, Any]] = []
    for record in registry.get("sources", []):
        source_path = Path(str(record["source_path"]))
        try:
            stat = source_path.stat()
            size_bytes = stat.st_size
            modified_ns = stat.st_mtime_ns
        except OSError:
            size_bytes = None
            modified_ns = None
        source_records.append(
            {
                "source_id": record.get("source_id"),
                "source_path": str(source_path.resolve()),
                "source_kind": record.get("source_kind"),
                "declared_month": record.get("declared_month"),
                "source_role": record.get("source_role"),
                "usage_scope": record.get("usage_scope"),
                "input_size_bytes": record.get("input_size_bytes"),
                "actual_size_bytes": size_bytes,
                "modified_ns": modified_ns,
            }
        )
    return {
        "stage": "hydration",
        "schema_version": _HYDRATION_STATE_SCHEMA_VERSION,
        "registry_path": str(registry_path.resolve()),
        "sources": source_records,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": file_sha256(selection_path),
        "limits": asdict(limits),
        "memory_policy_version": MEMORY_POLICY_VERSION,
        "max_records": max_records,
    }


def _restore_retainer(
    retainer: _RankRetainer,
    rows: object,
    candidate_count: object,
) -> None:
    if not isinstance(rows, list) or not isinstance(candidate_count, int):
        raise ValueError("hydration checkpoint has malformed retained-message state")
    for line_number, payload in enumerate(rows, start=1):
        retainer.offer(parse_selected_message(payload, line_number))
    if candidate_count < len(rows):
        raise ValueError("hydration checkpoint candidate count is smaller than retained rows")
    retainer.candidate_count = candidate_count


def _write_hydration_checkpoint(
    path: Path,
    *,
    run_id: str,
    completed_source_ids: set[str],
    active_source: dict[str, Any] | None,
    context: _RankRetainer,
    controls: _RankRetainer,
    context_from_context_only: int,
    scanned_total: int,
    tombstoned_count: int,
    source_reports: list[dict[str, Any]],
) -> None:
    atomic_write_json(
        path,
        {
            "kind": "hydration_checkpoint",
            "schema_version": _HYDRATION_STATE_SCHEMA_VERSION,
            "run_id": run_id,
            "completed_source_ids": sorted(completed_source_ids),
            "active_source": active_source,
            "context_rows": [serialize_message(message) for message in context.ordered()],
            "control_rows": [serialize_message(message) for message in controls.ordered()],
            "context_candidate_count": context.candidate_count,
            "control_candidate_count": controls.candidate_count,
            "context_from_context_only_sources": context_from_context_only,
            "tombstoned_count": tombstoned_count,
            "scanned_record_count": scanned_total,
            "sources": source_reports,
        },
    )


def _load_hydration_checkpoint(
    path: Path,
    *,
    run_id: str,
    context: _RankRetainer,
    controls: _RankRetainer,
) -> tuple[set[str], dict[str, Any] | None, int, int, int, list[dict[str, Any]]]:
    checkpoint = load_json(path)
    if checkpoint.get("kind") != "hydration_checkpoint":
        raise ValueError(f"invalid hydration checkpoint: {path}")
    if checkpoint.get("schema_version") != _HYDRATION_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported hydration checkpoint schema: {path}")
    if checkpoint.get("run_id") != run_id:
        raise ValueError("hydration checkpoint does not match current inputs or limits")
    completed_source_ids = checkpoint.get("completed_source_ids")
    active_source = checkpoint.get("active_source")
    source_reports = checkpoint.get("sources")
    if not isinstance(completed_source_ids, list) or not all(
        isinstance(source_id, str) for source_id in completed_source_ids
    ):
        raise ValueError("hydration checkpoint has malformed completed sources")
    if active_source is not None:
        if not isinstance(active_source, dict) or not isinstance(
            active_source.get("source_id"), str
        ):
            raise ValueError("hydration checkpoint has malformed active source")
        for key in (
            "last_line_number",
            "processed_records",
            "thread_matched",
            "rejected",
            "tombstoned",
        ):
            if not isinstance(active_source.get(key), int) or active_source[key] < 0:
                raise ValueError("hydration checkpoint has malformed active source")
    if not isinstance(source_reports, list) or not all(
        isinstance(report, dict) for report in source_reports
    ):
        raise ValueError("hydration checkpoint has malformed source reports")
    _restore_retainer(
        context,
        checkpoint.get("context_rows"),
        checkpoint.get("context_candidate_count"),
    )
    _restore_retainer(
        controls,
        checkpoint.get("control_rows"),
        checkpoint.get("control_candidate_count"),
    )
    context_from_context_only = checkpoint.get("context_from_context_only_sources")
    scanned_total = checkpoint.get("scanned_record_count")
    tombstoned_count = checkpoint.get("tombstoned_count")
    if (
        not isinstance(context_from_context_only, int)
        or not isinstance(scanned_total, int)
        or not isinstance(tombstoned_count, int)
        or tombstoned_count < 0
    ):
        raise ValueError("hydration checkpoint has malformed scan counters")
    return (
        set(completed_source_ids),
        dict(active_source) if active_source is not None else None,
        context_from_context_only,
        scanned_total,
        tombstoned_count,
        [dict(report) for report in source_reports],
    )


def _publish_hydration_outputs(
    context_path: Path,
    controls_path: Path,
    selected_rows: Iterable[dict[str, Any]],
    context_rows: Iterable[dict[str, Any]],
    control_rows: Iterable[dict[str, Any]],
) -> None:
    """Publish both hydration shards together or leave no new shard behind."""
    context_staging = context_path.with_name(context_path.name + ".staging")
    controls_staging = controls_path.with_name(controls_path.name + ".staging")
    published: list[Path] = []
    try:
        write_compressed_jsonl(context_staging, chain(selected_rows, context_rows))
        write_compressed_jsonl(controls_staging, control_rows)
        context_staging.replace(context_path)
        published.append(context_path)
        controls_staging.replace(controls_path)
        published.append(controls_path)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        context_staging.unlink(missing_ok=True)
        controls_staging.unlink(missing_ok=True)
def run_hydration(
    registry_path: Path,
    selection_path: Path,
    output_directory: Path,
    *,
    limits: HydrationLimits,
    max_records: int | None = None,
    tombstones_path: Path | None = None,
    checkpoint_path: Path | None = None,
    rss_sampler: Any = None,
    clock: Callable[[], float] = time.monotonic,
    rss_provider: Callable[[], int] | None = None,
    disk_provider: Callable[[Path], int] | None = None,
):
    """Reread registered sources with atomic output and source-level resume."""
    started_at = clock()
    if limits.max_context_messages < 0:
        raise ValueError("max_context_messages must not be negative")
    if limits.control_target < 0:
        raise ValueError("control_target must not be negative")
    if limits.checkpoint_interval_records <= 0:
        raise ValueError("checkpoint_interval_records must be positive")
    resource_limits = ResourceLimits(
        max_staging_bytes=limits.max_staging_bytes,
        minimum_free_disk_bytes=limits.minimum_free_disk_bytes,
        max_process_rss_bytes=limits.max_process_rss_bytes,
    )
    resource_limits.validate()
    output_directory.mkdir(parents=True, exist_ok=True)
    check_output_budget(output_directory, estimated_bytes=0, limits=resource_limits)
    tombstones = load_tombstone_ledger(tombstones_path)

    registry = load_source_registry(registry_path)
    identity = _hydration_identity(
        registry_path,
        registry,
        selection_path,
        limits,
        max_records,
    )
    identity["tombstones"] = tombstone_identity(tombstones)
    run_id = json_sha256(identity)
    telemetry_path = output_directory / "stage_telemetry.json"
    input_hashes = {
        "files": [
            {"path": str(registry_path), "sha256": file_sha256(registry_path)},
            {"path": str(selection_path), "sha256": file_sha256(selection_path)},
        ]
    }
    if tombstones_path is not None:
        input_hashes["files"].append(
            {"path": str(tombstones_path), "sha256": file_sha256(tombstones_path)}
        )
    manifest_path = output_directory / "hydration_manifest.json"
    checkpoint_path = checkpoint_path or output_directory / "hydration_checkpoint.json"
    if not manifest_path.exists() and any(
        path.exists()
        for path in (
            output_directory / "hydrated-context.jsonl.zst",
            output_directory / "rejected-controls.jsonl.zst",
        )
    ):
        raise ValueError("hydration output and manifest must be published together")
    if manifest_path.exists():
        existing_manifest = load_json(manifest_path)
        if existing_manifest.get("run_id") != run_id:
            raise ValueError("hydration manifest does not match current inputs or limits")
        if existing_manifest.get("complete"):
            context_path = output_directory / "hydrated-context.jsonl.zst"
            controls_path = output_directory / "rejected-controls.jsonl.zst"
            output_sha256 = existing_manifest.get("output_sha256")
            if not context_path.is_file() or not controls_path.is_file():
                raise ValueError("hydration outputs are missing")
            if not isinstance(output_sha256, dict):
                raise ValueError("complete hydration manifest has no output hashes")
            if output_sha256.get("context") != file_sha256(context_path) or output_sha256.get(
                "controls"
            ) != file_sha256(controls_path):
                raise ValueError("hydration outputs do not match their manifest")
            if not telemetry_path.is_file():
                raise ValueError("hydration telemetry is missing")
            telemetry = load_stage_telemetry(telemetry_path)
            if telemetry.get("stage") != "hydration":
                raise ValueError("hydration telemetry has an invalid stage")
            if telemetry.get("run_identity") != run_id:
                raise ValueError("hydration telemetry does not match current inputs or limits")
            if telemetry_file_identity(telemetry_path)["sha256"] != existing_manifest.get(
                "telemetry", {}
            ).get("sha256"):
                raise ValueError("hydration telemetry does not match its manifest")
            result = existing_manifest.get("result")
            if not isinstance(result, dict):
                raise ValueError("complete hydration manifest has no result")
            return result
    selected_candidates = list(read_selected_messages(selection_path))
    tombstoned_count = sum(
        tombstones.matches(message.fullname, message.source_revision_id)
        for message in selected_candidates
    )
    selected_messages = [
        message
        for message in selected_candidates
        if not tombstones.matches(message.fullname, message.source_revision_id)
    ]
    if not selected_candidates:
        raise ValueError(f"selection shard is empty: {selection_path}")
    has_active_selection = bool(selected_messages)
    selected_fullnames = {message.fullname for message in selected_messages}
    thread_ids = {message.thread_fullname for message in selected_messages}
    selected_bytes = sum(
        len(json.dumps(serialize_message(message), ensure_ascii=False, sort_keys=True).encode()) + 1
        for message in selected_messages
    )

    source_records = [
        source
        for source in registry["sources"]
        if source.get("source_role") in {"discovery", "context_only"}
    ]
    if not source_records and has_active_selection:
        raise ValueError("registry has no discovery or context-only source")
    if not has_active_selection:
        source_records = []

    context = _RankRetainer(limits.max_context_messages, limits.seed)
    controls = _RankRetainer(limits.control_target, limits.seed + 1)
    completed_source_ids: set[str] = set()
    active_source: dict[str, Any] | None = None
    source_reports: list[dict[str, Any]] = []
    context_from_context_only = 0
    scanned_total = 0
    resumed_from_checkpoint = checkpoint_path.exists()
    if resumed_from_checkpoint:
        (
            completed_source_ids,
            active_source,
            context_from_context_only,
            scanned_total,
            tombstoned_count,
            source_reports,
        ) = _load_hydration_checkpoint(
            checkpoint_path,
            run_id=run_id,
            context=context,
            controls=controls,
        )
    memory_budget: dict[str, int | str] | None = None

    interrupted: str | None = None
    budget_exhausted = False
    scan_limit_reached = False
    scan_started = time.monotonic()

    for source_record in source_records:
        source = SourceSpec(
            source_id=str(source_record["source_id"]),
            path=Path(str(source_record["source_path"])),
            source_kind=str(source_record["source_kind"]),  # type: ignore[arg-type]
            declared_month=str(source_record["declared_month"]),
            source_role=str(source_record["source_role"]),  # type: ignore[arg-type]
            usage_scope=str(source_record["usage_scope"]),
        )
        if source.source_id in completed_source_ids:
            continue
        if active_source is not None and active_source["source_id"] != source.source_id:
            raise ValueError("hydration checkpoint active source is out of order")
        if source_record.get("status") not in {"registered", "complete", "complete_with_errors"}:
            source_reports.append(
                {
                    "source_id": source.source_id,
                    "status": "skipped",
                    "reason": f"registry status {source_record.get('status')!r}",
                }
            )
            completed_source_ids.add(source.source_id)
            active_source = None
            _write_hydration_checkpoint(
                checkpoint_path,
                run_id=run_id,
                completed_source_ids=completed_source_ids,
                active_source=None,
                context=context,
                controls=controls,
                context_from_context_only=context_from_context_only,
                scanned_total=scanned_total,
                tombstoned_count=tombstoned_count,
                source_reports=source_reports,
            )
            continue
        progress = active_source
        active_source = None
        reader = ArchiveReader()
        processed = int(progress["processed_records"]) if progress else 0
        tombstoned = int(progress["tombstoned"]) if progress else 0
        thread_matched = int(progress["thread_matched"]) if progress else 0
        rejected = int(progress["rejected"]) if progress else 0
        resume_line_number = int(progress["last_line_number"]) if progress else 0
        try:
            for envelope in reader.iter_records(source):
                if envelope.line_number <= resume_line_number:
                    continue
                if max_records is not None and scanned_total >= max_records:
                    scan_limit_reached = True
                    break
                scanned_total += 1
                if scanned_total % _PROGRESS_INTERVAL == 0:
                    elapsed = time.monotonic() - scan_started
                    rate = scanned_total / elapsed if elapsed else 0.0
                    logger.info(
                        "hydration progress: source=%s scanned=%d context=%d "
                        "controls=%d rate=%.0f rec/s",
                        source.source_id,
                        scanned_total,
                        len(context.items),
                        len(controls.items),
                        rate,
                    )
                processed += 1
                try:
                    message = normalize_record(envelope)
                except NormalizationError:
                    rejected += 1
                else:
                    if tombstones.matches(message.fullname, message.source_revision_id):
                        tombstoned += 1
                        tombstoned_count += 1
                    elif message.fullname in selected_fullnames:
                        thread_matched += 1
                    elif message.thread_fullname not in thread_ids:
                        if limits.control_target:
                            controls.offer(message)
                        rejected += 1
                    else:
                        if source.source_role == "context_only":
                            context_from_context_only += 1
                        thread_matched += 1
                        context.offer(message)
                check_rss_budget(
                    limits=resource_limits,
                    sampler=rss_provider or rss_sampler or process_rss_bytes,
                    stage="hydration",
                    reason="retention_loop",
                )
                if processed % limits.checkpoint_interval_records == 0:
                    active_source = {
                        "source_id": source.source_id,
                        "last_line_number": envelope.line_number,
                        "processed_records": processed,
                        "thread_matched": thread_matched,
                        "tombstoned": tombstoned,
                        "rejected": rejected,
                    }
                    _write_hydration_checkpoint(
                        checkpoint_path,
                        run_id=run_id,
                        completed_source_ids=completed_source_ids,
                        active_source=active_source,
                        context=context,
                        controls=controls,
                        context_from_context_only=context_from_context_only,
                        scanned_total=scanned_total,
                        source_reports=source_reports,
                        tombstoned_count=tombstoned_count,
                    )
        except BudgetError as error:
            budget_exhausted = True
            memory_budget = error.metadata
            interrupted = str(error)
        except SourceReadError as error:
            record_validation_failure(source_record, error)
            interrupted = f"source read failed: {source.source_id}: {error}"
        if reader.stats.verified_sha256 is None and interrupted is None and not scan_limit_reached:
            interrupted = f"source read incomplete: {source.source_id}"
        source_reports.append(
            {
                "source_id": source.source_id,
                "source_role": source.source_role,
                "processed": processed,
                "thread_matched": thread_matched,
                "tombstoned": tombstoned,
                "rejected": rejected,
                "lines_seen": reader.stats.lines_seen,
                "invalid_records": reader.stats.invalid_records,
                "verified_sha256": reader.stats.verified_sha256,
                "complete": reader.stats.complete,
            }
        )
        if interrupted is not None or scan_limit_reached:
            break
        completed_source_ids.add(source.source_id)
        active_source = None
        _write_hydration_checkpoint(
            checkpoint_path,
            run_id=run_id,
            completed_source_ids=completed_source_ids,
            active_source=None,
            context=context,
            controls=controls,
            context_from_context_only=context_from_context_only,
            scanned_total=scanned_total,
            source_reports=source_reports,
            tombstoned_count=tombstoned_count,
        )

    selected_count = len(selected_messages)
    context_messages = context.ordered()
    control_messages = controls.ordered()
    context_count = len(context_messages)
    control_count = len(control_messages)
    staging_estimate_bytes = (
        selected_bytes + context.serialized_bytes() + controls.serialized_bytes()
    )
    if not interrupted:
        try:
            check_output_budget(
                output_directory, estimated_bytes=staging_estimate_bytes, limits=resource_limits
            )
            memory_budget = check_rss_budget(
                limits=resource_limits, sampler=rss_provider or rss_sampler or process_rss_bytes,
                stage="hydration", reason="before_publication",
            )
        except BudgetError as error:
            interrupted = str(error).split(":", 1)[0]
            memory_budget = error.metadata
            budget_exhausted = True
    complete = interrupted is None and not scan_limit_reached
    context_path = output_directory / "hydrated-context.jsonl.zst"
    controls_path = output_directory / "rejected-controls.jsonl.zst"
    if complete:
        _publish_hydration_outputs(
            context_path,
            controls_path,
            (serialize_message(message) for message in selected_messages),
            (serialize_message(message) for message in context_messages),
            (serialize_message(message) for message in control_messages),
        )
    else:
        context_path.unlink(missing_ok=True)
        controls_path.unlink(missing_ok=True)
        context_path.with_name(context_path.name + ".staging").unlink(missing_ok=True)
        controls_path.with_name(controls_path.name + ".staging").unlink(missing_ok=True)
    output_sha256 = {
        "context": file_sha256(context_path) if complete else None,
        "controls": file_sha256(controls_path) if complete else None,
    }

    telemetry = capture_stage_telemetry(
        stage="hydration",
        input_hashes=input_hashes,
        run_identity=run_id,
        limits=asdict(limits),
        path=output_directory,
        clock=clock,
        started_at=started_at,
        rss_provider=rss_provider or rss_sampler or process_rss_bytes,
        disk_provider=disk_provider or (lambda path: shutil.disk_usage(path).free),
        cache_provider=lambda: "not_used",
        cleanup_provider=lambda: "complete",
    )
    if not complete:
        telemetry["status"] = "incomplete"
        telemetry["errors"].append(
            {"field": "hydration", "error": interrupted or "hydration incomplete"}
        )
    write_stage_telemetry(telemetry_path, telemetry)
    telemetry_identity = telemetry_file_identity(telemetry_path) | {"status": telemetry["status"]}
    result = {
        "complete": complete,
        "interruption_reason": interrupted,
        "budget_exhausted": budget_exhausted,
        "scan_limit_records": max_records,
        "scanned_record_count": scanned_total,
        "selected_count": selected_count,
        "tombstoned_count": tombstoned_count,
        "context_count": context_count,
        "context_candidate_count": context.candidate_count,
        "control_count": control_count,
        "checkpoint_interval_records": limits.checkpoint_interval_records,
        "staging_estimate_bytes": staging_estimate_bytes,
        "context_path": str(context_path) if complete else None,
        "controls_path": str(controls_path) if complete else None,
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "run_id": run_id,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "sources": source_reports,
        "telemetry": telemetry_identity,
    }
    manifest = {
        "kind": "hydration_manifest",
        "schema_version": _HYDRATION_STATE_SCHEMA_VERSION,
        "run_id": run_id,
        "run_identity": identity,
        "complete": complete,
        "memory_policy_version": MEMORY_POLICY_VERSION,
        "telemetry": telemetry_identity,
        "memory_budget": memory_budget,
        "interruption_reason": interrupted,
        "scan_limit_records": max_records,
        "scanned_record_count": scanned_total,
        "selected_count": selected_count,
        "tombstoned_count": tombstoned_count,
        "context_count": context_count,
        "context_candidate_count": context.candidate_count,
        "context_from_context_only_sources": context_from_context_only,
        "output_sha256": output_sha256,
        "resource_limits": {
            "max_staging_bytes": limits.max_staging_bytes,
            "minimum_free_disk_bytes": limits.minimum_free_disk_bytes,
            "max_process_rss_bytes": limits.max_process_rss_bytes,
            "memory_policy_version": MEMORY_POLICY_VERSION,
            "seed": limits.seed,
        },
        "tombstones": tombstone_identity(tombstones),
        "budget_exhausted": budget_exhausted,
        "checkpoint_path": str(checkpoint_path),
        "sources": source_reports,
        "result": result,
    }
    atomic_write_json(manifest_path, manifest)
    persist_source_registry(registry_path, registry)
    if complete:
        checkpoint_path.unlink(missing_ok=True)
    return result
