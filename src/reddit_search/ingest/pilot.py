"""Bounded, explicitly partial discovery across registered source kinds."""

from __future__ import annotations

from dataclasses import replace
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
    serialized_json_bytes,
)

from .discovery import DiscoverySelector, load_discovery_rules
from .invalidation import load_tombstone_ledger, tombstone_identity
from .normalize import NormalizationError, normalize_record
from .reader import ArchiveReader, SourceReadError, SourceSpec
from .shard import retain_lowest_ranked as _retain_lowest_ranked
from .shard import serialize_message as _serialize_message
from .shard import stable_rank as _stable_rank
from .shard import write_compressed_jsonl as _write_compressed_jsonl
from .sources import (
    load_source_registry,
    persist_source_registry,
    record_reader_validation,
    record_validation_failure,
)
from .state import atomic_write_json, file_sha256, json_sha256, load_json

_DISCOVERY_STATE_SCHEMA_VERSION = 2


def _discovery_identity(
    registry_path: Path,
    source_records: list[dict[str, Any]],
    rules_path: Path,
    *,
    target: int,
    seed: int,
    max_records: int,
    max_staging_bytes: int,
    minimum_free_disk_bytes: int,
    max_process_rss_bytes: int,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for record in source_records:
        source_path = Path(str(record["source_path"]))
        try:
            stat = source_path.stat()
            size_bytes = stat.st_size
            modified_ns = stat.st_mtime_ns
        except OSError:
            size_bytes = None
            modified_ns = None
        sources.append(
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
        "stage": "discovery",
        "schema_version": _DISCOVERY_STATE_SCHEMA_VERSION,
        "registry_path": str(registry_path.resolve()),
        "sources": sources,
        "rules_path": str(rules_path.resolve()),
        "rules_sha256": file_sha256(rules_path),
        "target": target,
        "seed": seed,
        "max_records": max_records,
        "resource_limits": {
            "max_staging_bytes": max_staging_bytes,
            "minimum_free_disk_bytes": minimum_free_disk_bytes,
            "max_process_rss_bytes": max_process_rss_bytes,
            "memory_policy_version": MEMORY_POLICY_VERSION,
        },
    }


def _discovery_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".manifest.json")


def run_discovery(
    registry_path: Path,
    rules_path: Path,
    output_path: Path,
    *,
    target: int,
    seed: int,
    max_records: int = 1_000_000,
    max_staging_bytes: int = DEFAULT_MAX_STAGING_BYTES,
    minimum_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES,
    max_process_rss_bytes: int = DEFAULT_MAX_PROCESS_RSS_BYTES,
    rss_sampler: Any = None,
    tombstones_path: Path | None = None,
):
    """Select from all discovery sources with deterministic idempotent output."""
    if target <= 0:
        raise ValueError("target must be positive")
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    resource_limits = ResourceLimits(
        max_staging_bytes=max_staging_bytes,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
        max_process_rss_bytes=max_process_rss_bytes,
    )
    resource_limits.validate()
    registry = load_source_registry(registry_path)
    tombstones = load_tombstone_ledger(tombstones_path)
    source_records = [
        source for source in registry["sources"] if source.get("source_role") == "discovery"
    ]
    if not source_records:
        raise ValueError("registry has no discovery-role source")
    identity = _discovery_identity(
        registry_path,
        source_records,
        rules_path,
        target=target,
        seed=seed,
        max_records=max_records,
        max_staging_bytes=resource_limits.max_staging_bytes,
        minimum_free_disk_bytes=resource_limits.minimum_free_disk_bytes,
        max_process_rss_bytes=resource_limits.max_process_rss_bytes,
    )
    identity["tombstones"] = tombstone_identity(tombstones)
    run_id = json_sha256(identity)
    manifest_path = _discovery_manifest_path(output_path)
    if output_path.exists() or manifest_path.exists():
        if not output_path.exists() or not manifest_path.exists():
            raise ValueError("discovery output and manifest must be published together")
        existing_manifest = load_json(manifest_path)
        if existing_manifest.get("run_id") != run_id:
            raise ValueError("discovery manifest does not match current inputs or limits")
        if existing_manifest.get("output_sha256") != file_sha256(output_path):
            raise ValueError("discovery output does not match its manifest")
        result = existing_manifest.get("result")
        if not isinstance(result, dict):
            raise ValueError("complete discovery manifest has no result")
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    check_output_budget(output_path.parent, estimated_bytes=0, limits=resource_limits)
    selector = DiscoverySelector(load_discovery_rules(rules_path))
    selection_heap: list[tuple[int, str]] = []
    selected_by_fullname: dict[str, tuple[int, dict[str, Any]]] = {}
    tombstoned_count = 0
    normalization_error_count = 0
    scanned_record_count = 0
    scan_complete = True
    memory_budget: dict[str, int | str] | None = None
    budget_error: BudgetError | None = None

    for source_record in source_records:
        source_kind = str(source_record["source_kind"])
        if source_kind not in {"submission", "comment"}:
            raise ValueError(f"unknown source kind: {source_kind}")
        source = SourceSpec(
            source_id=str(source_record["source_id"]),
            path=Path(str(source_record["source_path"])),
            source_kind=source_kind,
            declared_month=str(source_record["declared_month"]),
            source_role="discovery",
            usage_scope=str(source_record["usage_scope"]),
        )
        if source_record.get("status") not in {"registered", "complete"}:
            raise ValueError(
                "discovery requires a registered or complete source: "
                f"{source_record.get('source_id')}"
            )
        reader = ArchiveReader()
        try:
            for envelope in reader.iter_records(source):
                if scanned_record_count >= max_records:
                    scan_complete = False
                    break
                scanned_record_count += 1
                try:
                    message = normalize_record(envelope)
                except NormalizationError:
                    normalization_error_count += 1
                    continue
                if tombstones.matches(message.fullname, message.source_revision_id):
                    tombstoned_count += 1
                    continue
                decision = selector.select(message)
                if not decision.selected:
                    continue
                selected_message = replace(
                    message,
                    selection_channels=tuple(decision.selection_channels),
                    matched_rule_ids=tuple(decision.matched_rule_ids),
                )
                try:
                    _retain_lowest_ranked(
                        selection_heap, selected_by_fullname,
                        _stable_rank(selected_message.fullname, seed),
                        selected_message.fullname, _serialize_message(selected_message), target,
                    )
                    memory_budget = check_rss_budget(
                        limits=resource_limits, sampler=rss_sampler or process_rss_bytes,
                        stage="discovery", reason="retention_loop",
                    )
                except BudgetError as error:
                    budget_error = error
                    memory_budget = error.metadata
                    scan_complete = False
                    break
        except SourceReadError as error:
            record_validation_failure(source_record, error)
            persist_source_registry(registry_path, registry)
            raise
        if reader.stats.verified_sha256 is None:
            scan_complete = False
        elif not record_reader_validation(source_record, reader.stats):
            persist_source_registry(registry_path, registry)
            raise ValueError(f"source validation did not complete: {source_record['source_id']}")
        if budget_error is not None:
            break
    if budget_error is not None:
        result = {
            "scope": "all_discovery_sources", "scan_complete": False, "complete": False,
            "budget_exhausted": True, "budget_breach": memory_budget,
            "manifest_path": str(manifest_path), "run_id": run_id,
            "resource_limits": {
                "max_process_rss_bytes": resource_limits.max_process_rss_bytes,
                "memory_policy_version": MEMORY_POLICY_VERSION,
            },
        }
        output_path.unlink(missing_ok=True)
        atomic_write_json(manifest_path, {"kind": "discovery_manifest",
            "schema_version": _DISCOVERY_STATE_SCHEMA_VERSION, "run_id": run_id,
            "run_identity": identity, "complete": False, "budget_exhausted": True,
            "memory_budget": memory_budget, "result": result})
        return result

    persist_source_registry(registry_path, registry)
    ordered_selection = sorted(
        ((rank, fullname, message) for fullname, (rank, message) in selected_by_fullname.items()),
        key=lambda item: (item[0], item[1]),
    )
    selected_rows = [item[2] for item in ordered_selection]
    estimated_bytes = sum(serialized_json_bytes(row) for row in selected_rows)
    try:
        check_rss_budget(
            limits=resource_limits, sampler=rss_sampler or process_rss_bytes,
            stage="discovery", reason="before_publication",
        )
    except BudgetError as error:
        resource_metadata = {
            "max_staging_bytes": resource_limits.max_staging_bytes,
            "minimum_free_disk_bytes": resource_limits.minimum_free_disk_bytes,
            "max_process_rss_bytes": resource_limits.max_process_rss_bytes,
            "memory_policy_version": MEMORY_POLICY_VERSION,
        }
        result = {
            "scope": "all_discovery_sources", "scan_complete": False, "complete": False,
            "budget_exhausted": True, "budget_breach": error.metadata,
            "memory_budget": error.metadata, "resource_limits": resource_metadata,
            "manifest_path": str(manifest_path), "run_id": run_id,
        }
        output_path.unlink(missing_ok=True)
        atomic_write_json(manifest_path, {
            "kind": "discovery_manifest", "schema_version": _DISCOVERY_STATE_SCHEMA_VERSION,
            "run_id": run_id, "run_identity": identity, "complete": False,
            "budget_exhausted": True, "memory_budget": error.metadata,
            "resource_limits": resource_metadata, "result": result,
        })
        return result
    check_output_budget(output_path.parent, estimated_bytes=estimated_bytes, limits=resource_limits)
    _write_compressed_jsonl(output_path, selected_rows)
    result = {
        "scope": "all_discovery_sources",
        "source_count": len(source_records),
        "scan_complete": scan_complete,
        "scan_limit_records": max_records,
        "scanned_record_count": scanned_record_count,
        "selected_count": len(ordered_selection),
        "tombstoned_count": tombstoned_count,
        "tombstones": tombstone_identity(tombstones),
        "normalization_error_count": normalization_error_count,
        "output_file": str(output_path),
        "manifest_path": str(manifest_path),
        "run_id": run_id,
        "staging_estimate_bytes": estimated_bytes,
        "resource_limits": {
            "max_staging_bytes": resource_limits.max_staging_bytes,
            "minimum_free_disk_bytes": resource_limits.minimum_free_disk_bytes,
            "max_process_rss_bytes": resource_limits.max_process_rss_bytes,
            "memory_policy_version": MEMORY_POLICY_VERSION,
        },
    }
    atomic_write_json(
        manifest_path,
        {
            "kind": "discovery_manifest",
            "schema_version": _DISCOVERY_STATE_SCHEMA_VERSION,
            "run_id": run_id,
            "run_identity": identity,
            "output_sha256": file_sha256(output_path),
            "result": result,
        },
    )
    return result
