"""Conservative reuse of human labels across identical scenario tasks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reddit_search.ingest.invalidation import TombstoneLedger, tombstone_blocks_row

from .labels import EvaluatorKind, LabelRecord

TaskKey = tuple[str, str | None]

def reuse_labels(
    source_pool_path: Path,
    source_labels_path: Path,
    target_pool_path: Path,
    output_directory: Path,
    *,
    rubric_version: str,
    tombstone_ledger: TombstoneLedger | None = None,
) -> dict[str, Any]:
    """Copy only human labels whose task and source fingerprint are unchanged."""
    if not rubric_version:
        raise ValueError("rubric_version must not be empty")
    source_pool = _read_jsonl(source_pool_path)
    target_pool = _read_jsonl(target_pool_path)
    source_by_task = _pool_by_task(source_pool)
    target_by_task = _pool_by_task(target_pool)
    labels = _read_jsonl(source_labels_path)
    reused: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[TaskKey] = set()
    for raw in labels:
        try:
            label = LabelRecord.model_validate(raw)
        except ValidationError:
            skipped["invalid_label"] += 1
            continue
        if label.evaluator_kind is not EvaluatorKind.HUMAN:
            skipped["non_human_label"] += 1
            continue
        task = _label_task(label, source_by_task)
        if task in seen:
            skipped["duplicate_label_task"] += 1
            continue
        seen.add(task)
        source_row = source_by_task.get(task)
        target_row = target_by_task.get(task)
        if source_row is None:
            skipped["source_task_missing"] += 1
            continue
        if target_row is None:
            skipped["target_task_missing"] += 1
            continue
        if label.rubric_version != rubric_version:
            skipped["rubric_mismatch"] += 1
            continue
        if tombstone_ledger is not None:
            if tombstone_blocks_row(tombstone_ledger, source_row) or tombstone_blocks_row(
                tombstone_ledger, target_row
            ):
                skipped["invalidated_source_or_context"] += 1
                continue
        if _context_dependency_identity(source_row) != _context_dependency_identity(target_row):
            skipped["context_dependency_identity_mismatch"] += 1
            continue
        if _fingerprint(source_row, rubric_version) != _fingerprint(target_row, rubric_version):
            skipped["source_fingerprint_mismatch"] += 1
            continue
        copied = label.model_dump(mode="json")
        copied.update(
            {
                "candidate_id": target_row["candidate_id"],
                "snapshot_id": target_row.get("snapshot_id"),
                "scenario_id": target_row.get("scenario_id"),
                "app_id": target_row.get("app_id"),
                "app_profile_version": target_row.get("app_profile_version"),
                "rubric_version": rubric_version,
            }
        )
        if "app_profile_sha256" in target_row:
            copied["app_profile_sha256"] = target_row["app_profile_sha256"]
        reused.append(copied)

    output_directory.mkdir(parents=True, exist_ok=True)
    labels_path = output_directory / "labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reused),
        encoding="utf-8",
    )
    report = {
        "kind": "reusable_label_cache_report",
        "schema_version": 1,
        "rubric_version": rubric_version,
        "source_pool_sha256": _sha(source_pool_path),
        "source_labels_sha256": _sha(source_labels_path),
        "target_pool_sha256": _sha(target_pool_path),
        "source_label_count": len(labels),
        "reused_label_count": len(reused),
        "skipped": dict(skipped),
        "labels_file": str(labels_path),
    }
    report_path = output_directory / "reuse_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "report_file": str(report_path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(value)
    return rows


def _pool_by_task(rows: list[dict[str, Any]]) -> dict[TaskKey, dict[str, Any]]:
    result: dict[TaskKey, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        scenario_id = row.get("scenario_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("pool row lacks candidate_id")
        if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
            raise ValueError("pool row scenario_id must be a non-empty string or null")
        task = (candidate_id, scenario_id)
        if task in result:
            raise ValueError("pool contains duplicate candidate/scenario tasks")
        result[task] = row
    return result


def _label_task(label: LabelRecord, source_by_task: dict[TaskKey, dict[str, Any]]) -> TaskKey:
    task = (label.candidate_id, label.scenario_id)
    if label.scenario_id is not None:
        return task
    candidates = [
        candidate_task
        for candidate_task in source_by_task
        if candidate_task[0] == label.candidate_id
    ]
    if len(candidates) == 1:
        return candidates[0]
    return task


def _context_dependency_identity(row: dict[str, Any]) -> str | None:
    value = row.get("context_dependency_identity")
    if value is None and isinstance(row.get("source"), dict):
        value = row["source"].get("context_dependency_identity")
    return value if value is None else str(value)


def _fingerprint(row: dict[str, Any], rubric_version: str) -> str:
    source = row.get("source")
    if not isinstance(source, dict):
        raise ValueError("pool row lacks source object")
    identity_fields = (
        "candidate_id",
        "scenario_id",
        "snapshot_id",
        "app_id",
        "app_profile_version",
        "app_profile_sha256",
        "source_revision_id",
        "source_fingerprint",
        "context_recipe_version",
        "context_dependency_identity",
        "context_text",
        "context_only_text",
        "missing_context_ids",
        "ancestors_truncated",
        "chunking_version",
    )
    payload = {
        "rubric_version": rubric_version,
        "row_identity": {key: row.get(key) for key in identity_fields if key in row},
        # Source carries the focus text and, for pooled context-bearing rows,
        # the complete parent-context identity.
        "source": source,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
