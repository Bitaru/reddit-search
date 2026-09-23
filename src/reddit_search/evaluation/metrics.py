"""Diagnostic measurement of saved lexical runs against judged labels (step 5).

Computes the judgment-coverage and pool-limited ranking metrics the plan
requires for deciding whether extra retrieval earns its cost, without
fabricating any judgment: rows without a human topic-fit decision count as
unjudged and are excluded from per-slot precision, never treated as invented
negatives. Metrics over development labels are diagnostic only; no held-out
test label has been frozen.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reddit_search.ingest.invalidation import (
    TombstoneLedger,
    tombstone_blocks_row,
    tombstone_identity,
)
from reddit_search.retrieval.comparison import load_comparison_manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_comparison_provenance(
    comparison_manifest_path: Path, runs_directory: Path, labels_path: Path
) -> None:
    """Reject tampered comparison inputs before scoring."""
    comparison = load_comparison_manifest(comparison_manifest_path)
    identity = comparison["identity"]
    if identity.get("labels_sha256") != _sha(labels_path):
        raise ValueError("labels sha256 does not match comparison manifest")
    run_manifest_path = runs_directory / "run_manifest.json"
    artifact_key = (
        "union" if "union" in runs_directory.name
        else runs_directory.name.split("-")[1].upper()
    )
    artifact = comparison.get("artifacts", {}).get(artifact_key)
    if (
        not isinstance(artifact, dict)
        or artifact.get("run_manifest_sha256") != _sha(run_manifest_path)
    ):
        raise ValueError("comparison artifact manifest does not match run manifest")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("parent_comparison_identity") != comparison["comparison_identity"]:
        raise ValueError("run manifest parent comparison identity mismatch")
    for key, entry in run_manifest.get("scenario_runs", {}).items():
        ranked = runs_directory / (f"{key}.jsonl" if artifact_key != "union" else f"{key}.jsonl")
        if not ranked.exists() or _sha(ranked) != entry.get("sha256"):
            raise ValueError(f"ranked artifact sha256 does not match run manifest: {key}")

_IDENTITY_FIELDS = (
    "candidate_id", "scenario_id", "snapshot_id", "app_id",
    "app_profile_version", "app_profile_sha256", "source_revision_id",
    "source_fingerprint", "context_recipe_version", "context_dependency_identity",
)


def _identity_projection(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _IDENTITY_FIELDS:
        value = row.get(key)
        if value is None and key in {"source_revision_id", "context_recipe_version"}:
            for container in (row.get("source"), row.get("source_bundle")):
                if isinstance(container, dict) and container.get(key) is not None:
                    value = container[key]
                    break
        if value is not None:
            result[key] = value
    return result


def _identity_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps([_identity_projection(row) for row in rows], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _identity_coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {key: sum(key in _identity_projection(row) for row in rows) for key in _IDENTITY_FIELDS}


def compute_run_report(
    runs_directory: Path,
    pool_path: Path,
    labels_path: Path,
    splits_path: Path | None = None,
    *,
    pool_key_path: Path | None = None,
    split_name: str | None = None,
    tombstone_ledger: TombstoneLedger | None = None,
) -> dict[str, Any]:
    """Measure a saved ranked run against scenario-aware human labels.
    Every measured row is identified by ``(candidate_id, scenario_id)``.
    Rows missing from the blinded pool remain unjudged rather than failing the
    report, which keeps corpus retrieval and pool-limited judgment separate.
    """
    manifest_path = runs_directory / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_rows = _read_jsonl(pool_path)
    pool_keys: set[TaskKey] = set()
    comparison_manifest_path = runs_directory.parent / "comparison_manifest.json"
    if comparison_manifest_path.exists():
        validate_comparison_provenance(comparison_manifest_path, runs_directory, labels_path)
        comparison_provenance = "trusted"
    else:
        comparison_provenance = "unknown"
    pool_by_candidate: dict[str, set[TaskKey]] = {}
    for row in pool_rows:
        task = _task_key(row)
        if task in pool_keys:
            raise ValueError("pool contains duplicate candidate/scenario task rows")
        pool_keys.add(task)
        pool_by_candidate.setdefault(task[0], set()).add(task)

    split_manifest_path = (
        splits_path.parent / "group_split_manifest.json" if splits_path is not None else None
    )
    split_manifest: dict[str, Any] | None = None
    split_provenance = "unknown"
    if split_manifest_path is not None and split_manifest_path.exists():
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        expected = split_manifest.get("candidate_splits_sha256")
        if expected and _sha(splits_path) != expected:
            raise ValueError("candidate split file sha256 does not match manifest")
        expected_pool = split_manifest.get("pool_sha256")
        if expected_pool and _sha(pool_path) != expected_pool:
            raise ValueError("pool sha256 does not match split manifest")
        expected_identity = split_manifest.get(
            "dependency_identity_digest", split_manifest.get("identity_digest")
        )
        if expected_identity and _identity_digest(pool_rows) != expected_identity:
            raise ValueError("pool dependency identity does not match split manifest")
        expected_coverage = split_manifest.get(
            "dependency_identity_coverage", split_manifest.get("identity_coverage")
        )
        if expected_coverage and _identity_coverage(pool_rows) != expected_coverage:
            raise ValueError("pool dependency coverage does not match split manifest")
        split_provenance = "trusted"
    split_keys: set[TaskKey] | None = None
    split_candidate_ids: set[str] | None = None
    if splits_path is not None:
        split_rows = _read_jsonl(splits_path)
        split_values = {row.get("split") for row in split_rows if row.get("split") is not None}
        if split_name is None and len(split_values) > 1:
            raise ValueError("split_name is required when splits contain multiple split values")
        split_keys = set()
        split_candidate_ids = set()
        for row in split_rows:
            if split_name is not None and row.get("split") != split_name:
                continue
            candidate_id = str(row["candidate_id"])
            split_candidate_ids.add(candidate_id)
            scenario_id = row.get("scenario_id")
            if scenario_id is None:
                split_keys.add((candidate_id, None))
            else:
                split_keys.add((candidate_id, str(scenario_id)))

    key_rows: dict[TaskKey, dict[str, Any]] = {}
    if pool_key_path is not None and pool_key_path.exists():
        payload = json.loads(pool_key_path.read_text(encoding="utf-8"))
        for entry in payload.values():
            if not isinstance(entry, dict) or "candidate_id" not in entry:
                continue
            task = _task_key(entry)
            key_rows[task] = entry

    label_rows = _read_jsonl(labels_path)
    scenarios: dict[str, dict[str, Any]] = {}
    judged: dict[TaskKey, str] = {}
    unknown_topic_fit: set[TaskKey] = set()
    evaluator_versions: Counter[str] = Counter()
    for row in label_rows:
        if str(row.get("evaluator_kind")) != "human":
            continue
        candidate_id = str(row["candidate_id"])
        scenario_id = row.get("scenario_id")
        if scenario_id is None:
            candidates = pool_by_candidate.get(candidate_id, set())
            if len(candidates) == 1:
                task = next(iter(candidates))
            else:
                task = (candidate_id, None)
        else:
            task = (candidate_id, str(scenario_id))
        fit = row.get("topic_fit")
        if fit in {"yes", "no"}:
            _record_label(judged, unknown_topic_fit, task, str(fit))
        elif fit == "unknown":
            _record_label(judged, unknown_topic_fit, task, "unknown")
        else:
            continue
        evaluator_versions[str(row.get("evaluator_version") or "unversioned")] += 1

    scenario_runs = manifest.get("scenario_runs") or manifest.get("scenario_sha256")
    if not isinstance(scenario_runs, dict) or not scenario_runs:
        raise ValueError("run manifest records no scenario runs")
    totals: Counter[str] = Counter()
    for scenario_id, recorded in sorted(scenario_runs.items()):
        run_path = runs_directory / f"{scenario_id}.jsonl"
        if not run_path.exists():
            raise ValueError(f"missing ranked run file: {run_path}")
        digest = _sha(run_path)
        if digest != recorded["sha256"]:
            raise ValueError(f"ranked run {scenario_id} sha256 {digest} does not match manifest")
        run_rows = _read_jsonl(run_path)
        if tombstone_ledger is not None:
            active_rows = []
            for row in run_rows:
                task = _task_key(row, str(scenario_id))
                candidates = [row]
                pool_row = next(
                    (pool for pool in pool_rows if _task_key(pool, str(scenario_id)) == task),
                    None,
                )
                key_row = key_rows.get(task)
                if pool_row is not None:
                    candidates.append(pool_row)
                if key_row is not None:
                    candidates.append(key_row)
                if any(tombstone_blocks_row(tombstone_ledger, item) for item in candidates):
                    continue
                active_rows.append(row)
            run_rows = active_rows
        scenario = _scenario_metrics(
            str(scenario_id),
            run_rows,
            judged,
            pool_keys,
            split_keys,
            split_candidate_ids,
            key_rows,
        )
        scenarios[str(scenario_id)] = scenario
        for field in (
            "returned_count",
            "judged_count",
            "relevant_count",
            "unjudged_count",
            "outside_pool_count",
            "fixed_slot_p20_relevant",
            "context_missing_count",
            "context_truncated_count",
        ):
            totals[field] += scenario[field]

    covered_tasks = {
        (item["candidate_id"], item["scenario_id"])
        for scenario in scenarios.values()
        for item in scenario["judged_task_keys"]
    }
    unjudged_tasks = {
        (item["candidate_id"], item["scenario_id"])
        for scenario in scenarios.values()
        for item in scenario["unjudged_task_keys"]
    }
    report: dict[str, Any] = {
        "kind": "lexical_run_metrics_report",
        "runs_directory": str(runs_directory),
        "inputs": {
            "run_manifest_sha256": _sha(manifest_path),
            "pool_path": str(pool_path),
            "pool_sha256": _sha(pool_path),
            "labels_path": str(labels_path),
            "labels_sha256": _sha(labels_path),
            "split_name": split_name,
            "split_manifest_path": str(split_manifest_path) if split_manifest_path else None,
            "split_provenance": split_provenance,
            "comparison_provenance": comparison_provenance,
            "tombstone_ledger": (
                tombstone_identity(tombstone_ledger)
                if tombstone_ledger is not None
                else "not_checked"
            ),
        },
        "measurement_limits": (
            "Diagnostic pool-limited measurement of human topic-fit judgments "
            "only. Rows without a human yes/no topic-fit decision are unjudged, "
            "not negatives. Rows outside the declared pool are retained as "
            "unjudged corpus results. Product fit is not_evaluated everywhere, "
            "so no confirmed-fit metric is reported. Not Reddit-wide recall and "
            "not a held-out test estimate."
        ),
        "judgment_coverage": {
            "label_rows": len(label_rows),
            "human_topic_fit_decisions": len(judged) + len(unknown_topic_fit),
            "yes_no_topic_fit_decisions": len(judged),
            "unknown_topic_fit_decisions": len(unknown_topic_fit),
            "relevant_human_labels": sum(value == "yes" for value in judged.values()),
            "evaluator_versions": dict(evaluator_versions),
            "product_fit_decisions": 0,
            "product_fit_note": (
                "No product_fit != not_evaluated label rows exist; confirmed-fit "
                "precision is unmeasurable until verified product evidence and "
                "human product-fit judgments exist."
            ),
        },
        "totals": dict(totals),
        "pool_limited_recall_note": (
            "Relevant judged questions retrieved / all relevant judged questions "
            "in the declared pool. The pool is a sample, so this is pool-limited "
            "recall, not total-corpus recall."
        ),
        "scenarios": scenarios,
        "judged_task_keys_covered": [
            {"candidate_id": candidate_id, "scenario_id": scenario_id}
            for candidate_id, scenario_id in sorted(
                covered_tasks, key=lambda task: (task[0], task[1] or "")
            )
        ],
        "judged_candidate_ids_covered": sorted({task[0] for task in covered_tasks}),
        "unjudged_top_candidates": sorted({task[0] for task in unjudged_tasks}),
        "unjudged_top_task_keys": [
            {"candidate_id": candidate_id, "scenario_id": scenario_id}
            for candidate_id, scenario_id in sorted(
                unjudged_tasks, key=lambda task: (task[0], task[1] or "")
            )
        ],
    }
    if split_keys is not None:
        report["inputs"]["split_restriction_candidate_count"] = len(split_candidate_ids or set())
        report["inputs"]["split_restriction_task_count"] = len(split_keys)
    return report


TaskKey = tuple[str, str | None]


def _task_key(row: dict[str, Any], default_scenario_id: str | None = None) -> TaskKey:
    candidate_id = row.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("row lacks candidate_id")
    scenario_id = row.get("scenario_id")
    if scenario_id is None:
        retrieval = row.get("retrieval")
        if isinstance(retrieval, dict):
            scenario_id = retrieval.get("scenario_id")
    if scenario_id is None:
        scenario_id = default_scenario_id
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
        raise ValueError("scenario_id must be a non-empty string or null")
    return candidate_id, scenario_id


def _record_label(
    judged: dict[TaskKey, str],
    unknown_topic_fit: set[TaskKey],
    task: TaskKey,
    verdict: str,
) -> None:
    prior = judged.get(task)
    if prior is not None and prior != verdict:
        raise ValueError(f"conflicting human topic-fit labels for {task}")
    if verdict == "unknown":
        if prior is not None:
            raise ValueError(f"conflicting human topic-fit labels for {task}")
        unknown_topic_fit.add(task)
        return
    if task in unknown_topic_fit:
        raise ValueError(f"conflicting human topic-fit labels for {task}")
    judged[task] = verdict


def _scenario_metrics(
    scenario_id: str,
    run_rows: list[dict[str, Any]],
    judged: dict[TaskKey, str],
    pool_keys: set[TaskKey],
    split_keys: set[TaskKey] | None,
    split_candidate_ids: set[str] | None,
    key_rows: dict[TaskKey, dict[str, Any]],
) -> dict[str, Any]:
    per_slot: list[str | None] = []
    fixed_slot_relevant = 0
    fixed_slot_count = 0
    relevant_returned = 0
    judged_task_keys: list[TaskKey] = []
    unjudged_task_keys: list[TaskKey] = []
    context_missing = 0
    context_truncated = 0
    outside_pool_count = 0
    for position, row in enumerate(run_rows, start=1):
        task = _task_key(row, scenario_id)
        if split_keys is not None and task not in split_keys and (task[0], None) not in split_keys:
            continue
        returned_rank = row.get("rank")
        rank = returned_rank if isinstance(returned_rank, int) and returned_rank > 0 else position
        verdict = judged.get(task) if task in pool_keys else None
        if task not in pool_keys:
            outside_pool_count += 1
        if verdict is None:
            unjudged_task_keys.append(task)
            per_slot.append(None)
        else:
            judged_task_keys.append(task)
            per_slot.append(verdict)
            if verdict == "yes":
                relevant_returned += 1
                if rank <= 20:
                    fixed_slot_relevant += 1
        if rank <= 20:
            fixed_slot_count += 1
        key = key_rows.get(task) or key_rows.get((task[0], None))
        if key is not None and key.get("stratum") == "unknown":
            context_missing += 1
        if row.get("context_complete") is False:
            context_truncated += 1

    judged_count = len(judged_task_keys)
    returned_count = len(per_slot)
    pool_relevant_total = sum(
        verdict == "yes"
        for (candidate_id, task_scenario), verdict in judged.items()
        if task_scenario == scenario_id and (candidate_id, task_scenario) in pool_keys
    )
    metrics: dict[str, Any] = {
        "scenario_id": scenario_id,
        "returned_count": returned_count,
        "judged_count": judged_count,
        "relevant_count": relevant_returned,
        "unjudged_count": len(unjudged_task_keys),
        "outside_pool_count": outside_pool_count,
        "precision_at_judged_count": (
            relevant_returned / judged_count if judged_count else None
        ),
        "precision_at_returned_count": (
            relevant_returned / returned_count
            if returned_count and judged_count == returned_count
            else None
        ),
        "judgment_coverage_at_returned_count": (
            judged_count / returned_count if returned_count else None
        ),
        "fixed_slot_p20_relevant": fixed_slot_relevant,
        "fixed_slot_p20": (
            fixed_slot_relevant / 20 if fixed_slot_count >= 20 else None
        ),
        "fixed_slot_p20_note": (
            "Fixed-slot P@20 uses the first 20 ranked display slots when they "
            "exist; unused or unjudged slots contribute no relevant count and "
            "are never invented negatives."
        ),
        "pool_relevant_count": pool_relevant_total,
        "pool_limited_recall": (
            relevant_returned / pool_relevant_total if pool_relevant_total else None
        ),
        "per_slot_verdicts": per_slot,
        "judged_task_keys": [
            {"candidate_id": candidate_id, "scenario_id": task_scenario}
            for candidate_id, task_scenario in judged_task_keys
        ],
        "unjudged_task_keys": [
            {"candidate_id": candidate_id, "scenario_id": task_scenario}
            for candidate_id, task_scenario in unjudged_task_keys
        ],
        "judged_candidate_ids": [candidate_id for candidate_id, _ in judged_task_keys],
        "unjudged_candidate_ids": [candidate_id for candidate_id, _ in unjudged_task_keys],
        "context_missing_count": context_missing,
        "context_truncated_count": context_truncated,
    }
    metrics["mean_reciprocal_rank"] = _mean_reciprocal_rank(
        run_rows,
        judged,
        pool_keys,
        split_keys,
        split_candidate_ids,
        scenario_id,
    )
    return metrics


def _mean_reciprocal_rank(
    run_rows: list[dict[str, Any]],
    judged: dict[TaskKey, str],
    pool_keys: set[TaskKey],
    split_keys: set[TaskKey] | None,
    split_candidate_ids: set[str] | None,
    scenario_id: str,
) -> float | None:
    """Reciprocal rank of the first relevant judged row; None when absent."""
    for position, row in enumerate(run_rows, start=1):
        task = _task_key(row, scenario_id)
        if split_keys is not None and task not in split_keys and (task[0], None) not in split_keys:
            continue
        rank = row.get("rank")
        rank = rank if isinstance(rank, int) and rank > 0 else position
        if task in pool_keys and judged.get(task) == "yes":
            return 1.0 / rank
    return None




def write_run_report(report: dict[str, Any], output_directory: Path) -> Path:
    """Atomically write the metrics report JSON."""
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / "run_metrics_report.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
