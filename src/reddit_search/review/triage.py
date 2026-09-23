"""Deterministic triage of validated Luna drafts into human-review queues."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

TaskKey = tuple[str, str | None]


def build_luna_triage(
    cards_path: Path,
    drafts_path: Path,
    validation_report_path: Path,
    output_directory: Path,
    *,
    negative_audit_rate: float = 0.1,
    seed: int = 20260912,
) -> dict[str, Any]:
    """Route drafts without promoting any model result to a human label.

    Valid model positives and abstentions always enter ``human_review``.
    Valid model negatives enter a deterministic seeded audit sample; remaining
    negatives are only ``not_selected`` and are not treated as judgments.
    Missing or invalid drafts always enter ``human_review``.
    """
    if not 0.0 <= negative_audit_rate <= 1.0:
        raise ValueError("negative_audit_rate must be between 0 and 1 inclusive")
    cards = _read_jsonl(cards_path)
    drafts = _index_rows(_read_jsonl(drafts_path), cards, label="draft")
    validation_payload = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if not isinstance(validation_payload, dict):
        raise ValueError("validation report must be a JSON object")
    validation_rows = validation_payload.get("rows")
    if not isinstance(validation_rows, list):
        raise ValueError("validation report lacks rows")
    validations = _index_rows(validation_rows, cards, label="validation")

    triage_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for card in cards:
        task = _task_key(card)
        draft = drafts.get(task)
        validation = validations.get(task)
        if draft is None:
            status, reason = "human_review", "missing_draft"
            model_topic_fit = None
            validation_status = "missing"
        elif validation is None:
            status, reason = "human_review", "missing_validation"
            model_topic_fit = _topic_fit(draft)
            validation_status = "missing"
        elif validation.get("validation_status") != "valid":
            status, reason = "human_review", "invalid_evidence"
            model_topic_fit = _topic_fit(draft)
            validation_status = str(validation.get("validation_status"))
        else:
            model_topic_fit = _topic_fit(draft)
            validation_status = "valid"
            if model_topic_fit in {"relevant", "uncertain"}:
                status, reason = "human_review", f"model_{model_topic_fit}"
            elif model_topic_fit == "not_relevant":
                if _sampled(task, seed=seed, rate=negative_audit_rate):
                    status, reason = "negative_audit", "seeded_negative_audit"
                else:
                    status, reason = "not_selected", "validated_negative"
            else:
                status, reason = "human_review", "invalid_topic_fit"
        counts[status] += 1
        triage_rows.append(
            {
                "candidate_id": task[0],
                "scenario_id": task[1],
                "triage_status": status,
                "reason": reason,
                "validation_status": validation_status,
                "model_topic_fit": model_topic_fit,
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    triage_path = output_directory / "luna_triage.jsonl"
    triage_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in triage_rows),
        encoding="utf-8",
    )
    manifest = {
        "kind": "luna_triage_manifest",
        "schema_version": 1,
        "automatic_labeling": False,
        "negative_audit_rate": negative_audit_rate,
        "seed": seed,
        "cards_sha256": _sha(cards_path),
        "drafts_sha256": _sha(drafts_path),
        "validation_report_sha256": _sha(validation_report_path),
        "card_count": len(cards),
        "counts": dict(counts),
        "triage_file": triage_path.name,
    }
    manifest_path = output_directory / "luna_triage_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "triage_file": str(triage_path), "manifest_file": str(manifest_path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} must be a JSON object")
            rows.append(value)
    return rows


def _index_rows(
    rows: list[dict[str, Any]], cards: list[dict[str, Any]], *, label: str
) -> dict[TaskKey, dict[str, Any]]:
    card_tasks_by_candidate: dict[str, set[TaskKey]] = {}
    card_tasks: set[TaskKey] = set()
    for card in cards:
        task = _task_key(card)
        if task in card_tasks:
            raise ValueError("cards contain duplicate candidate/scenario tasks")
        card_tasks.add(task)
        card_tasks_by_candidate.setdefault(task[0], set()).add(task)
    indexed: dict[TaskKey, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        if candidate_id not in card_tasks_by_candidate:
            continue
        scenario_id = row.get("scenario_id")
        if scenario_id is None:
            candidates = card_tasks_by_candidate.get(candidate_id, set())
            if len(candidates) == 1:
                task = next(iter(candidates))
            else:
                raise ValueError(f"{label} row for {candidate_id} requires scenario_id")
        else:
            task = _task_key(row)
        if task in indexed:
            raise ValueError(f"{label} rows contain duplicate candidate/scenario task")
        indexed[task] = row
    return indexed


def _task_key(row: dict[str, Any]) -> TaskKey:
    candidate_id = row.get("candidate_id")
    scenario_id = row.get("scenario_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("triage rows require a non-empty candidate_id")
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
        raise ValueError("triage scenario_id must be a non-empty string or null")
    return candidate_id, scenario_id


def _topic_fit(row: dict[str, Any]) -> str | None:
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        return None
    value = annotation.get("topic_fit")
    return value if isinstance(value, str) else None


def _sampled(task: TaskKey, *, seed: int, rate: float) -> bool:
    digest = hashlib.sha256(f"{seed}\0{task[0]}\0{task[1] or ''}".encode()).hexdigest()
    return int(digest[:16], 16) / 2**64 < rate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
