"""Deterministically construct a small, rule-balanced human-review queue."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reddit_search.ingest.state import file_sha256, json_sha256

_IDENTITY_FIELDS = (
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
)


def build_balanced_review_queue(
    cards_path: Path, output_directory: Path, *, limit: int
) -> dict[str, object]:
    """Select unique cards across discovery rules without rereading source archives."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    cards = list(_read_cards(cards_path))
    cards_by_task = {_task_key(card): card for card in cards}
    if len(cards_by_task) != len(cards):
        raise ValueError("review cards contain duplicate candidate/scenario tasks")
    if not cards:
        raise ValueError("review cards are empty")

    candidates_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        for rule_id in _rule_ids(card):
            candidates_by_rule[rule_id].append(card)
    report_rule_ids = sorted(candidates_by_rule)
    for rule_id in report_rule_ids:
        candidates_by_rule[rule_id].sort(key=_selection_key)
    sampling_rule_ids = sorted(
        report_rule_ids,
        key=lambda rule_id: (len(candidates_by_rule[rule_id]), rule_id),
    )

    quota_by_rule = _quota_by_rule(sampling_rule_ids, limit)
    queued_cards: list[dict[str, Any]] = []
    selected_tasks: set[tuple[str, str | None]] = set()
    queued_count_by_rule = {rule_id: 0 for rule_id in report_rule_ids}
    for rule_id in sampling_rule_ids:
        for card in candidates_by_rule[rule_id]:
            if queued_count_by_rule[rule_id] >= quota_by_rule[rule_id]:
                break
            task = _task_key(card)
            if task in selected_tasks:
                continue
            selected_tasks.add(task)
            queued_count_by_rule[rule_id] += 1
            queued_cards.append(_queued_card(card, len(queued_cards) + 1, rule_id))

    if len(queued_cards) < limit:
        for card in sorted(cards, key=_selection_key):
            if len(queued_cards) >= limit:
                break
            task = _task_key(card)
            if task in selected_tasks:
                continue
            selected_tasks.add(task)
            queued_cards.append(_queued_card(card, len(queued_cards) + 1, None))

    cards_file = _write_jsonl(output_directory / "review_cards.jsonl", queued_cards)
    manifest_file = _write_json(
        output_directory / "queue_manifest.json",
        {
            "schema_version": 1,
            "sampling": "deterministic_rarest_rule_quota_by_selection_order",
            "input_cards_file": str(cards_path),
            "input_cards_sha256": file_sha256(cards_path),
            "input_card_count": len(cards),
            "requested_card_count": limit,
            "queued_card_count": len(queued_cards),
            "dependency_identity_sha256": _dependency_identity_sha256(cards),
            "strata": [
                {
                    "rule_id": rule_id,
                    "available_card_count": len(candidates_by_rule[rule_id]),
                    "queued_card_count": queued_count_by_rule[rule_id],
                }
                for rule_id in report_rule_ids
            ],
        },
    )
    return {
        "input_card_count": len(cards),
        "queued_card_count": len(queued_cards),
        "cards_file": str(cards_file),
        "manifest_file": str(manifest_file),
    }


def _read_cards(path: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                card = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid review card row {line_number}") from error
            if not isinstance(card, dict):
                raise ValueError(f"review card row {line_number} must be an object")
            _candidate_id(card)
            _selection_key(card)
            _rule_ids(card)
            cards.append(card)
    return cards


def _candidate_id(card: dict[str, Any]) -> str:
    candidate_id = card.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("review card candidate_id must be a non-empty string")
    return candidate_id


def _selection_key(card: dict[str, Any]) -> tuple[int, str, str]:
    selection = _selection(card)
    selection_order = selection.get("selection_order")
    if (
        isinstance(selection_order, bool)
        or not isinstance(selection_order, int)
        or selection_order < 1
    ):
        raise ValueError("review card selection_order must be a positive integer")
    candidate_id, scenario_id = _task_key(card)
    return selection_order, candidate_id, scenario_id or ""


def _task_key(card: dict[str, Any]) -> tuple[str, str | None]:
    candidate_id = _candidate_id(card)
    scenario_id = card.get("scenario_id")
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
        raise ValueError("review card scenario_id must be a non-empty string or null")
    return candidate_id, scenario_id


def _rule_ids(card: dict[str, Any]) -> list[str]:
    rule_ids = _selection(card).get("matched_rule_ids")
    if (
        not isinstance(rule_ids, list)
        or not rule_ids
        or not all(isinstance(rule_id, str) and rule_id for rule_id in rule_ids)
    ):
        raise ValueError("review card matched_rule_ids must be a non-empty string list")
    return rule_ids


def _selection(card: dict[str, Any]) -> dict[str, Any]:
    selection = card.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("review card must contain a selection object")
    return selection


def _quota_by_rule(rule_ids: list[str], limit: int) -> dict[str, int]:
    base, remainder = divmod(limit, len(rule_ids))
    return {rule_id: base + (index < remainder) for index, rule_id in enumerate(rule_ids)}


def _queued_card(
    card: dict[str, Any], queue_order: int, stratum_rule_id: str | None
) -> dict[str, Any]:
    queued = dict(card)
    queued["review_queue"] = {
        "queue_order": queue_order,
        "stratum_rule_id": stratum_rule_id,
    }
    return queued


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True))
            stream.write("\n")
    temporary.replace(path)
    return path


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _dependency_identity_sha256(cards: list[dict[str, Any]]) -> str:
    identities = [
        {field: card[field] for field in _IDENTITY_FIELDS if field in card}
        for card in cards
    ]
    return json_sha256(identities)
