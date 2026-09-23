"""Export annotation stubs for a bounded human-review queue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reddit_search.ingest.state import file_sha256

from .queue import (
    _IDENTITY_FIELDS,
    _candidate_id,
    _dependency_identity_sha256,
    _rule_ids,
    _write_json,
    _write_jsonl,
)

_ANNOTATION_FIELDS = {
    "duplicate_of": "candidate_id or null",
    "need_clarity": ["clear", "unclear", "unreviewed"],
    "review_status": ["complete", "pending"],
    "reviewer_note": "string or null",
    "topic_fit": ["not_relevant", "relevant", "uncertain", "unreviewed"],
    "speaker_intent": [
        "seeking_solution",
        "describing_pain",
        "sharing_experience",
        "answering_someone",
        "self_promotion",
        "quoting",
        "unknown",
        "unreviewed",
    ],
    "product_fit": ["compatible", "incompatible", "needs_clarification", "not_evaluated"],
    "supported_claim_ids": "list of verified claim IDs (required for compatible)",
    "resolution_in_available_context": [
        "resolved",
        "no_resolution_observed",
        "unknown",
        "unreviewed",
    ],
}


def export_review_worksheet(cards_path: Path, output_directory: Path) -> dict[str, object]:
    """Write pending annotations that reference, rather than copy, queue evidence."""
    cards = _read_queued_cards(cards_path)
    rows = [_worksheet_row(card) for card in cards]
    worksheet_file = _write_jsonl(output_directory / "review_worksheet.jsonl", rows)
    manifest_file = _write_json(
        output_directory / "worksheet_manifest.json",
        {
            "schema_version": 1,
            "review_cards_file": str(cards_path),
            "review_cards_sha256": file_sha256(cards_path),
            "input_card_count": len(cards),
            "worksheet_row_count": len(rows),
            "dependency_identity_sha256": _dependency_identity_sha256(cards),
            "annotation_fields": _ANNOTATION_FIELDS,
        },
    )
    return {
        "input_card_count": len(cards),
        "worksheet_row_count": len(rows),
        "worksheet_file": str(worksheet_file),
        "manifest_file": str(manifest_file),
    }


def _read_queued_cards(path: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    task_keys: set[tuple[str, str | None]] = set()
    queue_orders: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                card = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid review card row {line_number}") from error
            if not isinstance(card, dict):
                raise ValueError(f"review card row {line_number} must be an object")
            candidate_id = _candidate_id(card)
            scenario_id = card.get("scenario_id")
            task = (candidate_id, scenario_id)
            if task in task_keys:
                raise ValueError("review cards contain duplicate candidate/scenario tasks")
            task_keys.add(task)
            _rule_ids(card)
            queue_order, _ = _review_queue(card)
            if queue_order in queue_orders:
                raise ValueError("review cards contain duplicate queue order")
            queue_orders.add(queue_order)
            cards.append(card)
    if not cards:
        raise ValueError("review cards are empty")
    return sorted(cards, key=lambda card: _review_queue(card)[0])


def _review_queue(card: dict[str, Any]) -> tuple[int, str | None]:
    review_queue = card.get("review_queue")
    if not isinstance(review_queue, dict):
        raise ValueError("review card must contain a review_queue object")
    queue_order = review_queue.get("queue_order")
    if isinstance(queue_order, bool) or not isinstance(queue_order, int) or queue_order < 1:
        raise ValueError("review card queue_order must be a positive integer")
    stratum_rule_id = review_queue.get("stratum_rule_id")
    if stratum_rule_id is not None and (
        not isinstance(stratum_rule_id, str) or not stratum_rule_id
    ):
        raise ValueError("review card stratum_rule_id must be a non-empty string or null")
    return queue_order, stratum_rule_id


def _worksheet_row(card: dict[str, Any]) -> dict[str, object]:
    queue_order, stratum_rule_id = _review_queue(card)
    row: dict[str, object] = {
        "schema_version": 1,
        "candidate_id": _candidate_id(card),
        "scenario_id": card.get("scenario_id"),
        "review_queue": {
            "queue_order": queue_order,
            "stratum_rule_id": stratum_rule_id,
        },
        "selection": {"matched_rule_ids": _rule_ids(card)},
        "annotation": {
            "review_status": "pending",
            "topic_fit": "unreviewed",
            "speaker_intent": card.get("speaker_intent", "unreviewed"),
            "product_fit": card.get("product_fit", "not_evaluated"),
            "supported_claim_ids": card.get("supported_claim_ids", []),
            "resolution_in_available_context": card.get(
                "resolution_in_available_context", "unreviewed"
            ),
            "need_clarity": "unreviewed",
            "duplicate_of": None,
            "reviewer_note": None,
        },
    }
    for field in _IDENTITY_FIELDS:
        if field not in {"candidate_id", "scenario_id"} and field in card:
            row[field] = card[field]
    return row
