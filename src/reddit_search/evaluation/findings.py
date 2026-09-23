"""Fail-closed findings export from reviewed label rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .labels import ProductFit, TopicFit

_FINDINGS_SCHEMA_VERSION = 1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path.name} line {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} line {line_number} must be an object")
        rows.append(value)
    return rows


def _card_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.jsonl"))


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def export_findings(
    labels_path: Path,
    cards_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Export judged-relevant findings, failing closed on incomplete input.

    Only label rows with ``topic_fit == "yes"`` and ``validation_status ==
    "valid"`` are exported. Every label must join to exactly one card by
    ``(candidate_id, scenario_id)``; an unmatched label is an error, never a
    silent skip. The manifest binds the SHA-256 of every input and the output.
    """
    labels_path = Path(labels_path)
    cards_path = Path(cards_path)
    output_directory = Path(output_directory)

    labels = _read_jsonl(labels_path)
    if not labels:
        raise ValueError(f"labels file is empty: {labels_path}")

    card_index: dict[tuple[str, str | None], dict[str, Any]] = {}
    for path in _card_files(cards_path):
        for card in _read_jsonl(path):
            candidate_id = _string(card.get("candidate_id"))
            if candidate_id is None:
                raise ValueError(f"card in {path.name} lacks candidate_id")
            key = (candidate_id, _string(card.get("scenario_id")))
            if key in card_index:
                raise ValueError(f"duplicate card for candidate {candidate_id}")
            card_index[key] = card

    findings: list[dict[str, Any]] = []
    skipped_uncertain = 0
    skipped_invalid = 0
    for index, label in enumerate(labels, 1):
        candidate_id = _string(label.get("candidate_id"))
        if candidate_id is None:
            raise ValueError(f"label row {index} lacks candidate_id")
        scenario_id = _string(label.get("scenario_id"))
        key = (candidate_id, scenario_id)
        card = card_index.get(key)
        if card is None:
            raise ValueError(
                f"label row {index} (candidate {candidate_id}) has no matching card"
            )

        validation_status = label.get("validation_status", "unknown")
        if validation_status != "valid":
            skipped_invalid += 1
            continue
        topic_fit = label.get("topic_fit")
        if topic_fit != TopicFit.YES.value:
            skipped_uncertain += 1
            continue

        source = card.get("source") if isinstance(card.get("source"), dict) else {}
        context = card.get("context") if isinstance(card.get("context"), dict) else {}
        selection = (
            card.get("selection") if isinstance(card.get("selection"), dict) else {}
        )
        retrieval = (
            card.get("retrieval") if isinstance(card.get("retrieval"), dict) else {}
        )
        product_fit = label.get("product_fit", ProductFit.NOT_EVALUATED.value)
        finding = {
            "finding_schema_version": _FINDINGS_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "scenario_id": scenario_id,
            "app_id": _string(label.get("app_id")),
            "snapshot_id": _string(label.get("snapshot_id")) or _string(card.get("snapshot_id")),
            "message_fullname": _string(source.get("message_fullname")),
            "thread_fullname": _string(card.get("thread_fullname"))
            or _string(source.get("thread_fullname")),
            "title": _string(source.get("title")),
            "focus_text": source.get("text"),
            "context_text": context.get("text"),
            "context_complete": context.get("context_complete"),
            "subreddit": _string(source.get("subreddit")),
            "created_utc": source.get("created_utc"),
            "permalink": _string(source.get("permalink")),
            "matched_rule_ids": selection.get("matched_rule_ids", []),
            "matched_queries": [retrieval["query"]] if _string(retrieval.get("query")) else [],
            "review_status": card.get("review_status", "unknown"),
            "topic_fit": topic_fit,
            "product_fit": product_fit,
            "supported_claim_ids": label.get("supported_claim_ids", []),
            "speaker_intent": label.get("speaker_intent"),
            "resolution_in_available_context": label.get("resolution_in_available_context"),
            "evidence": label.get("evidence", []),
            "source_revision_id": _string(source.get("source_revision_id")),
            "context_recipe_version": _string(source.get("context_recipe_version")),
        }
        if finding["message_fullname"] is None or finding["thread_fullname"] is None:
            raise ValueError(
                f"card for candidate {candidate_id} lacks message/thread identity"
            )
        findings.append(finding)

    output_directory.mkdir(parents=True, exist_ok=True)
    findings_path = output_directory / "findings.jsonl"
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in findings
    )
    temporary = findings_path.with_suffix(".jsonl.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(findings_path)

    manifest = {
        "kind": "findings_export_manifest",
        "schema_version": _FINDINGS_SCHEMA_VERSION,
        "labels_file": str(labels_path),
        "labels_sha256": _sha(labels_path),
        "cards_files": [str(path) for path in _card_files(cards_path)],
        "cards_sha256": {str(path): _sha(path) for path in _card_files(cards_path)},
        "label_row_count": len(labels),
        "card_count": len(card_index),
        "findings_count": len(findings),
        "skipped_uncertain": skipped_uncertain,
        "skipped_invalid": skipped_invalid,
        "findings_file": str(findings_path),
        "findings_sha256": _sha(findings_path),
    }
    manifest_path = output_directory / "findings_manifest.json"
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)

    return manifest
