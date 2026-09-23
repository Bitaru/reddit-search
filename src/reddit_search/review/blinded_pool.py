"""Convert a frozen blinded evaluation pool into reviewer cards."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reddit_search.ingest.state import file_sha256

from .queue import _write_json, _write_jsonl


def convert_blinded_pool(
    pool_path: Path, output_directory: Path, *, pool_manifest: Path | None = None
) -> dict[str, object]:
    rows = _read_pool(pool_path)
    ranked = [row for row in rows if row["scenario_id"] is not None]
    controls = [row for row in rows if row["scenario_id"] is None]
    ordered: list[dict[str, Any]] = []
    while ranked or controls:
        if ranked:
            ordered.append(ranked.pop(0))
        if controls:
            ordered.append(controls.pop(0))
    cards = [_card(row, index) for index, row in enumerate(ordered, 1)]
    cards_file = _write_jsonl(output_directory / "review_cards.jsonl", cards)
    manifest_data = _load_manifest(pool_manifest) if pool_manifest else {}
    ranked_total = sum(r["scenario_id"] is not None for r in rows)
    manifest = {
        "schema_version": 1,
        "kind": "blinded_pool_review_cards",
        "pool_file": str(pool_path),
        "pool_sha256": file_sha256(pool_path),
        "pool_count": len(rows),
        "ranked_count": len([r for r in rows if r["scenario_id"] is not None]),
        "control_count": len([r for r in rows if r["scenario_id"] is None]),
        "ordering": "alternating ranked and explicit control/audit rows, frozen within strata",
        "selection_schema": "blinded_pool_v2",
        "rubric_schema": "review_annotation_v1",
        "counts_by_stratum": {"ranked": ranked_total, "control_audit": len(rows) - ranked_total},
        "source_bindings": {
            key: manifest_data[key]
            for key in (
                "corpus_path",
                "corpus_sha256",
                "comparison_sha256",
                "comparison_identity_sha256",
                "profiles",
                "schema_version",
                "seed",
            )
            if key in manifest_data
        },
    }
    manifest_file = _write_json(output_directory / "review_cards_manifest.json", manifest)
    return {
        "card_count": len(cards),
        "cards_file": str(cards_file),
        "manifest_file": str(manifest_file),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("pool manifest must be an object")
    return value


def _read_pool(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str | None]] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid blinded pool row {number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"blinded pool row {number} must be an object")
        candidate = row.get("candidate_id")
        scenario = row.get("scenario_id")
        if not isinstance(candidate, str) or not candidate:
            raise ValueError("pool candidate_id must be non-empty")
        if scenario is not None and (not isinstance(scenario, str) or not scenario):
            raise ValueError("pool scenario_id must be non-empty or null")
        source = row.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("source_revision_id"), str):
            raise ValueError("pool source must contain source_revision_id")
        key = (candidate, scenario)
        if key in keys:
            raise ValueError("duplicate candidate/scenario task")
        keys.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("blinded pool is empty")
    return rows


def _card(row: dict[str, Any], order: int) -> dict[str, Any]:
    source = row["source"]
    control = row["scenario_id"] is None
    identity_keys = (
        "candidate_id",
        "scenario_id",
        "snapshot_id",
        "app_id",
        "app_profile_version",
        "app_profile_sha256",
        "context_complete",
    )
    card = {key: row[key] for key in identity_keys}
    rule_id = "control_audit" if control else "ranked_pool"
    card.update(
        {
            "source": source,
            "thread_fullname": row.get("thread_fullname"),
            "selection": {"selection_order": order, "matched_rule_ids": [rule_id]},
            "review_queue": {"queue_order": order, "stratum_rule_id": rule_id},
        }
    )
    card["source_revision_id"] = source["source_revision_id"]
    card["context_recipe_version"] = source.get("context_recipe_version")
    return card
