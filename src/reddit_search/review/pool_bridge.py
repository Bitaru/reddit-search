"""Convert blinded pool rows into deterministic review cards."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from reddit_search.ingest.state import file_sha256

_FORBIDDEN = {"rank", "score", "variant_id", "retrieval", "stratum"}
_REQUIRED = (
    "pool_id", "candidate_id", "scenario_id", "app_id", "app_profile_version",
    "app_profile_sha256", "thread_fullname", "snapshot_id", "context_complete", "source",
)


def convert_blinded_pool(
    pool_path: Path, output_directory: Path, *, seed: int | None = None
) -> dict[str, object]:
    """Convert pool JSONL to a deterministic, interleaved review-card queue."""
    rows: list[dict[str, Any]] = []
    seen_tasks: set[tuple[str, str | None]] = set()
    with pool_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{pool_path}:{line_number} must be an object")
            missing = [field for field in _REQUIRED if field not in value]
            if missing:
                raise ValueError(f"{pool_path}:{line_number} missing {', '.join(missing)}")
            pool_id = value["pool_id"]
            candidate_id = value["candidate_id"]
            scenario_id = value["scenario_id"]
            if not isinstance(pool_id, str) or not pool_id:
                raise ValueError(f"{pool_path}:{line_number} pool_id must be a non-empty string")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(
                    f"{pool_path}:{line_number} candidate_id must be a non-empty string"
                )
            if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
                raise ValueError(
                    f"{pool_path}:{line_number} scenario_id must be a non-empty string or null"
                )
            task = (candidate_id, scenario_id)
            if task in seen_tasks:
                raise ValueError(
                    f"{pool_path}:{line_number} duplicate candidate/scenario task {task!r}"
                )
            seen_tasks.add(task)
            rows.append(value)
    if not rows:
        raise ValueError(f"{pool_path} contains no pool rows")

    ranked = [row for row in rows if row["scenario_id"] is not None]
    controls = [row for row in rows if row["scenario_id"] is None]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(ranked)
        rng.shuffle(controls)
    ordered: list[tuple[dict[str, Any], str]] = []
    ri = ci = 0
    while ri < len(ranked) or ci < len(controls):
        if ri < len(ranked):
            ordered.append((ranked[ri], "operational_ranked"))
            ri += 1
        if ci < len(controls):
            ordered.append((controls[ci], "operational_control"))
            ci += 1

    cards: list[dict[str, Any]] = []
    for queue_order, (row, stratum) in enumerate(ordered, 1):
        card = {key: row.get(key) for key in _REQUIRED}
        card.update(
            {
                "schema_version": 1,
                "selection": {"matched_rule_ids": [stratum]},
                "review_queue": {"queue_order": queue_order, "stratum_rule_id": stratum},
                "annotation": {
                    "review_status": "pending",
                    "topic_fit": "unreviewed",
                    "speaker_intent": "unreviewed",
                    "product_fit": "not_evaluated",
                    "resolution_in_available_context": "unreviewed",
                    "need_clarity": "unreviewed",
                    "duplicate_of": None,
                    "reviewer_note": None,
                },
            }
        )
        if stratum == "operational_control":
            for key in ("scenario_id", "app_id", "app_profile_version", "app_profile_sha256"):
                card[key] = None
        cards.append(card)

    output_directory.mkdir(parents=True, exist_ok=True)
    cards_path = output_directory / "review_cards.jsonl"
    with cards_path.open("w", encoding="utf-8", newline="\n") as stream:
        for card in cards:
            stream.write(json.dumps(card, ensure_ascii=False, sort_keys=True) + "\n")
    order_note = (
        "seeded_round_robin_ranked_control"
        if seed is not None
        else "stable_round_robin_ranked_control"
    )
    manifest = {
        "schema_version": 1,
        "pool_sha256": file_sha256(pool_path),
        "card_sha256": file_sha256(cards_path),
        "pool_row_count": len(rows),
        "card_row_count": len(cards),
        "ranked_count": len(ranked),
        "control_count": len(controls),
        "order": order_note,
    }
    (output_directory / "pool_bridge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
