"""End-to-end discovery-review orchestration over in-process pipeline stages."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ingest.pilot import run_discovery
from .ingest.review_export import export_discovery_review_cards
from .ingest.sources import register_source_set
from .review.queue import build_balanced_review_queue
from .review.worksheet import export_review_worksheet

_RUN_MANIFEST_SCHEMA_VERSION = 1
_DISCOVERY_MANIFEST_SUFFIX = ".manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_discovery_review(
    *,
    sources_config: Path,
    rules_path: Path,
    output_directory: Path,
    snapshot_id: str,
    target: int = 30_000,
    max_records: int = 1_000_000,
    seed: int = 20_260_907,
    queue_limit: int = 200,
    runtime_limits: dict[str, int] | None = None,
    tombstones_path: Path | None = None,
) -> dict[str, Any]:
    """Chain sources register, discovery, cards, queue, and worksheet in-process.

    Each stage writes into a fresh subdirectory of ``output_directory`` and its
    result is recorded in the run manifest with the artifact SHA-256, keeping
    the provenance contract intact.
    """
    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(
            f"run output directory must be fresh or empty: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    limits = runtime_limits or {}
    stages: dict[str, Any] = {}

    registry_path = output_directory / "registry" / "source_registry.json"
    stages["sources_register"] = register_source_set(sources_config, registry_path)

    selection_path = output_directory / "discovery" / "discovery-selection.jsonl.zst"
    stages["ingest_discover"] = run_discovery(
        registry_path,
        rules_path,
        selection_path,
        target=target,
        seed=seed,
        max_records=max_records,
        max_staging_bytes=limits.get("max_staging_bytes", 4_294_967_296),
        minimum_free_disk_bytes=limits.get("minimum_free_disk_bytes", 5_368_709_120),
        max_process_rss_bytes=limits.get("max_process_rss_bytes", 2_147_483_648),
        tombstones_path=tombstones_path,
    )

    cards_directory = output_directory / "review-cards"
    stages["review_cards"] = export_discovery_review_cards(
        selection_path,
        cards_directory,
        snapshot_id=snapshot_id,
        tombstones_path=tombstones_path,
    )

    queue_directory = output_directory / "review-queue"
    stages["review_queue"] = build_balanced_review_queue(
        cards_directory / "review_cards.jsonl",
        queue_directory,
        limit=queue_limit,
    )

    worksheet_directory = output_directory / "review-worksheet"
    stages["review_worksheet"] = export_review_worksheet(
        queue_directory / "review_cards.jsonl",
        worksheet_directory,
    )

    manifest = {
        "kind": "run_manifest",
        "schema_version": _RUN_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "snapshot_id": snapshot_id,
        "sources_config": str(sources_config),
        "sources_config_sha256": _sha(sources_config),
        "rules_path": str(rules_path),
        "rules_sha256": _sha(rules_path),
        "options": {
            "target": target,
            "max_records": max_records,
            "seed": seed,
            "queue_limit": queue_limit,
        },
        "stages": {
            name: {**result, "_artifact_sha256": _artifact_hash(name, result)}
            for name, result in stages.items()
        },
    }
    manifest_path = output_directory / "run_manifest.json"
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)
    return {**manifest, "run_manifest": str(manifest_path)}


def _artifact_hash(stage: str, result: dict[str, Any]) -> str | None:
    candidates = {
        "sources_register": ("registry_file",),
        "ingest_discover": ("output_file", "selection_file", "output_path"),
        "review_cards": ("cards_file",),
        "review_queue": ("queue_file", "cards_file"),
        "review_worksheet": ("worksheet_file",),
    }
    for key in candidates.get(stage, ()):
        value = result.get(key)
        if isinstance(value, str):
            path = Path(value)
            if path.is_file():
                return _sha(path)
    # Discovery writes its manifest next to the selection shard.
    selection = result.get("output_file") or result.get("selection_file")
    if isinstance(selection, str):
        manifest = Path(selection + _DISCOVERY_MANIFEST_SUFFIX)
        if manifest.is_file():
            return _sha(manifest)
    return None
