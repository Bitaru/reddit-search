"""Persistent corpus build and deterministic lexical search runs (execution step 2).

Build indexes a frozen discovery-selection shard into a persistent
``LexicalStore`` using the same message, hydration, and unit construction paths
as the review exporter. Search runs every scenario's lexical queries against
one snapshot and writes atomically ranked, traceable result runs.

This is a diagnostic lexical baseline, not independent quality evaluation:
until hydration and grouped evaluation splits exist (execution step 3), results
carry the shard's known context-coverage limitations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reddit_search.config import load_scenarios
from reddit_search.corpus.sqlite_store import LexicalStore
from reddit_search.corpus.units import (
    CHUNKING_VERSION,
    CONTEXT_RECIPE_VERSION,
    build_message_units,
    estimate_focus_chunk_count,
)
from reddit_search.ingest.hydrate import ContextBuilder
from reddit_search.ingest.invalidation import load_tombstone_ledger, tombstone_identity
from reddit_search.ingest.review_export import _read_selected_messages
from reddit_search.ingest.state import atomic_write_json, file_sha256, json_sha256, load_json
from reddit_search.resources import (
    DEFAULT_MAX_STAGING_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    BudgetError,
    ResourceLimits,
    check_output_budget,
)

__all__ = ["BudgetError", "build_corpus_index", "run_lexical_scenarios"]

_CORPUS_STATE_SCHEMA_VERSION = 2


def _corpus_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".manifest.json")


_OUTPUT_BYTES_PER_UNIT = 8_192


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    scenario_id: str
    scenario_version: int
    queries: tuple[str, ...]
    query_hit_counts: tuple[int, ...]
    hits: list  # list[LexicalHit]; untyped to avoid a circular import concern is unnecessary


def build_corpus_index(
    selection_path: Path,
    output_path: Path,
    *,
    snapshot_id: str,
    tombstones_path: Path | None = None,
    context_shard_path: Path | None = None,
    context_recipe_version: str = CONTEXT_RECIPE_VERSION,
    chunking_version: str = CHUNKING_VERSION,
    max_staging_bytes: int = DEFAULT_MAX_STAGING_BYTES,
    minimum_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES,
    focus_chunk_tokens: int = 512,
    focus_overlap_tokens: int = 64,
) -> dict[str, Any]:
    """Build an atomic lexical corpus and reuse an identical completed build."""
    if not snapshot_id:
        raise ValueError("snapshot_id must not be empty")
    if not context_recipe_version.strip():
        raise ValueError("context_recipe_version must not be empty")
    resource_limits = ResourceLimits(
        max_staging_bytes=max_staging_bytes,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
    )
    resource_limits.validate()
    tombstones = load_tombstone_ledger(tombstones_path)
    manifest_path = _corpus_manifest_path(output_path)
    identity = {
        "stage": "corpus_build",
        "schema_version": _CORPUS_STATE_SCHEMA_VERSION,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": file_sha256(selection_path),
        "context_shard_path": (
            str(context_shard_path.resolve()) if context_shard_path is not None else None
        ),
        "context_shard_sha256": (
            file_sha256(context_shard_path) if context_shard_path is not None else None
        ),
        "snapshot_id": snapshot_id,
        "resource_limits": {
            "max_staging_bytes": resource_limits.max_staging_bytes,
            "minimum_free_disk_bytes": resource_limits.minimum_free_disk_bytes,
        },
        "tombstones": tombstone_identity(tombstones),
        "context_recipe_version": context_recipe_version,
        "chunking_version": chunking_version,
        "focus_chunk_tokens": focus_chunk_tokens,
        "focus_overlap_tokens": focus_overlap_tokens,
    }
    run_id = json_sha256(identity)
    if output_path.exists() or manifest_path.exists():
        if not output_path.exists() or not manifest_path.exists():
            raise ValueError("corpus output and manifest must be published together")
        existing_manifest = load_json(manifest_path)
        if existing_manifest.get("run_id") != run_id:
            raise ValueError("corpus manifest does not match current inputs or limits")
        if existing_manifest.get("corpus_sha256") != file_sha256(output_path):
            raise ValueError("corpus output does not match its manifest")
        result = existing_manifest.get("result")
        if not isinstance(result, dict):
            raise ValueError("complete corpus manifest has no result")
        return result
    selected_candidates = list(_read_selected_messages(selection_path))
    tombstoned_count = sum(
        tombstones.matches(message.fullname, message.source_revision_id)
        for message in selected_candidates
    )
    messages = [
        message
        for message in selected_candidates
        if not tombstones.matches(message.fullname, message.source_revision_id)
    ]
    if not selected_candidates:
        raise ValueError(f"selection shard is empty: {selection_path}")
    messages_by_fullname = {message.fullname: message for message in messages}
    if len(messages_by_fullname) != len(messages):
        raise ValueError("selection shard contains duplicate message fullnames")

    context_added = 0
    if context_shard_path is not None:
        for message in _read_selected_messages(context_shard_path):
            if tombstones.matches(message.fullname, message.source_revision_id):
                tombstoned_count += 1
                continue
            if message.fullname not in messages_by_fullname:
                context_added += 1
            messages_by_fullname[message.fullname] = message

    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimated_units = sum(
        estimate_focus_chunk_count(
            (message.raw_title or "") + "\n" + (message.raw_body or ""),
            max_tokens=focus_chunk_tokens,
            overlap_tokens=focus_overlap_tokens,
        )
        for message in messages
    )
    estimated_bytes = estimated_units * _OUTPUT_BYTES_PER_UNIT
    check_output_budget(
        output_path.parent,
        estimated_bytes=estimated_bytes,
        limits=resource_limits,
    )

    context_builder = ContextBuilder(messages_by_fullname)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    emitted_units = 0

    def build_units() -> Iterator[Any]:
        nonlocal emitted_units
        for message in messages:
            units = build_message_units(
                snapshot_id,
                context_builder.build(message.fullname),
                context_recipe_version=context_recipe_version,
                chunking_version=chunking_version,
                synthetic=False,
                max_tokens=focus_chunk_tokens,
                overlap_tokens=focus_overlap_tokens,
            )
            emitted_units += len(units)
            yield from units

    try:
        with LexicalStore(temporary_path) as store:
            store.index_units(build_units())
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    result = {
        "snapshot_id": snapshot_id,
        "indexed_unit_count": emitted_units,
        "tombstoned_count": tombstoned_count,
        "context_message_count": context_added,
        "corpus_path": str(output_path),
        "manifest_path": str(manifest_path),
        "run_id": run_id,
        "unit_ids_stable": True,
        "context_recipe_version": context_recipe_version,
        "chunking_version": chunking_version,
    }
    atomic_write_json(
        manifest_path,
        {
            "kind": "corpus_build_manifest",
            "schema_version": _CORPUS_STATE_SCHEMA_VERSION,
            "run_id": run_id,
            "run_identity": identity,
            "corpus_sha256": file_sha256(output_path),
            "result": result,
        },
    )
    return result


def run_lexical_scenarios(
    corpus_path: Path,
    scenarios_path: Path,
    output_path: Path,
    *,
    snapshot_id: str,
    candidates_per_scenario: int,
    output_limit: int,
) -> dict[str, Any]:
    """Run every scenario's lexical queries and persist deterministic rankings.

    Hits merge across a scenario's queries with a deterministic tie-break on
    (score, unit_id): the best score per unit wins, ranked ascending like the
    store's BM25 ordering. The run is written atomically with input identities
    so it is traceable to exact frozen inputs.
    """
    if candidates_per_scenario <= 0 or output_limit <= 0:
        raise ValueError("candidate and output limits must be positive")
    scenarios = load_scenarios(scenarios_path)
    if not scenarios:
        raise ValueError(f"no scenarios found in {scenarios_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    run_sha256s: dict[str, str] = {}
    total_hits = 0

    for scenario in sorted(scenarios, key=lambda item: item.scenario_id):
        with LexicalStore(corpus_path) as store:
            merged: dict[str, Any] = {}
            for query in scenario.lexical_queries:
                for hit in store.search(
                    snapshot_id=snapshot_id,
                    query=query,
                    limit=candidates_per_scenario,
                ):
                    existing = merged.get(hit.unit.unit_id)
                    if existing is None or hit.score < existing.score:
                        merged[hit.unit.unit_id] = hit
            hits = sorted(merged.values(), key=lambda hit: (hit.score, hit.unit.unit_id))
            hits = hits[:output_limit]
            total_hits += len(hits)

            destination = output_path / f"{scenario.scenario_id}.jsonl"
            temporary = destination.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for hit in hits:
                    stream.write(
                        json.dumps(
                            _run_row(
                                hit,
                                scenario_id=scenario.scenario_id,
                                app_id=scenario.app_id,
                            ),
                            sort_keys=True,
                        )
                    )
                    stream.write("\n")
            temporary.replace(destination)
            run_sha256s[scenario.scenario_id] = hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()

    hit_counts = _read_run_hit_counts(output_path)
    manifest = {
        "kind": "lexical_baseline_run_manifest",
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "corpus_path": str(corpus_path),
        "candidates_per_query": candidates_per_scenario,
        "output_limit_per_scenario": output_limit,
        "scenario_runs": {
            scenario_id: {"hit_count": hit_counts.get(scenario_id, 0), "sha256": digest}
            for scenario_id, digest in run_sha256s.items()
        },
    }
    manifest_path = output_path / "run_manifest.json"
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)

    return {
        "snapshot_id": snapshot_id,
        "scenario_count": len(scenarios),
        "total_output_hits": total_hits,
        "runs_directory": str(output_path),
        "scenario_sha256": run_sha256s,
        "run_manifest": str(manifest_path),
    }


def _read_run_hit_counts(output_path: Path) -> dict[str, int]:
    return {
        path.stem: sum(1 for line in path.open(encoding="utf-8") if line.strip())
        for path in sorted(output_path.glob("*.jsonl"))
    }


def _run_row(hit: Any, *, scenario_id: str, app_id: str | None) -> dict[str, Any]:
    unit = hit.unit
    return {
        "candidate_id": unit.unit_id,
        "rank": hit.rank,
        "score": hit.score,
        "app_id": app_id,
        "message_fullname": unit.message_fullname,
        "thread_fullname": unit.thread_fullname,
        "focus_text": unit.focus_text,
        "permalink": unit.permalink,
        "subreddit": unit.subreddit,
        "created_utc": unit.created_utc,
        "context_complete": not unit.context_missing,
        "retrieval": {
            "branch": "lexical",
            "scenario_id": scenario_id,
        },
    }
