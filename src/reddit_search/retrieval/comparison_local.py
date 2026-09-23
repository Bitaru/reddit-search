"""Run the pinned local Qwen A/B/C comparison on development scenarios."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reddit_search.config import configuration_hash, load_runtime_config, load_scenarios
from reddit_search.corpus.sqlite_store import LexicalStore
from reddit_search.ingest.invalidation import (
    TombstoneLedger,
    row_has_structured_context_identity,
    tombstone_blocks_identity,
    tombstone_blocks_row,
)
from reddit_search.retrieval.comparison import canonical_hash, load_comparison_config

from .comparison_run import ComparisonRun, ComparisonScenario, run_comparison
from .dense import EmbeddingRecipe, QdrantHttpIndex
from .feedback import FeedbackExample
from .fusion import RankedCandidate
from .indexing import DEFAULT_QUERY_INSTRUCTION
from .qwen import QwenEmbeddingAdapter, QwenRerankerAdapter

VERIFIED_REGISTRY_COLLECTION = "reddit_dense_e82fe3dcb467ee9e40b50b32-ctx2"


def _resolve_dense_collection(override: str | None) -> str | None:
    if override is None:
        return None
    if override != VERIFIED_REGISTRY_COLLECTION:
        raise ValueError(
            "dense_collection override is not the verified registry collection: "
            f"{VERIFIED_REGISTRY_COLLECTION}"
        )
    return override


def run_local_development_comparison(
    corpus_path: Path,
    runtime_path: Path,
    comparison_path: Path,
    labels_path: Path,
    output_path: Path,
    *,
    snapshot_id: str,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    reranker_batch_size: int = 1,
    seed_cards_path: Path | None = None,
    tombstone_ledger: TombstoneLedger | None = None,
    allow_test: bool = False,
    dense_collection: str | None = None,
) -> dict[str, Any]:
    """Run local Qwen A/B/C with development seeds."""
    dense_collection = _resolve_dense_collection(dense_collection)
    runtime = load_runtime_config(runtime_path)
    comparison = load_comparison_config(comparison_path)
    if comparison.snapshot_id != snapshot_id:
        raise ValueError(
            f"snapshot mismatch: comparison uses {comparison.snapshot_id}, requested {snapshot_id}"
        )
    recipe = EmbeddingRecipe.from_model_settings(
        runtime.models,
        query_instruction=DEFAULT_QUERY_INSTRUCTION,
    )
    dense_index = (
        QdrantHttpIndex(
            runtime.qdrant.url,
            collection=dense_collection,
            recipe=recipe,
        )
        if dense_collection is not None
        else QdrantHttpIndex.for_snapshot(
            runtime.qdrant.url,
            snapshot_id=snapshot_id,
            recipe=recipe,
        )
    )
    with LexicalStore(corpus_path) as store:
        units = store.units(snapshot_id=snapshot_id)
        if not units:
            raise ValueError(f"corpus has no units for snapshot {snapshot_id}: {corpus_path}")
        unit_by_id = {unit.unit_id: unit for unit in units}
        candidate_texts = {unit.unit_id: unit.context_text for unit in units}
        effective_device = device or runtime.models.device
        embedding = QwenEmbeddingAdapter(
            recipe,
            device=effective_device,
            batch_size=runtime.models.batch_size,
            cache_dir=cache_dir,
            max_length=runtime.context.retrieval_context_tokens,
        )
        reranker = QwenRerankerAdapter(
            runtime.models.reranker_id,
            runtime.models.reranker_revision or "",
            device=effective_device,
            batch_size=reranker_batch_size,
            cache_dir=cache_dir,
            max_length=runtime.context.evaluation_context_tokens,
        )
        positive_seeds, tombstoned_seed_cards, unverifiable_seed_cards = (
            _read_positive_seeds(
                labels_path, unit_by_id, seed_cards_path, tombstone_ledger=tombstone_ledger
            )
        )
        positive_seed_counts = {
            scenario_id: len(seeds) for scenario_id, seeds in positive_seeds.items()
        }
        seeded_scenarios: list[ComparisonScenario] = []
        skipped_scenarios: list[str] = []
        scenario_models = {
            scenario.scenario_id: scenario for scenario in load_scenarios(Path("configs/scenarios"))
        }
        for scenario_id in sorted(scenario_models):
            scenario = scenario_models[scenario_id]
            seeds = positive_seeds.get(scenario_id, [])
            if not seeds:
                skipped_scenarios.append(scenario_id)
                continue
            seeds = seeds[: runtime.feedback.max_positive_examples]
            vectors = embedding.encode_documents([item[1] for item in seeds])
            seeded_scenarios.append(
                ComparisonScenario(
                    scenario_id=scenario.scenario_id,
                    lexical_queries=tuple(scenario.lexical_queries),
                    semantic_queries=tuple(scenario.semantic_queries),
                    candidate_texts=candidate_texts,
                    feedback_examples=tuple(
                        FeedbackExample(
                            candidate_id=candidate_id,
                            scenario_id=scenario_id,
                            label="yes",
                            vector=tuple(vector),
                        )
                        for (candidate_id, _), vector in zip(seeds, vectors, strict=True)
                    ),
                )
            )

        def lexical_search(query: str, limit: int) -> list[RankedCandidate]:
            return [
                RankedCandidate(
                    candidate_id=hit.unit.unit_id,
                    rank=hit.rank,
                    raw_score=hit.score,
                    payload=_unit_payload(hit.unit),
                )
                for hit in store.search(snapshot_id=snapshot_id, query=query, limit=limit)
            ]

        def feedback_search(
            vector: tuple[float, ...], limit: int, payload_filter: dict[str, str] | None
        ) -> list[RankedCandidate]:
            return [
                RankedCandidate(
                    candidate_id=hit.candidate_id,
                    rank=hit.rank,
                    raw_score=hit.score,
                    payload=hit.payload,
                )
                for hit in dense_index.search(vector, limit=limit, payload_filter=payload_filter)
            ]

        run = run_comparison(
            comparison,
            seeded_scenarios,
            lexical_search=lexical_search,
            dense_adapter=embedding,
            dense_backend=dense_index,
            reranker=reranker,
            feedback_search=feedback_search,
            allow_test=allow_test,
            rerank_cache={},
            tombstone_ledger=tombstone_ledger,
        )
    corpus_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    identity = {
        "config_hash": run.config_hash,
        "snapshot_id": run.snapshot_id,
        "split": run.split,
        "runtime_configuration_hash": configuration_hash(runtime),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "seed_cards_sha256": (
            hashlib.sha256(seed_cards_path.read_bytes()).hexdigest()
            if seed_cards_path else None
        ),
        "corpus_sha256": corpus_sha256,
        "dense_recipe": {
            "model_id": recipe.model_id,
            "revision": recipe.revision,
            "dimension": recipe.dimension,
            "query_instruction": recipe.query_instruction,
            "document_text_field": recipe.document_text_field,
            "context_recipe_version": recipe.context_recipe_version,
        },
        "scenario_ids": [scenario.scenario_id for scenario in seeded_scenarios],
        "positive_seed_counts": positive_seed_counts,
    }
    comparison_identity = canonical_hash(identity)
    variant_run_directories = _write_variant_run_artifacts(
        output_path.parent, run, comparison_identity
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **run.as_dict(),
        "scenario_ids": [scenario.scenario_id for scenario in seeded_scenarios],
        "positive_seed_counts": positive_seed_counts,
        "positive_seed_limit": runtime.feedback.max_positive_examples,
        "skipped_scenarios": skipped_scenarios,
        "labels_sha256": identity["labels_sha256"],
        "seed_cards_path": str(seed_cards_path) if seed_cards_path else None,
        "seed_cards_sha256": identity["seed_cards_sha256"],
        "tombstoned_seed_cards_skipped": tombstoned_seed_cards
        if tombstone_ledger is not None
        else 0,
        "unverifiable_seed_cards_skipped": unverifiable_seed_cards
        if tombstone_ledger is not None
        else 0,
        "runtime_configuration_hash": identity["runtime_configuration_hash"],
        "dense_collection": dense_index.collection,
        "dense_point_count": dense_index.count(),
        "device": effective_device,
        "reranker_batch_size": reranker_batch_size,
        "variant_run_directories": variant_run_directories,
        "comparison_identity": comparison_identity,
    }
    _atomic_json(output_path, payload)
    manifest = {
        "kind": "retrieval_comparison_manifest",
        "schema_version": 1,
        "identity": identity,
        "comparison_identity": comparison_identity,
        "report_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "artifacts": _artifact_manifest(variant_run_directories),
    }
    _atomic_json(output_path.parent / "comparison_manifest.json", manifest)
    return payload

def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact_manifest(directories: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant, directory in sorted(directories.items()):
        manifest = Path(directory) / "run_manifest.json"
        result[variant] = {"run_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    return result


def _write_variant_run_artifacts(
    base_dir: Path, run: ComparisonRun, parent_identity: str | None = None
) -> dict[str, str]:
    """Persist pool-compatible ranked rows for each comparison variant."""
    directories: dict[str, str] = {}
    variant_ids = sorted({result.variant_id for result in run.variants})
    for variant_id in variant_ids:
        directory = base_dir / f"runs-{variant_id.lower()}-{run.config_hash[:12]}"
        directory.mkdir(parents=True, exist_ok=True)
        scenario_runs: dict[str, dict[str, Any]] = {}
        results = sorted(
            (result for result in run.variants if result.variant_id == variant_id),
            key=lambda result: result.scenario_id,
        )
        for result in results:
            path = directory / f"{result.scenario_id}.jsonl"
            rows = [
                {
                    "candidate_id": candidate_id,
                    "rank": rank,
                    "retrieval": {
                        "branch": variant_id,
                        "scenario_id": result.scenario_id,
                    },
                    "variant_id": variant_id,
                }
                for rank, candidate_id in enumerate(result.candidate_ids, start=1)
            ]
            content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
            scenario_runs[result.scenario_id] = {
                "hit_count": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest = {
            "config_hash": run.config_hash,
            "kind": "qwen_comparison_variant_run_manifest",
            "schema_version": 1,
            "scenario_runs": scenario_runs,
            "parent_comparison_identity": parent_identity,
            "parent_comparison_manifest": "comparison_manifest.json",
            "snapshot_id": run.snapshot_id,
            "variant_id": variant_id,
        }
        manifest_path = directory / "run_manifest.json"
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(manifest_path)
        directories[variant_id] = str(directory)
    union_directory = base_dir / f"runs-union-{run.config_hash[:12]}"
    union_directory.mkdir(parents=True, exist_ok=True)
    union_runs: dict[str, dict[str, Any]] = {}
    for result in sorted(run.variants, key=lambda item: (item.variant_id, item.scenario_id)):
        source = Path(directories[result.variant_id]) / f"{result.scenario_id}.jsonl"
        destination = union_directory / f"{result.variant_id}__{result.scenario_id}.jsonl"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(destination)
        union_runs[f"{result.variant_id}__{result.scenario_id}"] = {
            "hit_count": len(result.candidate_ids),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    union_manifest = {
        "config_hash": run.config_hash,
        "kind": "qwen_comparison_union_run_manifest",
        "schema_version": 1,
        "scenario_runs": union_runs,
        "snapshot_id": run.snapshot_id,
        "parent_comparison_identity": parent_identity,
        "parent_comparison_manifest": "comparison_manifest.json",
    }
    manifest_path = union_directory / "run_manifest.json"
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(union_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    directories["union"] = str(union_directory)
    return directories


def _read_positive_seeds(
    labels_path: Path,
    unit_by_id: dict[str, Any],
    seed_cards_path: Path | None = None,
    *,
    tombstone_ledger: TombstoneLedger | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Read human-positive seeds from the corpus or explicit dev source cards."""
    seed_cards = _read_seed_cards(seed_cards_path) if seed_cards_path else {}
    tombstoned_seed_cards = 0
    unverifiable_seed_cards = 0
    by_scenario: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line_number, line in enumerate(
        labels_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        scenario_id = row.get("scenario_id")
        candidate_id = row.get("candidate_id")
        if row.get("topic_fit") != "yes" or not isinstance(scenario_id, str):
            continue
        if not isinstance(candidate_id, str):
            raise ValueError(f"positive label line {line_number} has no candidate_id")
        unit = unit_by_id.get(candidate_id)
        if unit is not None:
            if tombstone_ledger is not None and tombstone_blocks_identity(
                tombstone_ledger,
                unit.message_fullname,
                unit.source_revision_id,
                unit.context_message_refs,
            ):
                continue
            text = unit.context_text
        else:
            card = seed_cards.get((candidate_id, scenario_id))
            if card is None:
                raise ValueError(
                    f"positive label line {line_number} is not a corpus unit "
                    "and has no matching seed card"
                )
            if tombstone_ledger is not None:
                if tombstone_blocks_row(tombstone_ledger, card):
                    tombstoned_seed_cards += 1
                    continue
                if not row_has_structured_context_identity(card):
                    # The ledger is identity-only: a card whose context is
                    # free text cannot prove its embedded contributors are
                    # still live, so it is never embedded.
                    unverifiable_seed_cards += 1
                    continue
            text = card["source"]["context_text"]
        by_scenario[scenario_id].append((candidate_id, text))
    return dict(by_scenario), tombstoned_seed_cards, unverifiable_seed_cards


def _read_seed_cards(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read source-bearing development cards keyed by exact task identity."""
    cards: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        candidate_id = row.get("candidate_id")
        scenario_id = row.get("scenario_id")
        source = row.get("source")
        context_text = source.get("context_text") if isinstance(source, dict) else None
        if (
            not isinstance(candidate_id, str)
            or not isinstance(scenario_id, str)
            or not isinstance(context_text, str)
            or not context_text.strip()
        ):
            raise ValueError(f"seed card line {line_number} lacks exact task source text")
        key = (candidate_id, scenario_id)
        if key in cards:
            raise ValueError(f"seed cards contain duplicate task {key}")
        cards[key] = row
    if not cards:
        raise ValueError(f"seed cards file is empty: {path}")
    return cards


def _unit_payload(unit: Any) -> dict[str, Any]:
    return {
        "snapshot_id": unit.snapshot_id,
        "message_fullname": unit.message_fullname,
        "source_revision_id": unit.source_revision_id,
        "context_message_refs": list(unit.context_message_refs),
        "thread_fullname": unit.thread_fullname,
        "focus_text": unit.focus_text,
        "context_text": unit.context_text,
        "context_missing": unit.context_missing,
        "permalink": unit.permalink,
        "subreddit": unit.subreddit,
        "created_utc": unit.created_utc,
    }
