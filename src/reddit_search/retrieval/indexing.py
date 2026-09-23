"""Build a pinned local dense index from the frozen lexical corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reddit_search.config import load_runtime_config
from reddit_search.corpus.sqlite_store import LexicalStore
from reddit_search.corpus.units import unit_content_hash
from reddit_search.retrieval.dense import (
    DenseIndexJobStore,
    EmbeddingRecipe,
    QdrantHttpIndex,
    write_index_manifest,
)
from reddit_search.retrieval.qwen import QwenEmbeddingAdapter

DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def build_dense_index(
    corpus_path: Path,
    runtime_path: Path,
    manifest_path: Path,
    *,
    snapshot_id: str,
    cache_dir: str | Path | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    job_store_path: Path | None = None,
) -> dict[str, Any]:
    """Encode pending frozen units and reconcile the full Qdrant point count."""
    runtime = load_runtime_config(runtime_path)
    recipe = EmbeddingRecipe.from_model_settings(
        runtime.models,
        query_instruction=DEFAULT_QUERY_INSTRUCTION,
    )
    with LexicalStore(corpus_path) as store:
        units = store.units(snapshot_id=snapshot_id)
    if not units:
        raise ValueError(f"corpus has no units for snapshot {snapshot_id}: {corpus_path}")
    effective_batch_size = batch_size or runtime.models.batch_size
    effective_device = device or runtime.models.device
    index = QdrantHttpIndex.for_snapshot(
        runtime.qdrant.url,
        snapshot_id=snapshot_id,
        recipe=recipe,
    )
    index.ensure_collection()
    documents = [
        (
            unit.unit_id,
            getattr(unit, recipe.document_text_field),
            {
                "snapshot_id": unit.snapshot_id,
                "unit_id": unit.unit_id,
                "focus_field": unit.focus_field,
                "focus_start": unit.focus_start,
                "focus_end": unit.focus_end,
                "message_fullname": unit.message_fullname,
                "source_revision_id": unit.source_revision_id,
                "thread_fullname": unit.thread_fullname,
                "focus_text": unit.focus_text,
                "context_text": unit.context_text,
                "chunking_version": unit.chunking_version,
                "context_recipe_version": unit.context_recipe_version,
                "context_missing": unit.context_missing,
                "permalink": unit.permalink,
                "subreddit": unit.subreddit,
                "created_utc": unit.created_utc,
            },
        )
        for unit in units
    ]
    job_path = job_store_path or Path("reports/dense-index-jobs.sqlite")
    with DenseIndexJobStore(job_path) as jobs:
        pending_ids = set(
            jobs.prepare(
                snapshot_id=snapshot_id,
                recipe=recipe,
                units=[(unit.unit_id, unit_content_hash(unit)) for unit in units],
            )
        )
        pending_documents = [document for document in documents if document[0] in pending_ids]
        indexed_count = 0

        def mark_batch(unit_ids: list[str] | tuple[str, ...]) -> None:
            for unit_id in unit_ids:
                jobs.mark_indexed(snapshot_id=snapshot_id, recipe=recipe, unit_id=unit_id)

        if pending_documents:
            adapter = QwenEmbeddingAdapter(
                recipe,
                device=effective_device,
                batch_size=effective_batch_size,
                cache_dir=cache_dir,
                max_length=runtime.context.retrieval_context_tokens,
            )
            indexed_count = index.index_documents(
                adapter,
                pending_documents,
                batch_size=effective_batch_size,
                on_batch_indexed=mark_batch,
            )
        actual_count = index.count()
        index.assert_count(len(documents))
    manifest = write_index_manifest(
        manifest_path,
        snapshot_id=snapshot_id,
        recipe=recipe,
        expected_count=len(documents),
        actual_count=actual_count,
    )
    return {
        **manifest,
        "collection": index.collection,
        "runtime_path": str(runtime_path),
        "batch_size": effective_batch_size,
        "device": effective_device,
        "job_store_path": str(job_path),
        "pending_count": len(pending_documents),
        "indexed_count": indexed_count,
    }


def write_index_result(path: Path, result: dict[str, Any]) -> None:
    """Persist the operator-facing indexing result without partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
