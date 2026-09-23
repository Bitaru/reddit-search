"""Additive dense-payload migration: legacy collections gain context refs.

Copies vectors from an existing dense collection into a new collection whose
payloads add a sorted ``context_message_refs`` key joined from corpus SQLite.
No re-embedding happens: vectors pass through untouched. The source
collection is never modified, and the new collection is only created after
every source point id has been proven to join the corpus (fail closed).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from reddit_search.retrieval.dense import (
    DenseBackendError,
    DensePoint,
    EmbeddingRecipe,
    QdrantHttpIndex,
)

LEGACY_PAYLOAD_KEY = "context_message_refs"


class DenseMigrationError(RuntimeError):
    """A dense payload migration could not satisfy its explicit contract."""

    def __init__(self, message: str, *, misses: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.misses = list(misses)


def load_refs_by_unit(corpus_path: Path | str, *, snapshot_id: str) -> dict[str, list[str]]:
    """Load unit_id -> sorted unique context refs for one snapshot, read-only."""
    uri = f"file:{Path(corpus_path)}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT unit_id, context_message_refs FROM search_units WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchall()
    refs: dict[str, list[str]] = {}
    for unit_id, raw in rows:
        loaded = json.loads(str(raw))
        if not isinstance(loaded, list) or not all(isinstance(ref, str) for ref in loaded):
            raise DenseMigrationError(
                f"corpus row {unit_id} has non-list or non-string context_message_refs"
            )
        refs[str(unit_id)] = sorted(set(loaded))
    return refs


def _scroll_points(
    index: QdrantHttpIndex,
    *,
    batch_size: int,
    with_payload: bool,
    with_vector: bool,
    snapshot_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield raw scroll points with offset pagination; bounded memory per batch."""
    offset: Any = None
    while True:
        body: dict[str, Any] = {
            "limit": batch_size,
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if offset is not None:
            body["offset"] = offset
        if snapshot_id is not None:
            body["filter"] = {"must": [{"key": "snapshot_id", "match": {"value": snapshot_id}}]}
        response = index._request("POST", f"{index.collection_path}/points/scroll", body)
        if response.get("status") != "ok":
            raise DenseBackendError("Qdrant scroll response status is not ok")
        result = response.get("result")
        points = result.get("points") if isinstance(result, dict) else result
        if not isinstance(points, list):
            raise DenseBackendError("Qdrant scroll response lacks a point list")
        yield from points
        next_offset = result.get("next_page_offset") if isinstance(result, dict) else None
        if next_offset is None:
            return
        offset = next_offset


def _target_collection_exists(index: QdrantHttpIndex) -> bool:
    try:
        index._request("GET", index.collection_path)
    except DenseBackendError as error:
        if error.status == 404:
            return False
        raise
    return True


def _existing_target_count(index: QdrantHttpIndex) -> int | None:
    """Return the target's point count, or None when it does not exist yet."""
    if not _target_collection_exists(index):
        return None
    return index.count()


def _build_target(index: QdrantHttpIndex, new_collection: str) -> QdrantHttpIndex:
    return QdrantHttpIndex(
        index._base_url,
        collection=new_collection,
        recipe=index.recipe,
        timeout_seconds=index.timeout_seconds,
    )


def _dense_points_from_batch(
    batch: Sequence[Mapping[str, Any]], refs_by_unit: Mapping[str, list[str]]
) -> list[DensePoint]:
    points: list[DensePoint] = []
    for raw in batch:
        unit_id = raw.get("id")
        if not isinstance(unit_id, str):
            raise DenseBackendError("Qdrant scroll point lacks a string id")
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            raise DenseBackendError(f"Qdrant point {unit_id} lacks a payload object")
        vector = raw.get("vector")
        if not isinstance(vector, list):
            raise DenseBackendError(f"Qdrant point {unit_id} lacks a vector")
        points.append(
            DensePoint(
                unit_id=unit_id,
                vector=tuple(float(component) for component in vector),
                payload=dict(payload),
                context_message_refs=refs_by_unit[unit_id],
            )
        )
    return points


def _sample_vector_mismatches(
    source_samples: Sequence[tuple[str, tuple[float, ...]]], target: QdrantHttpIndex
) -> list[str]:
    """Compare sampled source vectors against the target with exact equality."""
    if not source_samples:
        return []
    wanted = [unit_id for unit_id, _ in source_samples]
    response = target._request(
        "POST",
        f"{target.collection_path}/points/scroll",
        {
            "limit": len(wanted),
            "with_payload": False,
            "with_vector": True,
            "filter": {"must": [{"has_id": wanted}]},
        },
    )
    result = response.get("result")
    points = result.get("points") if isinstance(result, dict) else result
    if not isinstance(points, list):
        raise DenseBackendError("Qdrant scroll response lacks a point list")
    target_vectors = {
        point["id"]: tuple(float(component) for component in point["vector"])
        for point in points
        if isinstance(point, dict) and isinstance(point.get("vector"), list)
    }
    mismatches: list[str] = []
    for unit_id, source_vector in source_samples:
        target_vector = target_vectors.get(unit_id)
        if target_vector != source_vector:
            mismatches.append(unit_id)
    return mismatches


def _probe_payload_coverage(target: QdrantHttpIndex, *, sample_size: int = 8) -> int:
    """Fail closed unless every sampled target point carries the refs key."""
    sampled = 0
    for point in _scroll_points(
        target, batch_size=sample_size, with_payload=True, with_vector=False
    ):
        payload = point.get("payload")
        if not isinstance(payload, dict):
            raise DenseBackendError("Qdrant scroll point lacks a payload object")
        if LEGACY_PAYLOAD_KEY not in payload:
            raise DenseMigrationError(
                f"migrated point {point.get('id')} lacks the {LEGACY_PAYLOAD_KEY} payload key"
            )
        sampled += 1
        if sampled >= sample_size:
            break
    return sampled


def migrate_collection(
    index: QdrantHttpIndex,
    *,
    source_collection: str,
    new_collection: str,
    refs_by_unit: Mapping[str, list[str]],
    batch_size: int = 256,
    prevalidate_batch_size: int = 2048,
    force: bool = False,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Copy one legacy dense collection into ``new_collection`` with refs payloads.

    Refuses to touch the source. Pre-validates every point id against
    ``refs_by_unit`` before creating the new collection, then copies vectors
    verbatim in offset-paginated scroll batches. Re-runs are refused while the
    target exists and is non-empty unless ``force`` re-upserts over it.
    """
    if index.collection != source_collection:
        raise ValueError(
            f"index is bound to collection {index.collection!r}, not {source_collection!r}"
        )
    if batch_size <= 0 or prevalidate_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if new_collection == source_collection:
        raise ValueError("new_collection must differ from source_collection")
    started = time.monotonic()
    count_filter = (
        {"must": [{"key": "snapshot_id", "match": {"value": snapshot_id}}]}
        if snapshot_id is not None
        else None
    )
    source_count = index.count(count_filter)

    target = _build_target(index, new_collection)
    existing_target = _existing_target_count(target)
    if existing_target and not force:
        raise DenseMigrationError(
            f"target collection {new_collection!r} already exists with {existing_target} "
            "points; pass force=True to re-upsert over it"
        )

    # Pass 1: count-only scroll; fail closed on any missing join BEFORE creating
    # the target collection.
    missed: list[str] = []
    total_ids = 0
    for point in _scroll_points(
        index,
        batch_size=prevalidate_batch_size,
        with_payload=False,
        with_vector=False,
        snapshot_id=snapshot_id,
    ):
        total_ids += 1
        unit_id = point.get("id")
        if not isinstance(unit_id, str):
            raise DenseBackendError("Qdrant scroll point lacks a string id")
        if unit_id not in refs_by_unit:
            missed.append(unit_id)
    if total_ids != source_count:
        raise DenseMigrationError(
            f"source scroll yielded {total_ids} points but count reports {source_count}"
        )
    if missed:
        raise DenseMigrationError(
            f"{len(missed)} source points do not join corpus snapshot rows; "
            f"first misses: {sorted(missed)[:5]}",
            misses=sorted(missed),
        )

    # Pass 2: vector scroll + upsert with payload = legacy payload + refs.
    target.ensure_collection()
    upserted = 0
    vector_samples: list[tuple[str, tuple[float, ...]]] = []
    batch: list[Mapping[str, Any]] = []
    for point in _scroll_points(
        index,
        batch_size=batch_size,
        with_payload=True,
        with_vector=True,
        snapshot_id=snapshot_id,
    ):
        batch.append(point)
        if len(batch) >= batch_size:
            points = _dense_points_from_batch(batch, refs_by_unit)
            target.upsert(points, batch_size=batch_size)
            for dense_point in points:
                if len(vector_samples) < 16:
                    vector_samples.append((dense_point.unit_id, dense_point.vector))
            upserted += len(points)
            batch = []
    if batch:
        points = _dense_points_from_batch(batch, refs_by_unit)
        target.upsert(points, batch_size=batch_size)
        for dense_point in points:
            if len(vector_samples) < 16:
                vector_samples.append((dense_point.unit_id, dense_point.vector))
        upserted += len(points)
    if upserted != source_count:
        raise DenseMigrationError(
            f"upserted {upserted} points but source count is {source_count}"
        )

    # Verification.
    new_count = target.count()
    if new_count != source_count:
        raise DenseMigrationError(
            f"target count {new_count} does not match source count {source_count}"
        )
    payload_sampled = _probe_payload_coverage(target)
    vector_mismatches = _sample_vector_mismatches(vector_samples, target)
    if vector_mismatches:
        raise DenseMigrationError(
            f"{len(vector_mismatches)} sampled vectors differ after migration: "
            f"{vector_mismatches[:5]}"
        )

    return {
        "source": source_collection,
        "new": new_collection,
        "counts": {"source": source_count, "upserted": upserted, "new": new_count},
        "verified": {
            "count_match": new_count == source_count,
            "payload_coverage_sample": payload_sampled,
            "vector_sample": len(vector_samples),
            "vector_mismatches": 0,
        },
        "missed": [],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def migrate_snapshot(
    base_url: str,
    *,
    source_collection: str,
    new_collection: str,
    corpus_path: Path | str,
    snapshot_id: str,
    recipe: EmbeddingRecipe,
    expected_source_count: int | None = None,
    batch_size: int = 256,
    force: bool = False,
) -> dict[str, Any]:
    """Join-corpus, migrate, and verify one snapshot's dense collection.

    Loads ``unit_id -> context_message_refs`` from the corpus SQLite in
    read-only mode for ``snapshot_id``, pre-validates every source point id,
    then delegates to :func:`migrate_collection` with snapshot-scoped scroll.
    """
    refs_by_unit = load_refs_by_unit(corpus_path, snapshot_id=snapshot_id)
    index = QdrantHttpIndex(base_url, collection=source_collection, recipe=recipe)
    source_count = index.count()
    if expected_source_count is not None and source_count != expected_source_count:
        raise DenseMigrationError(
            f"source collection {source_collection!r} holds {source_count} points, "
            f"expected {expected_source_count}"
        )
    result = migrate_collection(
        index,
        source_collection=source_collection,
        new_collection=new_collection,
        refs_by_unit=refs_by_unit,
        batch_size=batch_size,
        force=force,
        snapshot_id=snapshot_id,
    )
    result["corpus"] = str(corpus_path)
    result["snapshot_id"] = snapshot_id
    return result
