"""Versioned dense-encoding and local Qdrant index primitives.

The real model adapter is deliberately injected. This module does not download
models or silently substitute a semantic mock; a dense run must provide a
resolved embedding revision and a working loopback Qdrant endpoint.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from reddit_search.ingest.invalidation import TombstoneProjection
from reddit_search.ingest.state import atomic_write_json, canonical_json_bytes

Vector = tuple[float, ...]


class DenseBackendError(RuntimeError):
    """A dense index or encoder could not satisfy its explicit contract."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class EmbeddingRecipe:
    """Immutable identity of one document/query embedding recipe."""

    model_id: str
    revision: str
    dimension: int
    query_instruction: str
    document_text_field: str = "context_text"
    context_recipe_version: str = "v2"
    chunking_version: str = "v2"

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("embedding model_id must not be empty")
        if not self.revision.strip():
            raise ValueError("embedding revision must be resolved before indexing")
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if not self.query_instruction.strip():
            raise ValueError("query instruction must not be empty")
        if not self.document_text_field.strip():
            raise ValueError("document text field must not be empty")
        if not self.context_recipe_version.strip():
            raise ValueError("context recipe version must not be empty")

    @classmethod
    def from_model_settings(
        cls,
        settings: Any,
        *,
        query_instruction: str,
        context_recipe_version: str = "v2",
    ) -> EmbeddingRecipe:
        """Build a recipe while refusing the unresolved config default."""
        revision = getattr(settings, "embedding_revision", None)
        if not isinstance(revision, str) or not revision.strip():
            raise DenseBackendError(
                "embedding_revision must be pinned before a dense index can be built"
            )
        return cls(
            model_id=str(settings.embedding_id),
            revision=revision,
            dimension=int(settings.embedding_dimension),
            query_instruction=query_instruction,
            context_recipe_version=context_recipe_version,
        )

    def manifest(self) -> dict[str, Any]:
        # chunking_version is payload provenance, not part of the registered
        # recipe identity: including it here repointed collection_name away
        # from the registered e82 collection to an unregistered 56008-point
        # index of a different corpus (49bd), which the dense registry and
        # preflight contract forbid.
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "query_instruction": self.query_instruction,
            "document_text_field": self.document_text_field,
            "context_recipe_version": self.context_recipe_version,
        }

    def collection_name(self, snapshot_id: str) -> str:
        if not snapshot_id.strip():
            raise ValueError("snapshot_id must not be empty")
        identity = json.dumps(
            {"snapshot_id": snapshot_id, "recipe": self.manifest()},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return f"reddit_dense_{digest}"


class EmbeddingAdapter(Protocol):
    """Separate document and prompted-query encoding methods."""

    recipe: EmbeddingRecipe

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode document context without the query instruction."""

    def encode_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode scenario queries with the recipe's query instruction."""


def validate_vector(vector: Sequence[float], dimension: int) -> Vector:
    """Return a finite, non-zero vector with the expected dimension."""
    if len(vector) != dimension:
        raise DenseBackendError(
            f"embedding dimension mismatch: expected {dimension}, got {len(vector)}"
        )
    values = tuple(float(value) for value in vector)
    if not all(isfinite(value) for value in values):
        raise DenseBackendError("embedding contains a non-finite value")
    norm = sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise DenseBackendError("embedding has zero norm")
    return values


def validate_vectors(
    vectors: Sequence[Sequence[float]], *, dimension: int, expected_count: int
) -> list[Vector]:
    if len(vectors) != expected_count:
        raise DenseBackendError(
            f"encoder returned {len(vectors)} vectors for {expected_count} inputs"
        )
    return [validate_vector(vector, dimension) for vector in vectors]


class DenseIndexJobStore:
    """SQLite resume ledger for one snapshot and embedding recipe."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dense_index_jobs (
                    snapshot_id TEXT NOT NULL,
                    recipe_hash TEXT NOT NULL,
                    unit_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (snapshot_id, recipe_hash, unit_id)
                )
                """
            )

    def __enter__(self) -> DenseIndexJobStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.connection.close()

    @staticmethod
    def recipe_hash(recipe: EmbeddingRecipe) -> str:
        encoded = json.dumps(
            recipe.manifest(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def prepare(
        self,
        *,
        snapshot_id: str,
        recipe: EmbeddingRecipe,
        units: Sequence[tuple[str, str]],
    ) -> list[str]:
        """Return units not yet indexed for this exact content and recipe."""
        if not snapshot_id.strip():
            raise ValueError("snapshot_id must not be empty")
        recipe_hash = self.recipe_hash(recipe)
        pending: list[str] = []
        now = int(time.time())
        with self.connection:
            for unit_id, content_hash in units:
                if not unit_id.strip() or not content_hash.strip():
                    raise ValueError("dense index jobs require unit and content hashes")
                row = self.connection.execute(
                    """
                    SELECT content_hash, status
                    FROM dense_index_jobs
                    WHERE snapshot_id = ? AND recipe_hash = ? AND unit_id = ?
                    """,
                    (snapshot_id, recipe_hash, unit_id),
                ).fetchone()
                if (
                    row is not None
                    and row["content_hash"] == content_hash
                    and row["status"] == "indexed"
                ):
                    continue
                self.connection.execute(
                    """
                    INSERT INTO dense_index_jobs (
                        snapshot_id, recipe_hash, unit_id, content_hash,
                        status, error, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', NULL, ?)
                    ON CONFLICT(snapshot_id, recipe_hash, unit_id) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        status = 'pending',
                        error = NULL,
                        updated_at = excluded.updated_at
                    """,
                    (snapshot_id, recipe_hash, unit_id, content_hash, now),
                )
                pending.append(unit_id)
        return pending

    def mark_indexed(self, *, snapshot_id: str, recipe: EmbeddingRecipe, unit_id: str) -> None:
        self._mark(snapshot_id=snapshot_id, recipe=recipe, unit_id=unit_id, status="indexed")

    def mark_failed(
        self,
        *,
        snapshot_id: str,
        recipe: EmbeddingRecipe,
        unit_id: str,
        error: str,
    ) -> None:
        self._mark(
            snapshot_id=snapshot_id,
            recipe=recipe,
            unit_id=unit_id,
            status="failed",
            error=error,
        )

    def status_counts(self, *, snapshot_id: str, recipe: EmbeddingRecipe) -> dict[str, int]:
        rows = self.connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM dense_index_jobs
            WHERE snapshot_id = ? AND recipe_hash = ?
            GROUP BY status
            """,
            (snapshot_id, self.recipe_hash(recipe)),
        ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def _mark(
        self,
        *,
        snapshot_id: str,
        recipe: EmbeddingRecipe,
        unit_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        if status not in {"indexed", "failed"}:
            raise ValueError(f"unsupported dense index job status: {status}")
        with self.connection:
            updated = self.connection.execute(
                """
                UPDATE dense_index_jobs
                SET status = ?, error = ?, updated_at = ?
                WHERE snapshot_id = ? AND recipe_hash = ? AND unit_id = ?
                """,
                (
                    status,
                    error,
                    int(time.time()),
                    snapshot_id,
                    self.recipe_hash(recipe),
                    unit_id,
                ),
            ).rowcount
        if updated != 1:
            raise ValueError(f"dense index job does not exist for unit {unit_id}")


def write_index_manifest(
    path: Path,
    *,
    snapshot_id: str,
    recipe: EmbeddingRecipe,
    expected_count: int,
    actual_count: int,
) -> dict[str, Any]:
    """Persist a ready/failed reconciliation record atomically."""
    if expected_count < 0 or actual_count < 0:
        raise ValueError("index counts must be non-negative")
    status = "ready" if expected_count == actual_count else "failed"
    manifest = {
        "kind": "dense_index_manifest",
        "schema_version": 1,
        "status": status,
        "snapshot_id": snapshot_id,
        "recipe": recipe.manifest(),
        "expected_count": expected_count,
        "actual_count": actual_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest


@dataclass(frozen=True, slots=True)
class DensePoint:
    """One canonical unit vector and its minimal retrieval payload."""

    unit_id: str
    vector: Vector
    payload: Mapping[str, Any]
    context_message_refs: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError("dense point unit_id must not be empty")
        try:
            uuid.UUID(self.unit_id)
        except ValueError as error:
            raise ValueError("dense point unit_id must be a UUID") from error


@dataclass(frozen=True, slots=True)
class DenseHit:
    """A Qdrant result with deterministic one-based rank."""

    candidate_id: str
    rank: int
    score: float
    payload: Mapping[str, Any]


class QdrantHttpIndex:
    """Small dependency-free client for a loopback Qdrant collection."""

    def __init__(
        self,
        base_url: str,
        *,
        collection: str,
        recipe: EmbeddingRecipe,
        timeout_seconds: float = 30.0,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("dense index URL must point to a loopback HTTP endpoint")
        if not collection.strip():
            raise ValueError("collection must not be empty")
        self._base_url = base_url.rstrip("/")
        self.collection = collection
        self.recipe = recipe
        self.dimension = recipe.dimension
        self.distance = "Cosine"
        self.timeout_seconds = timeout_seconds

    @classmethod
    def for_snapshot(
        cls,
        base_url: str,
        *,
        snapshot_id: str,
        recipe: EmbeddingRecipe,
        timeout_seconds: float = 30.0,
    ) -> QdrantHttpIndex:
        return cls(
            base_url,
            collection=recipe.collection_name(snapshot_id),
            recipe=recipe,
            timeout_seconds=timeout_seconds,
        )

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self._base_url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                raw = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise DenseBackendError(
                f"Qdrant {method} {path} failed with HTTP {error.code}: {detail}",
                status=error.code,
            ) from error
        except URLError as error:
            raise DenseBackendError(
                f"Qdrant {method} {path} unavailable: {error.reason}"
            ) from error
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise DenseBackendError(f"Qdrant returned invalid JSON for {method} {path}") from error
        if not isinstance(payload, dict):
            raise DenseBackendError(f"Qdrant returned a non-object response for {method} {path}")
        return payload

    @property
    def collection_path(self) -> str:
        return f"/collections/{quote(self.collection, safe='')}"

    def health(self) -> bool:
        self._request("GET", "/healthz")
        return True

    def ensure_collection(self) -> None:
        try:
            payload = self._request("GET", self.collection_path)
        except DenseBackendError as error:
            if error.status != 404:
                raise
            self._request(
                "PUT",
                self.collection_path,
                {"vectors": {"size": self.dimension, "distance": self.distance}},
            )
            return
        result = payload.get("result")
        config = result.get("config") if isinstance(result, dict) else None
        params = config.get("params") if isinstance(config, dict) else None
        vectors = params.get("vectors") if isinstance(params, dict) else None
        size = vectors.get("size") if isinstance(vectors, dict) else None
        distance = vectors.get("distance") if isinstance(vectors, dict) else None
        if size != self.dimension:
            raise DenseBackendError(
                f"Qdrant collection dimension mismatch: expected {self.dimension}, got {size}"
            )
        if distance is not None and str(distance).lower() != self.distance.lower():
            raise DenseBackendError(
                f"Qdrant collection distance mismatch: expected {self.distance}, got {distance}"
            )

    def upsert(self, points: Sequence[DensePoint], *, batch_size: int = 128) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for point in points:
            validate_vector(point.vector, self.dimension)
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            body = [
                {
                    "id": point.unit_id,
                    "vector": list(point.vector),
                    "payload": {
                        **dict(point.payload),
                        "context_message_refs": sorted(set(point.context_message_refs))
                        if point.context_message_refs is not None
                        else [],
                    },
                }
                for point in batch
            ]
            self._request(
                "PUT",
                f"{self.collection_path}/points?wait=true",
                {"points": body},
            )

    def count(self, filter: Mapping[str, Any] | None = None) -> int:
        body: dict[str, Any] = {"exact": True}
        if filter is not None:
            body["filter"] = dict(filter)
        payload = self._request("POST", f"{self.collection_path}/points/count", body)
        if payload.get("status") != "ok":
            raise DenseBackendError("Qdrant count response status is not ok")
        result = payload.get("result")
        count = result.get("count") if isinstance(result, dict) else None
        if not isinstance(count, int) or count < 0:
            raise DenseBackendError("Qdrant count response lacks a non-negative integer count")
        return count

    def _delete_filter(
        self,
        identities: Sequence[tuple[str, str | None] | tuple[str, str | None, str | None]],
        *,
        snapshot_id: str,
    ) -> tuple[dict[str, Any], list[tuple[str, str | None, str | None]]]:
        ordered = self._normalized_identities(identities)
        clauses = [self._identity_clause(identity) for identity in ordered]
        return (
            {
                "must": [
                    {"key": "snapshot_id", "match": {"value": snapshot_id}},
                    {"should": clauses},
                ]
            },
            ordered,
        )

    @staticmethod
    def _normalized_identities(
        identities: Sequence[tuple[str, str | None] | tuple[str, str | None, str | None]],
    ) -> list[tuple[str, str | None, str | None]]:
        """Validate identities and return them deduplicated and sorted."""
        normalized: set[tuple[str, str | None, str | None]] = set()
        for identity in identities:
            if len(identity) == 2:
                fullname, revision = identity
                unit_id = None
            elif len(identity) == 3:
                fullname, revision, unit_id = identity
            else:
                raise ValueError("dense deletion identities must have two or three fields")
            if (
                not isinstance(fullname, str)
                or not fullname
                or (revision is not None and (not isinstance(revision, str) or not revision))
                or (unit_id is not None and (not isinstance(unit_id, str) or not unit_id))
            ):
                raise ValueError("dense deletion identities must be non-empty pairs")
            normalized.add((fullname, revision, unit_id))
        return sorted(normalized, key=lambda item: (item[0], item[1] or "", item[2] or ""))

    @staticmethod
    def _identity_clause(
        identity: tuple[str, str | None, str | None],
    ) -> dict[str, Any]:
        fullname, revision, unit_id = identity
        must = [{"key": "message_fullname", "match": {"value": fullname}}]
        if revision is not None:
            must.append({"key": "source_revision_id", "match": {"value": revision}})
        if unit_id is not None:
            must.append({"key": "unit_id", "match": {"value": unit_id}})
        return {"must": must}

    def delete_identities(
        self,
        identities: Sequence[tuple[str, str | None] | tuple[str, str | None, str | None]],
        *,
        snapshot_id: str,
    ) -> int:
        """Delete explicit identities and return the observed count removed."""
        if not snapshot_id.strip():
            raise ValueError("snapshot_id must not be empty")
        selector, unique = self._delete_filter(identities, snapshot_id=snapshot_id)
        if not unique:
            return 0
        self.ensure_collection()
        before = self.count(selector)
        payload = self._request(
            "POST",
            f"{self.collection_path}/points/delete?wait=true",
            {"filter": selector},
        )
        if payload.get("status") != "ok":
            raise DenseBackendError("Qdrant delete response status is not ok")
        result = payload.get("result")
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise DenseBackendError("Qdrant delete response did not complete")
        after = self.count(selector)
        if after > before:
            raise DenseBackendError("Qdrant deletion count reconciliation increased")
        return before - after

    def _probe_context_payload_coverage(self, *, snapshot_id: str) -> None:
        """Fail closed unless every sampled snapshot point carries context refs.

        Detection (simplest reliable form): scroll a sample of up to 8
        snapshot-scoped points and require each payload to expose the
        ``context_message_refs`` key. An empty snapshot needs no coverage —
        the subsequent context delete is then a no-op. Collections whose
        points lack the key predate context-ref payloads and cannot support
        context-contributor deletion without a reindex.
        """
        response = self._request(
            "POST",
            f"{self.collection_path}/points/scroll",
            {
                "limit": 8,
                "with_payload": True,
                "with_vector": False,
                "filter": {"must": [{"key": "snapshot_id", "match": {"value": snapshot_id}}]},
            },
        )
        if response.get("status") != "ok":
            raise DenseBackendError("Qdrant scroll response status is not ok")
        result = response.get("result")
        points = result.get("points") if isinstance(result, dict) else result
        if not isinstance(points, list):
            raise DenseBackendError("Qdrant scroll response lacks a point list")
        sampled = 0
        for point in points:
            if not isinstance(point, dict):
                raise DenseBackendError("Qdrant scroll response contains a non-object point")
            point_payload = point.get("payload")
            if not isinstance(point_payload, dict):
                raise DenseBackendError("Qdrant scroll point lacks a payload object")
            sampled += 1
            if "context_message_refs" not in point_payload:
                raise DenseBackendError(
                    "dense collection points lack the context_message_refs payload; "
                    "context contributors are not coverable without a reindex"
                )
        if sampled == 0:
            return  # empty snapshot: context deletion is a no-op below

    def delete_context_contributors(
        self,
        fullnames: Sequence[str],
        *,
        snapshot_id: str,
        exclude_identities: Sequence[
            tuple[str, str | None] | tuple[str, str | None, str | None]
        ] = (),
    ) -> int:
        """Delete snapshot points whose payload embeds a fullname in context refs.

        Points are matched on the ``context_message_refs`` payload key and
        scoped to the snapshot. Direct identity matches can be excluded so a
        combined reconciliation never double-counts one point. The collection
        must expose ``context_message_refs`` in its point payloads (see
        ``_probe_context_payload_coverage``); otherwise this fails closed
        before any deletion.
        """
        if not snapshot_id.strip():
            raise ValueError("snapshot_id must not be empty")
        refs = sorted(fullnames)
        for ref in refs:
            if not isinstance(ref, str) or not ref:
                raise ValueError("context contributor fullnames must be non-empty strings")
        if not refs:
            return 0
        self.ensure_collection()
        self._probe_context_payload_coverage(snapshot_id=snapshot_id)
        clauses = [{"key": "context_message_refs", "match": {"value": ref}} for ref in refs]
        selector: dict[str, Any] = {
            "must": [
                {"key": "snapshot_id", "match": {"value": snapshot_id}},
                {"should": clauses},
            ]
        }
        exclusions = [
            self._identity_clause(identity)
            for identity in self._normalized_identities(exclude_identities)
        ]
        if exclusions:
            selector["must_not"] = exclusions
        before = self.count(selector)
        payload = self._request(
            "POST",
            f"{self.collection_path}/points/delete?wait=true",
            {"filter": selector},
        )
        if payload.get("status") != "ok":
            raise DenseBackendError("Qdrant delete response status is not ok")
        result = payload.get("result")
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise DenseBackendError("Qdrant delete response did not complete")
        after = self.count(selector)
        if after > before:
            raise DenseBackendError("Qdrant context deletion count reconciliation increased")
        return before - after

    def delete_tombstone_projection(
        self,
        projection: TombstoneProjection,
        *,
        snapshot_id: str,
        manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        """Apply an explicit tombstone projection without archive inference."""
        if projection.as_dict().get("propagation_status") != "not_claimed":
            raise ValueError("tombstone projection status is invalid")
        expected_digest = hashlib.sha256(
            canonical_json_bytes([dict(record) for record in projection.records])
        ).hexdigest()
        if projection.ledger_digest != expected_digest:
            raise ValueError("tombstone projection ledger identity is invalid")
        if projection.counts != {
            "records": len(projection.records),
            "artifacts": len(projection.source_artifact_hashes),
        }:
            raise ValueError("tombstone projection counts are invalid")
        identities: list[tuple[str, str | None] | tuple[str, str | None, str]] = []
        for record in projection.records:
            if record.get("status") in {"matched", "unresolved"} or record.get(
                "coverage_status"
            ) in {"matched", "unresolved"}:
                raise ValueError("archive coverage rows are not tombstones")
            fullname = record.get("message_fullname")
            revision = record.get("source_revision_id")
            unit_id = record.get("unit_id")
            if (
                not isinstance(fullname, str)
                or not fullname
                or (revision is not None and (not isinstance(revision, str) or not revision))
                or (unit_id is not None and (not isinstance(unit_id, str) or not unit_id))
            ):
                raise ValueError("tombstone projection identity is invalid")
            scoped_snapshot = record.get("snapshot_id")
            if scoped_snapshot is not None and scoped_snapshot != snapshot_id:
                raise ValueError("tombstone projection snapshot scope mismatch")
            identities.append(
                (fullname, revision) if unit_id is None else (fullname, revision, unit_id)
            )
        requested = sorted(
            set(identities),
            key=lambda item: (item[0], item[1] or "", item[2] if len(item) == 3 else ""),
        )
        fullnames = sorted({identity[0] for identity in requested})
        context_deleted = 0
        if fullnames:
            # Context deletion runs first so the fail-closed payload probe
            # raises before any direct deletion is applied on legacy indexes.
            context_deleted = self.delete_context_contributors(
                fullnames,
                snapshot_id=snapshot_id,
                exclude_identities=requested,
            )
        deleted = self.delete_identities(requested, snapshot_id=snapshot_id)
        result: dict[str, Any] = {
            "kind": "dense_tombstone_propagation",
            "snapshot_id": snapshot_id,
            "ledger_digest": projection.ledger_digest,
            "ledger_sha256": projection.ledger_digest,
            "counts": dict(projection.counts),
            "requested_identities": [
                {
                    "message_fullname": identity[0],
                    "source_revision_id": identity[1],
                    **({"unit_id": identity[2]} if len(identity) == 3 else {}),
                }
                for identity in requested
            ],
            "requested_count": len(requested),
            "deleted_count": deleted,
            "context_contributors": "covered",
            "context_deleted_count": context_deleted,
            "context_requested_fullnames": fullnames,
            "policy": "explicit_identity_and_payload_context_refs",
            "status": "applied",
            "propagation_status": "claimed",
        }
        if manifest_path is not None:
            atomic_write_json(manifest_path, result)
            result["manifest_path"] = str(manifest_path)
        return result

    def assert_count(self, expected: int) -> None:
        actual = self.count()
        if actual != expected:
            raise DenseBackendError(
                f"dense index count mismatch: expected {expected}, got {actual}"
            )

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        payload_filter: Mapping[str, str] | None = None,
    ) -> list[DenseHit]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        body: dict[str, Any] = {
            "vector": list(validate_vector(vector, self.dimension)),
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        if payload_filter:
            body["filter"] = {
                "must": [
                    {"key": key, "match": {"value": value}}
                    for key, value in sorted(payload_filter.items())
                ]
            }
        payload = self._request("POST", f"{self.collection_path}/points/search", body)
        result = payload.get("result")
        if not isinstance(result, list):
            raise DenseBackendError("Qdrant search response lacks a result list")
        candidates: list[tuple[str, float, Mapping[str, Any]]] = []
        for item in result:
            if not isinstance(item, dict):
                raise DenseBackendError("Qdrant search result contains a non-object item")
            candidate_id = item.get("id")
            score = item.get("score")
            item_payload = item.get("payload") or {}
            if not isinstance(candidate_id, str) or not candidate_id:
                raise DenseBackendError("Qdrant search result lacks a string ID")
            if not isinstance(score, (int, float)) or not isfinite(float(score)):
                raise DenseBackendError("Qdrant search result lacks a finite score")
            if not isinstance(item_payload, dict):
                raise DenseBackendError("Qdrant search payload must be an object")
            candidates.append((candidate_id, float(score), item_payload))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return [
            DenseHit(candidate_id=candidate_id, rank=rank, score=score, payload=item_payload)
            for rank, (candidate_id, score, item_payload) in enumerate(candidates, start=1)
        ]

    def index_documents(
        self,
        adapter: EmbeddingAdapter,
        documents: Sequence[
            tuple[str, str, Mapping[str, Any]] | tuple[str, str, Mapping[str, Any], Sequence[str]]
        ],
        *,
        batch_size: int = 8,
        expected_count: int | None = None,
        on_batch_indexed: Callable[[Sequence[str]], None] | None = None,
    ) -> int:
        """Encode context documents and upsert stable points, without silent drops.

        A document may carry an optional fourth element: the context
        contributor ``message_fullname`` values embedded in that unit. These
        are stored in the point payload under ``context_message_refs`` so
        tombstone propagation can delete context contributors later.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len({document[0] for document in documents}) != len(documents):
            raise DenseBackendError("dense documents contain duplicate unit IDs")
        if adapter.recipe != self.recipe:
            raise DenseBackendError("embedding adapter recipe does not match index recipe")
        self.ensure_collection()
        indexed = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            vectors = validate_vectors(
                adapter.encode_documents([document[1] for document in batch]),
                dimension=self.dimension,
                expected_count=len(batch),
            )
            self.upsert(
                [
                    DensePoint(
                        unit_id=first,
                        vector=vector,
                        payload=payload,
                        context_message_refs=list(rest[0]) if rest else None,
                    )
                    for ((first, _, payload, *rest), vector) in zip(batch, vectors, strict=True)
                ],
                batch_size=batch_size,
            )
            indexed += len(batch)
            if on_batch_indexed is not None:
                on_batch_indexed([document[0] for document in batch])
        if expected_count is not None:
            self.assert_count(expected_count)
        return indexed
