import pytest


def test_embedding_recipe_requires_pinned_revision_and_is_version_isolated() -> None:
    from reddit_search.retrieval.dense import DenseBackendError, EmbeddingRecipe

    with pytest.raises(DenseBackendError, match="embedding_revision"):
        EmbeddingRecipe.from_model_settings(
            type(
                "Settings",
                (),
                {"embedding_id": "qwen", "embedding_revision": None, "embedding_dimension": 4},
            )(),
            query_instruction="Represent the query.",
        )

    recipe = EmbeddingRecipe(
        model_id="qwen",
        revision="rev-1",
        dimension=4,
        query_instruction="Represent the query.",
    )
    assert recipe.collection_name("snapshot-a") != recipe.collection_name("snapshot-b")
    assert recipe.collection_name("snapshot-a") == recipe.collection_name("snapshot-a")


def test_validate_vector_rejects_dimension_nonfinite_and_zero_norm() -> None:
    from reddit_search.retrieval.dense import DenseBackendError, validate_vector

    assert validate_vector([3, 4], 2) == (3.0, 4.0)
    with pytest.raises(DenseBackendError, match="dimension mismatch"):
        validate_vector([1], 2)
    with pytest.raises(DenseBackendError, match="non-finite"):
        validate_vector([float("nan"), 1], 2)
    with pytest.raises(DenseBackendError, match="zero norm"):
        validate_vector([0, 0], 2)


def test_qdrant_client_is_loopback_only() -> None:
    from reddit_search.retrieval.dense import EmbeddingRecipe, QdrantHttpIndex

    recipe = EmbeddingRecipe(
        model_id="qwen",
        revision="rev-1",
        dimension=2,
        query_instruction="Represent the query.",
    )
    with pytest.raises(ValueError, match="loopback"):
        QdrantHttpIndex(
            "https://example.invalid",
            collection="collection",
            recipe=recipe,
        )


def test_dense_index_job_store_resumes_by_snapshot_recipe_and_content(tmp_path) -> None:
    from reddit_search.retrieval.dense import (
        DenseIndexJobStore,
        EmbeddingRecipe,
        write_index_manifest,
    )

    recipe = EmbeddingRecipe(
        model_id="qwen",
        revision="rev-1",
        dimension=2,
        query_instruction="Represent the query.",
    )
    unit_id = "123e4567-e89b-12d3-a456-426614174000"
    with DenseIndexJobStore(tmp_path / "jobs.db") as jobs:
        assert jobs.prepare(
            snapshot_id="snap",
            recipe=recipe,
            units=[(unit_id, "content-a")],
        ) == [unit_id]
        jobs.mark_indexed(snapshot_id="snap", recipe=recipe, unit_id=unit_id)
        assert (
            jobs.prepare(
                snapshot_id="snap",
                recipe=recipe,
                units=[(unit_id, "content-a")],
            )
            == []
        )
        assert jobs.prepare(
            snapshot_id="snap",
            recipe=recipe,
            units=[(unit_id, "content-b")],
        ) == [unit_id]
        assert jobs.status_counts(snapshot_id="snap", recipe=recipe) == {"pending": 1}

    manifest = write_index_manifest(
        tmp_path / "manifest.json",
        snapshot_id="snap",
        recipe=recipe,
        expected_count=1,
        actual_count=0,
    )
    assert manifest["status"] == "failed"
    assert (tmp_path / "manifest.json").exists()


def test_dense_tombstone_projection_validates_and_returns_deterministic_result(tmp_path) -> None:
    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.retrieval.dense import EmbeddingRecipe, QdrantHttpIndex

    recipe = EmbeddingRecipe("qwen", "rev-1", 2, "Represent the query.")
    index = QdrantHttpIndex("http://127.0.0.1:6333", collection="test", recipe=recipe)
    calls = []
    index.delete_context_contributors = (  # type: ignore[method-assign]
        lambda fullnames, *, snapshot_id, exclude_identities=(): 0
    )
    index.delete_identities = lambda identities, *, snapshot_id: (  # type: ignore[method-assign]
        calls.append((identities, snapshot_id)) or len(identities)
    )
    projection = project_tombstone_identities(
        [
            {
                "message_fullname": "t1_b",
                "source_revision_id": "rev-2",
                "snapshot_id": "snap",
            },
            {"message_fullname": "t1_a", "source_revision_id": "rev-1"},
        ]
    )
    result = index.delete_tombstone_projection(
        projection, snapshot_id="snap", manifest_path=tmp_path / "manifest.json"
    )
    assert result["status"] == "applied"
    assert result["ledger_digest"] == projection.ledger_digest
    assert result["ledger_sha256"] == projection.ledger_digest
    assert result["counts"] == projection.counts
    assert result["requested_identities"][0]["message_fullname"] == "t1_a"
    assert calls == [([("t1_a", "rev-1"), ("t1_b", "rev-2")], "snap")]

    tampered = type(projection)(
        projection.records, projection.source_artifact_hashes, "0" * 64, projection.counts
    )
    with pytest.raises(ValueError, match="ledger identity"):
        index.delete_tombstone_projection(tampered, snapshot_id="snap")


def test_qdrant_delete_transport_body_scope_and_reconciliation() -> None:
    from reddit_search.retrieval.dense import EmbeddingRecipe, QdrantHttpIndex

    recipe = EmbeddingRecipe("qwen", "rev-1", 2, "Represent the query.")
    index = QdrantHttpIndex("http://127.0.0.1:6333", collection="test", recipe=recipe)
    calls = []
    count_values = iter([2, 0])

    def request(method, path, body=None):
        calls.append((method, path, body))
        if method == "GET":
            return {
                "result": {
                    "config": {"params": {"vectors": {"size": 2, "distance": "Cosine"}}}
                }
            }
        if path.endswith("/points/count"):
            return {"status": "ok", "result": {"count": next(count_values)}}
        return {"status": "ok", "result": {"status": "completed"}}

    index._request = request  # type: ignore[method-assign]
    deleted = index.delete_identities(
        [("t1_all", None), ("t1_exact", "rev-1", "unit-1")],
        snapshot_id="snap",
    )
    assert deleted == 2
    assert calls[1][0:2] == ("POST", "/collections/test/points/count")
    assert calls[2][0:2] == ("POST", "/collections/test/points/delete?wait=true")
    assert "points" not in calls[2][2]
    assert calls[2][2]["filter"]["must"][0] == {
        "key": "snapshot_id",
        "match": {"value": "snap"},
    }
    assert calls[2][2]["filter"]["must"][1]["should"][0]["must"] == [
        {"key": "message_fullname", "match": {"value": "t1_all"}}
    ]
    assert calls[2][2]["filter"]["must"][1]["should"][1]["must"][-1] == {
        "key": "unit_id",
        "match": {"value": "unit-1"},
    }


@pytest.mark.parametrize(
    "response",
    [
        {"status": "ok", "result": {"status": "started"}},
        {"status": "ok", "result": {}},
        {"status": "error", "result": {"status": "completed"}},
    ],
)
def test_qdrant_delete_rejects_uncompleted_or_malformed_response(response) -> None:
    from reddit_search.retrieval.dense import DenseBackendError, EmbeddingRecipe, QdrantHttpIndex

    index = QdrantHttpIndex(
        "http://127.0.0.1:6333",
        collection="test",
        recipe=EmbeddingRecipe("qwen", "rev-1", 2, "Represent the query."),
    )
    responses = iter(
        [
            {"result": {"config": {"params": {"vectors": {"size": 2, "distance": "Cosine"}}}}},
            {"status": "ok", "result": {"count": 1}},
            response,
        ]
    )
    index._request = lambda method, path, body=None: next(responses)  # type: ignore[method-assign]
    with pytest.raises(DenseBackendError):
        index.delete_identities([("t1_x", "rev")], snapshot_id="snap")


def test_qdrant_delete_propagates_transport_errors_and_zero_match() -> None:
    from reddit_search.retrieval.dense import DenseBackendError, EmbeddingRecipe, QdrantHttpIndex

    index = QdrantHttpIndex(
        "http://127.0.0.1:6333",
        collection="test",
        recipe=EmbeddingRecipe("qwen", "rev-1", 2, "Represent the query."),
    )
    responses = iter(
        [
            {"result": {"config": {"params": {"vectors": {"size": 2, "distance": "Cosine"}}}}},
            {"status": "ok", "result": {"count": 0}},
            {"status": "ok", "result": {"status": "completed"}},
            {"status": "ok", "result": {"count": 0}},
        ]
    )
    index._request = lambda method, path, body=None: next(responses)  # type: ignore[method-assign]
    assert index.delete_identities([("t1_missing", "rev")], snapshot_id="snap") == 0

    index._request = lambda method, path, body=None: (_ for _ in ()).throw(
        DenseBackendError("timeout")
    )  # type: ignore[method-assign]
    with pytest.raises(DenseBackendError, match="timeout"):
        index.delete_identities([("t1_x", "rev")], snapshot_id="snap")


def test_qdrant_delete_tombstone_projection_rejects_wait_timeout() -> None:
    """A wait_timeout delete response must fail closed, never claim applied."""
    import json as json_module

    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.retrieval.dense import DenseBackendError, EmbeddingRecipe, QdrantHttpIndex

    index = QdrantHttpIndex(
        "http://127.0.0.1:6333",
        collection="synthetic-review",
        recipe=EmbeddingRecipe("synthetic", "r1", 2, "query"),
    )
    index.ensure_collection = lambda: None  # type: ignore[method-assign]
    requests: list[dict[str, object]] = []

    def request(method: str, path: str, payload: object = None) -> dict[str, object]:
        requests.append({"method": method, "path": path, "body": payload})
        if path.endswith("/points/scroll"):
            return {
                "status": "ok",
                "result": {
                    "points": [{"payload": {"context_message_refs": ["t1_parent"]}}]
                },
            }
        if path.endswith("/points/count"):
            return {"status": "ok", "result": {"count": 1}}
        return {"status": "ok", "result": {"status": "wait_timeout", "operation_id": 1}}

    index._request = request  # type: ignore[method-assign]
    projection = project_tombstone_identities(
        [
            {
                "message_fullname": "t1_parent",
                "source_revision_id": "r1",
                "unit_id": "only-this-unit",
                "snapshot_id": "snap",
            }
        ]
    )
    with pytest.raises(DenseBackendError):
        index.delete_tombstone_projection(projection, snapshot_id="snap")
    # The delete request must be scoped to the projection's unit_id too.
    delete_body = next(
        call["body"] for call in requests if str(call["path"]).endswith("/points/delete?wait=true")
    )
    clause = json_module.dumps(delete_body)
    assert "only-this-unit" in clause
    assert requests[-1]["path"].endswith("/points/delete?wait=true")


def test_qdrant_tombstone_projection_deletes_context_contributors() -> None:
    """Context deletion must run before direct deletion with separate counts."""
    import json as json_module

    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.retrieval.dense import EmbeddingRecipe, QdrantHttpIndex

    index = QdrantHttpIndex(
        "http://127.0.0.1:6333",
        collection="synthetic-review",
        recipe=EmbeddingRecipe("synthetic", "r1", 2, "query"),
    )
    deletes: list[dict[str, object]] = []
    counts = iter([2, 0, 2, 0])

    def request(method: str, path: str, payload: object = None) -> dict[str, object]:
        if path.endswith("/points/scroll"):
            return {
                "status": "ok",
                "result": {
                    "points": [
                        {"payload": {"context_message_refs": ["t1_parent"]}},
                        {"payload": {"context_message_refs": []}},
                    ]
                },
            }
        if path.endswith("/points/delete?wait=true"):
            deletes.append({"path": str(path), "body": payload})
            return {"status": "ok", "result": {"status": "completed"}}
        if path.endswith("/points/count"):
            return {"status": "ok", "result": {"count": next(counts)}}
        return {"result": {"config": {"params": {"vectors": {"size": 2, "distance": "Cosine"}}}}}

    index._request = request  # type: ignore[method-assign]
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_parent", "source_revision_id": "r1", "snapshot_id": "snap"}]
    )
    result = index.delete_tombstone_projection(projection, snapshot_id="snap")

    assert result["deleted_count"] == 2
    assert result["context_contributors"] == "covered"
    assert result["context_deleted_count"] == 2
    assert result["context_requested_fullnames"] == ["t1_parent"]
    assert result["policy"] == "explicit_identity_and_payload_context_refs"
    assert len(deletes) == 2
    context_filter = deletes[0]["body"]["filter"]
    assert context_filter["must"][0] == {"key": "snapshot_id", "match": {"value": "snap"}}
    assert {
        "key": "context_message_refs",
        "match": {"value": "t1_parent"},
    } in context_filter["must"][1]["should"]
    must_not = context_filter.get("must_not", [])
    assert any(
        {"key": "message_fullname", "match": {"value": "t1_parent"}} in clause["must"]
        for clause in must_not
    )
    direct_filter = deletes[1]["body"]["filter"]
    assert any(
        clause["must"][0]["key"] == "message_fullname"
        for clause in direct_filter["must"][1]["should"]
    )
    assert "context_message_refs" not in json_module.dumps(direct_filter["must"][1])


def test_qdrant_tombstone_projection_fails_closed_without_context_payload() -> None:
    """Legacy points without context_message_refs must abort before deleting."""
    import pytest

    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.retrieval.dense import (
        DenseBackendError,
        EmbeddingRecipe,
        QdrantHttpIndex,
    )

    index = QdrantHttpIndex(
        "http://127.0.0.1:6333",
        collection="synthetic-review",
        recipe=EmbeddingRecipe("synthetic", "r1", 2, "query"),
    )
    deletes: list[dict[str, object]] = []

    def request(method: str, path: str, payload: object = None) -> dict[str, object]:
        if path.endswith("/points/scroll"):
            return {"status": "ok", "result": {"points": [{"payload": {"snapshot_id": "snap"}}]}}
        if path.endswith("/points/delete?wait=true"):
            deletes.append({"path": str(path), "body": payload})
            return {"status": "ok", "result": {"status": "completed"}}
        if path.endswith("/points/count"):
            return {"status": "ok", "result": {"count": 1}}
        return {"result": {"config": {"params": {"vectors": {"size": 2, "distance": "Cosine"}}}}}

    index._request = request  # type: ignore[method-assign]
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_parent", "source_revision_id": "r1", "snapshot_id": "snap"}]
    )
    with pytest.raises(DenseBackendError, match="context_message_refs"):
        index.delete_tombstone_projection(projection, snapshot_id="snap")
    assert deletes == []


def test_qdrant_index_documents_stores_context_refs_in_payload() -> None:
    from reddit_search.retrieval.dense import EmbeddingRecipe, QdrantHttpIndex

    class Adapter:
        recipe = EmbeddingRecipe("qwen", "rev-1", 2, "Represent the query.")

        def encode_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def encode_queries(self, texts):
            return [[1.0, 0.0] for _ in texts]

    index = QdrantHttpIndex(
        "http://127.0.0.1:6333",
        collection="test",
        recipe=EmbeddingRecipe("qwen", "rev-1", 2, "Represent the query."),
    )
    upserts: list[dict[str, object]] = []

    def request(method: str, path: str, payload: object = None) -> dict[str, object]:
        if path.endswith("/points?wait=true"):
            upserts.append(payload)
            return {"status": "ok", "result": {"status": "completed"}}
        return {"result": {"config": {"params": {"vectors": {"size": 2, "distance": "Cosine"}}}}}

    index._request = request  # type: ignore[method-assign]
    indexed = index.index_documents(
        Adapter(),  # type: ignore[arg-type]
        [
            (
                "11111111-1111-4111-8111-111111111111",
                "text one",
                {"candidate_id": "c1"},
                ["t1_a", "t1_b"],
            ),
            ("22222222-2222-4222-8222-222222222222", "text two", {"candidate_id": "c2"}),
        ],
    )
    assert indexed == 2
    points = upserts[0]["points"]
    assert points[0]["payload"]["context_message_refs"] == ["t1_a", "t1_b"]
    assert points[1]["payload"]["context_message_refs"] == []
