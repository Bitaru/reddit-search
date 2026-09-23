# ruff: noqa: E501
"""Live Qdrant acceptance for the configured tombstone operator (Gate C.3).

Runs only against a real loopback Qdrant server and only against synthetic
collections named ``gatec3-live-<unique>``. Skipped entirely in the hermetic
suite; opt in with ``REDDIT_SEARCH_LIVE_QDRANT=1`` and
``REDDIT_SEARCH_LIVE_QDRANT_URL`` (default ``http://127.0.0.1:6333``).
"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from reddit_search.corpus.sqlite_store import LexicalStore
from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.ingest.state import file_sha256
from reddit_search.retrieval.dense import (
    DenseBackendError,
    EmbeddingRecipe,
    QdrantHttpIndex,
)

LIVE = os.environ.get("REDDIT_SEARCH_LIVE_QDRANT") == "1"
BASE_URL = os.environ.get("REDDIT_SEARCH_LIVE_QDRANT_URL", "http://127.0.0.1:6333")

pytestmark = pytest.mark.skipif(
    not LIVE, reason="live Qdrant acceptance requires REDDIT_SEARCH_LIVE_QDRANT=1"
)

SNAPSHOT = "gatec3-live-snap"
DIMENSION = 8
RECIPE = EmbeddingRecipe(
    model_id="synthetic/fake-encoder",
    revision="0000000000000000000000000000000000000000",
    dimension=DIMENSION,
    query_instruction="Represent the synthetic gatec3 query.",
)


def _rest(method: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        BASE_URL + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"text": raw.decode("utf-8", errors="replace")}


def _server_count(collection: str, flt: dict | None = None) -> int:
    body: dict = {"exact": True}
    if flt is not None:
        body["filter"] = flt
    return int(_rest("POST", f"/collections/{collection}/points/count", body)["result"]["count"])


def _delete_collection_quietly(collection: str) -> None:
    try:
        _rest("DELETE", f"/collections/{collection}")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise


class _FakeAdapter:
    """Deterministic fixed-dimension encoder; no model, no downloads."""

    recipe = RECIPE

    def encode_documents(self, texts):
        return [
            [float((len(text) * (index + 3)) % 97 + 1) / 97.0 for index in range(DIMENSION)]
            for text in texts
        ]

    def encode_queries(self, texts):
        return [
            [float((len(text) * (index + 5)) % 89 + 1) / 89.0 for index in range(DIMENSION)]
            for text in texts
        ]


def _unit(unit_id: str, fullname: str, revision: str, refs: tuple[str, ...] = ()):
    from reddit_search.corpus.units import SearchUnit

    return SearchUnit(
        unit_id=unit_id,
        snapshot_id=SNAPSHOT,
        message_fullname=fullname,
        source_revision_id=revision,
        thread_fullname="t3_thread",
        focus_field="body",
        focus_start=0,
        focus_end=4,
        focus_text="text",
        context_only_text="",
        context_text="text",
        missing_context_ids=(),
        permalink="/x",
        subreddit="test",
        created_utc=1,
        synthetic=True,
        context_message_refs=refs,
    )


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reddit_search", *arguments],
        capture_output=True,
        check=False,
        text=True,
    )


_DIRECT_SELECTOR = {
    "must": [
        {"key": "snapshot_id", "match": {"value": SNAPSHOT}},
        {
            "must": [
                {"key": "message_fullname", "match": {"value": "t1_drop"}},
                {"key": "source_revision_id", "match": {"value": "revdrop"}},
            ]
        },
    ]
}
_CONTEXT_SELECTOR = {
    "must": [
        {"key": "snapshot_id", "match": {"value": SNAPSHOT}},
        {"must": [{"key": "context_message_refs", "match": {"value": "t1_drop"}}]},
    ]
}


def test_operate_live_end_to_end_server_truth_idempotent_rerun(tmp_path: Path) -> None:
    _rest("GET", "/healthz")
    collection = f"gatec3-live-{uuid.uuid4().hex[:12]}"
    _delete_collection_quietly(collection)
    try:
        source = tmp_path / "source.db"
        with LexicalStore(source) as store:
            store.index_units(
                [
                    _unit(str(uuid.uuid5(uuid.NAMESPACE_URL, "gatec3-drop")), "t1_drop", "revdrop"),
                    _unit(str(uuid.uuid5(uuid.NAMESPACE_URL, "gatec3-keep")), "t1_keep", "revkeep"),
                    _unit(
                        str(uuid.uuid5(uuid.NAMESPACE_URL, "gatec3-contrib")),
                        "t1_contrib",
                        "revcontrib",
                        refs=("t1_drop",),
                    ),
                ]
            )
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text(
            json.dumps(
                {
                    "message_fullname": "t1_drop",
                    "source_revision_id": "revdrop",
                    "reason": "gatec3 live acceptance",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        index = QdrantHttpIndex(BASE_URL, collection=collection, recipe=RECIPE)
        indexed = index.index_documents(
            _FakeAdapter(),
            [
                (
                    str(uuid.uuid5(uuid.NAMESPACE_URL, "gatec3-drop")),
                    "text",
                    {
                        "snapshot_id": SNAPSHOT,
                        "message_fullname": "t1_drop",
                        "source_revision_id": "revdrop",
                    },
                ),
                (
                    str(uuid.uuid5(uuid.NAMESPACE_URL, "gatec3-keep")),
                    "text",
                    {
                        "snapshot_id": SNAPSHOT,
                        "message_fullname": "t1_keep",
                        "source_revision_id": "revkeep",
                    },
                ),
                (
                    str(uuid.uuid5(uuid.NAMESPACE_URL, "gatec3-contrib")),
                    "text",
                    {
                        "snapshot_id": SNAPSHOT,
                        "message_fullname": "t1_contrib",
                        "source_revision_id": "revcontrib",
                    },
                    ("t1_drop",),
                ),
            ],
            expected_count=3,
        )
        assert indexed == 3
        count_before = _server_count(collection)
        assert count_before == 3

        arguments = [
            "tombstones",
            "operate",
            "--outbox",
            str(tmp_path / "outbox.db"),
            "--ledger",
            str(ledger),
            "--snapshot-id",
            SNAPSHOT,
            "--sqlite-input",
            str(source),
            "--sqlite-output",
            str(tmp_path / "output.db"),
            "--reconciliation",
            str(tmp_path / "reconciliation.json"),
            "--dense-mode",
            "loopback",
            "--dense-url",
            BASE_URL,
            "--dense-collection",
            collection,
            "--dense-model-id",
            RECIPE.model_id,
            "--dense-revision",
            RECIPE.revision,
            "--dense-dimension",
            str(DIMENSION),
            "--dense-query-instruction",
            RECIPE.query_instruction,
        ]
        result = _run_cli(*arguments)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == "applied"

        manifest = json.loads(
            (tmp_path / "reconciliation.json").read_text(encoding="utf-8")
        )
        dense_result = manifest["dense"]["result"]
        assert manifest["executed"] == {"sqlite": True, "dense": True}
        assert manifest["outbox"]["deleted_count"] == 2

        # Server-observed truth must match the manifest exactly.
        assert _server_count(collection, _DIRECT_SELECTOR) == 0
        assert _server_count(collection, _CONTEXT_SELECTOR) == 0
        assert _server_count(collection) == count_before - 2
        assert dense_result["deleted_count"] == 1
        assert dense_result["context_deleted_count"] == 1

        # Idempotent rerun: outbox replay no-ops; server count unchanged.
        rerun = _run_cli(*arguments)
        assert rerun.returncode == 0, rerun.stderr
        manifest2 = json.loads(
            (tmp_path / "reconciliation.json").read_text(encoding="utf-8")
        )
        core_first = {k: v for k, v in manifest.items() if k != "observation"}
        core_second = {k: v for k, v in manifest2.items() if k != "observation"}
        assert core_first == core_second
        assert manifest["observation"] != manifest2["observation"]
        assert _server_count(collection) == count_before - 2
    finally:
        _delete_collection_quietly(collection)


def test_operate_live_legacy_collection_fails_closed_before_deletion(
    tmp_path: Path,
) -> None:
    _rest("GET", "/healthz")
    collection = f"gatec3-live-legacy-{uuid.uuid4().hex[:12]}"
    _delete_collection_quietly(collection)
    try:
        _rest(
            "PUT",
            f"/collections/{collection}",
            {"vectors": {"size": DIMENSION, "distance": "Cosine"}},
        )
        _rest(
            "PUT",
            f"/collections/{collection}/points?wait=true",
            {
                "points": [
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "gatec3-legacy-keep")),
                        "vector": [0.5] * DIMENSION,
                        "payload": {
                            "snapshot_id": SNAPSHOT,
                            "message_fullname": "t1_legacy_other",
                            "source_revision_id": "revl",
                        },
                    },
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "gatec3-legacy-drop")),
                        "vector": [0.6] * DIMENSION,
                        "payload": {
                            "snapshot_id": SNAPSHOT,
                            "message_fullname": "t1_legacy_target",
                            "source_revision_id": "revl2",
                        },
                    },
                ]
            },
        )
        count_before = _server_count(collection)

        source = tmp_path / "source.db"
        with LexicalStore(source) as store:
            store.index_units([_unit(str(uuid.uuid4()), "t1_legacy_target", "revl2")])
        projection = project_tombstone_identities(
            [{"message_fullname": "t1_legacy_target", "source_revision_id": "revl2"}],
            source_artifacts={"sqlite": (source, file_sha256(source))},
        )
        index = QdrantHttpIndex(BASE_URL, collection=collection, recipe=RECIPE)
        with pytest.raises(DenseBackendError, match="context_message_refs"):
            index.delete_tombstone_projection(projection, snapshot_id=SNAPSHOT)
        # Fail-closed: the probe raises before any deletion is applied.
        assert _server_count(collection) == count_before
        assert (
            _server_count(
                collection,
                {
                    "must": [
                        {"key": "message_fullname", "match": {"value": "t1_legacy_target"}}
                    ]
                },
            )
            == 1
        )
    finally:
        _delete_collection_quietly(collection)
