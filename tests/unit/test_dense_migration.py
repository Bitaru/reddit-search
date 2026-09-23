"""Hermetic tests for the dense payload migration (fake Qdrant + temp corpus)."""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from reddit_search.retrieval.dense import EmbeddingRecipe, QdrantHttpIndex
from reddit_search.retrieval.dense_migration import (
    DenseMigrationError,
    migrate_collection,
    migrate_snapshot,
)

RECIPE = EmbeddingRecipe(
    model_id="test-model",
    revision="rev0",
    dimension=4,
    query_instruction="test query instruction",
)

SNAPSHOT = "test-snapshot"
SOURCE = "reddit_dense_testsource"
TARGET = f"{SOURCE}-ctx2"


def _vector(unit_id: str) -> list[float]:
    seed = int(unit_id.replace("-", "")[:8], 16)
    return [((seed >> shift) & 0xFF) / 255.0 + 0.01 for shift in (24, 16, 8, 0)]


def _legacy_payload(unit_id: str) -> dict[str, Any]:
    return {
        "snapshot_id": SNAPSHOT,
        "message_fullname": f"t3_{unit_id[:8]}",
        "permalink": f"/r/test/comments/{unit_id[:8]}",
        "subreddit": "test",
        "created_utc": 1_750_000_000,
        "focus_text": f"focus {unit_id[:8]}",
        "context_text": f"context {unit_id[:8]}",
        "thread_fullname": f"t3_thread{unit_id[:6]}",
        "context_missing": unit_id.endswith("0"),
    }


class _FakeQdrantHandler(BaseHTTPRequestHandler):
    server: _FakeQdrant  # type: ignore[override]

    def log_message(self, *_args: object) -> None:
        return

    def _respond(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        name = self.path.removeprefix("/collections/").split("?")[0]
        collection = self.server.collections.get(name)
        if collection is None:
            self._respond(404, {"status": "error", "result": None})
            return
        self._respond(
            200,
            {
                "status": "ok",
                "result": {
                    "config": {"params": {"vectors": {"size": 4, "distance": "Cosine"}}}
                },
            },
        )

    def do_PUT(self) -> None:  # noqa: N802
        parts = self.path.split("?")[0].strip("/").split("/")
        if parts[0] != "collections" or len(parts) not in {2, 3}:
            self._respond(404, {"status": "error"})
            return
        name = parts[1]
        if len(parts) == 2:
            vectors = self._body().get("vectors", {})
            self.server.collections[name] = {
                "size": vectors.get("size"),
                "distance": vectors.get("distance"),
                "points": {},
            }
            self._respond(200, {"status": "ok", "result": {"operation_id": 0}})
            return
        if parts[2] != "points":
            self._respond(404, {"status": "error"})
            return
        collection = self.server.collections.get(name)
        if collection is None:
            self._respond(404, {"status": "error"})
            return
        for point in self._body().get("points", []):
            collection["points"][point["id"]] = {
                "vector": list(point["vector"]),
                "payload": dict(point["payload"]),
            }
        self._respond(200, {"status": "ok", "result": {"operation_id": 0}})

    def do_POST(self) -> None:  # noqa: N802
        parts = self.path.split("?")[0].strip("/").split("/")
        if len(parts) != 4 or parts[0] != "collections" or parts[2] != "points":
            self._respond(404, {"status": "error"})
            return
        name, action = parts[1], parts[3]
        collection = self.server.collections.get(name)
        if collection is None:
            self._respond(404, {"status": "error"})
            return
        body = self._body()
        if action == "count":
            musts = body.get("filter", {}).get("must", [])
            filters = [
                (item["key"], item["match"]["value"])
                for item in musts
                if isinstance(item, dict) and "match" in item
            ]
            count = sum(
                1
                for point in collection["points"].values()
                if all(point["payload"].get(key) == value for key, value in filters)
            )
            self._respond(200, {"status": "ok", "result": {"count": count}})
            return
        if action != "scroll":
            self._respond(404, {"status": "error"})
            return
        points = collection["points"]
        must = body.get("filter", {}).get("must", [])
        wanted_ids: list[str] = []
        for item in must:
            if isinstance(item, dict) and "has_id" in item:
                value = item["has_id"]
                wanted_ids.extend(value if isinstance(value, list) else [value])
        if wanted_ids:
            matches = [
                {"id": pid, "vector": list(points[pid]["vector"])}
                for pid in wanted_ids
                if pid in points
            ]
            self._respond(200, {"status": "ok", "result": {"points": matches}})
            return
        limit = int(body.get("limit", 10))
        with_payload = bool(body.get("with_payload", False))
        with_vector = bool(body.get("with_vector", False))
        offset = body.get("offset")
        musts = body.get("filter", {}).get("must", [])
        filters = [
            (item["key"], item["match"]["value"])
            for item in musts
            if isinstance(item, dict) and "match" in item
        ]
        ordered_ids = sorted(
            pid
            for pid in points
            if all(points[pid]["payload"].get(key) == value for key, value in filters)
        )
        start = ordered_ids.index(offset) + 1 if offset in ordered_ids else 0
        window = ordered_ids[start : start + limit]
        scrolled = [
            {
                "id": pid,
                **({"payload": dict(points[pid]["payload"])} if with_payload else {}),
                **({"vector": list(points[pid]["vector"])} if with_vector else {}),
            }
            for pid in window
        ]
        has_more = len(window) == limit and start + limit < len(ordered_ids)
        next_offset = window[-1] if has_more else None
        self._respond(
            200,
            {"status": "ok", "result": {"points": scrolled, "next_page_offset": next_offset}},
        )


class _FakeQdrant(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], handler: type[_FakeQdrantHandler]
    ) -> None:
        super().__init__(address, handler)
        self.collections: dict[str, dict[str, Any]] = {}

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture()
def fake_qdrant() -> _FakeQdrant:
    server = _FakeQdrant(("127.0.0.1", 0), _FakeQdrantHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture()
def units() -> list[tuple[str, list[str]]]:
    return [
        ("aaaa0001-0000-4000-8000-000000000001", ["t1_b", "t1_a", "t1_b"]),
        ("aaaa0002-0000-4000-8000-000000000002", ["t1_c"]),
        ("aaaa0003-0000-4000-8000-000000000003", []),
        ("aaaa0004-0000-4000-8000-000000000004", ["t1_a", "t1_d", "t1_a", "t1_d"]),
        ("aaaa0005-0000-4000-8000-000000000005", ["t1_e"]),
    ]


@pytest.fixture()
def corpus_path(tmp_path: Path, units: list[tuple[str, list[str]]]) -> Path:
    path = tmp_path / "corpus.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE search_units (unit_id TEXT, snapshot_id TEXT, "
            "message_fullname TEXT, context_message_refs TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO search_units (unit_id, snapshot_id, message_fullname, "
            "context_message_refs) VALUES (?, ?, ?, ?)",
            [
                (unit_id, SNAPSHOT, f"t3_{unit_id[:8]}", json.dumps(refs))
                for unit_id, refs in units
            ],
        )
    return path


def _seed_legacy_collection(
    server: _FakeQdrant, name: str, *, units: list[tuple[str, list[str]]]
) -> None:
    server.collections[name] = {
        "size": 4,
        "distance": "Cosine",
        "points": {
            unit_id: {"vector": _vector(unit_id), "payload": _legacy_payload(unit_id)}
            for unit_id, _ in units
        },
    }


def _run(server: _FakeQdrant, corpus_path: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "source_collection": SOURCE,
        "new_collection": TARGET,
        "corpus_path": corpus_path,
        "snapshot_id": SNAPSHOT,
        "recipe": RECIPE,
    }
    kwargs.update(overrides)
    return migrate_snapshot(server.base_url, **kwargs)


def test_happy_path_migrates_with_refs(fake_qdrant, corpus_path, units) -> None:
    _seed_legacy_collection(fake_qdrant, SOURCE, units=units)
    result = _run(fake_qdrant, corpus_path)

    assert result["source"] == SOURCE
    assert result["new"] == TARGET
    assert result["counts"] == {
        "source": len(units),
        "upserted": len(units),
        "new": len(units),
    }
    assert result["missed"] == []
    assert result["verified"]["vector_mismatches"] == 0
    assert result["verified"]["vector_sample"] >= 4
    target = fake_qdrant.collections[TARGET]
    assert len(target["points"]) == len(units)
    for unit_id, refs in units:
        point = target["points"][unit_id]
        assert point["payload"]["context_message_refs"] == sorted(set(refs))
        assert point["vector"] == _vector(unit_id)
        assert point["payload"]["snapshot_id"] == SNAPSHOT


def test_duplicate_refs_sorted_and_legacy_payload_preserved(
    fake_qdrant, corpus_path, units
) -> None:
    _seed_legacy_collection(fake_qdrant, SOURCE, units=units)
    _run(fake_qdrant, corpus_path)
    legacy = fake_qdrant.collections[SOURCE]["points"]
    migrated = fake_qdrant.collections[TARGET]["points"]
    duplicated = units[3][0]
    assert migrated[duplicated]["payload"]["context_message_refs"] == ["t1_a", "t1_d"]
    # Legacy payload survives verbatim; only the refs key is added.
    assert migrated[duplicated]["payload"] == {
        **legacy[duplicated]["payload"],
        "context_message_refs": ["t1_a", "t1_d"],
    }
    assert "context_message_refs" not in legacy[duplicated]["payload"]


def test_missing_join_fails_closed_before_creating_target(
    fake_qdrant, corpus_path, units
) -> None:
    _seed_legacy_collection(fake_qdrant, SOURCE, units=units)
    orphan = "aaaa0bad-0000-4bad-8000-000000000000"
    fake_qdrant.collections[SOURCE]["points"][orphan] = {
        "vector": _vector(orphan),
        "payload": {"snapshot_id": SNAPSHOT},
    }
    with pytest.raises(DenseMigrationError) as excinfo:
        _run(fake_qdrant, corpus_path)
    assert excinfo.value.misses == [orphan]
    assert TARGET not in fake_qdrant.collections


def test_missing_join_direct_api_reports_misses(fake_qdrant, corpus_path, units) -> None:
    _seed_legacy_collection(fake_qdrant, SOURCE, units=units)
    index = QdrantHttpIndex(fake_qdrant.base_url, collection=SOURCE, recipe=RECIPE)
    with pytest.raises(DenseMigrationError) as excinfo:
        migrate_collection(
            index,
            source_collection=SOURCE,
            new_collection=TARGET,
            refs_by_unit={unit_id: sorted(set(refs)) for unit_id, refs in units[:4]},
        )
    assert excinfo.value.misses == [units[4][0]]
    assert TARGET not in fake_qdrant.collections


def test_rerun_refused_when_target_nonempty(fake_qdrant, corpus_path, units) -> None:
    _seed_legacy_collection(fake_qdrant, SOURCE, units=units)
    _run(fake_qdrant, corpus_path)
    with pytest.raises(DenseMigrationError, match="force=True"):
        _run(fake_qdrant, corpus_path)
    result = _run(fake_qdrant, corpus_path, force=True)
    assert result["counts"]["new"] == len(units)


def test_snapshot_scoped_scroll_ignores_other_snapshots(
    fake_qdrant, corpus_path, units
) -> None:
    _seed_legacy_collection(fake_qdrant, SOURCE, units=units)
    stranger = "aaaa0a57-0000-4000-8000-000000000009"
    fake_qdrant.collections[SOURCE]["points"][stranger] = {
        "vector": _vector(stranger),
        "payload": {"snapshot_id": "other-snapshot"},
    }
    result = _run(fake_qdrant, corpus_path)
    assert result["counts"]["new"] == len(units)
    assert stranger not in fake_qdrant.collections[TARGET]["points"]


def test_filtered_out_points_do_not_stall_pagination(fake_qdrant, corpus_path, units) -> None:
    """Points outside the snapshot filter must be skipped without looping forever."""
    _seed_legacy_collection(fake_qdrant, SOURCE, units=units)
    filtered_out = "ffff0f00-0000-4000-8000-000000000000"
    fake_qdrant.collections[SOURCE]["points"][filtered_out] = {
        "vector": _vector(filtered_out),
        "payload": {"snapshot_id": "other-snapshot"},
    }
    result = _run(fake_qdrant, corpus_path, batch_size=2)
    assert result["counts"]["new"] == len(units)
    assert filtered_out not in fake_qdrant.collections[TARGET]["points"]
