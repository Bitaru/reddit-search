import json
import sqlite3
from pathlib import Path

import pytest

from reddit_search.operations.preflight import DenseCollectionProbe, PreflightConfig, _check_dense
from reddit_search.retrieval.dense_repair import DenseRepairError, repair_snapshot


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path]:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "alias": "alias",
                        "collection": "alias-ctx2",
                        "snapshot_id": "target",
                        "point_count": 2,
                    }
                ]
            }
        )
    )
    corpus = tmp_path / "corpus.db"
    db = sqlite3.connect(corpus)
    db.execute("create table search_units(snapshot_id text)")
    db.executemany("insert into search_units values (?)", [("target",), ("target",)])
    db.commit()
    db.close()
    return registry, corpus


def test_dry_run_emits_no_put_and_counts_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, corpus = _fixture_inputs(tmp_path)
    calls: list[tuple[str, str, dict | None]] = []

    def request(base: str, method: str, path: str, body: dict | None = None) -> dict:
        calls.append((method, path, body))
        if path.endswith("/count"):
            return {
                "status": "ok",
                "result": {
                    "count": 2
                    if "snapshot_id" not in json.dumps(body) or "full" in json.dumps(body)
                    else 0
                },
            }
        return {
            "status": "ok",
            "result": {
                "points": [
                    {"id": "u1", "payload": {"snapshot_id": "full"}},
                    {"id": "u2", "payload": {"snapshot_id": "full"}},
                ],
                "next_page_offset": None,
            },
        }

    monkeypatch.setattr("reddit_search.retrieval.dense_repair._request", request)
    result = repair_snapshot(
        qdrant_url="http://127.0.0.1:6333",
        registry_path=registry,
        corpus_path=corpus,
        alias="alias",
        source_snapshot="full",
        target_snapshot="target",
    )
    assert result["dry_run"] is True and result["changed"] == 2
    assert not [call for call in calls if call[0] == "PUT"]


def test_apply_emits_bounded_snapshot_only_payload_put(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, corpus = _fixture_inputs(tmp_path)
    calls: list[tuple[str, str, dict | None]] = []

    def request(base: str, method: str, path: str, body: dict | None = None) -> dict:
        calls.append((method, path, body))
        if path.endswith("/count"):
            count_calls = sum(
                1
                for method_name, path_name, _ in calls
                if method_name == "POST" and path_name.endswith("/count")
            )
            return {
                "status": "ok",
                "result": {"count": 0 if count_calls == 3 else 1 if count_calls == 5 else 2},
            }
        if path.endswith("/scroll"):
            return {
                "status": "ok",
                "result": {
                    "points": [{"id": "u1", "payload": {"snapshot_id": "full"}}],
                    "next_page_offset": None,
                },
            }
        return {"status": "ok", "result": {}}

    monkeypatch.setattr("reddit_search.retrieval.dense_repair._request", request)
    result = repair_snapshot(
        qdrant_url="http://127.0.0.1:6333",
        registry_path=registry,
        corpus_path=corpus,
        alias="alias",
        source_snapshot="full",
        target_snapshot="target",
        batch_size=1,
        dry_run=False,
    )
    puts = [body for method, path, body in calls if method == "PUT"]
    assert result["after_total"] == 2 and len(puts) == 1
    assert puts[0] == {"payload": {"snapshot_id": "target"}, "points": ["u1"]}


def test_preflight_resolves_alias_but_leaves_unmapped_test_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {"entries": [{"alias": "alias", "collection": "alias-ctx2", "snapshot_id": "target"}]}
        )
    )
    monkeypatch.setattr(
        "reddit_search.operations.preflight._probe_collection",
        lambda base, probe: ("ok", {"collection": probe.collection_name}, []),
    )
    monkeypatch.setattr(
        "reddit_search.operations.preflight._rest_json",
        lambda *args: {"result": {"collections": [{"name": "alias-ctx2"}, {"name": "test"}]}},
    )
    status, detail, reasons = _check_dense(
        PreflightConfig(
            qdrant_base_url="http://127.0.0.1:6333",
            dense_collections=(DenseCollectionProbe(collection_name="alias"),),
            dense_registry_path=registry,
        )
    )
    assert status == "warn"
    assert detail["collections"][0]["collection"] == "alias-ctx2"
    assert detail["unmapped_collections"] == ["test"]
    assert any("test" in reason for reason in reasons)


def test_loopback_rejection(tmp_path: Path) -> None:
    with pytest.raises(DenseRepairError, match="loopback"):
        repair_snapshot(
            qdrant_url="https://qdrant.example",
            registry_path=tmp_path / "r",
            corpus_path=tmp_path / "c",
            alias="a",
            source_snapshot="s",
            target_snapshot="t",
        )


def test_registry_mismatch_fails_closed(tmp_path: Path) -> None:
    reg = tmp_path / "r"
    reg.write_text(
        json.dumps(
            {"entries": [{"alias": "a", "collection": "x", "snapshot_id": "t", "point_count": 1}]}
        )
    )
    with pytest.raises(DenseRepairError):
        repair_snapshot(
            qdrant_url="http://127.0.0.1:6333",
            registry_path=reg,
            corpus_path=tmp_path / "c",
            alias="a",
            source_snapshot="s",
            target_snapshot="t",
        )


def test_missing_alias_fails_closed(tmp_path: Path) -> None:
    reg = tmp_path / "r"
    reg.write_text(json.dumps({"entries": []}))
    with pytest.raises(DenseRepairError, match="uniquely"):
        repair_snapshot(
            qdrant_url="http://127.0.0.1:6333",
            registry_path=reg,
            corpus_path=tmp_path / "c",
            alias="a",
            source_snapshot="s",
            target_snapshot="t",
        )
