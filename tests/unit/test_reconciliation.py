import json
from pathlib import Path

import pytest

from reddit_search.ingest.invalidation import (
    Tombstone,
    TombstoneLedger,
    project_tombstone_identities,
)
from reddit_search.operations.reconciliation import (
    reconcile_snapshot_collection,
    write_reconciliation_manifest,
)
from reddit_search.retrieval.dense import DenseIndexJobStore, EmbeddingRecipe


@pytest.fixture
def recipe() -> EmbeddingRecipe:
    return EmbeddingRecipe("model", "rev-1", 2, "Represent the query.")


def unit(
    unit_id: str, content: str, *, context: str = "context", fullname: str = "t1_root"
) -> dict:
    return {
        "unit_id": unit_id,
        "content_hash": content,
        "context_hash": context,
        "message_fullname": fullname,
        "source_revision_id": "rev",
    }


def test_reconciliation_is_byte_stable_and_reports_added_removed_changed(
    recipe: EmbeddingRecipe,
) -> None:
    current = [unit("a", "new"), unit("c", "same")]
    previous = [unit("a", "old"), unit("b", "gone"), unit("c", "same")]
    first = reconcile_snapshot_collection(
        snapshot_id="snap", recipe=recipe, current_units=current, previous_units=previous
    )
    second = reconcile_snapshot_collection(
        snapshot_id="snap", recipe=recipe, current_units=current, previous_units=previous
    )
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert first["units"]["added"] == []
    assert first["units"]["removed"] == ["b"]
    assert first["units"]["changed"] == ["a"]


def test_index_job_pending_and_indexed_content_mismatch(
    recipe: EmbeddingRecipe, tmp_path: Path
) -> None:
    jobs_path = tmp_path / "jobs.sqlite"
    with DenseIndexJobStore(jobs_path) as jobs:
        jobs.prepare(snapshot_id="snap", recipe=recipe, units=[("a", "old"), ("b", "ok")])
        jobs.mark_indexed(snapshot_id="snap", recipe=recipe, unit_id="b")
    result = reconcile_snapshot_collection(
        snapshot_id="snap",
        recipe=recipe,
        current_units=[unit("a", "new"), unit("b", "ok")],
        index_jobs_path=jobs_path,
    )
    assert result["index"]["pending_reindex"] == ["a"]
    assert result["index"]["indexed_unit_ids"] == ["b"]
    assert result["unresolved_mismatches"][0]["kind"] == "indexed_content_mismatch"


def test_snapshot_recipe_collection_mismatches_are_reported(recipe: EmbeddingRecipe) -> None:
    manifest = {
        "snapshot_id": "other",
        "recipe": {**recipe.manifest(), "revision": "other"},
        "collection": "wrong",
        "actual_count": 99,
    }
    result = reconcile_snapshot_collection(
        snapshot_id="snap", recipe=recipe, current_units=[unit("a", "x")], index_manifest=manifest
    )
    assert {item["field"] for item in result["collection"]["mismatches"]} == {
        "snapshot_id",
        "recipe",
        "collection",
        "actual_count",
    }


def test_tombstone_blocks_context_rows(recipe: EmbeddingRecipe) -> None:
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_blocked", "source_revision_id": None}]
    )
    ledger = TombstoneLedger(
        path=None,
        digest=projection.ledger_digest,
        records=(Tombstone("t1_blocked", None, "test"),),
        _all_revisions=frozenset({"t1_blocked"}),
        _by_revision={},
    )
    result = reconcile_snapshot_collection(
        snapshot_id="snap",
        recipe=recipe,
        current_units=[
            unit("a", "x", fullname="t1_live") | {"context_message_refs": ["t1_blocked"]}
        ],
        tombstone_ledger=ledger,
    )
    assert result["units"]["tombstone_blocked"] == ["a"]
    assert result["index"]["expected_count"] == 0


def test_manifest_write_is_atomic_and_replaces_previous(
    tmp_path: Path, recipe: EmbeddingRecipe
) -> None:
    path = tmp_path / "manifest.json"
    result = reconcile_snapshot_collection(snapshot_id="snap", recipe=recipe, current_units=[])
    write_reconciliation_manifest(path, result)
    assert json.loads(path.read_text()) == result
    write_reconciliation_manifest(path, {"kind": "replacement"})
    assert json.loads(path.read_text()) == {"kind": "replacement"}
    assert not path.with_suffix(".json.tmp").exists()
