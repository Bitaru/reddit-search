import json
from pathlib import Path


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_reuse_skips_invalidated_source_context(tmp_path: Path) -> None:
    from reddit_search.evaluation.reuse import reuse_labels
    from reddit_search.ingest.invalidation import load_tombstone_ledger

    pool_row = {
        "candidate_id": "c1",
        "scenario_id": "s1",
        "source": {"message_fullname": "t1_live", "source_revision_id": "r1", "text": "x"},
        "context": {"message_fullnames": ["t1_context"]},
    }
    source_pool = _write(tmp_path / "source.jsonl", [pool_row])
    target_pool = _write(tmp_path / "target.jsonl", [pool_row])
    label = {
        "candidate_id": "c1", "scenario_id": "s1", "topic_fit": "yes",
        "evaluator_kind": "human",
    }
    labels = _write(tmp_path / "labels.jsonl", [label])
    tombstone = {"message_fullname": "t1_context", "source_revision_id": None, "reason": "gone"}
    tombstones = _write(tmp_path / "tombstones.jsonl", [tombstone])
    report = reuse_labels(
        source_pool, labels, target_pool, tmp_path / "out", rubric_version="topic-fit-v1",
        tombstone_ledger=load_tombstone_ledger(tombstones),
    )
    assert report["skipped"]["invalidated_source_or_context"] == 1
    assert report["reused_label_count"] == 0
