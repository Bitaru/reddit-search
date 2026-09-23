"""Unit tests for deterministic group splits over pooled candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import zstandard

from reddit_search.evaluation.pooling import build_blinded_pool
from reddit_search.evaluation.splits import build_group_split


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _ranked(candidate_id: str, scenario: str, rank: int, thread: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "rank": rank,
        "score": -10.0 - rank,
        "message_fullname": candidate_id,
        "thread_fullname": thread,
        "focus_text": f"focus text {candidate_id}",
        "permalink": f"/r/test/comments/x/{candidate_id}/",
        "subreddit": "test",
        "created_utc": 1780000000,
        "context_complete": True,
        "retrieval": {"branch": "lexical", "scenario_id": scenario},
        "matched_rule_ids": ["rule.a"] if rank == 1 else [],
        "selection_channels": ["topic_rule"],
    }


@pytest.fixture()
def pool_file(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "scenario-a.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                _ranked("c1", "scenario-a", 1, "t3_a"),
                _ranked("c2", "scenario-a", 2, "t3_a"),
                _ranked("c3", "scenario-a", 4, "t3_b"),
                _ranked("c4", "scenario-a", 5, "t3_c"),
            ]
        )
    )
    controls = tmp_path / "rejected-controls.jsonl.zst"
    controls.write_bytes(zstandard.compress(json.dumps(_control()).encode()))
    output = tmp_path / "pool"
    build_blinded_pool(runs, controls, output, target_count=5)
    return output / "blinded_pool.jsonl"


def _control() -> dict[str, object]:
    return {
        "fullname": "k1",
        "kind": "comment",
        "thread_fullname": "t3_k1",
        "parent_fullname": None,
        "raw_title": "",
        "raw_body": "control body",
        "subreddit": "test",
        "created_utc": 1780000100,
        "source_revision_id": "rev-k1",
        "permalink": "/r/test/comments/x/k1/",
        "provenance": [{"source_id": "june-comments", "line_number": 1}],
        "archive_score": 4,
        "depth": 1,
        "selection_channels": [],
        "matched_rule_ids": [],
    }


def test_same_thread_candidates_share_group(pool_file: Path, tmp_path: Path) -> None:
    manifest = build_group_split(pool_file, tmp_path / "split", test_fraction=0.5)
    mapping = json.loads((tmp_path / "split" / "group_split_mapping.json").read_text())
    assert mapping["c1"]["group_id"] == mapping["c2"]["group_id"]
    assert mapping["c1"]["split"] == mapping["c2"]["split"]
    assert mapping["c3"]["group_id"] != mapping["c1"]["group_id"]
    assert manifest["dev_count"] + manifest["test_count"] == len(mapping)


def test_duplicate_links_merge_groups(pool_file: Path, tmp_path: Path) -> None:
    links = _write_jsonl(
        tmp_path / "dupes.jsonl",
        [{"candidate_id": "c3", "duplicate_of": "c1"}],
    )
    manifest = build_group_split(
        pool_file, tmp_path / "split", duplicate_links_path=links, test_fraction=0.5
    )
    mapping = json.loads((tmp_path / "split" / "group_split_mapping.json").read_text())
    assert mapping["c3"]["group_id"] == mapping["c1"]["group_id"]
    assert mapping["c3"]["split"] == mapping["c1"]["split"]
    assert manifest["group_count"] == 3  # {c1,c2,c3}, {c4}, {k1}


def test_split_is_deterministic_and_disjoint(pool_file: Path, tmp_path: Path) -> None:
    first = build_group_split(pool_file, tmp_path / "a", test_fraction=0.5)
    second = build_group_split(pool_file, tmp_path / "b", test_fraction=0.5)
    assert first == second
    mapping = json.loads((tmp_path / "a" / "group_split_mapping.json").read_text())
    dev = {c for c, entry in mapping.items() if entry["split"] == "dev"}
    test = {c for c, entry in mapping.items() if entry["split"] == "test"}
    assert set(dev) | set(test) == set(mapping)
    assert not (dev & test)

    pool_text = pool_file.read_text(encoding="utf-8")
    assert "stratum" not in pool_text


def test_invalid_test_fraction_rejected(pool_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_group_split(pool_file, tmp_path / "bad", test_fraction=0.0)
    with pytest.raises(ValueError):
        build_group_split(pool_file, tmp_path / "bad2", test_fraction=1.5)


def test_prior_review_forces_connected_group_to_development(
    pool_file: Path, tmp_path: Path
) -> None:
    prior = _write_jsonl(
        tmp_path / "prior.jsonl",
        [{"source": {"message_fullname": "c1"}}],
    )
    build_group_split(
        pool_file,
        tmp_path / "split",
        development_path=prior,
        test_fraction=0.5,
    )
    mapping = json.loads((tmp_path / "split" / "group_split_mapping.json").read_text())
    assert mapping["c1"]["split"] == "dev"
    assert mapping["c2"]["split"] == "dev"



def test_prior_thread_and_near_duplicate_force_connected_groups_to_development(
    tmp_path: Path,
) -> None:
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {
                "candidate_id": "current",
                "thread_fullname": "t3_current",
                "source": {
                    "message_fullname": "t1_current",
                    "text": (
                        "need a practical tool to track invoices and receipts "
                        "for my small business"
                    ),
                },
            },
            {
                "candidate_id": "thread-match",
                "thread_fullname": "t3_prior",
                "source": {"message_fullname": "t1_thread", "text": "unrelated short text"},
            },
        ],
    )
    prior = _write_jsonl(
        tmp_path / "prior.jsonl",
        [
            {
                "candidate_id": "prior-only",
                "thread_fullname": "t3_prior",
                "source": {
                    "message_fullname": "t1_prior",
                    "text": (
                        "need a practical tool to track invoices and receipts "
                        "for my small business today"
                    ),
                },
            }
        ],
    )

    manifest = build_group_split(pool, tmp_path / "split", development_path=prior)
    mapping = json.loads((tmp_path / "split" / "group_split_mapping.json").read_text())

    assert mapping["thread-match"]["split"] == "dev"
    assert mapping["current"]["split"] == "dev"
    assert manifest["development_thread_match_count"] == 1
    assert manifest["development_near_duplicate_match_count"] == 1

def test_near_duplicate_source_texts_share_group(tmp_path: Path) -> None:
    words = (
        "need a practical tool to track invoices and receipts for my small business "
        "while saving time every week for tax season"
    ).split()
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {
                "candidate_id": "c1",
                "thread_fullname": "t3_one",
                "source": {"message_fullname": "t1_c1", "text": " ".join(words)},
            },
            {
                "candidate_id": "c2",
                "thread_fullname": "t3_two",
                "source": {
                    "message_fullname": "t1_c2",
                    "text": " ".join(words + ["today"]),
                },
            },
        ],
    )
    manifest = build_group_split(pool, tmp_path / "split", test_fraction=0.5)
    mapping = json.loads((tmp_path / "split" / "group_split_mapping.json").read_text())
    assert mapping["c1"]["group_id"] == mapping["c2"]["group_id"]
    assert manifest["near_duplicate_link_count"] == 1


def test_same_candidate_scenarios_are_separate_split_tasks(tmp_path: Path) -> None:
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-a",
                "thread_fullname": "t3_one",
                "source": {"message_fullname": "t1_c1", "text": "need a practical tool"},
            },
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-b",
                "thread_fullname": "t3_one",
                "source": {"message_fullname": "t1_c1", "text": "need a practical tool"},
            },
        ],
    )

    manifest = build_group_split(pool, tmp_path / "split", test_fraction=0.5)
    split_rows = [
        json.loads(line)
        for line in (tmp_path / "split" / "candidate_splits.jsonl").read_text().splitlines()
    ]

    assert manifest["pool_count"] == 2
    assert manifest["pool_candidate_count"] == 1
    assert manifest["dev_task_count"] + manifest["test_task_count"] == 2
    assert {(row["candidate_id"], row["scenario_id"]) for row in split_rows} == {
        ("c1", "scenario-a"),
        ("c1", "scenario-b"),
    }
    assert len({row["split"] for row in split_rows}) == 1

def test_manifest_provenance_hashes_and_minimal_rows(tmp_path: Path) -> None:
    pool = _write_jsonl(tmp_path / "pool.jsonl", [
        {"candidate_id": "c1", "scenario_id": "s1", "snapshot_id": "snap",
         "app_id": "app", "source": {"source_revision_id": "rev"}},
    ])
    output = tmp_path / "split"
    manifest = build_group_split(pool, output)
    mapping = output / "group_split_mapping.json"
    splits = output / "candidate_splits.jsonl"
    assert manifest["dependency_identity_digest"]
    assert manifest["dependency_identity_coverage"]["candidate_id"] == 1
    assert manifest["mapping_sha256"] == hashlib.sha256(mapping.read_bytes()).hexdigest()
    assert manifest["candidate_splits_sha256"] == hashlib.sha256(splits.read_bytes()).hexdigest()
    row = json.loads(splits.read_text().splitlines()[0])
    assert set(row) == {"candidate_id", "scenario_id", "group_id", "split"}


def test_split_output_bytes_are_deterministic(tmp_path: Path) -> None:
    pool = _write_jsonl(tmp_path / "pool.jsonl", [{"candidate_id": "c1"}, {"candidate_id": "c2"}])
    first = tmp_path / "a"
    second = tmp_path / "b"
    build_group_split(pool, first)
    build_group_split(pool, second)
    for name in ("group_split_mapping.json", "candidate_splits.jsonl", "group_split_manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
