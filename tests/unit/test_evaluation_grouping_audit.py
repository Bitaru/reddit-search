from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from reddit_search.evaluation.grouping_audit import audit_corpus_grouping


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _build_corpus(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE search_units (
            unit_id TEXT, message_fullname TEXT, thread_fullname TEXT,
            focus_text TEXT, context_message_refs TEXT, snapshot_id TEXT
        )"""
    )
    long_a = (
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty"
    )
    long_b = long_a
    rows = [
        ("c1", "m1", "t1", "Alpha beta", json.dumps(["m4"]), "snap"),
        ("c2", "m2", "t1", " alpha   beta ", "[]", "snap"),
        ("c3", "m3", "t2", long_a, "[]", "snap"),
        ("c4", "m4", "t3", long_b, "[]", "snap"),
        ("c5", "m5", "t5", "unrelated words", "[]", "snap"),
    ]
    connection.executemany("INSERT INTO search_units VALUES (?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()


def test_grouping_audit_is_deterministic_and_complete(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.sqlite"
    _build_corpus(corpus)
    exposures = tmp_path / "exposures.jsonl"
    _write_jsonl(
        exposures,
        [
            {"candidate_id": "c1", "scenario_id": "s1"},
            {"candidate_id": "c1", "scenario_id": "s2"},
            {"candidate_id": "missing", "scenario_id": "s3"},
        ],
    )
    links = tmp_path / "links.jsonl"
    _write_jsonl(
        links,
        [
            {"candidate_id": "c1", "duplicate_of": "c2"},
            {"candidate_id": "c1", "duplicate_of": "missing"},
        ],
    )
    splits = tmp_path / "splits.jsonl"
    _write_jsonl(
        splits,
        [{"candidate_id": "c1", "split": "development"}, {"candidate_id": "c2", "split": "test"}],
    )
    split_bytes = splits.read_bytes()

    first = audit_corpus_grouping(
        corpus,
        [exposures],
        tmp_path / "out-a",
        duplicate_links_path=links,
        candidate_splits_path=splits,
        snapshot_id="snap",
    )
    second = audit_corpus_grouping(
        corpus,
        [exposures],
        tmp_path / "out-b",
        duplicate_links_path=links,
        candidate_splits_path=splits,
        snapshot_id="snap",
    )

    assert first["kind"] == "corpus_grouping_audit"
    assert first["complete"] is True
    counts = first["counts"]
    for relation in ("same_thread", "context_ref", "exact_text", "near_text", "duplicate_link"):
        assert counts["relation_rows"][relation] >= 1
    assert counts["matched_events"] == 2
    assert counts["unresolved_events"] == 1
    assert counts["split_crossing_groups"] == 1
    assert first["unresolved"]["duplicate_link_references"] == 1
    assert splits.read_bytes() == split_bytes

    events = [
        json.loads(line)
        for line in (tmp_path / "out-a" / "exposure_matches.jsonl").read_text().splitlines()
    ]
    assert [event["payload_identity"]["scenario_id"] for event in events] == ["s1", "s2", "s3"]
    for name in (
        "group_membership.jsonl",
        "exposure_matches.jsonl",
        "grouping_audit_manifest.json",
    ):
        assert (tmp_path / "out-a" / name).read_bytes() == (tmp_path / "out-b" / name).read_bytes()
    assert second["counts"] == first["counts"]


def test_grouping_audit_rejects_malformed_exposure(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.sqlite"
    _build_corpus(corpus)
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"candidate_id":\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        audit_corpus_grouping(corpus, [malformed], tmp_path / "out")
