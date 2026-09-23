from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import zstandard

from reddit_search.ingest.invalidation import (
    audit_tombstone_outputs,
    load_tombstone_ledger,
    project_tombstone_identities,
    tombstone_identity,
)


def _write_ledger(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_tombstone_ledger_matches_full_message_and_exact_revision(tmp_path: Path) -> None:
    ledger_path = _write_ledger(
        tmp_path / "tombstones.jsonl",
        [
            {
                "message_fullname": "t1_exact",
                "source_revision_id": "rev-2",
                "reason": "corrected source revision",
            },
            {
                "message_fullname": "t3_all",
                "source_revision_id": None,
                "reason": "operator removal",
            },
        ],
    )

    ledger = load_tombstone_ledger(ledger_path)

    assert ledger.matches("t1_exact", "rev-2") is True
    assert ledger.matches("t1_exact", "rev-1") is False
    assert ledger.matches("t3_all", "any-revision") is True
    assert ledger.count == 2
    assert tombstone_identity(ledger) == {
        "path": str(ledger_path.resolve()),
        "sha256": ledger.digest,
        "count": 2,
    }


def test_tombstone_ledger_rejects_unknown_fields_and_duplicates(tmp_path: Path) -> None:
    duplicate = _write_ledger(
        tmp_path / "duplicate.jsonl",
        [
            {
                "message_fullname": "t1_same",
                "source_revision_id": None,
                "reason": "first",
            },
            {
                "message_fullname": "t1_same",
                "source_revision_id": None,
                "reason": "second",
            },
        ],
    )
    with pytest.raises(ValueError, match="duplicate tombstone"):
        load_tombstone_ledger(duplicate)

    unknown = _write_ledger(
        tmp_path / "unknown.jsonl",
        [
            {
                "message_fullname": "t1_unknown",
                "source_revision_id": None,
                "reason": "test",
                "extra": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_tombstone_ledger(unknown)


def _write_selection(path: Path, fullname: str, revision: str) -> Path:
    payload = {
        "fullname": fullname,
        "kind": "comment",
        "thread_fullname": "t3_root",
        "parent_fullname": "t3_root",
        "raw_title": "",
        "raw_body": "Body.",
        "subreddit": "personalfinance",
        "created_utc": 1,
        "source_revision_id": revision,
        "permalink": "/r/personalfinance/comments/root/x/",
        "provenance": [{"source_id": "test", "line_number": 1}],
        "archive_score": 1,
        "depth": 1,
        "selection_channels": [],
        "matched_rule_ids": [],
    }
    with path.open("wb") as output:
        with zstandard.ZstdCompressor().stream_writer(output, closefd=False) as compressed:
            compressed.write(json.dumps(payload).encode() + b"\n")
    return path


def test_tombstone_audit_reports_stale_clean_and_missing_outputs(tmp_path: Path) -> None:
    ledger_path = _write_ledger(
        tmp_path / "tombstones.jsonl",
        [
            {
                "message_fullname": "t1_bad",
                "source_revision_id": None,
                "reason": "removed by operator",
            }
        ],
    )
    selection = _write_selection(tmp_path / "selection.jsonl.zst", "t1_bad", "rev-bad")
    review_cards = tmp_path / "review_cards.jsonl"
    review_cards.write_text(
        json.dumps(
            {
                "source": {
                    "message_fullname": "t3_good",
                    "source_revision_id": "rev-good",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.db"
    connection = sqlite3.connect(corpus)
    try:
        connection.execute(
            "CREATE TABLE search_units (message_fullname TEXT, source_revision_id TEXT)"
        )
        connection.execute(
            "INSERT INTO search_units VALUES (?, ?)",
            ("t3_good", "rev-good"),
        )
        connection.commit()
    finally:
        connection.close()

    report_path = tmp_path / "audit" / "report.json"
    report = audit_tombstone_outputs(
        ledger_path,
        report_path,
        selection_path=selection,
        hydration_directory=tmp_path / "missing-hydration",
        review_cards_path=review_cards,
        corpus_path=corpus,
    )

    statuses = {artifact["kind"]: artifact["status"] for artifact in report["artifacts"]}
    assert statuses == {
        "corpus_sqlite": "clean",
        "hydrated_context_shard": "missing",
        "rejected_controls_shard": "missing",
        "review_cards": "clean",
        "selection_shard": "stale",
    }
    assert report["summary"] == {
        "artifact_count": 5,
        "clean_artifact_count": 2,
        "stale_artifact_count": 1,
        "missing_artifact_count": 2,
        "invalidated_record_count": 1,
        "matched_tombstone_count": 1,
    }
    assert report["all_supplied_artifacts_clean"] is False
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_tombstone_projection_is_deterministic_and_does_not_mutate_inputs(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "source.jsonl"
    artifact.write_text("source\n", encoding="utf-8")
    rows = [
        {"message_fullname": "t3_b", "source_revision_id": "rev-2", "unit_id": "u2"},
        {"message_fullname": "t1_a", "source_revision_id": "rev-1", "candidate_id": "c1"},
    ]
    original = [dict(row) for row in rows]
    expected_hash = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()

    first = project_tombstone_identities(
        reversed(rows),
        source_artifacts={"source": (artifact, expected_hash)},
    )
    second = project_tombstone_identities(
        rows,
        source_artifacts={"source": (artifact, expected_hash)},
    )

    assert rows == original
    assert first == second
    assert [row["message_fullname"] for row in first.records] == ["t1_a", "t3_b"]
    assert first.as_dict()["propagation_status"] == "not_claimed"
    assert first.as_dict()["counts"] == {"records": 2, "artifacts": 1}
    assert first.as_dict()["source_artifact_hashes"] == {"source": expected_hash}



def test_projection_accepts_fullname_wide_tombstone_and_rejects_bad_revision() -> None:
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_all", "source_revision_id": None, "snapshot_id": "snap"}]
    )
    assert projection.records == (
        {"message_fullname": "t1_all", "source_revision_id": None, "snapshot_id": "snap"},
    )
    assert len(projection.ledger_digest) == 64
    with pytest.raises(ValueError, match="invalid identity"):
        project_tombstone_identities(
            [{"message_fullname": "t1_bad", "source_revision_id": 3}]
        )
def test_tombstone_projection_rejects_duplicates_conflicts_and_archive_coverage() -> None:
    duplicate = [
        {"message_fullname": "t3_same", "source_revision_id": "rev", "unit_id": "u1"},
        {"message_fullname": "t3_same", "source_revision_id": "rev", "unit_id": "u2"},
    ]
    with pytest.raises(ValueError, match="duplicate tombstone identity"):
        project_tombstone_identities(duplicate)

    for field in ("status", "coverage_status"):
        with pytest.raises(ValueError, match="archive coverage"):
            project_tombstone_identities(
                [{"message_fullname": "t3_archive", "source_revision_id": "rev", field: "matched"}]
            )
        with pytest.raises(ValueError, match="archive coverage"):
            project_tombstone_identities(
                [
                    {
                        "message_fullname": "t3_archive",
                        "source_revision_id": "rev",
                        field: "unresolved",
                    }
                ]
            )


def test_tombstone_projection_rejects_source_artifact_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "source.jsonl"
    artifact.write_text("source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source artifact hash mismatch"):
        project_tombstone_identities(
            [{"message_fullname": "t3_one", "source_revision_id": "rev"}],
            source_artifacts={"source": (artifact, "0" * 64)},
        )
