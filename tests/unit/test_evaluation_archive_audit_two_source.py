import hashlib
import json
import sqlite3
from pathlib import Path

from reddit_search.evaluation.archive_partitioned import (
    audit_registered_archives_partitioned,
)


def _submission(name: str, title: str, body: str) -> dict[str, object]:
    return {
        "name": name,
        "title": title,
        "selftext": body,
        "subreddit": "personalfinance",
        "created_utc": 1,
        "permalink": f"/r/personalfinance/comments/{name}/",
    }


def _revision(title: str, body: str) -> str:
    return hashlib.sha256(f"{title}\0{body}".encode()).hexdigest()


def _corpus(path: Path, revision: str) -> None:
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE search_units (unit_id TEXT, candidate_id TEXT, snapshot_id TEXT, "
        "message_fullname TEXT, source_revision_id TEXT, thread_fullname TEXT, focus_text TEXT)"
    )
    db.executemany(
        "INSERT INTO search_units VALUES (?,?,?,?,?,?,?)",
        [
            ("u1", "candidate-1", "snap", "t3_hit", revision, "t3_hit", "Need help\n\nBody"),
        ],
    )
    db.commit()
    db.close()


def _jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_partitioned_matches_probe_every_source_bucket_and_detect_cross_source_collisions(
    tmp_path: Path,
) -> None:
    """Identities stored only in a later source must still match.

    Connections are keyed by (source_id, bucket); a regression where they were
    keyed by bucket alone made matches_for probe only one source per bucket.
    """
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    duplicate = _submission("t3_hit", "Need help", "Body")
    _jsonl(first, [duplicate])
    _jsonl(second, [_submission("t3_hit", "Need help", "Body"), duplicate])
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {"source_id": "a", "kind": "submission", "source_path": first.name},
                    {"source_id": "b", "kind": "submission", "source_path": second.name},
                ]
            }
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    revision = _revision("Need help", "Body")
    _corpus(corpus, revision)
    exposure = tmp_path / "exposure.jsonl"
    _jsonl(exposure, [{"candidate_id": "candidate-1", "scenario_id": "s1"}])
    result = audit_registered_archives_partitioned(
        registry,
        corpus,
        [exposure],
        tmp_path / "out",
        max_records=10,
        index_directory=tmp_path / "index",
        checkpoint_path=tmp_path / "checkpoint.json",
        min_free_disk_bytes=0,
        max_process_rss_bytes=10**12,
    )
    assert result["complete"] is True
    assert result["counts"]["matched"] == 1
    assert result["counts"]["unresolved"] == 0
    assert result["counts"]["collisions"] >= 1
    matches = [
        json.loads(line)
        for line in (tmp_path / "out" / "archive_exposure_matches.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    archive_matches = matches[0]["archive_matches"]
    assert archive_matches == sorted(
        archive_matches, key=lambda item: (item["source_id"], item["line"])
    )
    assert {item["source_id"] for item in archive_matches} == {"a", "b"}
    collisions = [
        json.loads(line)
        for line in (tmp_path / "out" / "archive_identity_collisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert collisions
    assert any(item["archive_count"] >= 2 for item in collisions)
