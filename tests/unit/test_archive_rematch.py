import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from reddit_search.evaluation.archive_partitioned import (
    audit_registered_archives_partitioned,
)
from reddit_search.evaluation.archive_rematch import rematch_archive_audit


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


@pytest.fixture()
def audit_run(tmp_path: Path) -> dict[str, Path]:
    """Run a real two-source audit and return its artifact paths."""
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
    audit_registered_archives_partitioned(
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
    return {
        "audit": tmp_path / "out",
        "checkpoint": tmp_path / "checkpoint.json",
        "corpus": corpus,
        "exposure": exposure,
        "tmp": tmp_path,
    }


def test_rematch_matches_audit_counts_and_is_byte_stable(audit_run: dict[str, Path]) -> None:
    """A rematch over the audit's own artifacts reproduces the audit's counts."""
    out_a = audit_run["tmp"] / "rematch-a"
    out_b = audit_run["tmp"] / "rematch-b"
    result_a = rematch_archive_audit(audit_run["audit"], out_a, corpus_path=audit_run["corpus"])
    result_b = rematch_archive_audit(audit_run["audit"], out_b, corpus_path=audit_run["corpus"])
    assert result_a["counts"] == {
        "matched": 1,
        "unresolved": 0,
        "collisions": 1,
        "exposure_errors": 0,
    }
    assert result_a["counts"] == result_b["counts"]
    assert result_a["kind"] == "archive_audit_rematch"
    assert result_a["provenance"] == "recomputed_from_index_artifacts"
    assert result_a["fresh_archive_scan"] is False
    assert result_a["supersedes_counts"] is True
    for name in (
        "archive_exposure_matches.jsonl",
        "archive_index_coverage.jsonl",
        "archive_identity_collisions.jsonl",
    ):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    assert (out_a / "rematch_manifest.json").read_bytes() == (
        out_b / "rematch_manifest.json"
    ).read_bytes()


def test_rematch_output_paths_bind_index_and_inputs(audit_run: dict[str, Path]) -> None:
    """The rematch manifest binds index partition hashes and input hashes."""
    out = audit_run["tmp"] / "rematch"
    result = rematch_archive_audit(audit_run["audit"], out, corpus_path=audit_run["corpus"])
    partitions = result["input_hashes"]["index_partitions"]
    assert partitions, "index partition hashes must be recorded"
    for entry in partitions.values():
        assert set(entry) == {"bytes", "sha256"}
    assert result["input_hashes"]["corpus"]["sha256"] == hashlib.sha256(
        audit_run["corpus"].read_bytes()
    ).hexdigest()
    assert result["input_hashes"]["exposure"]  # bound through from the audit
    assert result["audit_manifest_sha256"] == hashlib.sha256(
        (audit_run["audit"] / "archive_audit_manifest.json").read_bytes()
    ).hexdigest()


def test_rematch_fails_when_bucket_file_missing(audit_run: dict[str, Path]) -> None:
    """A dropped bucket file must fail validation before any output is written."""
    index_dir = audit_run["tmp"] / "index"
    bucket = sorted(index_dir.rglob("bucket-*.sqlite"))[0]
    bucket.unlink()
    out = audit_run["tmp"] / "rematch"
    with pytest.raises(ValueError, match="archive index artifacts"):
        rematch_archive_audit(audit_run["audit"], out, corpus_path=audit_run["corpus"])
    assert not out.exists() or not any(out.iterdir())


def test_rematch_fails_when_bucket_file_corrupted(audit_run: dict[str, Path]) -> None:
    """A tampered bucket file must fail its recorded sha256 binding."""
    index_dir = audit_run["tmp"] / "index"
    bucket = sorted(index_dir.rglob("bucket-*.sqlite"))[0]
    with sqlite3.connect(bucket) as db:
        db.execute("INSERT INTO archive VALUES ('x','y','z','w','q',1)")
    out = audit_run["tmp"] / "rematch"
    with pytest.raises(ValueError, match="sha256 mismatch|do not match"):
        rematch_archive_audit(audit_run["audit"], out, corpus_path=audit_run["corpus"])
    assert not out.exists() or not any(out.iterdir())


def test_rematch_fails_when_corpus_hash_mismatches(audit_run: dict[str, Path]) -> None:
    """A corpus override that does not hash-match the audit's binding fails."""
    other = audit_run["tmp"] / "other.sqlite"
    _corpus(other, "unused-revision")
    out = audit_run["tmp"] / "rematch"
    with pytest.raises(ValueError, match="corpus input sha256 mismatch"):
        rematch_archive_audit(audit_run["audit"], out, corpus_path=other)


def test_rematch_fails_when_manifest_tampered(audit_run: dict[str, Path]) -> None:
    """A manifest edited after the audit must be rejected."""
    manifest_path = audit_run["audit"] / "archive_audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["matched"] = 999999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = audit_run["tmp"] / "rematch"
    with pytest.raises(ValueError, match="sha256 mismatch|counts do not match"):
        rematch_archive_audit(audit_run["audit"], out, corpus_path=audit_run["corpus"])


def test_rematch_fails_when_audit_incomplete(audit_run: dict[str, Path]) -> None:
    """An incomplete audit must be rejected."""
    manifest_path = audit_run["audit"] / "archive_audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        rematch_archive_audit(
            audit_run["audit"], audit_run["tmp"] / "rematch", corpus_path=audit_run["corpus"]
        )
