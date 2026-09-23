from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reddit_search.cli import app
from reddit_search.evaluation.unresolved_diagnostic import diagnose_unresolved

runner = CliRunner()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path, *, duplicate: bool = False, index_hit: bool = False
) -> tuple[Path, Path, Path, Path]:
    audit = tmp_path / "audit"
    audit.mkdir()
    corpus = tmp_path / "corpus.sqlite"
    connection = sqlite3.connect(corpus)
    connection.execute(
        "CREATE TABLE search_units (snapshot_id TEXT, unit_id TEXT PRIMARY KEY, "
        "message_fullname TEXT, source_revision_id TEXT, thread_fullname TEXT, focus_text TEXT)"
    )
    rows = [("snap", "unit-1", "t3_submission", "revision", "t3_submission", "Title\n\nBody")]
    if duplicate:
        rows.append(
            ("snap", "unit-2", "t3_submission", "revision", "t3_submission", "Title\n\nBody")
        )
    connection.executemany("INSERT INTO search_units VALUES (?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()

    registry = tmp_path / "registry.json"
    registry.write_text("{}", encoding="utf-8")
    exposure = tmp_path / "exposure.jsonl"
    _write_jsonl(
        exposure,
        [
            {
                "candidate_id": "unit-1",
                "source": {
                    "message_fullname": "t3_submission",
                    "source_revision_id": "revision",
                    "thread_fullname": "t3_submission",
                    "text": "Title\n\nBody",
                },
                "status": "unresolved",
                "archive_matches": [],
                "snapshot_id": "snap",
                "line": 1,
            }
        ],
    )

    index = tmp_path / "index"
    source_dir = index / "source-submissions"
    source_dir.mkdir(parents=True)
    bucket = source_dir / "bucket-000.sqlite"
    index_db = sqlite3.connect(bucket)
    index_db.execute(
        "CREATE TABLE archive (fullname TEXT, revision TEXT, thread TEXT, text TEXT, "
        "source_id TEXT, line INTEGER)"
    )
    if index_hit:
        index_db.execute(
            "INSERT INTO archive VALUES (?,?,?,?,?,?)",
            ("t3_submission", "revision", "t3_submission", "Title\n\nBody", "src", 1),
        )
    index_db.commit()
    index_db.close()

    outputs = {
        "archive_exposure_matches.jsonl": audit / "archive_exposure_matches.jsonl",
        "archive_index_coverage.jsonl": audit / "archive_index_coverage.jsonl",
        "archive_identity_collisions.jsonl": audit / "archive_identity_collisions.jsonl",
    }
    outputs["archive_exposure_matches.jsonl"].write_bytes(
        exposure.read_bytes()
    )
    outputs["archive_index_coverage.jsonl"].write_text(
        '{"source_id":"src","matched_records":0}\n', encoding="utf-8"
    )
    outputs["archive_identity_collisions.jsonl"].write_bytes(b"")
    output_meta = {
        name: {"sha256": _sha(path), "rows": path.read_bytes().count(b"\n")}
        for name, path in outputs.items()
    }
    inputs = {
        "registry": {"path": str(registry), "sha256": _sha(registry)},
        "corpus": {"path": str(corpus), "sha256": _sha(corpus)},
        "exposure": [{"path": str(exposure), "sha256": _sha(exposure)}],
    }
    identity = {
        "schema_version": 3,
        "algorithm_version": "archive-audit-exact-match-v1",
        "snapshot_id": "snap",
        "input_hashes": inputs,
        "index_directory": str(index),
    }
    (audit / "archive_audit_manifest.json").write_text(
        json.dumps(
            {
                "kind": "registered_archive_audit",
                "schema_version": 3,
                "complete": True,
                "snapshot_id": "snap",
                "identity": identity,
                "algorithm": {"retained_exact_match_only": True, "near_text": False},
                "counts": {"matched": 0, "unresolved": 1, "collisions": 0, "exposure_errors": 0},
                "input_hashes": inputs,
                "outputs": output_meta,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return audit, corpus, index, exposure


def test_diagnostic_is_deterministic_and_does_not_mutate_inputs(tmp_path: Path) -> None:
    audit, corpus, index, exposure = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in audit.rglob("*") if path.is_file()}
    first = diagnose_unresolved(audit, corpus, tmp_path / "out-a")
    diagnose_unresolved(audit, corpus, tmp_path / "out-b")
    assert first["counts"] == {"unresolved": 1, "submission:present": 1}
    assert first["cause"] == "unknown_exact_identity_or_normalization_mismatch"
    assert first["tombstone_safe"] is False
    assert (tmp_path / "out-a/unresolved_diagnostics.jsonl").read_bytes() == (
        tmp_path / "out-b/unresolved_diagnostics.jsonl"
    ).read_bytes()
    assert (tmp_path / "out-a/unresolved_diagnostic_manifest.json").read_bytes() == (
        tmp_path / "out-b/unresolved_diagnostic_manifest.json"
    ).read_bytes()
    assert {path: path.read_bytes() for path in audit.rglob("*") if path.is_file()} == before
    assert exposure.exists() and index.exists()


@pytest.mark.parametrize("tamper", ["audit", "index"])
def test_rejects_tampered_or_missing_inputs(tmp_path: Path, tamper: str) -> None:
    audit, corpus, index, _ = _fixture(tmp_path)
    if tamper == "audit":
        (audit / "archive_exposure_matches.jsonl").write_text("{}\n", encoding="utf-8")
    else:
        for path in index.rglob("*"):
            if path.is_file():
                path.unlink()
    with pytest.raises(ValueError):
        diagnose_unresolved(audit, corpus, tmp_path / "out")


def test_rejects_duplicate_corpus_identity(tmp_path: Path) -> None:
    audit, corpus, _index, _ = _fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicated"):
        diagnose_unresolved(audit, corpus, tmp_path / "out")


def test_rejects_unresolved_row_that_is_in_index(tmp_path: Path) -> None:
    audit, corpus, _index, _ = _fixture(tmp_path, index_hit=True)
    with pytest.raises(ValueError, match="matches retained archive index"):
        diagnose_unresolved(audit, corpus, tmp_path / "out")


def test_partial_index_identity_is_not_an_exact_hit(tmp_path: Path) -> None:
    audit, corpus, index, _ = _fixture(tmp_path)
    bucket = next(index.rglob("bucket-*.sqlite"))
    connection = sqlite3.connect(bucket)
    connection.execute(
        "INSERT INTO archive VALUES (?,?,?,?,?,?)",
        ("t3_submission", "revision", "other-thread", "other text", "src", 2),
    )
    connection.commit()
    connection.close()
    result = diagnose_unresolved(audit, corpus, tmp_path / "out")
    assert result["counts"]["submission:present"] == 1


def test_cli_reports_success_and_structured_failure(tmp_path: Path) -> None:
    audit, corpus, _index, _ = _fixture(tmp_path)
    result = runner.invoke(
        app,
        [
            "dataset",
            "diagnose-unresolved",
            "--audit",
            str(audit),
            "--corpus",
            str(corpus),
            "--output",
            str(tmp_path / "cli-out"),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["diagnosed"] is True

    (audit / "archive_exposure_matches.jsonl").write_text("{}\n", encoding="utf-8")
    failed = runner.invoke(
        app,
        [
            "dataset",
            "diagnose-unresolved",
            "--audit",
            str(audit),
            "--corpus",
            str(corpus),
            "--output",
            str(tmp_path / "cli-fail"),
        ],
    )
    assert failed.exit_code == 2
    assert json.loads(failed.stderr)["diagnosed"] is False
