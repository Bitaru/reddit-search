from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reddit_search.cli import app
from reddit_search.evaluation.archive_audit import audit_registered_archives
from reddit_search.evaluation.archive_partitioned import (
    audit_registered_archives_partitioned,
)
from reddit_search.telemetry import load_stage_telemetry

runner = CliRunner()


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
            ("u2", "candidate-2", "snap", "t3_hit", revision, "t3_hit", "Need help\n\nBody"),
            ("u3", "candidate-3", "snap", "t3_other", "other", "t3_other", "Other"),
        ],
    )
    db.commit()
    db.close()


def _jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_archive_audit_manifest_matches_exposures_and_preserves_inputs(tmp_path: Path) -> None:
    source = tmp_path / "submissions.jsonl"
    rows = [_submission("t3_hit", "Need help", "Body"), _submission("t3_hit", "Need help", "Body")]
    _jsonl(source, rows)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "src",
                        "kind": "submission",
                        "source_path": source.name,
                    }
                ]
            }
        )
    )
    corpus = tmp_path / "corpus.sqlite"
    revision = _revision("Need help", "Body")
    _corpus(corpus, revision)
    exposure = tmp_path / "exposure.jsonl"
    _jsonl(
        exposure,
        [
            {"candidate_id": "candidate-1", "scenario_id": "s1"},
            {
                "message_fullname": "t3_hit",
                "source_revision_id": revision,
                "thread_fullname": "t3_hit",
                "focus_text": "Need help\n\nBody",
                "scenario_id": "s2",
            },
            {"candidate_id": "missing", "scenario_id": "s3"},
            "{malformed",
        ],
    )
    optional = tmp_path / "links.jsonl"
    optional.write_text("{}\n")
    before = {p: p.read_bytes() for p in (registry, corpus, exposure, optional)}

    first = audit_registered_archives(
        registry,
        corpus,
        [exposure],
        tmp_path / "out-a",
        max_records=10,
        snapshot_id="snap",
        duplicate_links_path=optional,
        min_free_disk_bytes=1,
    )
    second = audit_registered_archives(
        registry,
        corpus,
        [exposure],
        tmp_path / "out-b",
        max_records=10,
        snapshot_id="snap",
        duplicate_links_path=optional,
        min_free_disk_bytes=1,
    )

    assert first["complete"] is False  # malformed exposure is surfaced, never hidden
    first_telemetry = load_stage_telemetry(tmp_path / "out-a" / "stage_telemetry.json")
    assert first_telemetry["observations"]["cleanup"] == "complete"
    assert first_telemetry["status"] == "incomplete"
    assert first["counts"] == {
        "matched": 2,
        "unresolved": 1,
        "collisions": 1,
        "exposure_errors": 1,
    }
    assert first["optional_inputs_unused"][0]["kind"] == "duplicate_links"
    events = [
        json.loads(line)
        for line in (tmp_path / "out-a/archive_exposure_matches.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in events] == [
        "matched",
        "matched",
        "unresolved",
        "error",
    ]
    assert events[0]["archive_matches"] == [
        {"source_id": "src", "line": 1},
        {"source_id": "src", "line": 2},
    ]
    assert events[0]["corpus_matches"] == 1
    collision = json.loads((tmp_path / "out-a/archive_identity_collisions.jsonl").read_text())
    assert collision["archive_count"] == 2 and collision["corpus_count"] == 2
    for name in (
        "archive_exposure_matches.jsonl",
        "archive_index_coverage.jsonl",
        "archive_identity_collisions.jsonl",
    ):
        assert (tmp_path / "out-a" / name).read_bytes() == (tmp_path / "out-b" / name).read_bytes()
    first_telemetry = json.loads((tmp_path / "out-a/stage_telemetry.json").read_text())
    second_telemetry = json.loads((tmp_path / "out-b/stage_telemetry.json").read_text())
    assert first_telemetry["input_hashes"] == second_telemetry["input_hashes"]
    assert first_telemetry["status"] == second_telemetry["status"] == "incomplete"
    assert {p: p.read_bytes() for p in before} == before
    assert second["outputs"] == first["outputs"]


def test_archive_audit_cap_is_incomplete_and_does_not_verify_source(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _jsonl(
        source,
        [
            _submission("t3_one", "One", "Body"),
            _submission("t3_two", "Two", "Body"),
        ],
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "src",
                        "kind": "submission",
                        "source_path": source.name,
                    }
                ]
            }
        )
    )
    corpus = tmp_path / "corpus.sqlite"
    _corpus(corpus, _revision("One", "Body"))
    checkpoint = tmp_path / "audit.checkpoint.json"
    result = audit_registered_archives(
        registry,
        corpus,
        [],
        tmp_path / "out",
        max_records=1,
        checkpoint_path=checkpoint,
        min_free_disk_bytes=1,
    )
    assert result["complete"] is False
    assert result["budget"]["exhausted"] is True
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_id"] == "src"
    assert result["sources"][0]["reader_stats"]["complete"] is False
    checkpoint_data = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_data["active_source"] == "src"
    assert checkpoint_data["completed_source_ids"] == []
    assert checkpoint_data["source_stats"] == []
    assert checkpoint_data["records_used"] == 0


def test_archive_audit_cap_counts_malformed_physical_source_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "{malformed\n" + json.dumps(_submission("t3_one", "One", "Body")) + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {"sources": [{"source_id": "src", "kind": "submission", "source_path": source.name}]}
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    _corpus(corpus, _revision("One", "Body"))
    result = audit_registered_archives(
        registry, corpus, [], tmp_path / "out", max_records=1, min_free_disk_bytes=1
    )


    assert result["complete"] is False
    assert result["budget"]["exhausted"] is True
    assert result["budget"]["records_used"] >= 2
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_id"] == "src"
    assert result["sources"][0]["reader_stats"]["invalid_records"] == 1
    assert result["sources"][0]["reader_stats"]["complete"] is False


def _operational_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source.jsonl"
    _jsonl(source, [_submission("t3_hit", "Need help", "Body")])
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "src",
                        "kind": "submission",
                        "source_path": source.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    _corpus(corpus, _revision("Need help", "Body"))
    exposure = tmp_path / "exposure.jsonl"
    _jsonl(
        exposure,
        [
            {
                "candidate_id": "candidate-1",
                "scenario_id": "s1",
            }
        ],
    )
    return registry, corpus, exposure, source


def test_archive_audit_operational_artifacts_bind_identity_and_preserve_inputs(
    tmp_path: Path,
) -> None:
    registry, corpus, exposure, source = _operational_fixture(tmp_path)
    checkpoint = tmp_path / "audit.checkpoint.json"
    progress = tmp_path / "audit.progress.json"
    before = {path: path.read_bytes() for path in (registry, corpus, exposure, source)}

    result = audit_registered_archives(
        registry,
        corpus,
        [exposure],
        tmp_path / "out",
        checkpoint_path=checkpoint,
        progress_path=progress,
        max_process_rss_bytes=10**12,
        min_free_disk_bytes=0,
        max_records=10,
    )
    assert result["complete"] is True
    assert result["operational"]["index_path"] == str(
        checkpoint.with_name(checkpoint.name + ".sqlite")
    )
    assert result["operational"]["checkpoint_path"] == str(checkpoint)
    assert result["operational"]["progress_path"] == str(progress)
    assert result["operational"]["resume"] == "source-boundary"
    persistent_telemetry = load_stage_telemetry(
        tmp_path / "out" / "stage_telemetry.json"
    )
    assert persistent_telemetry["observations"]["cleanup"] == "not_applicable"
    telemetry_identity = result["operational"]["telemetry"]
    assert telemetry_identity["status"] == "complete"
    assert Path(telemetry_identity["path"]).is_file()
    assert telemetry_identity["sha256"]
    assert result["identity"]["input_hashes"] == result["input_hashes"]
    assert checkpoint.exists() and progress.exists()
    assert {path: path.read_bytes() for path in before} == before


def test_archive_audit_resume_checks_identity_before_scanning(tmp_path: Path) -> None:
    registry, corpus, exposure, _ = _operational_fixture(tmp_path)
    checkpoint = tmp_path / "audit.checkpoint.json"
    kwargs = dict(
        checkpoint_path=checkpoint,
        max_process_rss_bytes=10**12,
        max_records=10,
        min_free_disk_bytes=0,
    )
    first = audit_registered_archives(registry, corpus, [exposure], tmp_path / "out", **kwargs)
    assert first["complete"] is True
    originals = {path: path.read_bytes() for path in (registry, corpus, exposure)}
    for path in (registry, corpus, exposure):
        path.write_bytes(originals[path] + b"\n")
        with pytest.raises(ValueError, match="checkpoint identity does not match"):
            audit_registered_archives(registry, corpus, [exposure], tmp_path / "out", **kwargs)
        path.write_bytes(originals[path])


def test_archive_audit_completed_rerun_uses_index_without_opening_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, corpus, exposure, _ = _operational_fixture(tmp_path)
    checkpoint = tmp_path / "audit.checkpoint.json"
    kwargs = dict(
        checkpoint_path=checkpoint,
        max_process_rss_bytes=10**12,
        max_records=10,
        min_free_disk_bytes=0,
    )
    first = audit_registered_archives(registry, corpus, [exposure], tmp_path / "out", **kwargs)
    output_before = (tmp_path / "out" / "archive_exposure_matches.jsonl").read_bytes()

    import reddit_search.evaluation.archive_audit as module

    def fail_reader() -> object:
        raise AssertionError("completed sources must not be reopened")

    monkeypatch.setattr(module, "ArchiveReader", fail_reader)
    second = audit_registered_archives(registry, corpus, [exposure], tmp_path / "out", **kwargs)
    assert first["complete"] is True
    assert second["complete"] is True
    assert (tmp_path / "out" / "archive_exposure_matches.jsonl").read_bytes() == output_before


def test_archive_audit_resumes_at_source_boundary_and_replays_active_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, corpus, exposure, source = _operational_fixture(tmp_path)
    source_b = tmp_path / "source-b.jsonl"
    source_b.write_bytes(source.read_bytes())
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {"source_id": "a", "kind": "submission", "source_path": source.name},
                    {"source_id": "b", "kind": "submission", "source_path": source_b.name},
                ]
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "resume.checkpoint.json"
    kwargs = {
        "checkpoint_path": checkpoint,
        "max_process_rss_bytes": 10**12,
        "min_free_disk_bytes": 0,
        "max_records": 10,
    }
    import reddit_search.evaluation.archive_audit as module

    original_iter_records = module.ArchiveReader.iter_records
    first_seen: list[str] = []

    def fail_second(reader: object, spec: object) -> object:
        first_seen.append(spec.source_id)  # type: ignore[attr-defined]
        if spec.source_id == "b":
            raise module.SourceReadError("simulated second-source failure")
        return original_iter_records(reader, spec)  # type: ignore[arg-type]

    monkeypatch.setattr(module.ArchiveReader, "iter_records", fail_second)
    first = audit_registered_archives(registry, corpus, [exposure], tmp_path / "partial", **kwargs)
    assert first["complete"] is False
    assert first_seen == ["a", "b"]
    resumed_seen: list[str] = []

    def record(reader: object, spec: object) -> object:
        resumed_seen.append(spec.source_id)  # type: ignore[attr-defined]
        return original_iter_records(reader, spec)  # type: ignore[arg-type]

    monkeypatch.setattr(module.ArchiveReader, "iter_records", record)
    resumed = audit_registered_archives(
        registry, corpus, [exposure], tmp_path / "partial", **kwargs
    )
    audit_registered_archives(registry, corpus, [exposure], tmp_path / "clean", **kwargs)
    assert resumed["complete"] is True
    assert resumed_seen == ["b"]
    for name in (
        "archive_exposure_matches.jsonl",
        "archive_index_coverage.jsonl",
        "archive_identity_collisions.jsonl",
    ):
        assert (tmp_path / "partial" / name).read_bytes() == (
            tmp_path / "clean" / name
        ).read_bytes()

    def no_sources(reader: object, spec: object) -> object:
        raise AssertionError("completed sources must not be reopened")

    monkeypatch.setattr(module.ArchiveReader, "iter_records", no_sources)
    output_before = (tmp_path / "partial" / "archive_exposure_matches.jsonl").read_bytes()
    completed = audit_registered_archives(
        registry, corpus, [exposure], tmp_path / "partial", **kwargs
    )
    assert completed["complete"] is True
    assert (tmp_path / "partial" / "archive_exposure_matches.jsonl").read_bytes() == output_before

def test_archive_audit_mutated_completed_source_replays_from_earliest_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, corpus, exposure, source = _operational_fixture(tmp_path)
    source_b = tmp_path / "source-b.jsonl"
    source_b.write_bytes(source.read_bytes())
    registry.write_text(json.dumps({"sources": [
        {"source_id": "a", "kind": "submission", "source_path": source.name},
        {"source_id": "b", "kind": "submission", "source_path": source_b.name},
    ]}), encoding="utf-8")
    checkpoint = tmp_path / "resume.json"
    kwargs = {"checkpoint_path": checkpoint, "max_process_rss_bytes": 10**12,
              "min_free_disk_bytes": 0, "max_records": 10}
    import reddit_search.evaluation.archive_audit as module
    original = module.ArchiveReader.iter_records
    def fail_b(reader: object, spec: object) -> object:
        if spec.source_id == "b":
            raise module.SourceReadError("stop")
        return original(reader, spec)  # type: ignore[arg-type]
    monkeypatch.setattr(module.ArchiveReader, "iter_records", fail_b)
    first = audit_registered_archives(
        registry, corpus, [exposure], tmp_path / "out", **kwargs
    )
    assert first["complete"] is False
    source.write_text(source.read_text().replace("Body", "B0dy"), encoding="utf-8")
    seen: list[str] = []
    def record(reader: object, spec: object) -> object:
        seen.append(spec.source_id)
        return original(reader, spec)  # type: ignore[arg-type]
    monkeypatch.setattr(module.ArchiveReader, "iter_records", record)
    result = audit_registered_archives(registry, corpus, [exposure], tmp_path / "out", **kwargs)
    assert result["complete"] is True
    assert seen == ["a", "b"]


def test_archive_audit_missing_completed_source_fails_closed(tmp_path: Path) -> None:
    registry, corpus, exposure, source = _operational_fixture(tmp_path)
    checkpoint = tmp_path / "resume.json"
    kwargs = {"checkpoint_path": checkpoint, "max_process_rss_bytes": 10**12,
              "min_free_disk_bytes": 0, "max_records": 10}
    first = audit_registered_archives(
        registry, corpus, [exposure], tmp_path / "out", **kwargs
    )
    assert first["complete"] is True
    source.unlink()
    result = audit_registered_archives(registry, corpus, [exposure], tmp_path / "out", **kwargs)
    assert result["complete"] is False


def test_archive_audit_expected_checksum_mismatch_is_incomplete(tmp_path: Path) -> None:
    registry, corpus, exposure, source = _operational_fixture(tmp_path)
    registry.write_text(json.dumps({"sources": [{
        "source_id": "src", "kind": "submission", "source_path": source.name,
        "expected_checksum": "0" * 64,
    }]}), encoding="utf-8")
    result = audit_registered_archives(
        registry, corpus, [exposure], tmp_path / "out",
        max_process_rss_bytes=10**12, min_free_disk_bytes=0, max_records=10,
    )
    assert result["complete"] is False
    assert "expected checksum mismatch" in result["sources"][0]["read_error"]


def test_archive_audit_rss_breach_returns_incomplete_operational_manifest(
    tmp_path: Path,
) -> None:
    registry, corpus, exposure, source = _operational_fixture(tmp_path)
    checkpoint = tmp_path / "audit.checkpoint.json"
    progress = tmp_path / "audit.progress.json"
    result = audit_registered_archives(
        registry,
        corpus,
        [exposure],
        tmp_path / "out",
        checkpoint_path=checkpoint,
        progress_path=progress,
        max_process_rss_bytes=1,
        min_free_disk_bytes=0,
        max_records=10,
    )
    assert result["complete"] is False
    assert result["operational_error"]
    rss_telemetry = load_stage_telemetry(tmp_path / "out" / "stage_telemetry.json")
    assert rss_telemetry["status"] == "incomplete"
    assert rss_telemetry["observations"]["rss_bytes"] is not None
    assert rss_telemetry["errors"]
    assert all(error["field"] == "archive_audit" for error in rss_telemetry["errors"])
    assert result["operational"]["checkpoint_path"] == str(checkpoint)
    assert checkpoint.exists() and progress.exists()
    checkpoint_data = json.loads(checkpoint.read_text(encoding="utf-8"))
    progress_data = json.loads(progress.read_text(encoding="utf-8"))
    assert progress_data["current_source"] == checkpoint_data["active_source"]
    assert progress_data["completed_sources"] == len(checkpoint_data["completed_source_ids"])


def test_archive_audit_disk_reserve_breach_returns_incomplete_manifest(
    tmp_path: Path,
) -> None:
    registry, corpus, exposure, source = _operational_fixture(tmp_path)
    result = audit_registered_archives(
        registry,
        corpus,
        [exposure],
        tmp_path / "out",
        max_process_rss_bytes=10**12,
        min_free_disk_bytes=10**18,
        max_records=10,
    )
    assert result["complete"] is False
    assert result["operational_error"]
    disk_telemetry = load_stage_telemetry(tmp_path / "out" / "stage_telemetry.json")
    assert disk_telemetry["status"] == "incomplete"
    assert disk_telemetry["observations"]["free_disk_bytes"] is not None
    assert disk_telemetry["errors"]
    assert all(error["field"] == "archive_audit" for error in disk_telemetry["errors"])


def test_partitioned_archive_audit_filters_targets_and_is_deterministic(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    other = tmp_path / "other.jsonl"
    _jsonl(target, [_submission("t3_hit", "Need help", "Body")] * 2)
    _jsonl(other, [_submission("t3_other", "Other", "Text")])
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {"source_id": "b", "kind": "submission", "source_path": "other.jsonl"},
                    {"source_id": "a", "kind": "submission", "source_path": "target.jsonl"},
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
    checkpoint = tmp_path / "checkpoint.json"
    kwargs = {
        "max_records": 10,
        "index_directory": tmp_path / "index",
        "checkpoint_path": checkpoint,
        "min_free_disk_bytes": 0,
        "max_process_rss_bytes": 10**12,
    }
    before = {path: path.read_bytes() for path in (registry, corpus, exposure, target, other)}
    first = audit_registered_archives_partitioned(
        registry, corpus, [exposure], tmp_path / "out-a", **kwargs
    )
    _second = audit_registered_archives_partitioned(
        registry, corpus, [exposure], tmp_path / "out-b", **kwargs
    )
    assert first["complete"] is True
    assert [item["source_id"] for item in first["sources"]] == ["a", "b"]
    assert [item["records_scanned"] for item in first["sources"]] == [2, 0]
    assert first["counts"]["matched"] == 1
    assert first["counts"]["collisions"] == 1
    assert first["counts"]["unresolved"] == 0
    for name in (
        "archive_exposure_matches.jsonl",
        "archive_index_coverage.jsonl",
        "archive_identity_collisions.jsonl",
    ):
        assert (tmp_path / "out-a" / name).read_bytes() == (tmp_path / "out-b" / name).read_bytes()
    telemetry_a = load_stage_telemetry(tmp_path / "out-a/stage_telemetry.json")
    telemetry_b = load_stage_telemetry(tmp_path / "out-b/stage_telemetry.json")
    assert telemetry_a["status"] == telemetry_b["status"] == "complete"
    assert telemetry_a["input_hashes"] == telemetry_b["input_hashes"]
    assert telemetry_a["observations"]["cleanup"] == "not_applicable"
    assert {path: path.read_bytes() for path in before} == before

def test_partitioned_archive_audit_telemetry_identity_and_observations(tmp_path: Path) -> None:
    registry, corpus, exposure, _ = _operational_fixture(tmp_path)
    output = tmp_path / "out"
    checkpoint = tmp_path / "checkpoint.json"
    result = audit_registered_archives_partitioned(
        registry,
        corpus,
        [exposure],
        output,
        max_records=10,
        index_directory=tmp_path / "index",
        checkpoint_path=checkpoint,
        min_free_disk_bytes=0,
        max_process_rss_bytes=10**12,
        clock=iter([10.0, 12.5]).__next__,
        rss_provider=lambda: 123,
        disk_provider=lambda _path: 456,
    )
    telemetry = load_stage_telemetry(output / "stage_telemetry.json")
    assert result["complete"] is True
    assert telemetry["run_identity"]
    assert telemetry["input_hashes"] == result["input_hashes"]
    assert telemetry["limits"]["max_records"] == 10
    assert telemetry["observations"] == {
        "elapsed_seconds": 2.5,
        "rss_bytes": 123,
        "free_disk_bytes": 456,
        "sampled_path": str(output),
        "cache": "not_used",
        "cleanup": "not_applicable",
    }
    manifest = json.loads((output / "archive_audit_manifest.json").read_text(encoding="utf-8"))
    assert telemetry["stage"] == "archive_audit"
    assert manifest["operational"]["telemetry"] == {
        "path": str(output / "stage_telemetry.json"),
        "sha256": hashlib.sha256((output / "stage_telemetry.json").read_bytes()).hexdigest(),
        "status": telemetry["status"],
    }

def test_partitioned_archive_audit_rejects_wrong_snapshot_before_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.jsonl"
    _jsonl(target, [_submission("t3_hit", "Need help", "Body")])
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {"sources": [{"source_id": "a", "kind": "submission", "source_path": target.name}]}
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    _corpus(corpus, _revision("Need help", "Body"))
    exposure = tmp_path / "exposure.jsonl"
    _jsonl(exposure, [{"candidate_id": "candidate-1", "scenario_id": "s1"}])
    before = {path: path.read_bytes() for path in (registry, corpus, exposure, target)}

    import reddit_search.evaluation.archive_partitioned as module

    def fail_scan(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("archive source scan should not start")

    monkeypatch.setattr(module.ArchiveReader, "iter_records", fail_scan)
    output = tmp_path / "out"
    index = tmp_path / "index"
    checkpoint = tmp_path / "checkpoint.json"
    with pytest.raises(ValueError, match="no corpus identities.*wrong-snapshot"):
        audit_registered_archives_partitioned(
            registry,
            corpus,
            [exposure],
            output,
            snapshot_id="wrong-snapshot",
            max_records=10,
            index_directory=index,
            checkpoint_path=checkpoint,
            min_free_disk_bytes=0,
            max_process_rss_bytes=10**12,
        )
    assert {path: path.read_bytes() for path in before} == before
    assert output.exists() and not any(output.iterdir())
    assert index.exists() and not any(index.iterdir())
    assert not checkpoint.exists()
def test_partitioned_publication_budget_preserves_existing_outputs_and_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, corpus, exposure, source = _operational_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    output_names = (
        "archive_exposure_matches.jsonl",
        "archive_index_coverage.jsonl",
        "archive_identity_collisions.jsonl",
    )
    prior = {}
    for name in output_names:
        path = output / name
        prior[name] = f"prior:{name}\n".encode()
        path.write_bytes(prior[name])

    import reddit_search.evaluation.archive_partitioned as module

    calls = 0

    def disk_usage(path: Path) -> object:
        nonlocal calls
        calls += 1
        free = 10**12 if calls == 1 else 0
        return type("Usage", (), {"free": free})()

    monkeypatch.setattr(module.shutil, "disk_usage", disk_usage)
    result = audit_registered_archives_partitioned(
        registry,
        corpus,
        [exposure],
        output,
        max_records=10,
        index_directory=tmp_path / "index",
        checkpoint_path=tmp_path / "checkpoint.json",
        max_index_bytes=10**12,
        min_free_disk_bytes=1,
        max_process_rss_bytes=10**12,
    )

    assert result["complete"] is False
    assert result["operational_error"]
    assert {name: (output / name).read_bytes() for name in output_names} == prior
    assert not any(output.glob("*.tmp"))

def test_partitioned_cap_checkpoints_active_source_without_publishing_partial_index(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _jsonl(source, [_submission("t3_hit", "Need help", "Body")] * 2)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "src",
                        "kind": "submission",
                        "source_path": source.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    _corpus(corpus, _revision("Need help", "Body"))
    index = tmp_path / "index"
    checkpoint = tmp_path / "checkpoint.json"
    result = audit_registered_archives_partitioned(
        registry,
        corpus,
        [],
        tmp_path / "out",
        max_records=1,
        index_directory=index,
        checkpoint_path=checkpoint,
        min_free_disk_bytes=0,
        max_process_rss_bytes=10**12,
    )
    telemetry = load_stage_telemetry(tmp_path / "out/stage_telemetry.json")
    assert telemetry["status"] == "incomplete"
    assert telemetry["errors"][-1]["field"] == "archive_audit"
    assert telemetry["observations"]["cleanup"] == "not_applicable"
    assert result["complete"] is False
    assert result["budget"]["exhausted"] is True
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["active_source"] == "src"
    assert state["completed_source_ids"] == []
    assert not any(index.glob("*.staging"))
    assert not any(index.glob("source-*"))


def test_partitioned_min_free_disk_breach_does_not_publish_partial_index(tmp_path: Path) -> None:
    registry, corpus, exposure, _ = _operational_fixture(tmp_path)
    index = tmp_path / "index"
    result = audit_registered_archives_partitioned(
        registry,
        corpus,
        [exposure],
        tmp_path / "out",
        max_records=10,
        index_directory=index,
        checkpoint_path=tmp_path / "checkpoint.json",
        min_free_disk_bytes=10**18,
        max_process_rss_bytes=10**12,
    )
    assert result["complete"] is False
    assert result["operational_error"]
    assert not any(index.glob("*.staging"))
    assert not any(index.glob("source-*"))


def test_cli_partitioned_archive_options_forward_to_partitioned_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry.json"
    corpus = tmp_path / "corpus.sqlite"
    exposure = tmp_path / "exposure.jsonl"
    for path in (registry, corpus, exposure):
        path.touch()
    captured: dict[str, object] = {}

    def fake_audit(*args: object, **kwargs: object) -> dict[str, object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"complete": True}

    monkeypatch.setattr("reddit_search.cli.audit_registered_archives_partitioned", fake_audit)
    result = runner.invoke(
        app,
        [
            "dataset", "audit-archives",
            "--registry", str(registry), "--corpus", str(corpus),
            "--exposure", str(exposure), "--output", str(tmp_path / "out"),
            "--max-records", "7", "--index-directory", str(tmp_path / "indexes"),
            "--max-index-bytes", "1234", "--max-process-rss-bytes", "5678",
            "--min-free-disk-bytes", "0",
        ],
    )
    assert result.exit_code == 0
    assert captured["kwargs"] == {
        "max_records": 7,
        "index_directory": tmp_path / "indexes",
        "snapshot_id": None,
        "max_bytes": None,
        "checkpoint_path": None,
        "progress_path": None,
        "max_index_bytes": 1234,
        "max_process_rss_bytes": 5678,
        "min_free_disk_bytes": 0,
    }



def test_cli_legacy_index_forwards_to_single_file_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry.json"
    corpus = tmp_path / "corpus.sqlite"
    exposure = tmp_path / "exposure.jsonl"
    for path in (registry, corpus, exposure):
        path.touch()
    captured: dict[str, object] = {}

    def fake_audit(*args: object, **kwargs: object) -> dict[str, object]:
        captured["kwargs"] = kwargs
        return {"complete": True}

    monkeypatch.setattr("reddit_search.cli.audit_registered_archives", fake_audit)
    result = runner.invoke(
        app,
        [
            "dataset", "audit-archives",
            "--registry", str(registry), "--corpus", str(corpus),
            "--exposure", str(exposure), "--output", str(tmp_path / "out"),
            "--max-records", "7", "--checkpoint", str(tmp_path / "checkpoint.json"),
            "--index", str(tmp_path / "index.sqlite"),
        ],
    )
    assert result.exit_code == 0
    assert captured["kwargs"]["index_path"] == tmp_path / "index.sqlite"  # type: ignore[index]
