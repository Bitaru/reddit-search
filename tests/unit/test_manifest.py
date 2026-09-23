from datetime import UTC, datetime
from pathlib import Path


def test_manifest_marks_invalid_input_complete_with_errors_not_complete(tmp_path: Path) -> None:
    from reddit_search.ingest.manifest import manifest_from_reader
    from reddit_search.ingest.reader import ReaderStats, SourceSpec

    source = SourceSpec("comments", tmp_path / "RC.zst", "comment", "2026-09")
    source.path.write_bytes(b"source")
    stats = ReaderStats(lines_seen=2, valid_records=1, invalid_records=1, complete=False)

    manifest = manifest_from_reader(
        source,
        stats,
        run_id="run-1",
        configuration_hash="config",
        started_at=datetime.now(UTC),
    )

    assert manifest.status == "complete_with_errors"
    assert manifest.completed_at is not None


def test_manifest_marks_decompression_failure_failed(tmp_path: Path) -> None:
    from reddit_search.ingest.manifest import manifest_from_reader
    from reddit_search.ingest.reader import ReaderStats, SourceSpec

    source = SourceSpec("comments", tmp_path / "RC.zst", "comment", "2026-09")
    source.path.write_bytes(b"source")
    stats = ReaderStats(complete=False, decompression_error="truncated frame")

    manifest = manifest_from_reader(
        source,
        stats,
        run_id="run-1",
        configuration_hash="config",
        started_at=datetime.now(UTC),
    )

    assert manifest.status == "failed"
    assert manifest.completed_at is None
