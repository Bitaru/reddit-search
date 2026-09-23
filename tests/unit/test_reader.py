import json
from pathlib import Path

import pytest
import zstandard


def write_zstd(path: Path, *frames: bytes) -> None:
    compressor = zstandard.ZstdCompressor()
    path.write_bytes(b"".join(compressor.compress(frame) for frame in frames))


def test_reader_recovers_concatenated_frames_and_final_line_without_newline(tmp_path: Path) -> None:
    from reddit_search.ingest.reader import ArchiveReader, SourceSpec

    source_file = tmp_path / "RC_2026-09.zst"
    first = json.dumps({"id": "a", "body": "café"}).encode() + b"\n"
    second = json.dumps({"id": "b", "body": "final record"}).encode()
    write_zstd(source_file, first, second)

    reader = ArchiveReader()
    records = list(reader.iter_records(SourceSpec("comments", source_file, "comment", "2026-09")))

    assert [record.line_number for record in records] == [1, 2]
    assert [record.payload["body"] for record in records] == ["café", "final record"]
    assert reader.stats.valid_records == 2
    assert reader.stats.complete is True


def test_reader_quarantines_bad_json_and_marks_source_incomplete(tmp_path: Path) -> None:
    from reddit_search.ingest.reader import ArchiveReader, SourceSpec

    source_file = tmp_path / "RS_2026-09.zst"
    write_zstd(source_file, b'{"id":"good"}\nnot-json\n')

    reader = ArchiveReader()
    records = list(
        reader.iter_records(SourceSpec("submissions", source_file, "submission", "2026-09"))
    )

    assert [record.payload["id"] for record in records] == ["good"]
    assert reader.stats.invalid_records == 1
    assert reader.stats.invalid_line_numbers == [2]
    assert reader.stats.complete is False


def test_truncated_zstd_cannot_complete_a_manifest(tmp_path: Path) -> None:
    from reddit_search.ingest.reader import ArchiveReader, SourceReadError, SourceSpec

    source_file = tmp_path / "truncated.zst"
    compressor = zstandard.ZstdCompressor()
    source_file.write_bytes(compressor.compress(b'{"id":"one"}\n')[:-3])

    reader = ArchiveReader()
    with pytest.raises(SourceReadError):
        list(reader.iter_records(SourceSpec("comments", source_file, "comment", "2026-09")))

    assert reader.stats.complete is False
    assert reader.stats.decompression_error is not None
