import json
from pathlib import Path

import pytest
import zstandard


def write_zstd(path: Path, *frames: bytes) -> None:
    compressor = zstandard.ZstdCompressor()
    path.write_bytes(b"".join(compressor.compress(frame) for frame in frames))


def set_decompressed_chunks(monkeypatch: pytest.MonkeyPatch, reader, *chunks: bytes) -> None:
    def iter_chunks(_path: Path):
        yield from chunks

    monkeypatch.setattr(reader, "_iter_zstd_chunks", iter_chunks)


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


def test_reader_splits_lines_across_decompressed_chunk_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reddit_search.ingest.reader import ArchiveReader

    source_file = tmp_path / "boundary.zst"
    write_zstd(source_file, b"synthetic compressed fixture")

    first_line = b"first\n"
    crossing_line = b"x" * (131_072 - len(first_line))
    raw = first_line + crossing_line + b"\nsecond\nfinal"
    reader = ArchiveReader()
    set_decompressed_chunks(monkeypatch, reader, raw[:131_072], raw[131_072:])

    assert list(reader._iter_source_lines(source_file)) == [
        "first\n",
        crossing_line.decode() + "\n",
        "second\n",
        "final",
    ]


def test_reader_yields_long_line_once_and_quarantines_over_limit(
    tmp_path: Path,
) -> None:
    from reddit_search.ingest.reader import ArchiveReader, SourceSpec

    source_file = tmp_path / "long-line.zst"
    raw_line = b"x" * 131_073 + b"\n"
    write_zstd(source_file, raw_line)

    reader = ArchiveReader(max_line_bytes=131_072)
    assert list(reader._iter_source_lines(source_file)) == [raw_line.decode()]

    records = list(
        reader.iter_records(
            SourceSpec("comments", source_file, "comment", "2026-09")
        )
    )
    assert records == []
    assert reader.stats.lines_seen == 1
    assert reader.stats.invalid_records == 1
    assert reader.stats.invalid_line_numbers == [1]


def test_reader_preserves_carriage_returns_in_crlf_lines(tmp_path: Path) -> None:
    from reddit_search.ingest.reader import ArchiveReader

    source_file = tmp_path / "crlf.zst"
    write_zstd(source_file, b'{"id":"one"}\r\n{"id":"two"}\r\n')

    assert list(ArchiveReader()._iter_source_lines(source_file)) == [
        '{"id":"one"}\r\n',
        '{"id":"two"}\r\n',
    ]


def test_reader_raises_for_invalid_utf8_straddling_chunk_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reddit_search.ingest.reader import ArchiveReader, SourceReadError, SourceSpec

    source_file = tmp_path / "invalid-utf8.zst"
    write_zstd(source_file, b"synthetic compressed fixture")

    valid = b'{"id":"ok"}\n'
    prefix = b'{"id":"'
    padding = b"x" * (131_072 - len(valid) - len(prefix) - 1)
    first_chunk = valid + prefix + padding + b"\xe2"
    second_chunk = b'("}\n'
    reader = ArchiveReader()
    set_decompressed_chunks(monkeypatch, reader, first_chunk, second_chunk)

    with pytest.raises(SourceReadError):
        list(
            reader.iter_records(
                SourceSpec("comments", source_file, "comment", "2026-09")
            )
        )

    assert reader.stats.complete is False
    assert reader.stats.decompression_error is not None
