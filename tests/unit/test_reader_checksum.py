import hashlib
from pathlib import Path


def test_complete_reader_records_compressed_source_checksum(tmp_path: Path) -> None:
    from reddit_search.ingest.reader import ArchiveReader, SourceSpec

    source = tmp_path / "RS_2026-06.jsonl"
    source.write_text('{"id":"one"}\n')
    reader = ArchiveReader()
    list(reader.iter_records(SourceSpec("source", source, "submission", "2026-06")))

    assert reader.stats.complete is True
    assert reader.stats.verified_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
