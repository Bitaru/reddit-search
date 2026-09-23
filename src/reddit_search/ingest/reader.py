"""Bounded, line-oriented readers for Reddit-style JSONL and JSONL.zst sources."""

from __future__ import annotations

import codecs
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import zstandard


class SourceReadError(RuntimeError):
    """Raised when decompression or UTF-8 decoding prevents a complete read."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    path: Path
    source_kind: Literal["submission", "comment"]
    declared_month: str
    source_role: Literal["discovery", "context_only"] = "discovery"
    usage_scope: str = "synthetic"


@dataclass(frozen=True, slots=True)
class RawRecordEnvelope:
    source_id: str
    source_kind: Literal["submission", "comment"]
    line_number: int
    payload: dict[str, Any]
    raw_line: str


@dataclass(slots=True)
class ReaderStats:
    compressed_bytes_read: int = 0
    decompressed_bytes_read: int = 0
    lines_seen: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    invalid_line_numbers: list[int] = field(default_factory=list)
    complete: bool = False
    decompression_error: str | None = None
    verified_sha256: str | None = None


class ArchiveReader:
    """Iterate input one JSON object at a time without materializing the source."""

    def __init__(self, *, max_line_bytes: int = 8_388_608) -> None:
        if max_line_bytes <= 0:
            raise ValueError("max_line_bytes must be positive")
        self.max_line_bytes = max_line_bytes
        self.stats = ReaderStats()
        self._compressed_hasher = hashlib.sha256()

    def iter_records(self, source: SourceSpec) -> Iterator[RawRecordEnvelope]:
        """Yield valid object rows while quarantining malformed JSON rows in stats."""
        self.stats = ReaderStats()
        self._compressed_hasher = hashlib.sha256()
        try:
            for line_number, raw_line in enumerate(self._iter_source_lines(source.path), start=1):
                self.stats.lines_seen += 1
                encoded_line = raw_line.encode("utf-8")
                if len(encoded_line) > self.max_line_bytes:
                    self._quarantine(line_number)
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    self._quarantine(line_number)
                    continue
                if not isinstance(payload, dict):
                    self._quarantine(line_number)
                    continue
                self.stats.valid_records += 1
                yield RawRecordEnvelope(
                    source_id=source.source_id,
                    source_kind=source.source_kind,
                    line_number=line_number,
                    payload=payload,
                    raw_line=raw_line,
                )
        except (OSError, UnicodeDecodeError, zstandard.ZstdError) as error:
            self.stats.decompression_error = str(error)
            self.stats.complete = False
            raise SourceReadError(f"could not completely read {source.path}: {error}") from error
        else:
            self.stats.verified_sha256 = self._compressed_hasher.hexdigest()
            self.stats.complete = self.stats.invalid_records == 0

    def _quarantine(self, line_number: int) -> None:
        self.stats.invalid_records += 1
        self.stats.invalid_line_numbers.append(line_number)

    def _iter_source_lines(self, path: Path) -> Iterator[str]:
        if path.suffix != ".zst":
            with path.open("rb") as source:
                for raw_line in source:
                    self.stats.compressed_bytes_read += len(raw_line)
                    self._compressed_hasher.update(raw_line)
                    self.stats.decompressed_bytes_read += len(raw_line)
                    yield raw_line.decode("utf-8")
            return

        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        buffered_text = ""
        for chunk in self._iter_zstd_chunks(path):
            buffered_text += decoder.decode(chunk)
            while "\n" in buffered_text:
                line, buffered_text = buffered_text.split("\n", maxsplit=1)
                yield f"{line}\n"
        buffered_text += decoder.decode(b"", final=True)
        if buffered_text:
            yield buffered_text

    def _iter_zstd_chunks(self, path: Path) -> Iterator[bytes]:
        decompressor = zstandard.ZstdDecompressor()
        frame = None
        saw_complete_frame = False
        with path.open("rb") as source:
            while compressed_chunk := source.read(131_072):
                self._compressed_hasher.update(compressed_chunk)
                self.stats.compressed_bytes_read += len(compressed_chunk)
                pending = compressed_chunk
                while pending:
                    if frame is None:
                        frame = decompressor.decompressobj()
                    decompressed_chunk = frame.decompress(pending)
                    pending = frame.unused_data
                    if decompressed_chunk:
                        self.stats.decompressed_bytes_read += len(decompressed_chunk)
                        yield decompressed_chunk
                    if frame.eof:
                        saw_complete_frame = True
                        frame = None
                        continue
                    if pending:
                        raise zstandard.ZstdError("decompressor retained unconsumed input")
                    break
        if frame is not None or not saw_complete_frame:
            raise zstandard.ZstdError("truncated or empty zstandard input")
