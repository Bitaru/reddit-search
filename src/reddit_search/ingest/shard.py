"""Shared JSONL.zst shard I/O for selection, context, and control shards.

One home for the message-serialization shape used by every persisted shard:
discovery selections (execution step 1), hydration context and rejected
controls (execution step 3). Read and write paths are strict and symmetric so
a shard written by one stage is parseable by the next without inference.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import zstandard

from .normalize import NormalizedMessage, SourceProvenance


def read_selected_messages(path: Path) -> Iterator[NormalizedMessage]:
    """Yield normalized messages from a JSONL.zst shard one row at a time."""
    with path.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as decompressed:
            with io.TextIOWrapper(decompressed, encoding="utf-8") as text:
                for line_number, line in enumerate(text, start=1):
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid selected JSONL row {line_number}") from error
                    yield parse_selected_message(payload, line_number)


def parse_selected_message(payload: object, line_number: int) -> NormalizedMessage:
    """Parse one strict shard row, refusing fields that would require invention."""
    if not isinstance(payload, dict):
        raise ValueError(f"selected JSONL row {line_number} must be an object")
    kind = payload.get("kind")
    if kind not in {"submission", "comment"}:
        raise ValueError(f"selected JSONL row {line_number} has unknown kind {kind!r}")

    raw_provenance = payload.get("provenance")
    if not isinstance(raw_provenance, list) or not raw_provenance:
        raise ValueError(f"selected JSONL row {line_number} lacks provenance")
    provenance = tuple(
        SourceProvenance(
            _required_string(item, "source_id", line_number),
            _required_int(item, "line_number", line_number),
        )
        for item in raw_provenance
        if isinstance(item, dict)
    )
    if len(provenance) != len(raw_provenance):
        raise ValueError(f"selected JSONL row {line_number} has malformed provenance")

    fullname = _required_string(payload, "fullname", line_number)
    return NormalizedMessage(
        fullname=fullname,
        kind=kind,
        thread_fullname=_required_string(payload, "thread_fullname", line_number),
        parent_fullname=_optional_string(payload, "parent_fullname", line_number),
        raw_title=_required_string(payload, "raw_title", line_number)
        if kind == "submission"
        else "",
        raw_body=_required_string(payload, "raw_body", line_number),
        subreddit=_required_string(payload, "subreddit", line_number),
        created_utc=_required_int(payload, "created_utc", line_number),
        source_revision_id=_required_string(payload, "source_revision_id", line_number),
        permalink=_required_string(payload, "permalink", line_number),
        provenance=provenance,
        archive_score=_optional_number(payload, "archive_score", line_number),
        depth=_optional_int(payload, "depth", line_number),
        selection_channels=_string_list(payload, "selection_channels", line_number),
        matched_rule_ids=_string_list(payload, "matched_rule_ids", line_number),
    )


def serialize_message(message: NormalizedMessage) -> dict[str, Any]:
    """Serialize one message with provenance and channel tuples as lists."""
    serialized = asdict(message)
    serialized["provenance"] = [asdict(provenance) for provenance in message.provenance]
    serialized["selection_channels"] = list(message.selection_channels)
    serialized["matched_rule_ids"] = list(message.matched_rule_ids)
    return serialized


def write_compressed_jsonl(output_path: Path, messages: Iterable[dict[str, Any]]) -> None:
    """Atomically write messages to a compressed JSONL shard."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        with temporary_path.open("wb") as file_handle:
            with zstandard.ZstdCompressor().stream_writer(file_handle, closefd=False) as compressed:
                for message in messages:
                    compressed.write(
                        json.dumps(message, ensure_ascii=False, sort_keys=True).encode()
                    )
                    compressed.write(b"\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def stable_rank(fullname: str, seed: int) -> int:
    """Deterministic content rank independent of input order."""
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{fullname}".encode()).digest(),
        byteorder="big",
    )


def retain_lowest_ranked(
    selection_heap: list[tuple[int, str]],
    selected_by_fullname: dict[str, tuple[int, dict[str, Any]]],
    rank: int,
    fullname: str,
    message: dict[str, Any],
    target: int,
) -> None:
    """Keep the target lowest (rank, fullname) messages in a bounded heap."""
    if fullname in selected_by_fullname:
        return
    if len(selected_by_fullname) < target:
        selected_by_fullname[fullname] = (rank, message)
        heapq.heappush(selection_heap, (-rank, fullname))
        return

    while selection_heap:
        negative_rank, worst_fullname = selection_heap[0]
        current = selected_by_fullname.get(worst_fullname)
        if current and current[0] == -negative_rank:
            break
        heapq.heappop(selection_heap)
    worst_rank = -selection_heap[0][0]
    worst_fullname = selection_heap[0][1]
    if (rank, fullname) >= (worst_rank, worst_fullname):
        return
    heapq.heappop(selection_heap)
    del selected_by_fullname[worst_fullname]
    selected_by_fullname[fullname] = (rank, message)
    heapq.heappush(selection_heap, (-rank, fullname))


def _required_string(payload: dict[str, Any], key: str, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"selected JSONL row {line_number} has invalid {key}")
    return value


def _required_int(payload: dict[str, Any], key: str, line_number: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"selected JSONL row {line_number} has invalid {key}")
    return value


def _optional_string(payload: dict[str, Any], key: str, line_number: int) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"selected JSONL row {line_number} has invalid {key}")
    return value


def _optional_number(payload: dict[str, Any], key: str, line_number: int) -> int | float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"selected JSONL row {line_number} has invalid {key}")
    return value


def _optional_int(payload: dict[str, Any], key: str, line_number: int) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"selected JSONL row {line_number} has invalid {key}")
    return value


def _string_list(payload: dict[str, Any], key: str, line_number: int) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"selected JSONL row {line_number} has invalid {key}")
    return tuple(value)
