"""Normalize Reddit-style source rows without mutating evidence text."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from .reader import RawRecordEnvelope


class NormalizationError(ValueError):
    """Raised for malformed source fields that cannot be truthfully normalized."""


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    source_id: str
    line_number: int


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    fullname: str
    kind: Literal["submission", "comment"]
    thread_fullname: str
    parent_fullname: str | None
    raw_title: str
    raw_body: str
    subreddit: str
    created_utc: int
    source_revision_id: str
    permalink: str
    provenance: tuple[SourceProvenance, ...]
    archive_score: int | float | None = None
    depth: int | None = None
    selection_channels: tuple[str, ...] = field(default_factory=tuple)
    matched_rule_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def focus_text(self) -> str:
        return "\n\n".join(part for part in (self.raw_title, self.raw_body) if part)


def normalize_record(envelope: RawRecordEnvelope) -> NormalizedMessage:
    """Create one typed message, refusing fields that would require invention."""
    payload = envelope.payload
    expected_prefix = "t3_" if envelope.source_kind == "submission" else "t1_"
    fullname = _fullname(payload, "name", expected_prefix, fallback_key="id")
    thread_fullname = (
        fullname if envelope.source_kind == "submission" else _fullname(payload, "link_id", "t3_")
    )
    parent_fullname = _optional_fullname(payload, "parent_id")
    raw_title = _required_string(payload, "title") if envelope.source_kind == "submission" else ""
    raw_body = (
        _submission_body(payload)
        if envelope.source_kind == "submission"
        else _required_string(payload, "body")
    )
    subreddit = _required_string(payload, "subreddit")
    permalink = _required_string(payload, "permalink")
    created_utc = _utc_seconds(payload.get("created_utc"))
    score = _optional_number(payload.get("score"), "score")
    depth = _optional_int(payload.get("depth"), "depth")
    revision = hashlib.sha256(f"{raw_title}\0{raw_body}".encode()).hexdigest()

    return NormalizedMessage(
        fullname=fullname,
        kind=envelope.source_kind,
        thread_fullname=thread_fullname,
        parent_fullname=parent_fullname,
        raw_title=raw_title,
        raw_body=raw_body,
        subreddit=subreddit,
        created_utc=created_utc,
        source_revision_id=revision,
        permalink=permalink,
        provenance=(SourceProvenance(envelope.source_id, envelope.line_number),),
        archive_score=score,
        depth=depth,
    )


def _fullname(
    payload: dict[str, Any], key: str, expected_prefix: str, *, fallback_key: str | None = None
) -> str:
    value = payload.get(key)
    if value is None and fallback_key is not None:
        raw_id = payload.get(fallback_key)
        value = f"{expected_prefix}{raw_id}" if isinstance(raw_id, str) else raw_id
    if (
        not isinstance(value, str)
        or not value.startswith(expected_prefix)
        or len(value) == len(expected_prefix)
    ):
        raise NormalizationError(f"{key} must be a full {expected_prefix} identifier")
    return value


def _optional_fullname(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith(("t1_", "t3_")):
        raise NormalizationError(f"{key} must be a full t1_ or t3_ identifier when supplied")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise NormalizationError(f"{key} must be a string")
    return value


def _submission_body(payload: dict[str, Any]) -> str:
    value = payload.get("selftext")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise NormalizationError("selftext must be a string or null")
    return value


def _utc_seconds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise NormalizationError("created_utc must be an integer UTC timestamp")
    return int(value)


def _optional_number(value: Any, field_name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizationError(f"{field_name} must be numeric when supplied")
    return value


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise NormalizationError(f"{field_name} must be an integer when supplied")
    return value
