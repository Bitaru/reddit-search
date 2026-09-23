"""Source-preserving message units for lexical and later dense retrieval."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass

from reddit_search.ingest.hydrate import EvidenceBundle

CONTEXT_RECIPE_VERSION = "v2"
CHUNKING_VERSION = "v2"


@dataclass(frozen=True, slots=True)
class SearchUnit:
    unit_id: str
    snapshot_id: str
    message_fullname: str
    source_revision_id: str
    thread_fullname: str
    focus_field: str
    focus_start: int
    focus_end: int
    focus_text: str
    context_only_text: str
    context_text: str
    missing_context_ids: tuple[str, ...]
    permalink: str
    subreddit: str
    created_utc: int
    synthetic: bool
    context_message_refs: tuple[str, ...] = ()
    chunking_version: str = "v1"
    context_recipe_version: str = CONTEXT_RECIPE_VERSION
    ancestors_truncated: bool = False

    @property
    def context_missing(self) -> bool:
        return bool(self.missing_context_ids) or self.ancestors_truncated


def estimate_focus_chunk_count(text: str, *, max_tokens: int, overlap_tokens: int) -> int:
    """Deterministic upper-bound count matching paragraph/oversized chunk rules."""
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("invalid focus chunk limits")
    paragraphs = re.findall(r"[^\n]+", text) or [""]
    step = max_tokens - overlap_tokens
    return sum(
        1
        if (n := len(re.findall(r"\S+", paragraph))) <= max_tokens
        else (1 + (n - max_tokens + step - 1) // step)
        for paragraph in paragraphs
    )


def build_message_units(
    snapshot_id: str,
    evidence: EvidenceBundle,
    *,
    chunking_version: str = CHUNKING_VERSION,
    context_recipe_version: str = CONTEXT_RECIPE_VERSION,
    synthetic: bool = True,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[SearchUnit]:
    """Build deterministic paragraph-aligned chunks without dropping tail text."""
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("invalid focus chunk limits")
    _, text = _focus_field_and_text(evidence)
    paragraphs = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+(?:\n|$)", text)]
    if not paragraphs:
        return [
            build_message_unit(
                snapshot_id,
                evidence,
                chunking_version=chunking_version,
                context_recipe_version=context_recipe_version,
                synthetic=synthetic,
            )
        ]
    spans = []
    current_start = paragraphs[0][0]
    current_end = current_start
    count = 0
    for para_start, para_end in paragraphs:
        words = list(re.finditer(r"\S+", text[para_start:para_end]))
        if len(words) > max_tokens:
            if current_end > current_start:
                spans.append((current_start, current_end))
            for i in range(0, len(words), max_tokens - overlap_tokens):
                a = para_start + words[i].start()
                b = para_start + words[min(i + max_tokens - 1, len(words) - 1)].end()
                spans.append((a, b))
                if i + max_tokens >= len(words):
                    break
            current_start = current_end = para_end
            count = 0
            continue
        if count and count + len(words) > max_tokens:
            spans.append((current_start, current_end))
            current_start = para_start
            count = 0
        if not count:
            current_start = para_start
        current_end = para_end
        count += len(words)
    if current_end > current_start:
        spans.append((current_start, current_end))
    result = [
        _build_unit(
            snapshot_id, evidence, a, b, chunking_version, context_recipe_version, synthetic
        )
        for a, b in spans
    ]
    assert all(len(re.findall(r"\S+", unit.focus_text)) <= max_tokens for unit in result)
    return result


def build_message_unit(
    snapshot_id: str,
    evidence: EvidenceBundle,
    *,
    chunking_version: str = "v1",
    context_recipe_version: str = CONTEXT_RECIPE_VERSION,
    synthetic: bool = True,
) -> SearchUnit:
    """Compatibility builder for short messages; chunk-aware callers use build_message_units."""
    _, text = _focus_field_and_text(evidence)
    return _build_unit(
        snapshot_id, evidence, 0, len(text), chunking_version, context_recipe_version, synthetic
    )


def _build_unit(
    snapshot_id, evidence, start, end, chunking_version, context_recipe_version, synthetic
):
    focus = evidence.focus
    focus_field, full_text = _focus_field_and_text(evidence)
    text = full_text[start:end]
    context_only_text = "\n\n".join(_labeled_text(message) for message in evidence.ancestors)
    label = "COMMENT" if focus.kind == "comment" else "SUBMISSION"
    focus_marker = f" {start}:{end}" if (start != 0 or end != len(full_text)) else ""
    context_text = "\n\n".join(
        part
        for part in (context_only_text, f"[FOCUS {label} {focus.fullname}{focus_marker}]\n{text}")
        if part
    )
    unit_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "\0".join(
                (
                    focus.fullname,
                    focus.source_revision_id,
                    focus_field,
                    str(start),
                    str(end),
                    chunking_version,
                )
            ),
        )
    )
    return SearchUnit(
        unit_id,
        snapshot_id,
        focus.fullname,
        focus.source_revision_id,
        focus.thread_fullname,
        focus_field,
        start,
        end,
        text,
        context_only_text,
        context_text,
        tuple(evidence.missing_parent_ids),
        focus.permalink,
        focus.subreddit,
        focus.created_utc,
        synthetic,
        tuple(message.fullname for message in evidence.ancestors),
        chunking_version,
        context_recipe_version,
        evidence.ancestors_truncated,
    )


def unit_content_hash(unit: SearchUnit) -> str:
    return hashlib.sha256(
        "\0".join(
            (
                unit.focus_text,
                unit.context_only_text,
                unit.chunking_version,
                unit.context_recipe_version,
            )
        ).encode()
    ).hexdigest()


def _focus_field_and_text(evidence):
    focus = evidence.focus
    if focus.kind == "comment":
        return "body", focus.raw_body
    if focus.raw_body:
        return "selftext", focus.raw_body
    return "title", focus.raw_title


def _labeled_text(message):
    label = "SUBMISSION" if message.kind == "submission" else "COMMENT"
    return f"[{label} {message.fullname}]\n" + "\n".join(
        part for part in (message.raw_title, message.raw_body) if part
    )
