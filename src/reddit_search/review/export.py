"""Deterministic JSONL review-card export for human evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from reddit_search.corpus.sqlite_store import LexicalHit
from reddit_search.ingest.invalidation import TombstoneLedger, tombstone_blocks_identity


def write_lexical_review_cards(
    output: Path,
    hits: list[LexicalHit],
    *,
    scenario_id: str,
    query: str,
    tombstone_ledger: TombstoneLedger | None = None,
) -> Path:
    """Atomically write evidence-backed pending cards without product-fit invention."""
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "review_cards.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for hit in hits:
            unit = hit.unit
            if tombstone_ledger is not None and tombstone_blocks_identity(
                tombstone_ledger,
                unit.message_fullname,
                unit.source_revision_id,
                unit.context_message_refs,
            ):
                continue
            stream.write(
                json.dumps(_card(hit, scenario_id=scenario_id, query=query), sort_keys=True)
            )
            stream.write("\n")
    temporary.replace(destination)
    return destination


def _card(hit: LexicalHit, *, scenario_id: str, query: str) -> dict[str, object]:
    unit = hit.unit
    return {
        "candidate_id": unit.unit_id,
        "synthetic": unit.synthetic,
        "source": {
            "message_fullname": unit.message_fullname,
            "source_revision_id": unit.source_revision_id,
            "chunking_version": unit.chunking_version,
            "context_recipe_version": unit.context_recipe_version,
            "field": unit.focus_field,
            "start": unit.focus_start,
            "end": unit.focus_end,
            "text": unit.focus_text,
            "permalink": unit.permalink,
            "subreddit": unit.subreddit,
            "created_utc": unit.created_utc,
        },
        "context": {
            "text": unit.context_text,
            "message_fullnames": list(unit.context_message_refs),
            "missing_parent_ids": list(unit.missing_context_ids),
            "ancestors_truncated": unit.ancestors_truncated,
            "context_complete": not unit.context_missing,
        },
        "retrieval": {
            "scenario_id": scenario_id,
            "branch": "lexical",
            "query": query,
            "rank": hit.rank,
            "score": hit.score,
            "snippet": hit.snippet,
        },
        "topic_fit": "unknown",
        "speaker_intent": "unknown",
        "product_fit": "not_evaluated",
        "resolution_in_available_context": "unknown",
        "live_status": "unverified",
        "policy_status": "unverified",
        "review_status": "pending",
    }
