"""Export bounded discovery selections of any message kind as review cards."""

import json
from collections.abc import Iterable
from pathlib import Path

import zstandard

from reddit_search.corpus.units import (
    CHUNKING_VERSION,
    CONTEXT_RECIPE_VERSION,
    SearchUnit,
    build_message_units,
)
from reddit_search.ingest.hydrate import ContextBuilder
from reddit_search.ingest.invalidation import load_tombstone_ledger, tombstone_identity
from reddit_search.ingest.normalize import NormalizedMessage
from reddit_search.ingest.shard import read_selected_messages as _read_selected_messages


def export_discovery_review_cards(
    selection_path: Path,
    output_directory: Path,
    *,
    snapshot_id: str,
    tombstones_path: Path | None = None,
    context_recipe_version: str = CONTEXT_RECIPE_VERSION,
    chunking_version: str = CHUNKING_VERSION,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> dict[str, object]:
    """Write review cards for a bounded mixed-kind discovery shard."""
    if not snapshot_id:
        raise ValueError("snapshot_id must not be empty")
    if not context_recipe_version.strip():
        raise ValueError("context_recipe_version must not be empty")
    tombstones = load_tombstone_ledger(tombstones_path)

    try:
        candidates = list(_read_selected_messages(selection_path))
    except (UnicodeDecodeError, zstandard.ZstdError) as error:
        raise ValueError(f"could not read selection shard: {selection_path}") from error
    tombstoned_count = sum(
        tombstones.matches(message.fullname, message.source_revision_id) for message in candidates
    )
    messages = [
        message
        for message in candidates
        if not tombstones.matches(message.fullname, message.source_revision_id)
    ]
    messages_by_fullname = {message.fullname: message for message in messages}
    if len(messages_by_fullname) != len(messages):
        raise ValueError("selection shard contains duplicate message fullnames")

    context_builder = ContextBuilder(messages_by_fullname)
    cards_path = _write_cards(
        output_directory,
        (
            _discovery_card(unit, message, selection_order=index)
            for index, message in enumerate(messages, start=1)
            for unit in build_message_units(
                snapshot_id,
                context_builder.build(message.fullname),
                context_recipe_version=context_recipe_version,
                chunking_version=chunking_version,
                synthetic=False,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        ),
    )
    return {
        "scope": "all_discovery_sources",
        "snapshot_id": snapshot_id,
        "selected_message_count": len(messages),
        "tombstoned_count": tombstoned_count,
        "tombstones": tombstone_identity(tombstones),
        "context_recipe_version": context_recipe_version,
        "card_count": len(cards_path.read_text(encoding="utf-8").splitlines()),
        "cards_file": str(cards_path),
    }


def _discovery_card(
    unit: SearchUnit, message: NormalizedMessage, *, selection_order: int
) -> dict[str, object]:
    return {
        "candidate_id": unit.unit_id,
        "synthetic": unit.synthetic,
        "source": {
            "message_fullname": unit.message_fullname,
            "context_recipe_version": unit.context_recipe_version,
            "chunking_version": unit.chunking_version,
            "source_revision_id": unit.source_revision_id,
            "field": unit.focus_field,
            "title": message.raw_title or None,
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
            "ancestors_truncated": unit.ancestors_truncated,
            "missing_parent_ids": list(unit.missing_context_ids),
            "context_complete": not unit.context_missing,
        },
        "selection": {
            "channels": list(message.selection_channels),
            "matched_rule_ids": list(message.matched_rule_ids),
            "selection_order": selection_order,
        },
        "topic_fit": "unknown",
        "product_fit": "not_evaluated",
        "live_status": "unverified",
        "policy_status": "unverified",
        "review_status": "pending",
    }


def _write_cards(output_directory: Path, cards: Iterable[dict[str, object]]) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / "review_cards.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for card in cards:
            stream.write(json.dumps(card, sort_keys=True))
            stream.write("\n")
    temporary.replace(destination)
    return destination
