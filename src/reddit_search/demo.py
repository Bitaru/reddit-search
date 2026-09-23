"""The deterministic, network-free lexical demonstration."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard

from .corpus.sqlite_store import LexicalStore
from .corpus.units import build_message_units
from .ingest.discovery import DiscoveryRule, DiscoverySelector
from .ingest.hydrate import ContextBuilder
from .ingest.manifest import manifest_from_reader
from .ingest.normalize import NormalizedMessage, normalize_record
from .ingest.reader import ArchiveReader, SourceSpec
from .review.export import write_lexical_review_cards

_DEMO_SNAPSHOT_ID = "synthetic-v1"
_DEMO_CHUNKING_VERSION = "v2"
_DEMO_FOCUS_CHUNK_TOKENS = 512
_DEMO_FOCUS_OVERLAP_TOKENS = 64
_DEMO_SCENARIO_ID = "demo.expense_tracking_without_bank_sync"
_DEMO_QUERY = "expense AND bank"


def run_synthetic_demo(output: Path) -> dict[str, Any]:
    """Run the lexical vertical slice using only freshly created synthetic data."""
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reddit-search-demo-") as temporary_directory:
        temporary = Path(temporary_directory)
        sources = _create_synthetic_sources(temporary)
        selector = DiscoverySelector(
            [
                DiscoveryRule(
                    _DEMO_SCENARIO_ID,
                    (("expense", "spending", "budget"), ("bank", "sync", "link")),
                )
            ]
        )
        messages: dict[str, NormalizedMessage] = {}
        selected_messages: list[NormalizedMessage] = []
        source_summaries: list[dict[str, Any]] = []

        for source in sources:
            reader = ArchiveReader()
            started_at = datetime.now(UTC)
            for envelope in reader.iter_records(source):
                message = normalize_record(envelope)
                messages[message.fullname] = message
                decision = selector.select(message)
                if decision.selected:
                    selected_messages.append(
                        replace(
                            message,
                            selection_channels=tuple(decision.selection_channels),
                            matched_rule_ids=tuple(decision.matched_rule_ids),
                        )
                    )
            manifest = manifest_from_reader(
                source,
                reader.stats,
                run_id="synthetic-demo",
                configuration_hash="synthetic-demo-v1",
                started_at=started_at,
            )
            source_summaries.append(
                {
                    "source_id": manifest.source_id,
                    "source_kind": manifest.source_kind,
                    "status": manifest.status,
                    "valid_records": manifest.valid_records,
                }
            )

        context_builder = ContextBuilder(messages)
        units = [
            unit
            for message in selected_messages
            for unit in build_message_units(
                _DEMO_SNAPSHOT_ID,
                context_builder.build(message.fullname),
                synthetic=True,
                chunking_version=_DEMO_CHUNKING_VERSION,
                max_tokens=_DEMO_FOCUS_CHUNK_TOKENS,
                overlap_tokens=_DEMO_FOCUS_OVERLAP_TOKENS,
            )
        ]
        with LexicalStore(temporary / "lexical.sqlite") as store:
            store.index_units(units)
            hits = store.search(snapshot_id=_DEMO_SNAPSHOT_ID, query=_DEMO_QUERY, limit=20)

        if not hits:
            raise RuntimeError("synthetic demo did not find its known relevant comment")
        cards_path = write_lexical_review_cards(
            output,
            hits,
            scenario_id=_DEMO_SCENARIO_ID,
            query=_DEMO_QUERY,
        )
        summary = {
            "synthetic": True,
            "network": "off",
            "snapshot_id": _DEMO_SNAPSHOT_ID,
            "selected_message_count": len(selected_messages),
            "card_count": len(hits),
            "cards_file": cards_path.name,
            "sources": source_summaries,
        }
        _write_json(output / "run_summary.json", summary)
        return summary


def _create_synthetic_sources(directory: Path) -> list[SourceSpec]:
    submissions = directory / "RS_synthetic.zst"
    comments = directory / "RC_synthetic.zst"
    _write_zstd_jsonl(
        submissions,
        [
            {
                "name": "t3_weekly_chat",
                "title": "Weekly chat",
                "selftext": "General discussion unrelated to budgeting.",
                "subreddit": "personalfinance",
                "created_utc": 1_725_148_700,
                "permalink": "/r/personalfinance/comments/weekly_chat/",
            }
        ],
    )
    _write_zstd_jsonl(
        comments,
        [
            {
                "name": "t1_irrelevant",
                "link_id": "t3_weekly_chat",
                "parent_id": "t3_weekly_chat",
                "body": "I like the weekly thread format.",
                "subreddit": "personalfinance",
                "created_utc": 1_725_148_750,
                "permalink": "/r/personalfinance/comments/weekly_chat/comment/irrelevant/",
            },
            {
                "name": "t1_hidden_need",
                "link_id": "t3_weekly_chat",
                "parent_id": "t1_older_month_parent",
                "body": "I need an expense tracker without linking my bank account.",
                "subreddit": "personalfinance",
                "created_utc": 1_725_148_800,
                "permalink": "/r/personalfinance/comments/weekly_chat/comment/hidden_need/",
                "score": 0,
                "depth": 12,
            },
        ],
    )
    return [
        SourceSpec("synthetic-submissions", submissions, "submission", "2026-09"),
        SourceSpec("synthetic-comments", comments, "comment", "2026-09"),
    ]


def _write_zstd_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode(
        "utf-8"
    )
    path.write_bytes(zstandard.ZstdCompressor().compress(payload))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
