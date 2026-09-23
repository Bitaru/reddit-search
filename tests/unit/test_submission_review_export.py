import json
from pathlib import Path

import zstandard


def test_discovery_review_export_preserves_discovery_evidence(tmp_path: Path) -> None:
    from reddit_search.ingest.review_export import export_discovery_review_cards

    selection = tmp_path / "selected.jsonl.zst"
    payload = {
        "fullname": "t3_need",
        "kind": "submission",
        "thread_fullname": "t3_need",
        "parent_fullname": None,
        "raw_title": "Need an expense tracker",
        "raw_body": "I will not link a bank account.",
        "subreddit": "personalfinance",
        "created_utc": 1,
        "source_revision_id": "revision-1",
        "permalink": "/r/personalfinance/comments/need/",
        "provenance": [{"source_id": "june-submissions", "line_number": 7}],
        "archive_score": None,
        "depth": None,
        "selection_channels": ["topic_rule"],
        "matched_rule_ids": ["mieru.no_bank_link"],
    }
    with selection.open("wb") as output:
        with zstandard.ZstdCompressor().stream_writer(output, closefd=False) as compressed:
            compressed.write(json.dumps(payload).encode() + b"\n")

    report = export_discovery_review_cards(
        selection, tmp_path / "review", snapshot_id="june-2026-submissions-prefix"
    )

    cards_path = tmp_path / "review" / "review_cards.jsonl"
    card = json.loads(cards_path.read_text())
    assert report == {
        "scope": "all_discovery_sources",
        "snapshot_id": "june-2026-submissions-prefix",
        "selected_message_count": 1,
        "tombstoned_count": 0,
        "tombstones": {"path": None, "sha256": None, "count": 0},
        "context_recipe_version": "v2",
        "card_count": 1,
        "cards_file": str(cards_path),
    }
    assert card["synthetic"] is False
    assert card["source"]["context_recipe_version"] == "v2"
    assert card["source"]["message_fullname"] == "t3_need"
    assert card["source"]["text"] == "I will not link a bank account."
    assert card["source"]["title"] == "Need an expense tracker"
    assert card["context"] == {
        "text": "[FOCUS SUBMISSION t3_need]\nI will not link a bank account.",
        "message_fullnames": [],
        "missing_parent_ids": [],
        "ancestors_truncated": False,
        "context_complete": True,
    }
    assert card["selection"] == {
        "channels": ["topic_rule"],
        "matched_rule_ids": ["mieru.no_bank_link"],
        "selection_order": 1,
    }
    assert "retrieval" not in card


def test_discovery_review_export_hydrates_comment_context(tmp_path: Path) -> None:
    from reddit_search.ingest.review_export import export_discovery_review_cards

    selection = tmp_path / "selected.jsonl.zst"
    parent = {
        "fullname": "t3_need",
        "kind": "submission",
        "thread_fullname": "t3_need",
        "parent_fullname": None,
        "raw_title": "Need an expense tracker",
        "raw_body": "I will not link a bank account.",
        "subreddit": "personalfinance",
        "created_utc": 1,
        "source_revision_id": "revision-1",
        "permalink": "/r/personalfinance/comments/need/",
        "provenance": [{"source_id": "june-comments", "line_number": 3}],
        "archive_score": None,
        "depth": None,
        "selection_channels": [],
        "matched_rule_ids": [],
    }
    comment = {
        "fullname": "t1_reply",
        "kind": "comment",
        "thread_fullname": "t3_need",
        "parent_fullname": "t3_need",
        "raw_title": "",
        "raw_body": "Same here, manual entry is a dealbreaker for me.",
        "subreddit": "personalfinance",
        "created_utc": 2,
        "source_revision_id": "revision-2",
        "permalink": "/r/personalfinance/comments/need/x/",
        "provenance": [{"source_id": "june-comments", "line_number": 9}],
        "archive_score": 4,
        "depth": 1,
        "selection_channels": ["topic_rule"],
        "matched_rule_ids": ["mieru.manual_entry"],
    }
    with selection.open("wb") as output:
        with zstandard.ZstdCompressor().stream_writer(output, closefd=False) as compressed:
            compressed.write(json.dumps(parent).encode() + b"\n")
            compressed.write(json.dumps(comment).encode() + b"\n")

    report = export_discovery_review_cards(
        selection, tmp_path / "review", snapshot_id="june-2026-comments-prefix"
    )

    cards_path = tmp_path / "review" / "review_cards.jsonl"
    cards = [json.loads(line) for line in cards_path.read_text().splitlines()]
    assert report["selected_message_count"] == 2
    assert report["card_count"] == 2
    comment_card = next(card for card in cards if card["source"]["message_fullname"] == "t1_reply")
    assert comment_card["source"]["title"] is None
    assert comment_card["source"]["text"] == "Same here, manual entry is a dealbreaker for me."
    assert comment_card["context"]["message_fullnames"] == ["t3_need"]
    assert comment_card["context"]["context_complete"] is True
    assert "[SUBMISSION t3_need]" in comment_card["context"]["text"]


def test_discovery_review_export_excludes_tombstoned_message(tmp_path: Path) -> None:
    from reddit_search.ingest.review_export import export_discovery_review_cards

    selection = tmp_path / "selected.jsonl.zst"
    payload = {
        "fullname": "t1_removed",
        "kind": "comment",
        "thread_fullname": "t3_root",
        "parent_fullname": "t3_root",
        "raw_title": "",
        "raw_body": "Removed context.",
        "subreddit": "personalfinance",
        "created_utc": 1,
        "source_revision_id": "revision-1",
        "permalink": "/r/personalfinance/comments/root/x/",
        "provenance": [{"source_id": "june-comments", "line_number": 1}],
        "archive_score": 1,
        "depth": 1,
        "selection_channels": [],
        "matched_rule_ids": [],
    }
    with selection.open("wb") as output:
        with zstandard.ZstdCompressor().stream_writer(output, closefd=False) as compressed:
            compressed.write(json.dumps(payload).encode() + b"\n")
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t1_removed",
                "source_revision_id": None,
                "reason": "removed by operator",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = export_discovery_review_cards(
        selection,
        tmp_path / "review",
        snapshot_id="snapshot",
        tombstones_path=tombstones,
    )

    assert report["selected_message_count"] == 0
    assert report["card_count"] == 0
    assert report["tombstoned_count"] == 1


def test_review_export_emits_configured_chunks_and_version(tmp_path: Path) -> None:
    from reddit_search.ingest.review_export import export_discovery_review_cards

    selection = tmp_path / "selected.jsonl.zst"
    payload = {
        "fullname": "t3_long",
        "kind": "submission",
        "thread_fullname": "t3_long",
        "parent_fullname": None,
        "raw_title": "Long",
        "raw_body": "one two three four five six seven eight nine",
        "subreddit": "test",
        "created_utc": 1,
        "source_revision_id": "rev",
        "permalink": "/x/",
        "provenance": [{"source_id":"x","line_number":1}],
        "archive_score": None,
        "depth": None,
        "selection_channels": [],
        "matched_rule_ids": [],
    }
    with selection.open("wb") as output:
        with zstandard.ZstdCompressor().stream_writer(output, closefd=False) as compressed:
            compressed.write(json.dumps(payload).encode() + b"\n")
    report = export_discovery_review_cards(
        selection,
        tmp_path / "review",
        snapshot_id="s",
        max_tokens=3,
        overlap_tokens=1,
        chunking_version="test-v9",
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "review" / "review_cards.jsonl").read_text().splitlines()
    ]
    assert report["card_count"] == len(rows) > 1
    assert {row["source"]["chunking_version"] for row in rows} == {"test-v9"}
