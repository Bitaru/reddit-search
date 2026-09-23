import json
import subprocess
import sys
from pathlib import Path

import zstandard


def test_review_cards_exports_selected_shard(tmp_path: Path) -> None:
    selection = tmp_path / "selected.jsonl.zst"
    payload = {
        "fullname": "t3_need",
        "kind": "submission",
        "thread_fullname": "t3_need",
        "parent_fullname": None,
        "raw_title": "Need an expense tracker",
        "raw_body": "No bank link.",
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

    review_output = tmp_path / "review"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "review",
            "cards",
            "--selection",
            str(selection),
            "--output",
            str(review_output),
            "--snapshot-id",
            "june-2026-submissions-prefix",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["card_count"] == 1
    assert (review_output / "review_cards.jsonl").exists()


def test_review_cards_rejects_corrupt_selected_shard(tmp_path: Path) -> None:
    selection = tmp_path / "selected.jsonl.zst"
    selection.write_bytes(b"not a zstandard stream")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "review",
            "cards",
            "--selection",
            str(selection),
            "--output",
            str(tmp_path / "review"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert json.loads(result.stderr)["exported"] is False
