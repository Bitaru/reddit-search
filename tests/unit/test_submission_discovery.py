import json
from pathlib import Path

import pytest
import zstandard


def complete_submission_registry(source: Path, registry: Path) -> None:
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "source_id": "june-submissions",
                        "source_path": str(source),
                        "source_kind": "submission",
                        "declared_month": "2026-06",
                        "source_role": "discovery",
                        "usage_scope": "submission_only_test",
                        "input_size_bytes": source.stat().st_size,
                        "status": "complete",
                    }
                ],
            }
        )
    )


def test_submission_discovery_writes_bounded_selected_shard(tmp_path: Path) -> None:
    from reddit_search.ingest.pilot import run_discovery

    source = tmp_path / "RS_2026-06.jsonl"
    source.write_text(
        json.dumps(
            {
                "name": "t3_need",
                "title": "Need an expense tracker",
                "selftext": "I do not want to link a bank account.",
                "subreddit": "personalfinance",
                "created_utc": 1,
                "permalink": "/r/personalfinance/comments/need/",
            }
        )
        + "\n"
    )
    registry = tmp_path / "registry.json"
    complete_submission_registry(source, registry)
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """rules:
  - rule_id: mieru.no_bank_link
    required_term_groups:
      - [expense, spending]
      - [bank, sync, link]
"""
    )
    output = tmp_path / "selected.jsonl.zst"

    summary = run_discovery(registry, rules, output, target=10, seed=7)
    repeated = run_discovery(registry, rules, output, target=10, seed=7)
    assert repeated == summary
    assert output.with_name(output.name + ".manifest.json").exists()

    with pytest.raises(ValueError, match="does not match current inputs"):
        run_discovery(registry, rules, output, target=10, seed=8)

    with output.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as decompressed:
            rows = decompressed.read().decode().splitlines()
    selected = json.loads(rows[0])
    assert summary["scope"] == "all_discovery_sources"
    assert summary["selected_count"] == 1
    assert summary["normalization_error_count"] == 0
    assert summary["output_file"] == str(output)
    assert selected["fullname"] == "t3_need"
    assert selected["matched_rule_ids"] == ["mieru.no_bank_link"]


def test_discovery_withholds_output_when_staging_budget_is_exceeded(tmp_path: Path) -> None:
    from reddit_search.ingest.pilot import run_discovery
    from reddit_search.resources import BudgetError

    source = tmp_path / "RS_2026-06.jsonl"
    source.write_text(
        json.dumps(
            {
                "name": "t3_need",
                "title": "Expense tracker request",
                "selftext": "No bank link please.",
                "subreddit": "personalfinance",
                "created_utc": 1,
                "permalink": "/r/personalfinance/comments/need/",
            }
        )
        + "\n"
    )
    registry = tmp_path / "registry.json"
    complete_submission_registry(source, registry)
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """rules:
  - rule_id: mieru.no_bank_link
    required_term_groups:
      - [expense, spending]
      - [bank, sync, link]
""",
        encoding="utf-8",
    )
    output = tmp_path / "selected.jsonl.zst"

    with pytest.raises(BudgetError, match="staging budget exceeded"):
        run_discovery(
            registry,
            rules,
            output,
            target=10,
            seed=7,
            max_staging_bytes=1,
            minimum_free_disk_bytes=0,
        )
    assert not output.exists()
    assert not output.with_suffix(".zst.tmp").exists()


def test_discovery_leaves_registered_source_partial_at_record_cap(tmp_path: Path) -> None:
    from reddit_search.ingest.pilot import run_discovery

    source = tmp_path / "RS_2026-06.jsonl"
    source.write_text(
        json.dumps(
            {
                "name": "t3_need",
                "title": "Expense tracker request",
                "selftext": "No bank link please.",
                "subreddit": "personalfinance",
                "created_utc": 1,
                "permalink": "/r/personalfinance/comments/need/",
            }
        )
        + "\n"
        + json.dumps(
            {
                "name": "t3_later",
                "title": "Later submission",
                "selftext": "Irrelevant.",
                "subreddit": "personalfinance",
                "created_utc": 2,
                "permalink": "/r/personalfinance/comments/later/",
            }
        )
        + "\n"
    )
    registry = tmp_path / "registry.json"
    complete_submission_registry(source, registry)
    payload = json.loads(registry.read_text())
    payload["sources"][0]["status"] = "registered"
    registry.write_text(json.dumps(payload))
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """rules:
  - rule_id: mieru.no_bank_link
    required_term_groups:
      - [expense, spending]
      - [bank, sync, link]
"""
    )

    summary = run_discovery(
        registry,
        rules,
        tmp_path / "selected.jsonl.zst",
        target=10,
        seed=7,
        max_records=1,
    )

    persisted = json.loads(registry.read_text())["sources"][0]
    assert summary["scan_complete"] is False
    assert summary["scanned_record_count"] == 1
    assert summary["selected_count"] == 1
    assert persisted["status"] == "registered"
    assert persisted.get("verified_sha256") is None


def test_discovery_deduplicates_replayed_fullnames(tmp_path: Path) -> None:
    from reddit_search.ingest.pilot import run_discovery

    message = {
        "name": "t3_need",
        "title": "Need an expense tracker",
        "selftext": "I do not want to link a bank account.",
        "subreddit": "personalfinance",
        "created_utc": 1,
        "permalink": "/r/personalfinance/comments/need/",
    }
    source = tmp_path / "RS_2026-06.jsonl"
    source.write_text("\n".join(json.dumps(message) for _ in range(2)) + "\n")
    registry = tmp_path / "registry.json"
    complete_submission_registry(source, registry)
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """rules:
  - rule_id: mieru.no_bank_link
    required_term_groups:
      - [expense, spending]
      - [bank, sync, link]
"""
    )

    summary = run_discovery(registry, rules, tmp_path / "selected.jsonl.zst", target=10, seed=7)

    assert summary["selected_count"] == 1


def test_discovery_selects_comments_and_submissions_independently(tmp_path: Path) -> None:
    from reddit_search.ingest.pilot import run_discovery

    submissions = tmp_path / "RS_2026-06.jsonl"
    submissions.write_text(
        json.dumps(
            {
                "name": "t3_need",
                "title": "Need an expense tracker",
                "selftext": "I do not want to link a bank account.",
                "subreddit": "personalfinance",
                "created_utc": 1,
                "permalink": "/r/personalfinance/comments/need/",
            }
        )
        + "\n"
    )
    comments = tmp_path / "RC_2026-06.jsonl"
    comments.write_text(
        json.dumps(
            {
                "name": "t1_need",
                "link_id": "t3_other",
                "parent_id": "t3_other",
                "body": "I track spending manually but every app forces a bank link.",
                "subreddit": "personalfinance",
                "created_utc": 2,
                "permalink": "/r/personalfinance/comments/other/x/",
            }
        )
        + "\n"
        + json.dumps(
            {
                "name": "t1_noise",
                "link_id": "t3_other",
                "parent_id": "t3_other",
                "body": "The weather is nice today.",
                "subreddit": "personalfinance",
                "created_utc": 3,
                "permalink": "/r/personalfinance/comments/other/y/",
            }
        )
        + "\n"
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "source_id": "june-submissions",
                        "source_path": str(submissions),
                        "source_kind": "submission",
                        "declared_month": "2026-06",
                        "source_role": "discovery",
                        "usage_scope": "test",
                        "input_size_bytes": submissions.stat().st_size,
                        "status": "complete",
                    },
                    {
                        "source_id": "june-comments",
                        "source_path": str(comments),
                        "source_kind": "comment",
                        "declared_month": "2026-06",
                        "source_role": "discovery",
                        "usage_scope": "test",
                        "input_size_bytes": comments.stat().st_size,
                        "status": "complete",
                    },
                ],
            }
        )
    )
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """rules:
  - rule_id: mieru.no_bank_link
    required_term_groups:
      - [expense, spending]
      - [bank, sync, link]
"""
    )
    output = tmp_path / "selected.jsonl.zst"

    summary = run_discovery(registry, rules, output, target=10, seed=7)

    with output.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as decompressed:
            rows = [json.loads(line) for line in decompressed.read().decode().splitlines()]
    fullnames = {row["fullname"] for row in rows}
    assert summary["scope"] == "all_discovery_sources"
    assert summary["source_count"] == 2
    assert fullnames == {"t3_need", "t1_need"}
    assert all(row["matched_rule_ids"] == ["mieru.no_bank_link"] for row in rows)
    comment_row = next(row for row in rows if row["fullname"] == "t1_need")
    assert comment_row["kind"] == "comment"
    assert comment_row["thread_fullname"] == "t3_other"


def test_discovery_excludes_tombstoned_messages(tmp_path: Path) -> None:
    from reddit_search.ingest.pilot import run_discovery

    source = tmp_path / "RS_2026-06.jsonl"
    source.write_text(
        json.dumps(
            {
                "name": "t3_need",
                "title": "Need an expense tracker",
                "selftext": "I do not want to link a bank account.",
                "subreddit": "personalfinance",
                "created_utc": 1,
                "permalink": "/r/personalfinance/comments/need/",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    complete_submission_registry(source, registry)
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """rules:
  - rule_id: mieru.no_bank_link
    required_term_groups:
      - [expense, spending]
      - [bank, sync, link]
""",
        encoding="utf-8",
    )
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t3_need",
                "source_revision_id": None,
                "reason": "removed by operator",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_discovery(
        registry,
        rules,
        tmp_path / "selected.jsonl.zst",
        target=10,
        seed=7,
        tombstones_path=tombstones,
    )

    assert summary["selected_count"] == 0
    assert summary["tombstoned_count"] == 1

def test_discovery_rss_budget_withholds_output_and_records_manifest(tmp_path: Path) -> None:
    from reddit_search.ingest.pilot import run_discovery

    source = tmp_path / "RS_2026-06.jsonl"
    source.write_text(json.dumps({"name": "t3_need", "title": "expense bank link", "selftext": "",
                                  "subreddit": "personalfinance", "created_utc": 1,
                                  "permalink": "/r/personalfinance/comments/need/"}) + "\n")
    registry = tmp_path / "registry.json"
    complete_submission_registry(source, registry)
    rules = tmp_path / "rules.yaml"
    rules.write_text("rules:\n  - rule_id: test\n    required_term_groups:\n      - [expense]\n")
    output = tmp_path / "selected.jsonl.zst"
    result = run_discovery(registry, rules, output, target=1, seed=7,
                           max_process_rss_bytes=1, rss_sampler=lambda: 2)
    report = json.loads(output.with_name(output.name + ".manifest.json").read_text())
    assert result["complete"] is False
    assert result["budget_exhausted"] is True
    assert not output.exists()
    assert report["complete"] is False and report["budget_exhausted"] is True
    assert report["memory_budget"]["max_process_rss_bytes"] == 1
