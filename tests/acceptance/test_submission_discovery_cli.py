import json
import subprocess
import sys
from pathlib import Path


def test_ingest_discover_runs_submission_only_source(tmp_path: Path) -> None:
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

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "ingest",
            "discover",
            "--registry",
            str(registry),
            "--rules",
            str(rules),
            "--output",
            str(output),
            "--target",
            "10",
            "--max-records",
            "1",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["scope"] == "all_discovery_sources"
    assert json.loads(result.stdout)["selected_count"] == 1


def test_ingest_discover_uses_runtime_staging_limit(tmp_path: Path) -> None:
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
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        """schema_version: 1
ingestion:
  max_staging_bytes: 1
  minimum_free_disk_bytes: 1
""",
        encoding="utf-8",
    )
    output = tmp_path / "selected.jsonl.zst"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "ingest",
            "discover",
            "--registry",
            str(registry),
            "--rules",
            str(rules),
            "--runtime",
            str(runtime),
            "--output",
            str(output),
            "--target",
            "10",
            "--max-records",
            "1",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "staging budget exceeded" in result.stderr
    assert not output.exists()
