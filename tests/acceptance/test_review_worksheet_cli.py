import json
import subprocess
import sys
from pathlib import Path


def test_review_worksheet_exports_pending_annotations(tmp_path: Path) -> None:
    cards = tmp_path / "review_cards.jsonl"
    cards.write_text(
        json.dumps(
            {
                "candidate_id": "need-one",
                "review_queue": {"queue_order": 1, "stratum_rule_id": "rule.a"},
                "selection": {"matched_rule_ids": ["rule.a"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "worksheet"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "review",
            "worksheet",
            "--cards",
            str(cards),
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["worksheet_row_count"] == 1
    assert (output / "review_worksheet.jsonl").exists()
    assert (output / "worksheet_manifest.json").exists()
