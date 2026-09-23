import json
import subprocess
import sys
from pathlib import Path


def test_review_queue_exports_balanced_card_subset(tmp_path: Path) -> None:
    cards = tmp_path / "review_cards.jsonl"
    cards.write_text(
        "".join(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "selection": {
                        "matched_rule_ids": [rule_id],
                        "selection_order": order,
                    },
                }
            )
            + "\n"
            for candidate_id, order, rule_id in (
                ("a-one", 1, "rule.a"),
                ("a-two", 2, "rule.a"),
                ("b-one", 3, "rule.b"),
                ("b-two", 4, "rule.b"),
            )
        )
    )
    output = tmp_path / "queue"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "review",
            "queue",
            "--cards",
            str(cards),
            "--output",
            str(output),
            "--limit",
            "2",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["queued_card_count"] == 2
    assert (output / "review_cards.jsonl").exists()
    assert (output / "queue_manifest.json").exists()
