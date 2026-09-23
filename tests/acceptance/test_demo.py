import json
import subprocess
import sys
from pathlib import Path


def run_demo(output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "demo",
            "--synthetic",
            "--network",
            "off",
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def read_cards(output: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (output / "review_cards.jsonl").read_text().splitlines()]


def test_network_off_synthetic_demo_finds_hidden_comment_with_traceable_context(
    tmp_path: Path,
) -> None:
    output = tmp_path / "demo"
    result = run_demo(output)

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    cards = read_cards(output)

    assert summary["synthetic"] is True
    assert summary["network"] == "off"
    assert summary["card_count"] == 1
    assert cards[0]["source"]["message_fullname"] == "t1_hidden_need"
    assert (
        cards[0]["source"]["text"] == "I need an expense tracker without linking my bank account."
    )
    assert "Weekly chat" in cards[0]["context"]["text"]
    assert cards[0]["context"]["missing_parent_ids"] == ["t1_older_month_parent"]
    assert cards[0]["product_fit"] == "not_evaluated"
    assert cards[0]["live_status"] == "unverified"


def test_synthetic_demo_export_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert run_demo(first).returncode == 0
    assert run_demo(second).returncode == 0

    assert (first / "review_cards.jsonl").read_bytes() == (
        second / "review_cards.jsonl"
    ).read_bytes()
