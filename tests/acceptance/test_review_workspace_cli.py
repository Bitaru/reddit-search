import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


def test_review_serve_starts_local_workspace(tmp_path: Path) -> None:
    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    cards_path.write_text(json.dumps(_card()) + "\n", encoding="utf-8")
    worksheet_path.write_text(json.dumps(_worksheet()) + "\n", encoding="utf-8")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "review",
            "serve",
            "--cards",
            str(cards_path),
            "--worksheet",
            str(worksheet_path),
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        launch = json.loads(process.stdout.readline())
        page = urlopen(launch["url"], timeout=5).read().decode()
    finally:
        process.terminate()
        process.wait(timeout=5)

    assert launch["host"] == "127.0.0.1"
    assert launch["card_count"] == 1
    assert "Local review workspace" in page


def test_review_serve_limits_workspace_to_candidate_id_file(tmp_path: Path) -> None:
    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    candidate_ids_path = tmp_path / "candidate_ids.json"
    cards_path.write_text(json.dumps(_card()) + "\n", encoding="utf-8")
    worksheet_path.write_text(json.dumps(_worksheet()) + "\n", encoding="utf-8")
    candidate_ids_path.write_text(json.dumps({"candidate_ids": ["candidate-1"]}), encoding="utf-8")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "review",
            "serve",
            "--cards",
            str(cards_path),
            "--worksheet",
            str(worksheet_path),
            "--candidate-ids",
            str(candidate_ids_path),
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        launch = json.loads(process.stdout.readline())
    finally:
        process.terminate()
        process.wait(timeout=5)

    assert launch["card_count"] == 1


def _card() -> dict[str, object]:
    return {
        "candidate_id": "candidate-1",
        "review_queue": {"queue_order": 1, "stratum_rule_id": "mieru.no_bank_link"},
        "selection": {"matched_rule_ids": ["mieru.no_bank_link"]},
        "source": {"title": "Need a tracker", "text": "No bank sync please."},
    }


def _worksheet() -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_id": "candidate-1",
        "review_queue": {"queue_order": 1, "stratum_rule_id": "mieru.no_bank_link"},
        "selection": {"matched_rule_ids": ["mieru.no_bank_link"]},
        "annotation": {
            "review_status": "pending",
            "topic_fit": "unreviewed",
            "need_clarity": "unreviewed",
            "duplicate_of": None,
            "reviewer_note": None,
        },
    }
