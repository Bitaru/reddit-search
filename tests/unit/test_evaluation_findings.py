import json
from pathlib import Path

import pytest

from reddit_search.evaluation.findings import export_findings


def _card(candidate_id: str, scenario_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "snapshot_id": "snap1",
        "app_id": None,
        "source": {
            "message_fullname": f"t1_{candidate_id}",
            "thread_fullname": "t3_root",
            "source_revision_id": f"rev_{candidate_id}",
            "field": "selftext",
            "title": "Need tracker",
            "text": "I need to track spending without bank sync.",
            "permalink": f"/r/test/comments/{candidate_id}/",
            "subreddit": "test",
            "created_utc": 1_780_000_000,
            "context_recipe_version": "v1",
        },
        "context": {"text": "context text", "context_complete": True},
        "selection": {"matched_rule_ids": ["expense.no_bank_link"], "selection_order": 1},
        "review_status": "complete",
    }


def _label(candidate_id: str, scenario_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "snapshot_id": "snap1",
        "topic_fit": "yes",
        "validation_status": "valid",
        "product_fit": "not_evaluated",
        "supported_claim_ids": [],
        "evidence": [],
        "speaker_intent": "seeking_solution",
        "resolution_in_available_context": "unknown",
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_export_findings_joins_relevant_rows_with_manifest(tmp_path: Path) -> None:
    cards = _write_jsonl(tmp_path / "cards.jsonl", [_card("u1", "example_app.need")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            _label(
                "u1",
                "example_app.need",
                supported_claim_ids=["example_app.claim"],
                evidence=[{"quote": "spending", "start": 18, "end": 26}],
            )
        ],
    )
    output = tmp_path / "findings"

    manifest = export_findings(labels, cards, output)

    assert manifest["findings_count"] == 1
    assert manifest["skipped_uncertain"] == 0
    assert manifest["skipped_invalid"] == 0
    assert manifest["labels_sha256"] == _sha(labels)
    assert manifest["cards_sha256"][str(cards)] == _sha(cards)

    rows = [json.loads(line) for line in (output / "findings.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    finding = rows[0]
    assert finding["scenario_id"] == "example_app.need"
    assert finding["message_fullname"] == "t1_u1"
    assert finding["thread_fullname"] == "t3_root"
    assert finding["permalink"] == "/r/test/comments/u1/"
    assert finding["snapshot_id"] == "snap1"
    assert finding["subreddit"] == "test"
    assert finding["created_utc"] == 1_780_000_000
    assert finding["matched_rule_ids"] == ["expense.no_bank_link"]
    assert finding["supported_claim_ids"] == ["example_app.claim"]
    assert finding["evidence"] == [{"quote": "spending", "start": 18, "end": 26}]
    assert finding["source_revision_id"] == "rev_u1"
    assert (output / "findings_manifest.json").is_file()


def test_export_findings_skips_uncertain_rows(tmp_path: Path) -> None:
    cards = _write_jsonl(tmp_path / "cards.jsonl", [_card("u1", "example_app.need")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl", [_label("u1", "example_app.need", topic_fit="unknown")]
    )
    output = tmp_path / "findings"

    manifest = export_findings(labels, cards, output)

    assert manifest["findings_count"] == 0
    assert manifest["skipped_uncertain"] == 1
    assert (output / "findings.jsonl").read_text() == ""


def test_export_findings_never_exports_invalid_validation_status(tmp_path: Path) -> None:
    cards = _write_jsonl(tmp_path / "cards.jsonl", [_card("u1", "example_app.need")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [_label("u1", "example_app.need", validation_status="unknown")],
    )
    output = tmp_path / "findings"

    manifest = export_findings(labels, cards, output)

    assert manifest["findings_count"] == 0
    assert manifest["skipped_invalid"] == 1


def test_export_findings_rejects_label_without_matching_card(tmp_path: Path) -> None:
    cards = _write_jsonl(tmp_path / "cards.jsonl", [_card("u1", "example_app.need")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl", [_label("ghost", "example_app.other")]
    )
    output = tmp_path / "findings"

    with pytest.raises(ValueError, match="no matching card"):
        export_findings(labels, cards, output)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
