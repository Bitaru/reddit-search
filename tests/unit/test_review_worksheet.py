import hashlib
import json
from pathlib import Path

import pytest


def test_review_worksheet_preserves_queue_identity_without_copying_evidence(
    tmp_path: Path,
) -> None:
    from reddit_search.review.worksheet import export_review_worksheet

    cards_path = tmp_path / "review_cards.jsonl"
    cards_path.write_text(
        "".join(
            json.dumps(card) + "\n"
            for card in (
                _card("second", queue_order=2, stratum_rule_id="rule.b"),
                _card("first", queue_order=1, stratum_rule_id="rule.a"),
            )
        ),
        encoding="utf-8",
    )

    report = export_review_worksheet(cards_path, tmp_path / "worksheet")

    worksheet_path = tmp_path / "worksheet" / "review_worksheet.jsonl"
    rows = [json.loads(line) for line in worksheet_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((tmp_path / "worksheet" / "worksheet_manifest.json").read_text())
    assert report == {
        "input_card_count": 2,
        "worksheet_row_count": 2,
        "worksheet_file": str(worksheet_path),
        "manifest_file": str(tmp_path / "worksheet" / "worksheet_manifest.json"),
    }
    assert [row["candidate_id"] for row in rows] == ["first", "second"]
    assert rows == [
        {
            "annotation": {
                "duplicate_of": None,
                "need_clarity": "unreviewed",
                "review_status": "pending",
                "reviewer_note": None,
                "topic_fit": "unreviewed",
                "speaker_intent": "unreviewed",
                "product_fit": "not_evaluated",
                "supported_claim_ids": [],
                "resolution_in_available_context": "unreviewed",
            },
            "candidate_id": "first",
            "review_queue": {"queue_order": 1, "stratum_rule_id": "rule.a"},
            "scenario_id": None,
            "schema_version": 1,
            "selection": {"matched_rule_ids": ["rule.a"]},
        },
        {
            "annotation": {
                "duplicate_of": None,
                "need_clarity": "unreviewed",
                "review_status": "pending",
                "reviewer_note": None,
                "topic_fit": "unreviewed",
                "speaker_intent": "unreviewed",
                "product_fit": "not_evaluated",
                "supported_claim_ids": [],
                "resolution_in_available_context": "unreviewed",
            },
            "candidate_id": "second",
            "review_queue": {"queue_order": 2, "stratum_rule_id": "rule.b"},
            "scenario_id": None,
            "schema_version": 1,
            "selection": {"matched_rule_ids": ["rule.b"]},
        },
    ]
    assert manifest["annotation_fields"] == {
        "duplicate_of": "candidate_id or null",
        "need_clarity": ["clear", "unclear", "unreviewed"],
        "review_status": ["complete", "pending"],
        "reviewer_note": "string or null",
        "topic_fit": ["not_relevant", "relevant", "uncertain", "unreviewed"],
        "speaker_intent": [
            "seeking_solution",
            "describing_pain",
            "sharing_experience",
            "answering_someone",
            "self_promotion",
            "quoting",
            "unknown",
            "unreviewed",
        ],
        "product_fit": ["compatible", "incompatible", "needs_clarification", "not_evaluated"],
        "supported_claim_ids": "list of verified claim IDs (required for compatible)",
        "resolution_in_available_context": [
            "resolved",
            "no_resolution_observed",
            "unknown",
            "unreviewed",
        ],
    }
    assert "focus" not in rows[0]
    assert manifest["review_cards_file"] == str(cards_path)


def test_worksheet_manifest_and_rows_preserve_dependency_identity(tmp_path: Path) -> None:
    from reddit_search.review.worksheet import export_review_worksheet

    card = _card("identified", queue_order=1, stratum_rule_id="rule.a")
    identity = {
        "snapshot_id": "snap-1",
        "app_id": "app-1",
        "app_profile_version": 2,
        "app_profile_sha256": "a" * 64,
        "source_revision_id": "rev-1",
        "source_fingerprint": "fingerprint-1",
        "context_recipe_version": "recipe-1",
        "context_dependency_identity": {"context": "dep-1"},
    }
    card.update(identity)
    card["evidence"] = {"text": "private evidence"}
    cards_path = tmp_path / "cards.jsonl"
    raw = (json.dumps(card, separators=(",", ":")) + "\n").encode()
    cards_path.write_bytes(raw)

    report = export_review_worksheet(cards_path, tmp_path / "worksheet")
    manifest = json.loads(Path(report["manifest_file"]).read_text())
    row = json.loads(Path(report["worksheet_file"]).read_text().strip())
    assert manifest["review_cards_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["dependency_identity_sha256"]
    assert {key: row[key] for key in identity} == identity
    assert "focus" not in row
    assert "evidence" not in row


def test_review_worksheet_rejects_duplicate_queue_order(tmp_path: Path) -> None:
    from reddit_search.review.worksheet import export_review_worksheet

    cards_path = tmp_path / "review_cards.jsonl"
    cards_path.write_text(
        "".join(
            json.dumps(card) + "\n"
            for card in (
                _card("first", queue_order=1, stratum_rule_id="rule.a"),
                _card("second", queue_order=1, stratum_rule_id="rule.b"),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate queue order"):
        export_review_worksheet(cards_path, tmp_path / "worksheet")


def _card(candidate_id: str, *, queue_order: int, stratum_rule_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "focus": {"text": "This source evidence must not be copied."},
        "review_queue": {
            "queue_order": queue_order,
            "stratum_rule_id": stratum_rule_id,
        },
        "selection": {"matched_rule_ids": [stratum_rule_id]},
    }
