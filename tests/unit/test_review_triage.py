"""Unit tests for conservative Luna draft triage."""

from __future__ import annotations

import json
from pathlib import Path

from reddit_search.review.triage import build_luna_triage


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _draft(candidate_id: str, scenario_id: str, topic_fit: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "annotation": {"topic_fit": topic_fit},
    }


def test_triage_sends_positive_invalid_and_missing_rows_to_humans(tmp_path: Path) -> None:
    cards = _write_jsonl(
        tmp_path / "cards.jsonl",
        [
            {"candidate_id": "c1", "scenario_id": "s1"},
            {"candidate_id": "c2", "scenario_id": "s2"},
            {"candidate_id": "c3", "scenario_id": "s3"},
            {"candidate_id": "c4", "scenario_id": "s4"},
        ],
    )
    drafts = _write_jsonl(
        tmp_path / "drafts.jsonl",
        [
            _draft("c1", "s1", "relevant"),
            _draft("c2", "s2", "not_relevant"),
            _draft("c3", "s3", "not_relevant"),
        ],
    )
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "rows": [
                    {"candidate_id": "c1", "scenario_id": "s1", "validation_status": "valid"},
                    {"candidate_id": "c2", "scenario_id": "s2", "validation_status": "valid"},
                    {"candidate_id": "c3", "scenario_id": "s3", "validation_status": "invalid"},
                    {"candidate_id": "<file>", "validation_status": "invalid"},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_luna_triage(
        cards,
        drafts,
        validation,
        tmp_path / "triage",
        negative_audit_rate=1.0,
        seed=7,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "triage" / "luna_triage.jsonl").read_text().splitlines()
    ]

    assert report["automatic_labeling"] is False
    assert report["counts"] == {"human_review": 3, "negative_audit": 1}
    assert {(row["candidate_id"], row["triage_status"]) for row in rows} == {
        ("c1", "human_review"),
        ("c2", "negative_audit"),
        ("c3", "human_review"),
        ("c4", "human_review"),
    }


def test_validated_negative_can_be_excluded_without_becoming_a_label(tmp_path: Path) -> None:
    cards = _write_jsonl(tmp_path / "cards.jsonl", [{"candidate_id": "c1", "scenario_id": "s1"}])
    drafts = _write_jsonl(tmp_path / "drafts.jsonl", [_draft("c1", "s1", "not_relevant")])
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "rows": [
                    {"candidate_id": "c1", "scenario_id": "s1", "validation_status": "valid"}
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_luna_triage(
        cards,
        drafts,
        validation,
        tmp_path / "triage",
        negative_audit_rate=0.0,
    )

    assert report["counts"] == {"not_selected": 1}
    assert report["automatic_labeling"] is False
