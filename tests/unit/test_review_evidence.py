"""Deterministic machine-draft evidence validation (spec T09)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reddit_search.review.evidence import validate_draft_rows

TITLE = "Need a tracker without bank sync"
BODY = "I want to log spending in two currencies, but every app demands a bank link."
REVISION = "a" * 64


def _card(candidate_id: str, *, scenario_id: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "source": {
            "message_fullname": "t3_abc",
            "source_revision_id": REVISION,
            "field": "selftext",
            "title": TITLE,
            "start": 0,
            "end": len(BODY),
            "text": BODY,
            "permalink": "/r/example/comments/abc/x/",
            "subreddit": "example",
            "created_utc": 1_780_000_000,
        },
    }
    if scenario_id is not None:
        row["scenario_id"] = scenario_id
    return row


def _evidence(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "message_fullname": "t3_abc",
        "source_revision_id": REVISION,
        "field": "selftext",
        "start": 0,
        "end": len(BODY),
        "quote": BODY,
        "supports": "topic_fit",
        "attribution": "focal_author",
    }
    base.update(overrides)
    return base


def _draft(candidate_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "model_provenance": {"model": "test/model", "label_kind": "machine_draft"},
        "annotation": {
            "review_status": "machine_draft",
            "topic_fit": "relevant",
            "need_clarity": "clear",
            "duplicate_of": None,
            "reviewer_note": None,
        },
        "evidence": [_evidence()],
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _run(
    tmp_path: Path,
    cards: list[dict[str, object]],
    drafts: list[dict[str, object]],
) -> dict[str, object]:
    cards_path = tmp_path / "cards.jsonl"
    drafts_path = tmp_path / "drafts.jsonl"
    _write_jsonl(cards_path, cards)
    _write_jsonl(drafts_path, drafts)
    return validate_draft_rows(cards_path, drafts_path)


def test_valid_relevant_draft_with_matching_quote_passes(tmp_path: Path) -> None:
    report = _run(tmp_path, [_card("c1")], [_draft("c1")])
    assert report["summary"] == {
        "candidate_count": 1,
        "task_count": 1,
        "draft_row_count": 1,
        "valid": 1,
        "invalid": 0,
    }
    assert report["rows"][0]["validation_status"] == "valid"


def test_fabricated_quote_fails(tmp_path: Path) -> None:
    draft = _draft("c1", evidence=[_evidence(quote=BODY + " Fabricated tail.")])
    report = _run(tmp_path, [_card("c1")], [draft])
    assert report["rows"][0]["validation_status"] == "invalid"
    assert any("quote does not match" in f for f in report["rows"][0]["failures"])


def test_offset_beyond_field_length_fails(tmp_path: Path) -> None:
    draft = _draft("c1", evidence=[_evidence(end=len(BODY) + 5)])
    report = _run(tmp_path, [_card("c1")], [draft])
    assert any("exceeds field length" in f for f in report["rows"][0]["failures"])


def test_mismatched_revision_fails(tmp_path: Path) -> None:
    draft = _draft("c1", evidence=[_evidence(source_revision_id="b" * 64)])
    report = _run(tmp_path, [_card("c1")], [draft])
    assert any("source_revision_id" in f for f in report["rows"][0]["failures"])


def test_unknown_and_duplicate_candidate_ids_fail(tmp_path: Path) -> None:
    report = _run(tmp_path, [_card("c1")], [_draft("c1"), _draft("c1"), _draft("ghost")])
    by_id = {row["candidate_id"]: row["failures"] for row in report["rows"]}
    assert any("duplicate candidate_id" in f for f in by_id["c1"])
    assert any("unknown candidate_id" in f for f in by_id["ghost"])


def test_missing_row_fails_when_complete_and_passes_with_allow_incomplete(
    tmp_path: Path,
) -> None:
    cards_path = tmp_path / "cards.jsonl"
    _write_jsonl(cards_path, [_card("c1"), _card("c2")])
    drafts_path = tmp_path / "drafts.jsonl"
    _write_jsonl(drafts_path, [_draft("c1")])
    strict = validate_draft_rows(cards_path, drafts_path)
    assert strict["rows"][1]["failures"] == ["missing draft row for candidate"]
    lenient = validate_draft_rows(cards_path, drafts_path, require_complete=False)
    assert len(lenient["rows"]) == 1
    assert lenient["summary"]["invalid"] == 0
    assert lenient["summary"]["candidate_count"] == 2


def test_uncertain_topic_fit_needs_no_evidence(tmp_path: Path) -> None:
    annotation = {
        "review_status": "machine_draft",
        "topic_fit": "uncertain",
        "need_clarity": "unclear",
    }
    report = _run(tmp_path, [_card("c1")], [_draft("c1", annotation=annotation, evidence=[])])
    assert report["rows"][0]["validation_status"] == "valid"


def test_relevant_topic_fit_requires_evidence(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [_card("c1")],
        [_draft("c1", evidence=[])],
    )
    assert any("requires at least one evidence entry" in f for f in report["rows"][0]["failures"])


def test_draft_identity_includes_scenario_id(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [_card("c1", scenario_id="scenario-a"), _card("c1", scenario_id="scenario-b")],
        [_draft("c1", scenario_id="scenario-a"), _draft("c1", scenario_id="scenario-b")],
    )

    assert report["summary"]["candidate_count"] == 1
    assert report["summary"]["task_count"] == 2
    assert {(row["candidate_id"], row["scenario_id"]) for row in report["rows"]} == {
        ("c1", "scenario-a"),
        ("c1", "scenario-b"),
    }


def test_title_evidence_validates_against_title_field(tmp_path: Path) -> None:
    draft = _draft(
        "c1",
        evidence=[
            _evidence(
                field="title",
                start=0,
                end=len(TITLE),
                quote=TITLE,
            )
        ],
    )
    report = _run(tmp_path, [_card("c1")], [draft])
    assert report["rows"][0]["validation_status"] == "valid"


def test_unsupported_field_fails(tmp_path: Path) -> None:
    draft = _draft("c1", evidence=[_evidence(field="selftext_html")])
    report = _run(tmp_path, [_card("c1")], [draft])
    assert any("unsupported field" in f for f in report["rows"][0]["failures"])


def test_author_requirement_with_wrong_attribution_fails(tmp_path: Path) -> None:
    draft = _draft(
        "c1",
        evidence=[_evidence(supports="author_requirement", attribution="quoted_other")],
    )
    report = _run(tmp_path, [_card("c1")], [draft])
    assert any("attributed to the focal author" in f for f in report["rows"][0]["failures"])


def test_report_is_source_free_and_hashed(tmp_path: Path) -> None:
    report = _run(tmp_path, [_card("c1")], [_draft("c1")])
    assert "text" not in json.dumps(report)
    assert "quote" not in json.dumps(report)
    assert (
        report["cards_sha256"]
        == hashlib.sha256(json.dumps(_card("c1"), sort_keys=True).encode() + b"\n").hexdigest()
    )


def test_line_level_parse_failures_are_reported(tmp_path: Path) -> None:
    cards_path = tmp_path / "cards.jsonl"
    _write_jsonl(cards_path, [_card("c1")])
    drafts_path = tmp_path / "drafts.jsonl"
    drafts_path.write_text("not json\n", encoding="utf-8")
    report = validate_draft_rows(cards_path, drafts_path)
    file_row = next(row for row in report["rows"] if row["candidate_id"] == "<file>")
    assert file_row["validation_status"] == "invalid"


def test_malformed_draft_label_value_fails(tmp_path: Path) -> None:
    annotation = {
        "review_status": "machine_draft",
        "topic_fit": "maybe",
        "need_clarity": "clear",
    }
    report = _run(tmp_path, [_card("c1")], [_draft("c1", annotation=annotation, evidence=[])])
    assert any("topic_fit must be one of" in f for f in report["rows"][0]["failures"])


def test_offset_into_middle_of_body_matches(tmp_path: Path) -> None:
    needle = "two currencies"
    start = BODY.index(needle)
    draft = _draft(
        "c1",
        evidence=[_evidence(start=start, end=start + len(needle), quote=needle)],
    )
    report = _run(tmp_path, [_card("c1")], [draft])
    assert report["rows"][0]["validation_status"] == "valid"
def test_nonzero_start_quote_matches_canonical_full_field(tmp_path: Path) -> None:
    prefix = "prefix: "
    quote = "two currencies"
    full_text = prefix + quote + " are hard to track."
    card = _card("c1")
    card["source"] = {
        **card["source"],
        "text": full_text,
        "start": len(prefix),
        "end": len(prefix) + len(quote),
    }
    evidence = _evidence(start=len(prefix), end=len(prefix) + len(quote), quote=quote)
    report = _run(tmp_path, [card], [_draft("c1", evidence=[evidence])])
    assert report["rows"][0]["validation_status"] == "valid"


def test_altered_nonzero_start_quote_is_rejected(tmp_path: Path) -> None:
    prefix = "prefix: "
    quote = "two currencies"
    full_text = prefix + quote + " are hard to track."
    card = _card("c1")
    card["source"] = {**card["source"], "text": full_text}
    evidence = _evidence(start=len(prefix), end=len(prefix) + len(quote), quote="altered quote")
    report = _run(tmp_path, [card], [_draft("c1", evidence=[evidence])])
    assert any("quote does not match" in f for f in report["rows"][0]["failures"])


def test_nonzero_start_out_of_range_is_rejected(tmp_path: Path) -> None:
    card = _card("c1")
    full_text = "prefix: " + BODY
    card["source"] = {**card["source"], "text": full_text}
    evidence = _evidence(start=len(full_text) + 1, end=len(full_text) + 5, quote="x")
    report = _run(tmp_path, [card], [_draft("c1", evidence=[evidence])])
    assert any("exceeds field length" in f for f in report["rows"][0]["failures"])



def test_title_evidence_on_comment_card_without_title_fails(tmp_path: Path) -> None:
    card = _card("c1")
    comment_source = dict(card["source"])  # type: ignore[arg-type]
    comment_source["message_fullname"] = "t1_abc"
    comment_source["title"] = None
    card["source"] = comment_source
    draft = _draft(
        "c1",
        evidence=[_evidence(message_fullname="t1_abc", field="title", start=0, end=2, quote="Ne")],
    )
    report = _run(tmp_path, [card], [draft])
    row = report["rows"][0]
    assert row["validation_status"] == "invalid"
    assert any("cited field is unavailable" in f for f in row["failures"])


def test_schema_failure_report_does_not_echo_draft_text(tmp_path: Path) -> None:
    marker = "SYNTHETIC_PRIVATE_SOURCE_MARKER"
    draft = _draft(
        "c1",
        annotation={
            "review_status": "machine_draft",
            "topic_fit": "maybe",
            "need_clarity": "clear",
            "reviewer_note": marker,
        },
    )
    report = _run(tmp_path, [_card("c1")], [draft])
    file_row = next(row for row in report["rows"] if row["candidate_id"] == "<file>")
    assert file_row["validation_status"] == "invalid"
    assert marker not in json.dumps(report)


def test_unknown_field_failure_does_not_echo_field_text(tmp_path: Path) -> None:
    marker = "INJECTED_FIELD_NAME_WITH_PRIVATE_TEXT"
    draft = _draft("c1", evidence=[_evidence(field=marker)])
    report = _run(tmp_path, [_card("c1")], [draft])
    assert marker not in json.dumps(report)
    assert any("unsupported field" in f for f in report["rows"][0]["failures"])


def test_unsupported_structured_judgment_is_rejected(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [_card("c1")],
        [
            _draft(
                "c1",
                annotation={
                    "review_status": "machine_draft",
                    "topic_fit": "uncertain",
                    "need_clarity": "uncertain",
                    "speaker_intent": "invented",
                    "product_fit": "not_evaluated",
                    "resolution_in_available_context": "unknown",
                },
            )
        ],
    )
    assert report["rows"][0]["validation_status"] == "invalid"
    assert any("speaker_intent must be one" in item for item in report["rows"][0]["failures"])
