from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import zstandard

from reddit_search.contracts import ClaimStatus, ProductClaim, ProductProfile
from reddit_search.engagement.drafting import (
    Qualification,
    ReplyDraft,
    _append_record,
    append_human_edit,
)
from reddit_search.engagement.handoff import export_handoff, summarize_pilot
from reddit_search.engagement.review import (
    EngagementReview,
    FalsePositiveAdjudication,
    ManualCheck,
    append_false_positive_adjudication,
    append_manual_check,
    append_review,
    review_sha256,
)
from reddit_search.ingest.state import file_sha256


def _profile(app_id: str = "app") -> ProductProfile:
    return ProductProfile(
        app_id=app_id,
        profile_version=1,
        verified_at="2026-09-01T00:00:00Z",
        verification_owner="fictional test owner",
        verification_status=ClaimStatus.VERIFIED,
        synthetic=False,
        capabilities=[
            ProductClaim(
                claim_id="claim_workflow",
                status=ClaimStatus.VERIFIED,
                evidence_ref="fictional://test/claim_workflow",
                statement="Fictional test claim: organizes workflow notes.",
            )
        ],
    )


def _review(
    *,
    app_id: str = "app",
    thread: str = "t3_thread",
    target: str = "t1_target",
    verdict: str = "provisional_candidate",
    reviewed_at: datetime | None = None,
    bundle_sha256: str = "0" * 64,
) -> EngagementReview:
    return EngagementReview(
        app_id=app_id,
        thread_fullname=thread,
        reply_target_fullname=target,
        bundle_sha256=bundle_sha256,
        source_message_evidence_ids=[target],
        product_fit=("compatible" if verdict == "provisional_candidate" else "not_evaluated"),
        supporting_claim_ids=(["claim_workflow"] if verdict == "provisional_candidate" else []),
        reviewer="reviewer",
        reviewed_at=reviewed_at or datetime(2000, 9, 1, tzinfo=UTC),
        verdict=verdict,
        reason=f"Recorded reviewer decision: {verdict}.",
        author_need=("first_person_relevant" if verdict == "provisional_candidate" else "unclear"),
        answered_in_available_context=("no" if verdict == "provisional_candidate" else "unknown"),
        context_complete=verdict == "provisional_candidate",
        reason_codes=(
            ["author_has_current_need"]
            if verdict == "provisional_candidate"
            else ["context_incomplete"]
        ),
    )


def _artifacts(
    tmp_path: Path, *, incomplete: bool = False, reply_target: str = "t1_target"
) -> dict[str, Path | EngagementReview | str]:
    repo = tmp_path / "repo"
    reports = repo / "reports"
    reports.mkdir(parents=True)
    (repo / ".gitignore").write_text("reports/\n", encoding="utf-8")
    root = {
        "fullname": "t3_thread",
        "thread_fullname": "t3_thread",
        "parent_fullname": None,
        "raw_title": "A workflow question",
        "raw_body": "I am looking for a way to organize this work.",
        "permalink": "/r/demo/comments/thread/a_workflow_question/",
        "provenance": [{"source_id": "archive", "line_number": 1}],
        "archived_status": {},
    }
    target = {
        "fullname": "t1_target",
        "thread_fullname": "t3_thread",
        "parent_fullname": "t3_thread",
        "raw_title": None,
        "raw_body": "Could someone suggest a simple approach?",
        "permalink": "/r/demo/comments/thread/a_workflow_question/target/",
        "provenance": [{"source_id": "archive", "line_number": 2}],
        "archived_status": {},
    }
    alternate_target = {
        "fullname": "t1_alternate",
        "thread_fullname": "t3_thread",
        "parent_fullname": "t3_thread",
        "raw_title": None,
        "raw_body": "I also have a workflow question for the team.",
        "permalink": "/r/demo/comments/thread/a_workflow_question/alternate/",
        "provenance": [{"source_id": "archive", "line_number": 3}],
        "archived_status": {},
    }
    bundle = reports / "thread.jsonl.zst"
    with bundle.open("wb") as output:
        with zstandard.ZstdCompressor().stream_writer(output) as compressed:
            compressed.write(
                "".join(
                    json.dumps(message) + "\n" for message in (root, target, alternate_target)
                ).encode()
            )
    registry = reports / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "archive",
                        "source_kind": "submission",
                        "declared_month": "2026-08",
                        "source_role": "discovery",
                        "usage_scope": "authorized_local_research",
                        "status": "complete",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    context_manifest = reports / "context_manifest.json"
    context_manifest.write_text(
        json.dumps(
            {
                "scan_complete": True,
                "output_sha256": file_sha256(bundle),
                "inputs": {"registry_sha256": file_sha256(registry)},
                "sources": [
                    {
                        "source_id": "archive",
                        "source_kind": "submission",
                        "declared_month": "2026-08",
                        "complete": True,
                    }
                ],
                "threads": [
                    {
                        "thread_fullname": "t3_thread",
                        "coverage_status": (
                            "needs_context" if incomplete else "complete_for_scanned_sources"
                        ),
                        "thread_context_complete_for_sources": not incomplete,
                        "root_missing": False,
                        "missing_parents": [],
                        "reply_targets": ["t1_target"],
                        "missing_reply_targets": ["t1_target"] if incomplete else [],
                        "coverage_reasons": (
                            ["missing_reply_target: t1_target"] if incomplete else []
                        ),
                        "coverage_reason_counts": (
                            {"missing_reply_target": 1} if incomplete else {}
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    review_path = reports / "decisions.jsonl"
    review = _review(target=reply_target, bundle_sha256=file_sha256(bundle))
    append_review(
        review_path,
        review,
        profile=_profile(),
        expected_bundle_sha256=file_sha256(bundle),
    )
    history = reports / "reply_drafts.jsonl"
    qualification = Qualification(
        decision="provisional_candidate",
        reply_target_fullname=reply_target,
        cited_message_ids=[reply_target],
        claim_ids=["claim_workflow"],
        explanation="The exact target and stated need are available in captured context.",
    )
    _append_record(
        history,
        {
            "record_type": "machine",
            "created_at": datetime(2000, 9, 1, 0, 1, tzinfo=UTC),
            "app_id": "app",
            "thread_fullname": "t3_thread",
            "reply_target_fullname": reply_target,
            "outcome": "draft",
            "context_bundle_sha256": file_sha256(bundle),
            "context_manifest_sha256": file_sha256(context_manifest),
            "review_sha256": hashlib.sha256(review.model_dump_json().encode()).hexdigest(),
            "review_journal_sha256": file_sha256(review_path),
            "product_profile_sha256": "1" * 64,
            "product_profile_version": 1,
            "qualification_prompt_version": "qualification-v1",
            "qualification_prompt_sha256": "2" * 64,
            "draft_prompt_version": "draft-v1",
            "draft_prompt_sha256": "3" * 64,
            "model_version": "local-model-v1",
            "model_sha256": "4" * 64,
            "qualification": qualification,
            "used_message_ids": [reply_target],
            "used_claim_ids": ["claim_workflow"],
            "claim_statements": {"claim_workflow": "The app helps organize workflow notes."},
            "operator_affiliation": "I work with this product team.",
            "text": (
                "Thanks for explaining your workflow. The app helps organize workflow notes. "
                "I work with this product team. What have you tried so far?"
            ),
            "abstention_reasons": [],
            "human_editor": None,
            "human_edit_valid": None,
            "human_edit_issues": [],
        },
    )
    return {
        "repo": repo,
        "reports": reports,
        "bundle": bundle,
        "manifest": context_manifest,
        "registry": registry,
        "review_path": review_path,
        "reply_target": reply_target,
        "history": history,
        "review": review,
    }


def _export(artifacts: dict[str, Path | EngagementReview | str], path: Path) -> dict[str, object]:
    return export_handoff(
        app_id="app",
        thread_fullname="t3_thread",
        reply_target_fullname=artifacts["reply_target"],
        context_bundle_path=artifacts["bundle"],
        context_manifest_path=artifacts["manifest"],
        expected_context_manifest_sha256=file_sha256(artifacts["manifest"]),
        review_journal_path=artifacts["review_path"],
        draft_history_path=artifacts["history"],
        registry_path=artifacts["registry"],
        output_path=path,
        repo_root=artifacts["repo"],
    )


def test_export_is_hash_bound_private_and_live_unverified(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    output = artifacts["reports"] / "handoff.json"
    result = _export(artifacts, output)
    packet = json.loads(output.read_text(encoding="utf-8"))
    assert result["output_sha256"] == file_sha256(output)
    assert (
        packet["source_message_evidence"][0]["text"] == "Could someone suggest a simple approach?"
    )
    assert packet["provisional_reply"].startswith("Thanks for explaining")
    assert packet["source_message_evidence"][0]["message_id"] == "t1_target"
    assert packet["local_permissions"]["external_transmission_permitted"] is False
    assert packet["local_permissions"]["posting_permitted"] is False
    assert packet["manual_check_form"]["live_status"] == "live_unverified"
    assert packet["manual_check_form"]["reviewer"] == ""
    assert packet["manual_check_form"]["final_human_decision"] is None
    assert "replyability inference" in packet["manual_check_form"]["instruction"]


def test_export_accepts_captured_alternate_reply_target(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, reply_target="t1_alternate")
    packet_path = artifacts["reports"] / "alternate-handoff.json"

    _export(artifacts, packet_path)

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["identity"]["reply_target_fullname"] == "t1_alternate"
    assert packet["source_message_evidence"][0]["message_id"] == "t1_alternate"
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert "t1_alternate" not in manifest["threads"][0]["reply_targets"]


@pytest.mark.parametrize(
    "edited_text",
    [
        "Thanks. The app helps organize workflow notes. What have you tried so far?",
        "Thanks for explaining your workflow. I work with this product team. "
        "What have you tried so far?",
    ],
)
def test_export_rejects_invalid_human_edit(tmp_path: Path, edited_text: str) -> None:
    artifacts = _artifacts(tmp_path)
    draft = ReplyDraft.model_validate_json(artifacts["history"].read_text().splitlines()[0])
    edited = append_human_edit(
        artifacts["history"],
        prior_record_sha256=draft.record_sha256,
        edited_text=edited_text,
        editor="human reviewer",
    )
    assert edited.human_edit_valid is False

    with pytest.raises(ValueError, match="draft fails deterministic safety validation"):
        _export(artifacts, artifacts["reports"] / "unsafe-handoff.json")


def test_export_rejects_missing_draft(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    artifacts["history"].write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="draft history has no record"):
        _export(artifacts, artifacts["reports"] / "missing-draft.json")


def test_export_rejects_changed_context_manifest_hash(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    artifacts["manifest"].write_text(artifacts["manifest"].read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="context manifest hash"):
        _export(artifacts, artifacts["reports"] / "rejected.json")


def test_export_rejects_incomplete_thread_coverage(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, incomplete=True)
    with pytest.raises(ValueError, match="context is incomplete"):
        _export(artifacts, artifacts["reports"] / "incomplete.json")


def test_pilot_summary_uses_journal_decisions_edits_and_abstentions(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    history = artifacts["history"]
    original = json.loads(history.read_text().splitlines()[0])

    draft = ReplyDraft.model_validate(original)
    append_human_edit(
        history,
        prior_record_sha256=draft.record_sha256,
        edited_text=(
            "Thanks for explaining your workflow. The app helps organize workflow notes. "
            "I work with this product team. What have you tried already?"
        ),
        editor="human reviewer",
    )
    # An authentic appended abstention is retained as a separate identity/version.
    abstention = dict(original)
    abstention.update(
        {
            "app_id": "app",
            "thread_fullname": "t3_other",
            "reply_target_fullname": "t1_other",
            "outcome": "abstain",
            "context_bundle_sha256": "5" * 64,
            "qualification": None,
            "used_message_ids": [],
            "used_claim_ids": [],
            "claim_statements": {},
            "text": "",
            "abstention_reasons": ["insufficient context", "uncertain author intent"],
            "human_editor": None,
            "human_edit_valid": None,
            "human_edit_issues": [],
        }
    )
    _append_record(history, abstention)

    review_path = artifacts["reports"] / "pilot-decisions.jsonl"
    append_review(
        review_path,
        _review(verdict="provisional_candidate"),
        profile=_profile(),
        expected_bundle_sha256="0" * 64,
    )
    append_review(
        review_path,
        _review(
            thread="t3_rejected",
            target="t1_rejected",
            verdict="reject",
            reviewed_at=datetime(2026, 9, 2, tzinfo=UTC),
        ),
        expected_bundle_sha256="0" * 64,
    )
    append_review(
        review_path,
        _review(
            thread="t3_uncertain",
            target="t1_uncertain",
            verdict="needs_context",
            reviewed_at=datetime(2026, 9, 3, tzinfo=UTC),
        ),
        expected_bundle_sha256="0" * 64,
    )
    check_path = artifacts["reports"] / "manual-checks.jsonl"
    check = ManualCheck(
        app_id="app",
        thread_fullname="t3_thread",
        reply_target_fullname="t1_target",
        bundle_sha256="0" * 64,
        reviewer="human reviewer",
        observed_at=datetime(2026, 9, 4, tzinfo=UTC),
        live_status="human_checked_replyable",
        permalink="https://www.reddit.com/r/demo/comments/thread/a_workflow_question/target/",
        observation="Target and current thread reviewed.",
        rule_evidence=["Current subreddit rules reviewed manually."],
        final_human_decision="do_not_post",
    )
    append_manual_check(check_path, check, expected_bundle_sha256="0" * 64)

    summary = summarize_pilot(
        review_journal_path=review_path,
        draft_history_path=history,
        manual_check_journal_path=check_path,
    )
    assert summary["reviewer_decisions_latest_per_identity"] == {
        "identity_count": 3,
        "provisional_candidate": 1,
        "reject": 1,
        "needs_context": 1,
    }
    assert summary["human_edits"] == {"record_count": 1, "valid_count": 1, "invalid_count": 0}
    assert summary["latest_abstention_reasons"] == {
        "insufficient context": 1,
        "uncertain author intent": 1,
    }
    assert summary["manual_final_human_decisions"] == {"do_not_post": 1}
    assert summary["false_positive_adjudications"]["criterion_version"] == "engagement-fp-v1"
    assert summary["false_positive_adjudications"]["criterion_description"]
    assert summary["false_positive_adjudications"]["status"] == "not_recorded"
    assert summary["false_positive_adjudications"]["false_positive_rate"] is None
    assert summary["false_positive_adjudications"]["rate_denominator"] is None


def test_pilot_summary_uses_explicit_adjudication_denominator_and_labels(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    review_path = artifacts["reports"] / "pilot-decisions.jsonl"
    first = _review(
        verdict="reject",
        thread="t3_rejected",
        target="t1_rejected",
        reviewed_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    second = _review(
        verdict="needs_context",
        thread="t3_uncertain",
        target="t1_uncertain",
        reviewed_at=datetime(2026, 9, 3, tzinfo=UTC),
    )
    append_review(review_path, first, expected_bundle_sha256="0" * 64)
    append_review(review_path, second, expected_bundle_sha256="0" * 64)
    adjudication_path = artifacts["reports"] / "adjudications.jsonl"
    for reviewed, label in ((first, "false_positive"), (second, "true_positive")):
        append_false_positive_adjudication(
            adjudication_path,
            FalsePositiveAdjudication(
                app_id=reviewed.app_id,
                thread_fullname=reviewed.thread_fullname,
                reply_target_fullname=reviewed.reply_target_fullname,
                bundle_sha256=reviewed.bundle_sha256,
                review_sha256=review_sha256(reviewed),
                review_verdict=reviewed.verdict,
                label=label,
                criterion_version="engagement-fp-v1",
                adjudicator="adjudicator",
                adjudicated_at=datetime(2026, 9, 5, tzinfo=UTC),
            ),
            expected_bundle_sha256="0" * 64,
        )
    summary = summarize_pilot(
        review_journal_path=review_path,
        draft_history_path=artifacts["history"],
        manual_check_journal_path=artifacts["reports"] / "missing-manual.jsonl",
        false_positive_adjudication_journal_path=adjudication_path,
    )
    metrics = summary["false_positive_adjudications"]
    assert metrics["criterion_versions"] == ["engagement-fp-v1"]
    assert metrics["adjudicated_identity_count"] == 2
    assert metrics["unadjudicated_reviewed_identity_count"] == 0
    assert metrics["label_counts"] == {
        "true_positive": 1,
        "false_positive": 1,
        "unclear": 0,
    }
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["rate_denominator"] == 2



def test_pilot_summary_rejects_stale_adjudication_and_counts_current_review(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    review_path = artifacts["reports"] / "revision-decisions.jsonl"
    older = _review(
        verdict="reject",
        reviewed_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    current = older.model_copy(
        update={
            "reason": "Revised decision",
            "reviewed_at": datetime(2026, 9, 3, tzinfo=UTC),
        }
    )
    append_review(review_path, older, expected_bundle_sha256="0" * 64)
    append_review(review_path, current, expected_bundle_sha256="0" * 64)

    def adjudication(reviewed: EngagementReview) -> FalsePositiveAdjudication:
        return FalsePositiveAdjudication(
            app_id=reviewed.app_id,
            thread_fullname=reviewed.thread_fullname,
            reply_target_fullname=reviewed.reply_target_fullname,
            bundle_sha256=reviewed.bundle_sha256,
            review_sha256=review_sha256(reviewed),
            review_verdict=reviewed.verdict,
            label="false_positive",
            criterion_version="engagement-fp-v1",
            adjudicator="adjudicator",
            adjudicated_at=datetime(2026, 9, 4, tzinfo=UTC),
        )

    stale_path = artifacts["reports"] / "stale-adjudications.jsonl"
    append_false_positive_adjudication(
        stale_path, adjudication(older), expected_bundle_sha256="0" * 64
    )
    with pytest.raises(
        ValueError, match="stale adjudication references a superseded review"
    ):
        summarize_pilot(
            review_journal_path=review_path,
            draft_history_path=artifacts["history"],
            manual_check_journal_path=artifacts["reports"] / "missing-checks.jsonl",
            false_positive_adjudication_journal_path=stale_path,
        )

    current_path = artifacts["reports"] / "current-adjudications.jsonl"
    current_adjudication = adjudication(current)
    append_false_positive_adjudication(
        current_path, current_adjudication, expected_bundle_sha256="0" * 64
    )
    summary = summarize_pilot(
        review_journal_path=review_path,
        draft_history_path=artifacts["history"],
        manual_check_journal_path=artifacts["reports"] / "missing-checks.jsonl",
        false_positive_adjudication_journal_path=current_path,
    )
    assert summary["false_positive_adjudications"]["adjudicated_identity_count"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        append_false_positive_adjudication(
            current_path, current_adjudication, expected_bundle_sha256="0" * 64
        )


def test_pilot_summary_reports_zero_false_positives_with_denominator(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    review_path = artifacts["reports"] / "pilot-decisions.jsonl"
    reviewed = _review(
        verdict="reject",
        thread="t3_rejected",
        target="t1_rejected",
        reviewed_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    append_review(review_path, reviewed, expected_bundle_sha256="0" * 64)
    adjudication_path = artifacts["reports"] / "adjudications.jsonl"
    append_false_positive_adjudication(
        adjudication_path,
        FalsePositiveAdjudication(
            app_id=reviewed.app_id,
            thread_fullname=reviewed.thread_fullname,
            reply_target_fullname=reviewed.reply_target_fullname,
            bundle_sha256=reviewed.bundle_sha256,
            review_sha256=review_sha256(reviewed),
            review_verdict=reviewed.verdict,
            label="true_positive",
            criterion_version="engagement-fp-v1",
            adjudicator="adjudicator",
            adjudicated_at=datetime(2026, 9, 5, tzinfo=UTC),
        ),
        expected_bundle_sha256="0" * 64,
    )
    summary = summarize_pilot(
        review_journal_path=review_path,
        draft_history_path=artifacts["history"],
        manual_check_journal_path=artifacts["reports"] / "missing-manual.jsonl",
        false_positive_adjudication_journal_path=adjudication_path,
    )
    metrics = summary["false_positive_adjudications"]
    assert metrics["status"] == "recorded"
    assert metrics["false_positive_rate"] == 0
    assert metrics["rate_denominator"] == 1
