from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import zstandard

from reddit_search.contracts import ClaimStatus, ProductClaim, ProductProfile
from reddit_search.engagement.drafting import Qualification, _append_record
from reddit_search.engagement.review import (
    EngagementReview,
    append_review,
    read_false_positive_adjudications,
    read_reviews,
    review_sha256,
)
from reddit_search.ingest.state import file_sha256

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _profile() -> ProductProfile:
    return ProductProfile(
        app_id="app",
        profile_version=1,
        verified_at="2026-09-01T00:00:00Z",
        verification_owner="fictional acceptance owner",
        verification_status=ClaimStatus.VERIFIED,
        synthetic=False,
        capabilities=[
            ProductClaim(
                claim_id="claim_workflow",
                status=ClaimStatus.VERIFIED,
                evidence_ref="fictional://acceptance/claim_workflow",
                statement="The app helps organize workflow notes.",
            )
        ],
    )


def _review(
    *,
    bundle_sha256: str,
    thread: str = "t3_thread",
    target: str = "t1_target",
    verdict: str = "provisional_candidate",
    reviewed_at: datetime | None = None,
) -> EngagementReview:
    provisional = verdict == "provisional_candidate"
    return EngagementReview(
        app_id="app",
        thread_fullname=thread,
        reply_target_fullname=target,
        bundle_sha256=bundle_sha256,
        source_message_evidence_ids=[target] if provisional else [],
        product_fit="compatible" if provisional else "not_evaluated",
        supporting_claim_ids=["claim_workflow"] if provisional else [],
        reviewer="acceptance-reviewer",
        reviewed_at=reviewed_at or datetime(2000, 9, 1, tzinfo=UTC),
        verdict=verdict,
        reason=f"Synthetic acceptance decision: {verdict}.",
        author_need="first_person_relevant" if provisional else "unclear",
        answered_in_available_context="no" if provisional else "unknown",
        context_complete=provisional,
        reason_codes=["author_has_current_need"] if provisional else ["context_incomplete"],
    )


def _fixture(tmp_path: Path, *, include_rejection: bool = False) -> dict[str, Path | str]:
    repo = tmp_path / "repo"
    reports = repo / "reports"
    reports.mkdir(parents=True)
    (repo / ".gitignore").write_text("reports/\n", encoding="utf-8")

    rows = [
        {
            "fullname": "t3_thread",
            "thread_fullname": "t3_thread",
            "parent_fullname": None,
            "raw_title": "Synthetic workflow question",
            "raw_body": "Looking for a simple way to organize workflow notes.",
            "permalink": "/r/demo/comments/thread/workflow/",
            "provenance": [{"source_id": "acceptance-source", "line_number": 1}],
            "archived_status": {},
        },
        {
            "fullname": "t1_target",
            "thread_fullname": "t3_thread",
            "parent_fullname": "t3_thread",
            "raw_title": None,
            "raw_body": "Could someone suggest a workflow notes tool?",
            "permalink": "/r/demo/comments/thread/workflow/target/",
            "provenance": [{"source_id": "acceptance-source", "line_number": 2}],
            "archived_status": {},
        },
    ]
    bundle = reports / "context.jsonl.zst"
    bundle.write_bytes(
        zstandard.ZstdCompressor().compress(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
        )
    )
    registry = reports / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "acceptance-source",
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
    manifest = reports / "context-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scan_complete": True,
                "output_sha256": file_sha256(bundle),
                "inputs": {"registry_sha256": file_sha256(registry)},
                "sources": [
                    {
                        "source_id": "acceptance-source",
                        "source_kind": "submission",
                        "declared_month": "2026-08",
                        "complete": True,
                    }
                ],
                "threads": [
                    {
                        "thread_fullname": "t3_thread",
                        "coverage_status": "complete_for_scanned_sources",
                        "thread_context_complete_for_sources": True,
                        "root_missing": False,
                        "missing_parents": [],
                        "missing_reply_targets": [],
                        "coverage_reasons": [],
                        "coverage_reason_counts": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle_hash = file_sha256(bundle)
    review_path = reports / "reviews.jsonl"
    review = _review(bundle_sha256=bundle_hash)
    append_review(
        review_path,
        review,
        profile=_profile(),
        expected_bundle_sha256=bundle_hash,
    )
    if include_rejection:
        append_review(
            review_path,
            _review(
                bundle_sha256=bundle_hash,
                thread="t3_rejected",
                target="t1_rejected",
                verdict="reject",
                reviewed_at=datetime(2000, 9, 2, tzinfo=UTC),
            ),
            expected_bundle_sha256=bundle_hash,
        )

    history = reports / "draft-history.jsonl"
    qualification = Qualification(
        decision="provisional_candidate",
        reply_target_fullname="t1_target",
        cited_message_ids=["t1_target"],
        claim_ids=["claim_workflow"],
        explanation="Synthetic acceptance fixture has the target evidence.",
    )
    base_record = {
        "record_type": "machine",
        "created_at": datetime(2000, 9, 1, 0, 1, tzinfo=UTC),
        "app_id": "app",
        "thread_fullname": "t3_thread",
        "reply_target_fullname": "t1_target",
        "outcome": "draft",
        "context_bundle_sha256": bundle_hash,
        "context_manifest_sha256": file_sha256(manifest),
        "review_sha256": hashlib.sha256(review.model_dump_json().encode("utf-8")).hexdigest(),
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
        "used_message_ids": ["t1_target"],
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
    }
    _append_record(history, base_record)
    return {
        "repo": repo,
        "bundle": bundle,
        "manifest": manifest,
        "registry": registry,
        "review": review_path,
        "history": history,
        "manifest_sha256": file_sha256(manifest),
    }


def _append_abstention(history: Path) -> None:
    record = json.loads(history.read_text(encoding="utf-8").splitlines()[0])
    for field in ("history_sequence", "previous_record_sha256", "record_sha256"):
        record.pop(field)
    record.update(
        {
            "created_at": datetime(2000, 9, 2, tzinfo=UTC).isoformat(),
            "thread_fullname": "t3_abstained",
            "reply_target_fullname": "t1_abstained",
            "outcome": "abstain",
            "qualification": None,
            "used_message_ids": [],
            "used_claim_ids": [],
            "claim_statements": {},
            "text": "",
            "abstention_reasons": ["insufficient context"],
            "human_editor": None,
            "human_edit_valid": None,
            "human_edit_issues": [],
        }
    )
    _append_record(history, record)


def _invoke(command: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reddit_search", "engagement", command, *args],
        capture_output=True,
        check=False,
        text=True,
        cwd=_REPO_ROOT,
    )


def _report_output(tmp_path: Path, name: str) -> Path:
    return _REPO_ROOT / "reports" / "engagement" / f"{name}-{tmp_path.name}.json"


def _edit_args(fixture: dict[str, Path | str], text_file: Path) -> list[str]:
    first = json.loads(Path(fixture["history"]).read_text(encoding="utf-8").splitlines()[0])
    return [
        "--history",
        str(fixture["history"]),
        "--prior-record-sha256",
        first["record_sha256"],
        "--text-file",
        str(text_file),
        "--editor",
        "synthetic acceptance reviewer",
    ]


def _handoff_args(fixture: dict[str, Path | str], output: Path) -> list[str]:
    return [
        "--app-id",
        "app",
        "--thread-fullname",
        "t3_thread",
        "--reply-target-fullname",
        "t1_target",
        "--context-bundle",
        str(fixture["bundle"]),
        "--context-manifest",
        str(fixture["manifest"]),
        "--expected-context-manifest-sha256",
        str(fixture["manifest_sha256"]),
        "--review-journal",
        str(fixture["review"]),
        "--draft-history",
        str(fixture["history"]),
        "--registry",
        str(fixture["registry"]),
        "--output",
        str(output),
    ]


def test_edit_cli_and_handoff_export_private_packet_without_stdout_text(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    edited_text = (
        "Thanks for explaining your workflow. The app helps organize workflow notes. "
        "I work with this product team. What have you tried already?"
    )
    text_file = tmp_path / "edited-text.txt"
    text_file.write_text(edited_text, encoding="utf-8")
    edit_result = _invoke("edit", _edit_args(fixture, text_file))
    assert edit_result.returncode == 0, edit_result.stderr
    assert edited_text not in edit_result.stdout
    edit_report = json.loads(edit_result.stdout)
    assert edit_report["status"] == "valid"
    assert edit_report["human_edit_findings"] == []

    output = _report_output(tmp_path, "handoff")
    try:
        handoff_result = _invoke("handoff", _handoff_args(fixture, output))
        assert handoff_result.returncode == 0, handoff_result.stderr
        assert edited_text not in handoff_result.stdout
        assert json.loads(handoff_result.stdout)["exported"] is True
        packet = json.loads(output.read_text(encoding="utf-8"))
        assert packet["kind"] == "offline_engagement_handoff"
        assert packet["provisional_reply"] == edited_text
        assert packet["draft_validation"]["human_edit_valid"] is True
    finally:
        output.unlink(missing_ok=True)
        try:
            output.parent.rmdir()
        except OSError:
            pass


def test_invalid_edit_is_audited_and_blocks_handoff(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    edited_text = "The app helps organize workflow notes."
    text_file = tmp_path / "unsafe-edit.txt"
    text_file.write_text(edited_text, encoding="utf-8")
    edit_result = _invoke("edit", _edit_args(fixture, text_file))
    assert edit_result.returncode == 0, edit_result.stderr
    assert edited_text not in edit_result.stdout
    edit_report = json.loads(edit_result.stdout)
    assert edit_report["status"] == "invalid"
    assert edit_report["human_edit_findings"]
    appended = json.loads(Path(fixture["history"]).read_text(encoding="utf-8").splitlines()[-1])
    assert appended["record_type"] == "human_edit"
    assert appended["human_edit_valid"] is False
    assert appended["human_edit_issues"]

    output = _report_output(tmp_path, "blocked")
    try:
        handoff_result = _invoke("handoff", _handoff_args(fixture, output))
        assert handoff_result.returncode == 2
        assert "draft fails deterministic safety validation" in handoff_result.stderr
        assert not output.exists()
    finally:
        output.unlink(missing_ok=True)
        try:
            output.parent.rmdir()
        except OSError:
            pass


def test_pilot_report_cli_writes_actual_review_edit_and_abstention_metrics(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, include_rejection=True)
    edited_text = (
        "Thanks for explaining your workflow. The app helps organize workflow notes. "
        "I work with this product team. What have you tried already?"
    )
    text_file = tmp_path / "edited-text.txt"
    text_file.write_text(edited_text, encoding="utf-8")
    edit_result = _invoke("edit", _edit_args(fixture, text_file))
    assert edit_result.returncode == 0, edit_result.stderr
    assert json.loads(edit_result.stdout)["status"] == "valid"
    _append_abstention(Path(fixture["history"]))

    output = _report_output(tmp_path, "pilot-summary")
    try:
        result = _invoke(
            "pilot-report",
            [
                "--review-journal",
                str(fixture["review"]),
                "--draft-history",
                str(fixture["history"]),
                "--output",
                str(output),
            ],
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["written"] is True
        summary = json.loads(output.read_text(encoding="utf-8"))
        assert summary["reviewer_decisions_latest_per_identity"] == {
            "identity_count": 2,
            "provisional_candidate": 1,
            "reject": 1,
            "needs_context": 0,
        }
        assert summary["human_edits"] == {"record_count": 1, "valid_count": 1, "invalid_count": 0}
        assert summary["latest_draft_outcomes"] == {"identity_count": 2, "draft": 1, "abstain": 1}
        assert summary["latest_abstention_reasons"] == {"insufficient context": 1}
        assert summary["false_positive_adjudications"]["status"] == "not_recorded"
        assert summary["false_positive_adjudications"]["false_positive_rate"] is None
    finally:
        output.unlink(missing_ok=True)
        try:
            output.parent.rmdir()
        except OSError:
            pass


def test_adjudicate_cli_and_pilot_report_include_false_positive_denominator(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, include_rejection=True)
    reviews = read_reviews(Path(fixture["review"]))
    rejected = next(review for review in reviews if review.verdict == "reject")
    adjudications = tmp_path / "adjudications.jsonl"
    recorded = _invoke(
        "adjudicate",
        [
            "--review-journal",
            str(fixture["review"]),
            "--adjudication-journal",
            str(adjudications),
            "--review-sha256",
            review_sha256(rejected),
            "--label",
            "false_positive",
            "--adjudicator",
            "acceptance-adjudicator",
            "--adjudicated-at",
            "2026-09-30T12:00:00Z",
        ],
    )
    assert recorded.returncode == 0, recorded.stderr
    assert json.loads(recorded.stdout)["recorded"] is True
    output = _report_output(tmp_path, "adjudicated-summary")
    try:
        result = _invoke(
            "pilot-report",
            [
                "--review-journal",
                str(fixture["review"]),
                "--draft-history",
                str(fixture["history"]),
                "--adjudication-journal",
                str(adjudications),
                "--output",
                str(output),
            ],
        )
        assert result.returncode == 0, result.stderr
        summary = json.loads(output.read_text(encoding="utf-8"))
        metrics = summary["false_positive_adjudications"]
        assert metrics["criterion_versions"] == ["engagement-fp-v1"]
        assert metrics["adjudicated_identity_count"] == 1
        assert metrics["unadjudicated_reviewed_identity_count"] == 1
        assert metrics["label_counts"] == {
            "true_positive": 0,
            "false_positive": 1,
            "unclear": 0,
        }
        assert metrics["false_positive_rate"] == 1
        assert metrics["rate_denominator"] == 1
    finally:
        output.unlink(missing_ok=True)
        try:
            output.parent.rmdir()
        except OSError:
            pass


def test_adjudicate_cli_identity_selects_latest_review(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prior = read_reviews(Path(fixture["review"]))[0]
    latest = _review(
        bundle_sha256=prior.bundle_sha256,
        reviewed_at=datetime(2000, 9, 2, tzinfo=UTC),
        verdict="needs_context",
    )
    append_review(
        Path(fixture["review"]),
        latest,
        expected_bundle_sha256=prior.bundle_sha256,
    )
    journal = tmp_path / "identity-adjudications.jsonl"
    result = _invoke(
        "adjudicate",
        [
            "--review-journal",
            str(fixture["review"]),
            "--adjudication-journal",
            str(journal),
            "--app-id",
            "app",
            "--thread-fullname",
            "t3_thread",
            "--reply-target-fullname",
            "t1_target",
            "--label",
            "unclear",
            "--adjudicator",
            "acceptance-adjudicator",
            "--adjudicated-at",
            "2026-09-30T12:00:00Z",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["review_sha256"] == review_sha256(latest)
    assert read_false_positive_adjudications(journal)[0].review_verdict == "needs_context"


def test_adjudicate_cli_rejects_missing_ambiguous_and_mixed_selection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    reviews = read_reviews(Path(fixture["review"]))
    review = reviews[0]

    def invoke(*selection: str) -> subprocess.CompletedProcess[str]:
        return _invoke(
            "adjudicate",
            [
                "--review-journal",
                str(fixture["review"]),
                "--adjudication-journal",
                str(tmp_path / f"rejected-{len(list(tmp_path.glob('rejected-*')))}.jsonl"),
                *selection,
                "--label",
                "unclear",
                "--adjudicator",
                "acceptance-adjudicator",
                "--adjudicated-at",
                "2026-09-30T12:00:00Z",
            ],
        )

    missing = invoke(
        "--app-id",
        "missing",
        "--thread-fullname",
        "t3_thread",
        "--reply-target-fullname",
        "t1_target",
    )
    assert missing.returncode == 2
    assert "no review found" in missing.stderr

    append_review(
        Path(fixture["review"]),
        _review(
            bundle_sha256=review.bundle_sha256,
            reviewed_at=review.reviewed_at,
            verdict="reject",
        ),
        expected_bundle_sha256=review.bundle_sha256,
    )
    ambiguous = invoke(
        "--app-id",
        "app",
        "--thread-fullname",
        "t3_thread",
        "--reply-target-fullname",
        "t1_target",
    )
    assert ambiguous.returncode == 2
    assert "ambiguous same-time" in ambiguous.stderr

    mixed = invoke(
        "--review-sha256",
        review_sha256(review),
        "--app-id",
        "app",
        "--thread-fullname",
        "t3_thread",
        "--reply-target-fullname",
        "t1_target",
    )
    assert mixed.returncode == 2
    assert "either --review-sha256" in mixed.stderr
