import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from reddit_search.contracts import ProductClaim, ProductProfile
from reddit_search.engagement.review import (
    EngagementReview,
    FalsePositiveAdjudication,
    ManualCheck,
    append_false_positive_adjudication,
    append_manual_check,
    append_review,
    read_false_positive_adjudications,
    read_manual_checks,
    read_reviews,
    review_sha256,
)

HASH = "a" * 64
NOW = datetime(2026, 9, 25, tzinfo=UTC)


def test_reviews_append_and_keep_target_revisions_separate(tmp_path):
    path = tmp_path / "decisions.jsonl"
    first = review()
    second = review(
        reply_target_fullname="t1_comment2",
        verdict="needs_context",
        product_fit="not_evaluated",
        supporting_claim_ids=[],
    )
    profile = verified_profile()
    append_review(path, first, profile=profile, expected_bundle_sha256=HASH)
    append_review(
        path,
        first.model_copy(update={"reason": "revised"}),
        profile=profile,
        expected_bundle_sha256=HASH,
    )
    append_review(path, second, profile=profile, expected_bundle_sha256=HASH)
    assert len(read_reviews(path, expected_bundle_sha256=HASH)) == 3
    assert read_reviews(path)[1].reason == "revised"
    assert read_reviews(path)[2].reply_target_fullname == "t1_comment2"


def test_rejected_review_can_be_reopened_without_overwriting_history(tmp_path):
    path = tmp_path / "decisions.jsonl"
    profile = verified_profile()
    rejected = review(
        verdict="reject",
        product_fit="not_evaluated",
        supporting_claim_ids=[],
        reason_codes=["no_useful_reply"],
        reviewed_at=NOW,
    )
    reopened = review(reviewed_at=datetime(2026, 9, 26, tzinfo=UTC), reason="Reconsidered")

    append_review(path, rejected, profile=profile, expected_bundle_sha256=HASH)
    append_review(path, reopened, profile=profile, expected_bundle_sha256=HASH)
    records = read_reviews(path, expected_bundle_sha256=HASH)

    assert [record.verdict for record in records] == ["reject", "provisional_candidate"]
    assert records[0].reason == rejected.reason
    assert records[1].reason == "Reconsidered"


def test_product_fit_requires_verified_profile_claims(tmp_path):
    with pytest.raises(ValueError, match="ProductProfile"):
        append_review(tmp_path / "decisions.jsonl", review(), expected_bundle_sha256=HASH)
    with pytest.raises(ValueError, match="not verified"):
        append_review(
            tmp_path / "decisions.jsonl",
            review(supporting_claim_ids=["made.up"]),
            profile=verified_profile(),
            expected_bundle_sha256=HASH,
        )
    with pytest.raises(ValidationError, match="requires supporting_claim_ids"):
        review(supporting_claim_ids=[])
    with pytest.raises(ValidationError, match="requires compatible product_fit"):
        review(product_fit="incompatible")


def test_provisional_requires_relevant_unanswered_need_and_complete_context(tmp_path):
    with pytest.raises(ValidationError, match="first_person_relevant"):
        review(author_need="unclear")
    with pytest.raises(ValidationError, match="answered_in_available_context=no"):
        review(answered_in_available_context="yes")
    with pytest.raises(ValidationError, match="requires complete context"):
        review(context_complete=False)
    with pytest.raises(ValidationError, match="requires reject verdict"):
        review(author_need="tester_or_self_promotion")
    with pytest.raises(ValidationError, match="matching reason_code"):
        review(
            verdict="reject",
            author_need="tester_or_self_promotion",
            product_fit="not_evaluated",
            supporting_claim_ids=[],
            reason_codes=[],
        )
    with pytest.raises(ValidationError):
        review(reason_codes=["not-a-code"])
    unsafe_copy = review().model_copy(update={"author_need": "unclear"})
    with pytest.raises(ValidationError, match="first_person_relevant"):
        append_review(
            tmp_path / "blocked.jsonl",
            unsafe_copy,
            profile=verified_profile(),
            expected_bundle_sha256=HASH,
        )


def test_legacy_reviews_parse_with_safe_defaults_without_rewriting_verdict():
    legacy = review(
        verdict="needs_context",
        product_fit="not_evaluated",
        supporting_claim_ids=[],
    ).model_dump(mode="json")
    for field in (
        "author_need",
        "answered_in_available_context",
        "context_complete",
        "reason_codes",
    ):
        legacy.pop(field)
    parsed = EngagementReview.model_validate(legacy)
    assert parsed.verdict == "needs_context"
    assert parsed.author_need == "unclear"
    assert parsed.answered_in_available_context == "unknown"
    assert parsed.context_complete is False

    legacy["verdict"] = "provisional_candidate"
    with pytest.raises(ValidationError, match="first_person_relevant"):
        EngagementReview.model_validate(legacy)


def test_review_rejects_mismatched_submission_target():
    with pytest.raises(ValidationError, match="must equal thread_fullname"):
        review(reply_target_fullname="t3_other")


def test_unknown_live_status_defaults_unverified_and_human_check_requires_evidence(tmp_path):
    check = manual()
    assert check.live_status == "live_unverified"
    path = tmp_path / "checks.jsonl"
    append_manual_check(path, check, expected_bundle_sha256=HASH)
    assert read_manual_checks(path)[0] == check
    with pytest.raises(ValidationError, match="rule_evidence"):
        manual(
            live_status="human_checked_replyable",
            final_human_decision="post",
            permalink="https://reddit.example/r/x/comments/thread/",
            observation="Reply button available",
        )



def test_unknown_verdict_and_unsubstantiated_live_check_are_rejected() -> None:
    with pytest.raises(ValidationError):
        review(verdict="approved")

    with pytest.raises(ValidationError, match="rule_evidence"):
        manual(
            live_status="human_checked_replyable",
            final_human_decision="post",
            permalink="https://reddit.example/r/x/comments/thread/",
            observation="Reply button available",
        )

def test_journal_rejects_unknown_fields_and_bundle_mismatch(tmp_path):
    with pytest.raises(ValidationError):
        review(unexpected="value")
    with pytest.raises(ValueError, match="bundle hash mismatch"):
        append_review(
            tmp_path / "decisions.jsonl",
            review(),
            profile=verified_profile(),
            expected_bundle_sha256="b" * 64,
        )
    path = tmp_path / "decisions.jsonl"
    path.write_text(json.dumps(review().model_dump(mode="json")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bundle hash mismatch"):
        read_reviews(path, expected_bundle_sha256="b" * 64)


def test_false_positive_adjudication_round_trip_is_private_and_bundle_bound(tmp_path):
    path = tmp_path / "private" / "adjudications.jsonl"
    reviewed = review(verdict="reject", reply_target_fullname="t1_reply")
    record = FalsePositiveAdjudication(
        app_id=reviewed.app_id,
        thread_fullname=reviewed.thread_fullname,
        reply_target_fullname=reviewed.reply_target_fullname,
        bundle_sha256=reviewed.bundle_sha256,
        review_sha256=review_sha256(reviewed),
        review_verdict=reviewed.verdict,
        label="false_positive",
        criterion_version="engagement-fp-v1",
        adjudicator="reviewer",
        adjudicated_at=NOW,
    )
    append_false_positive_adjudication(path, record, expected_bundle_sha256=HASH)
    assert read_false_positive_adjudications(path, expected_bundle_sha256=HASH) == [record]
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="duplicate"):
        append_false_positive_adjudication(path, record, expected_bundle_sha256=HASH)
    with pytest.raises(ValueError, match="bundle hash mismatch"):
        append_false_positive_adjudication(path, record, expected_bundle_sha256="b" * 64)


def test_false_positive_adjudication_rejects_invalid_model_and_blank_journal_lines(tmp_path):
    reviewed = review(verdict="reject", reply_target_fullname="t1_reply")
    values = dict(
        app_id=reviewed.app_id,
        thread_fullname=reviewed.thread_fullname,
        reply_target_fullname=reviewed.reply_target_fullname,
        bundle_sha256=reviewed.bundle_sha256,
        review_sha256=review_sha256(reviewed),
        review_verdict=reviewed.verdict,
        label="unclear",
        criterion_version="engagement-fp-v1",
        adjudicator="reviewer",
        adjudicated_at=NOW,
    )
    with pytest.raises(ValidationError):
        FalsePositiveAdjudication(**values, unexpected=True)
    with pytest.raises(ValidationError, match="UTC"):
        FalsePositiveAdjudication(**{**values, "adjudicated_at": datetime(2026, 9, 25)})
    with pytest.raises(ValidationError, match="nonblank"):
        FalsePositiveAdjudication(**{**values, "adjudicator": " "})


    path = tmp_path / "adjudications.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank line"):
        read_false_positive_adjudications(path)


def test_false_positive_adjudication_rejects_missing_required_field():
    reviewed = review(verdict="reject", reply_target_fullname="t1_reply")
    values = dict(
        app_id=reviewed.app_id,
        thread_fullname=reviewed.thread_fullname,
        reply_target_fullname=reviewed.reply_target_fullname,
        bundle_sha256=reviewed.bundle_sha256,
        review_sha256=review_sha256(reviewed),
        review_verdict=reviewed.verdict,
        label="false_positive",
        criterion_version="engagement-fp-v1",
        adjudicator="reviewer",
        adjudicated_at=NOW,
    )
    del values["adjudicator"]

    with pytest.raises(ValidationError, match="adjudicator"):
        FalsePositiveAdjudication(**values)


def test_manual_check_rejects_missing_required_reviewer():
    values = manual().model_dump(mode="python")
    del values["reviewer"]

    with pytest.raises(ValidationError, match="reviewer"):
        ManualCheck.model_validate(values)


def verified_profile():
    return ProductProfile(
        app_id="app",
        profile_version=1,
        verification_status="verified",
        capabilities=[
            ProductClaim(
                claim_id="app.claim",
                status="verified",
                evidence_ref="https://example.test/evidence",
            )
        ],
    )


def review(**changes):
    values = dict(
        schema_version=1,
        app_id="app",
        thread_fullname="t3_thread",
        reply_target_fullname="t3_thread",
        bundle_sha256=HASH,
        source_message_evidence_ids=["t3_thread"],
        product_fit="compatible",
        supporting_claim_ids=["app.claim"],
        reviewer="human",
        reviewed_at=NOW,
        verdict="provisional_candidate",
        reason="Relevant request",
        author_need="first_person_relevant",
        answered_in_available_context="no",
        context_complete=True,
        reason_codes=["author_has_current_need"],
    )
    values.update(changes)
    return EngagementReview(**values)


def manual(**changes):
    values = dict(
        app_id="app",
        thread_fullname="t3_thread",
        reply_target_fullname="t3_thread",
        bundle_sha256=HASH,
        reviewer="human",
        observed_at=NOW,
    )
    values.update(changes)
    return ManualCheck(**values)
