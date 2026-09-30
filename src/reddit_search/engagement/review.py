from __future__ import annotations

import os
import stat
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, StrictBool, model_validator

from reddit_search.contracts import ProductProfile, StrictModel


class EngagementReview(StrictModel):
    """Immutable, versioned reviewer decision for one app/thread/target."""

    schema_version: Literal[1] = 1
    app_id: str = Field(min_length=1)
    thread_fullname: str = Field(pattern=r"^t3_[A-Za-z0-9]+$")
    reply_target_fullname: str = Field(pattern=r"^t[13]_[A-Za-z0-9]+$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_message_evidence_ids: list[str] = Field(default_factory=list)
    product_fit: Literal["compatible", "incompatible", "needs_clarification", "not_evaluated"]
    supporting_claim_ids: list[str] = Field(default_factory=list)
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    verdict: Literal["reject", "needs_context", "provisional_candidate"]
    reason: str = Field(min_length=1)
    author_need: Literal[
        "first_person_relevant",
        "tester_or_self_promotion",
        "quoted_or_other_person",
        "no_need",
        "unclear",
    ] = "unclear"
    answered_in_available_context: Literal["yes", "no", "unknown"] = "unknown"
    context_complete: StrictBool = False
    reason_codes: list[
        Literal[
            "author_has_current_need",
            "need_already_answered",
            "context_incomplete",
            "product_fit_unverified",
            "product_incompatible",
            "self_promotion_or_tester_request",
            "quoted_or_other_person",
            "no_useful_reply",
            "historical_removal",
            "other",
        ]
    ] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_claim_support(self) -> EngagementReview:
        if self.product_fit == "compatible" and not self.supporting_claim_ids:
            raise ValueError("compatible product fit requires supporting_claim_ids")
        if self.product_fit != "compatible" and self.supporting_claim_ids:
            raise ValueError("supporting_claim_ids requires compatible product_fit")
        if (
            self.reply_target_fullname.startswith("t3_")
            and self.reply_target_fullname != self.thread_fullname
        ):
            raise ValueError("submission reply_target_fullname must equal thread_fullname")
        if (
            self.reviewed_at.tzinfo is None
            or self.reviewed_at.utcoffset() is None
            or self.reviewed_at.utcoffset().total_seconds() != 0
        ):
            raise ValueError("reviewed_at must be a UTC timestamp")
        if not self.reviewer.strip() or not self.reason.strip():
            raise ValueError("reviewer and reason must be nonblank")
        return self

    @model_validator(mode="after")
    def validate_provisional_decision(self) -> EngagementReview:
        if self.author_need == "tester_or_self_promotion" and self.verdict != "reject":
            raise ValueError("tester/self-promotion author need requires reject verdict")
        if (
            self.author_need == "tester_or_self_promotion"
            and "self_promotion_or_tester_request" not in self.reason_codes
        ):
            raise ValueError("tester/self-promotion author need requires matching reason_code")
        if self.verdict == "provisional_candidate":
            if self.author_need != "first_person_relevant":
                raise ValueError("provisional_candidate requires first_person_relevant author_need")
            if self.answered_in_available_context != "no":
                raise ValueError("provisional_candidate requires answered_in_available_context=no")
            if not self.context_complete:
                raise ValueError("provisional_candidate requires complete context")
            if not self.source_message_evidence_ids:
                raise ValueError("provisional_candidate requires captured message evidence")
            if self.product_fit != "compatible" or not self.supporting_claim_ids:
                raise ValueError(
                    "provisional_candidate requires compatible product_fit and supporting claims"
                )
        return self


FALSE_POSITIVE_CRITERION_VERSION = "engagement-fp-v1"
FALSE_POSITIVE_CRITERION = (
    "Label false_positive only when the reviewed item clearly lacks a genuine, "
    "first-person, currently unanswered need that the app can address using its "
    "verified claims; label unclear when evidence is insufficient."
)


class FalsePositiveAdjudication(StrictModel):
    """Immutable, versioned outcome label for one exact engagement review."""

    schema_version: Literal[1] = 1
    app_id: str = Field(min_length=1)
    thread_fullname: str = Field(pattern=r"^t3_[A-Za-z0-9]+$")
    reply_target_fullname: str = Field(pattern=r"^t[13]_[A-Za-z0-9]+$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_verdict: Literal["reject", "needs_context", "provisional_candidate"]
    label: Literal["true_positive", "false_positive", "unclear"]
    criterion_version: str = Field(min_length=1)
    adjudicator: str = Field(min_length=1)
    adjudicated_at: datetime
    note: str | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_adjudication(self) -> FalsePositiveAdjudication:
        if (
            self.adjudicated_at.tzinfo is None
            or self.adjudicated_at.utcoffset() is None
            or self.adjudicated_at.utcoffset().total_seconds() != 0
        ):
            raise ValueError("adjudicated_at must be a UTC timestamp")
        if (
            not self.app_id.strip()
            or not self.criterion_version.strip()
            or not self.adjudicator.strip()
        ):
            raise ValueError("app_id, criterion_version, and adjudicator must be nonblank")
        if (
            self.reply_target_fullname.startswith("t3_")
            and self.reply_target_fullname != self.thread_fullname
        ):
            raise ValueError("submission reply_target_fullname must equal thread_fullname")
        return self


def review_sha256(review: EngagementReview) -> str:
    """Hash the exact canonical JSON record representation used by the review journal."""
    return sha256(review.model_dump_json().encode("utf-8")).hexdigest()


def append_false_positive_adjudication(
    path: str | Path,
    adjudication: FalsePositiveAdjudication,
    *,
    expected_bundle_sha256: str,
) -> None:
    if adjudication.bundle_sha256 != expected_bundle_sha256:
        raise ValueError("bundle hash mismatch: record does not match expected bundle_sha256")
    destination = Path(path)
    existing = read_false_positive_adjudications(path) if destination.exists() else []
    if any(
        (item.app_id, item.thread_fullname, item.reply_target_fullname, item.review_sha256)
        == (
            adjudication.app_id,
            adjudication.thread_fullname,
            adjudication.reply_target_fullname,
            adjudication.review_sha256,
        )
        for item in existing
    ):
        raise ValueError("duplicate false-positive adjudication for this review")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(adjudication.model_dump_json() + "\n")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def read_false_positive_adjudications(
    path: str | Path, *, expected_bundle_sha256: str | None = None
) -> list[FalsePositiveAdjudication]:
    records: list[FalsePositiveAdjudication] = []
    seen: set[tuple[str, str, str, str]] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(
                    "blank line in false-positive adjudication journal "
                    f"at line {line_number}"
                )
            try:
                record = FalsePositiveAdjudication.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid false-positive adjudication at line {line_number}: {exc}"
                ) from exc
            if (
                expected_bundle_sha256 is not None
                and record.bundle_sha256 != expected_bundle_sha256
            ):
                raise ValueError(f"bundle hash mismatch at line {line_number}")
            key = (
                record.app_id,
                record.thread_fullname,
                record.reply_target_fullname,
                record.review_sha256,
            )
            if key in seen:
                raise ValueError(f"duplicate false-positive adjudication at line {line_number}")
            seen.add(key)
            records.append(record)
    return records


class ManualCheck(StrictModel):
    """Append-only record of a human's current-page observation and decision."""

    schema_version: Literal[1] = 1
    app_id: str = Field(min_length=1)
    thread_fullname: str = Field(pattern=r"^t3_[A-Za-z0-9]+$")
    reply_target_fullname: str = Field(pattern=r"^t[13]_[A-Za-z0-9]+$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1)
    observed_at: datetime
    live_status: Literal[
        "live_unverified", "live_unavailable", "live_not_replyable", "human_checked_replyable"
    ] = "live_unverified"
    permalink: str | None = None
    observation: str = ""
    rule_evidence: list[str] = Field(default_factory=list)
    final_human_decision: Literal["post", "do_not_post", "defer"] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_observation(self) -> ManualCheck:
        if (
            self.reply_target_fullname.startswith("t3_")
            and self.reply_target_fullname != self.thread_fullname
        ):
            raise ValueError("submission reply_target_fullname must equal thread_fullname")
        if (
            self.observed_at.tzinfo is None
            or self.observed_at.utcoffset() is None
            or self.observed_at.utcoffset().total_seconds() != 0
        ):
            raise ValueError("observed_at must be a UTC timestamp")
        if not self.reviewer.strip():
            raise ValueError("reviewer must be nonblank")
        if self.live_status == "human_checked_replyable":
            if (
                not self.permalink
                or not self.observation.strip()
                or not self.rule_evidence
                or self.final_human_decision is None
            ):
                raise ValueError(
                    "human_checked_replyable requires permalink, observation, "
                    "rule_evidence, and final_human_decision"
                )
        elif self.final_human_decision is not None:
            raise ValueError("final_human_decision requires human_checked_replyable")
        return self


def engagement_identity(record: EngagementReview | ManualCheck) -> tuple[str, str, str]:
    return (record.app_id, record.thread_fullname, record.reply_target_fullname)


def append_review(
    path: str | Path,
    review: EngagementReview,
    *,
    profile: ProductProfile | None = None,
    expected_bundle_sha256: str,
) -> None:
    supplied_fields = review.model_fields_set
    review = EngagementReview.model_validate(review.model_dump(mode="python"))
    required_decisions = {
        "author_need",
        "answered_in_available_context",
        "context_complete",
        "reason_codes",
    }
    if required_decisions - supplied_fields:
        raise ValueError(
            "review requires explicit author_need, answered_in_available_context, "
            "context_complete, and reason_codes"
        )
    if review.product_fit == "compatible":
        if profile is None:
            raise ValueError("compatible product fit requires a verified ProductProfile")
        if (
            profile.app_id != review.app_id
            or profile.verification_status != "verified"
            or profile.synthetic
        ):
            raise ValueError("ProductProfile must match app_id and be verified, non-synthetic")
        claims = {claim.claim_id: claim for claim in profile.claims}
        for claim_id in review.supporting_claim_ids:
            claim = claims.get(claim_id)
            if claim is None or claim.status != "verified" or not claim.evidence_ref:
                raise ValueError(f"supporting claim {claim_id!r} is not verified with evidence")
    _append(path, review, expected_bundle_sha256)


def append_manual_check(
    path: str | Path, check: ManualCheck, *, expected_bundle_sha256: str
) -> None:
    _append(path, check, expected_bundle_sha256)


def _append(
    path: str | Path, record: EngagementReview | ManualCheck, expected_bundle_sha256: str
) -> None:
    if record.bundle_sha256 != expected_bundle_sha256:
        raise ValueError("bundle hash mismatch: record does not match expected bundle_sha256")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(record.model_dump_json() + "\n")


def read_reviews(
    path: str | Path, *, expected_bundle_sha256: str | None = None
) -> list[EngagementReview]:
    return _read(path, EngagementReview, expected_bundle_sha256)


def read_manual_checks(
    path: str | Path, *, expected_bundle_sha256: str | None = None
) -> list[ManualCheck]:
    return _read(path, ManualCheck, expected_bundle_sha256)


def _read(
    path: str | Path,
    model: type[EngagementReview] | type[ManualCheck],
    expected_bundle_sha256: str | None,
):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = model.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid journal record at line {line_number}: {exc}") from exc
            if (
                expected_bundle_sha256 is not None
                and record.bundle_sha256 != expected_bundle_sha256
            ):
                raise ValueError(f"bundle hash mismatch at line {line_number}")
            records.append(record)
    return records
