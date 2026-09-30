from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import zstandard
from pydantic import ValidationError

from reddit_search.engagement.drafting import ReplyDraft, validate_human_edit
from reddit_search.engagement.review import (
    FALSE_POSITIVE_CRITERION,
    FALSE_POSITIVE_CRITERION_VERSION,
    EngagementReview,
    FalsePositiveAdjudication,
    ManualCheck,
    engagement_identity,
    read_false_positive_adjudications,
    review_sha256,
)
from reddit_search.ingest.state import file_sha256

_SHA256_LENGTH = 64
_ALLOWED_SCOPES = frozenset(
    {
        "authorized_local_research",
        "user_provided_local_research_no_external_transmission",
    }
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MAX_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_THREAD_MESSAGES = 100_000
_MAX_THREAD_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_draft_history(path: Path) -> tuple[list[ReplyDraft], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[ReplyDraft] = []
    previous: ReplyDraft | None = None
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"draft history has a blank line at {line_number}")
            try:
                record = ReplyDraft.model_validate_json(line)
            except (ValueError, ValidationError):
                raise ValueError(f"invalid draft history record at line {line_number}") from None
            if record.history_sequence != line_number:
                raise ValueError(f"draft history sequence mismatch at line {line_number}")
            expected_previous = previous.record_sha256 if previous else None
            if record.previous_record_sha256 != expected_previous:
                raise ValueError(f"draft history chain mismatch at line {line_number}")
            canonical = json.dumps(
                record.model_dump(mode="json", exclude={"record_sha256"}),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != record.record_sha256:
                raise ValueError(f"draft history hash mismatch at line {line_number}")
            records.append(record)
            previous = record
    return records, file_sha256(path)


def _read_reviews(path: Path) -> tuple[list[EngagementReview], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[EngagementReview] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(EngagementReview.model_validate_json(line))
            except (ValueError, ValidationError):
                raise ValueError(f"invalid review journal record at line {line_number}") from None
    return records, file_sha256(path)


def _read_manual_checks(path: Path) -> tuple[list[ManualCheck], str | None]:
    if not path.exists():
        return [], None
    records: list[ManualCheck] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(ManualCheck.model_validate_json(line))
            except (ValueError, ValidationError):
                raise ValueError(
                    f"invalid manual-check journal record at line {line_number}"
                ) from None
    return records, file_sha256(path)


def _read_context_thread(
    bundle_path: Path,
    thread_fullname: str,
    *,
    max_bundle_bytes: int,
    max_thread_bytes: int,
    max_thread_messages: int,
) -> list[dict[str, Any]]:
    if bundle_path.stat().st_size > max_bundle_bytes:
        raise ValueError("compressed context bundle exceeds configured limit")
    rows: list[dict[str, Any]] = []
    thread_bytes = 0
    total_bytes = 0
    buffer = bytearray()

    def consume(line: bytes) -> None:
        nonlocal thread_bytes
        if not line.strip():
            return
        try:
            item = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("context bundle contains invalid JSONL") from error
        if not isinstance(item, dict):
            raise ValueError("context bundle rows must be JSON objects")
        if item.get("thread_fullname") != thread_fullname:
            return
        thread_bytes += len(line)
        if thread_bytes > max_thread_bytes:
            raise ValueError("thread context exceeds configured limit")
        rows.append(item)
        if len(rows) > max_thread_messages:
            raise ValueError("thread context exceeds configured message limit")

    try:
        with (
            bundle_path.open("rb") as raw,
            zstandard.ZstdDecompressor().stream_reader(raw) as reader,
        ):
            while chunk := reader.read(_READ_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > max_bundle_bytes:
                    raise ValueError("decompressed context bundle exceeds configured limit")
                buffer.extend(chunk)
                while (newline := buffer.find(b"\n")) >= 0:
                    if newline > max_thread_bytes:
                        raise ValueError("context bundle contains an oversized record")
                    consume(bytes(buffer[:newline]))
                    del buffer[: newline + 1]
                if len(buffer) > max_thread_bytes:
                    raise ValueError("context bundle contains an oversized record")
    except zstandard.ZstdError as error:
        raise ValueError("context bundle is not a valid zstd stream") from error
    if buffer:
        consume(bytes(buffer))
    if not rows:
        raise ValueError("context bundle does not contain the requested thread")
    return rows


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{description} must be a JSON object")
    return data


def _source_permissions(
    registry_path: Path, context_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    inputs = context_manifest.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("registry_sha256") != file_sha256(registry_path):
        raise ValueError("source registry hash does not match context manifest")
    registry = _read_json_object(registry_path, "source registry")
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source registry has no valid sources list")
    source_by_id = {
        record.get("source_id"): record
        for record in sources
        if isinstance(record, dict) and isinstance(record.get("source_id"), str)
    }
    reports = context_manifest.get("sources")
    if not isinstance(reports, list) or not reports:
        raise ValueError("context manifest has no source reports")
    permissions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for report in reports:
        if not isinstance(report, dict) or report.get("complete") is not True:
            raise ValueError("context contains an incomplete source")
        source_id = report.get("source_id")
        record = source_by_id.get(source_id)
        if not isinstance(source_id, str) or source_id in seen or record is None:
            raise ValueError("context source identity is missing or ambiguous in registry")
        scope = record.get("usage_scope")
        if scope not in _ALLOWED_SCOPES or record.get("status") != "complete":
            raise ValueError("source registry does not authorize local handoff")
        if record.get("source_kind") != report.get("source_kind"):
            raise ValueError("context source kind does not match registry")
        if record.get("declared_month") != report.get("declared_month"):
            raise ValueError("context source month does not match registry")
        seen.add(source_id)
        permissions.append(
            {
                "source_id": source_id,
                "source_kind": record["source_kind"],
                "declared_month": record.get("declared_month"),
                "source_role": record.get("source_role"),
                "usage_scope": scope,
                "status": record["status"],
            }
        )
    return permissions


def _private_destination(output_path: Path, repo_root: Path) -> Path:
    root = repo_root.resolve()
    ignored_reports = root / "reports"
    try:
        ignore_lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("repository .gitignore is required for private handoff output") from error
    if "reports/" not in {line.strip() for line in ignore_lines}:
        raise ValueError("repository does not ignore reports/ output")
    destination = output_path.resolve()
    try:
        destination.relative_to(ignored_reports.resolve())
    except ValueError as error:
        raise ValueError(
            "handoff output must be under the gitignored reports/ directory"
        ) from error
    if destination.exists():
        raise FileExistsError("handoff output already exists")
    return destination


def _write_private_json(path: Path, packet: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=".handoff-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def export_handoff(
    *,
    app_id: str,
    thread_fullname: str,
    reply_target_fullname: str,
    context_bundle_path: str | Path,
    context_manifest_path: str | Path,
    expected_context_manifest_sha256: str,
    review_journal_path: str | Path,
    draft_history_path: str | Path,
    registry_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path = _PROJECT_ROOT,
    max_bundle_bytes: int = _MAX_BUNDLE_BYTES,
    max_thread_bytes: int = _MAX_THREAD_BYTES,
    max_thread_messages: int = _MAX_THREAD_MESSAGES,
) -> dict[str, Any]:
    """Export a private offline JSON packet from hash-bound, validated artifacts.

    The destination must be new and under ``repo_root/reports/``, which must
    appear in that repository's ``.gitignore``. No network or posting operation
    is performed. The included blank manual form always starts live-unverified.
    """
    identity = (app_id, thread_fullname, reply_target_fullname)
    if (
        not app_id.strip()
        or not thread_fullname.startswith("t3_")
        or not reply_target_fullname.startswith(("t1_", "t3_"))
    ):
        raise ValueError("engagement identity is malformed")
    if min(max_bundle_bytes, max_thread_bytes, max_thread_messages) <= 0:
        raise ValueError("handoff read limits must be positive")
    bundle = Path(context_bundle_path)
    manifest_path = Path(context_manifest_path)
    reviews_path = Path(review_journal_path)
    history_path = Path(draft_history_path)
    registry = Path(registry_path)
    destination = _private_destination(Path(output_path), Path(repo_root))
    for path in (bundle, manifest_path, reviews_path, history_path, registry):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not _sha256(expected_context_manifest_sha256):
        raise ValueError("expected context manifest hash must be a SHA-256")
    manifest_hash = file_sha256(manifest_path)
    if manifest_hash != expected_context_manifest_sha256:
        raise ValueError("context manifest hash does not match expected hash")
    context_manifest = _read_json_object(manifest_path, "context manifest")
    bundle_hash = file_sha256(bundle)
    if context_manifest.get("output_sha256") != bundle_hash:
        raise ValueError("context bundle hash does not match context manifest")
    if context_manifest.get("scan_complete") is not True:
        raise ValueError("context scan is incomplete")
    permissions = _source_permissions(registry, context_manifest)
    reviews, review_journal_hash = _read_reviews(reviews_path)
    matching_reviews = [review for review in reviews if engagement_identity(review) == identity]
    if not matching_reviews:
        raise ValueError("review journal has no record for the requested identity")
    latest_review = max(matching_reviews, key=lambda item: item.reviewed_at)
    if sum(item.reviewed_at == latest_review.reviewed_at for item in matching_reviews) != 1:
        raise ValueError("newest review timestamp is ambiguous for the requested identity")
    review_hash = hashlib.sha256(latest_review.model_dump_json().encode("utf-8")).hexdigest()
    history, history_hash = _read_draft_history(history_path)
    matching_drafts = [
        record
        for record in history
        if (record.app_id, record.thread_fullname, record.reply_target_fullname) == identity
    ]
    if not matching_drafts:
        raise ValueError("draft history has no record for the requested identity")
    draft = matching_drafts[-1]
    if draft.created_at < latest_review.reviewed_at:
        raise ValueError("latest draft predates the latest review")
    if latest_review.verdict != "provisional_candidate" or not latest_review.context_complete:
        raise ValueError("latest review is not a context-complete provisional candidate")
    if draft.context_bundle_sha256 != bundle_hash or latest_review.bundle_sha256 != bundle_hash:
        raise ValueError("draft or review context bundle hash mismatch")
    if draft.context_manifest_sha256 != manifest_hash:
        raise ValueError("draft context manifest hash mismatch")
    if draft.review_journal_sha256 != review_journal_hash:
        raise ValueError("draft review journal hash mismatch")
    if draft.review_sha256 != review_hash:
        raise ValueError("draft does not reference the latest review")
    thread_reports = context_manifest.get("threads")
    if not isinstance(thread_reports, list):
        raise ValueError("context manifest has no thread coverage reports")
    matching_reports = [
        report
        for report in thread_reports
        if isinstance(report, dict) and report.get("thread_fullname") == thread_fullname
    ]
    if len(matching_reports) != 1:
        raise ValueError("context manifest does not contain exactly one requested thread")
    thread_report = matching_reports[0]
    if (
        thread_report.get("coverage_status") != "complete_for_scanned_sources"
        or thread_report.get("thread_context_complete_for_sources") is not True
        or thread_report.get("root_missing") is not False
        or thread_report.get("missing_parents") != []
        or reply_target_fullname in thread_report.get("missing_reply_targets", [])
    ):
        raise ValueError("context is incomplete or reply target is unresolved")
    rows = _read_context_thread(
        bundle,
        thread_fullname,
        max_bundle_bytes=max_bundle_bytes,
        max_thread_bytes=max_thread_bytes,
        max_thread_messages=max_thread_messages,
    )
    by_id: dict[str, dict[str, Any]] = {}
    ambiguous_ids: set[str] = set()
    for row in rows:
        message_id = row.get("fullname")
        if isinstance(message_id, str):
            if message_id in by_id:
                ambiguous_ids.add(message_id)
            by_id[message_id] = row
    if reply_target_fullname not in by_id or reply_target_fullname in ambiguous_ids:
        raise ValueError("context bundle does not resolve a unique reply target")
    if thread_fullname not in by_id or thread_fullname in ambiguous_ids:
        raise ValueError("context bundle does not resolve a unique thread root")
    ancestor_ids: set[str] = set()
    current_id = reply_target_fullname
    while current_id not in ancestor_ids:
        ancestor_ids.add(current_id)
        if current_id == thread_fullname:
            break
        parent_id = by_id[current_id].get("parent_fullname")
        if not isinstance(parent_id, str) or parent_id not in by_id or parent_id in ambiguous_ids:
            raise ValueError("reply target parent chain is incomplete")
        current_id = parent_id
    if thread_fullname not in ancestor_ids:
        raise ValueError("reply target parent chain does not reach the thread root")
    evidence_ids = sorted(
        ancestor_ids | set(draft.used_message_ids) | set(latest_review.source_message_evidence_ids)
    )
    if not set(evidence_ids).issubset(by_id):
        raise ValueError("draft or review references a message absent from context bundle")
    if any(message_id in ambiguous_ids for message_id in evidence_ids):
        raise ValueError("draft or review references an ambiguous context message")
    if not set(draft.used_claim_ids).issubset(draft.claim_statements):
        raise ValueError("draft claim statements do not cover all used claims")
    target_permalink = by_id[reply_target_fullname].get("permalink")
    if not isinstance(target_permalink, str) or not target_permalink.strip():
        raise ValueError("reply target has no captured permalink")
    permalink = (
        "https://www.reddit.com" + target_permalink
        if target_permalink.startswith("/")
        else target_permalink
    )
    evidence = []
    for message_id in evidence_ids:
        row = by_id[message_id]
        evidence.append(
            {
                "message_id": message_id,
                "parent_message_id": row.get("parent_fullname"),
                "title": row.get("raw_title"),
                "text": row.get("raw_body", ""),
                "permalink": row.get("permalink"),
                "provenance": row.get("provenance", []),
                "archived_status": row.get("archived_status", {}),
            }
        )
    validation_findings = validate_human_edit(draft, draft.text)
    if validation_findings:
        raise ValueError("draft fails deterministic safety validation")
    if draft.human_edit_valid is False:
        raise ValueError("latest human edit is marked invalid")
    packet = {
        "kind": "offline_engagement_handoff",
        "schema_version": 1,
        "identity": {
            "app_id": app_id,
            "thread_fullname": thread_fullname,
            "reply_target_fullname": reply_target_fullname,
        },
        "provisional_reply": draft.text,
        "source_message_evidence": evidence,
        "claims": {
            "profile_version": draft.product_profile_version,
            "profile_sha256": draft.product_profile_sha256,
            "claim_statements": draft.claim_statements,
            "used_claim_ids": draft.used_claim_ids,
        },
        "operator_affiliation": draft.operator_affiliation,
        "permalink": permalink,
        "archive_coverage": {
            "status": thread_report["coverage_status"],
            "complete_for_scanned_sources": thread_report["thread_context_complete_for_sources"],
            "coverage_reasons": thread_report.get("coverage_reasons", []),
            "coverage_reason_counts": thread_report.get("coverage_reason_counts", {}),
            "archive_content_notes": thread_report.get("archive_content_notes", []),
            "archive_content_counts": thread_report.get("archive_content_counts", {}),
        },
        "local_permissions": {
            "sources": permissions,
            "external_transmission_permitted": False,
            "posting_permitted": False,
        },
        "draft_validation": {
            "deterministic_findings": validation_findings,
            "human_edit_valid": draft.human_edit_valid,
            "human_edit_issues": draft.human_edit_issues,
        },
        "provenance_hashes": {
            "context_bundle_sha256": bundle_hash,
            "context_manifest_sha256": manifest_hash,
            "review_sha256": review_hash,
            "review_journal_sha256": review_journal_hash,
            "draft_record_sha256": draft.record_sha256,
            "draft_history_sha256": history_hash,
        },
        "manual_check_form": {
            "schema_version": 1,
            "bundle_sha256": bundle_hash,
            "app_id": app_id,
            "thread_fullname": thread_fullname,
            "reply_target_fullname": reply_target_fullname,
            "permalink": permalink,
            "reviewer": "",
            "observed_at": None,
            "live_status": "live_unverified",
            "observation": "",
            "rule_evidence": [],
            "final_human_decision": None,
            "allowed_live_statuses": [
                "live_unverified",
                "live_unavailable",
                "live_not_replyable",
                "human_checked_replyable",
            ],
            "instruction": (
                "Check the current page, target, new replies, removals/locks, and subreddit rules. "
                "Recheck immediately before any manual posting. "
                "This packet makes no live-status or replyability inference."
            ),
        },
        "posting": "manual only; no posting API or automated action",
    }
    _write_private_json(destination, packet)
    return {
        "output_path": str(destination),
        "output_sha256": file_sha256(destination),
        "draft_record_sha256": draft.record_sha256,
        "thread_fullname": thread_fullname,
        "reply_target_fullname": reply_target_fullname,
    }


def summarize_pilot(
    *,
    review_journal_path: str | Path,
    draft_history_path: str | Path,
    manual_check_journal_path: str | Path,
    false_positive_adjudication_journal_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize recorded reviewer, manual, and explicit adjudication outcomes."""
    reviews, review_hash = _read_reviews(Path(review_journal_path))
    drafts, draft_hash = _read_draft_history(Path(draft_history_path))
    checks, check_hash = _read_manual_checks(Path(manual_check_journal_path))
    adjudications: list[FalsePositiveAdjudication] = []
    adjudication_hash: str | None = None
    if false_positive_adjudication_journal_path is not None:
        adjudication_path = Path(false_positive_adjudication_journal_path)
        adjudications = read_false_positive_adjudications(adjudication_path)
        adjudication_hash = file_sha256(adjudication_path)
    all_reviews_by_hash = {review_sha256(review): review for review in reviews}
    latest_by_identity: dict[tuple[str, str, str], EngagementReview] = {}
    for review in reviews:
        identity = engagement_identity(review)
        previous = latest_by_identity.get(identity)
        if previous is None or review.reviewed_at > previous.reviewed_at:
            latest_by_identity[identity] = review
        elif review.reviewed_at == previous.reviewed_at:
            raise ValueError("ambiguous same-time review records for one identity")
    latest_review_hashes = {review_sha256(review) for review in latest_by_identity.values()}
    for adjudication in adjudications:
        reviewed = all_reviews_by_hash.get(adjudication.review_sha256)
        if reviewed is None:
            raise ValueError("adjudication references an unknown review record")
        if (
            engagement_identity(reviewed)
            != (
                adjudication.app_id,
                adjudication.thread_fullname,
                adjudication.reply_target_fullname,
            )
            or reviewed.bundle_sha256 != adjudication.bundle_sha256
            or reviewed.verdict != adjudication.review_verdict
        ):
            raise ValueError("adjudication identity or outcome does not match its review")
        if adjudication.review_sha256 not in latest_review_hashes:
            raise ValueError("stale adjudication references a superseded review")
    adjudication_labels = Counter(item.label for item in adjudications)
    adjudicated_identities = {
        (item.app_id, item.thread_fullname, item.reply_target_fullname) for item in adjudications
    }
    criterion_versions = sorted({item.criterion_version for item in adjudications})
    verdicts = Counter(review.verdict for review in latest_by_identity.values())
    latest_drafts: dict[tuple[str, str, str], ReplyDraft] = {}
    for draft in drafts:
        identity = (draft.app_id, draft.thread_fullname, draft.reply_target_fullname)
        latest_drafts[identity] = draft
    abstentions = Counter(
        reason
        for draft in latest_drafts.values()
        if draft.outcome == "abstain"
        for reason in draft.abstention_reasons
    )
    manual_decisions = Counter(
        check.final_human_decision for check in checks if check.final_human_decision is not None
    )
    manual_statuses = Counter(check.live_status for check in checks)
    human_edits = [draft for draft in drafts if draft.record_type == "human_edit"]
    false_positive_count = adjudication_labels["false_positive"]
    adjudicated_count = len(adjudications)
    return {
        "kind": "engagement_pilot_summary",
        "schema_version": 1,
        "reviewer_decisions_latest_per_identity": {
            "identity_count": len(latest_by_identity),
            "provisional_candidate": verdicts["provisional_candidate"],
            "reject": verdicts["reject"],
            "needs_context": verdicts["needs_context"],
        },
        "manual_check_records": len(checks),
        "manual_live_status_observations": dict(sorted(manual_statuses.items())),
        "manual_final_human_decisions": dict(sorted(manual_decisions.items())),
        "human_edits": {
            "record_count": len(human_edits),
            "valid_count": sum(draft.human_edit_valid is True for draft in human_edits),
            "invalid_count": sum(draft.human_edit_valid is False for draft in human_edits),
        },
        "latest_draft_outcomes": {
            "identity_count": len(latest_drafts),
            "draft": sum(draft.outcome == "draft" for draft in latest_drafts.values()),
            "abstain": sum(draft.outcome == "abstain" for draft in latest_drafts.values()),
        },
        "latest_abstention_reasons": dict(sorted(abstentions.items())),
        "false_positive_adjudications": {
            "criterion_versions": criterion_versions,
            "criterion_version": FALSE_POSITIVE_CRITERION_VERSION,
            "criterion_description": FALSE_POSITIVE_CRITERION,
            "adjudicated_identity_count": adjudicated_count,
            "unadjudicated_reviewed_identity_count": (
                len(latest_by_identity) - len(adjudicated_identities)
            ),
            "label_counts": {
                label: adjudication_labels[label]
                for label in ("true_positive", "false_positive", "unclear")
            },
            "false_positive_rate": (
                false_positive_count / adjudicated_count if adjudicated_count else None
            ),
            "rate_denominator": adjudicated_count if adjudicated_count else None,
            "status": "recorded" if adjudicated_count else "not_recorded",
        },
        "journal_hashes": {
            "review_journal_sha256": review_hash,
            "draft_history_sha256": draft_hash,
            "manual_check_journal_sha256": check_hash,
            "false_positive_adjudication_journal_sha256": adjudication_hash,
        },
    }
