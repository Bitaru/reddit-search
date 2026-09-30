from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import stat
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any

import yaml
import zstandard
from pydantic import ValidationError

from reddit_search.contracts import ProductProfile
from reddit_search.engagement.contracts import ShortlistCandidate, ShortlistManifest
from reddit_search.engagement.drafting import ReplyDraft
from reddit_search.engagement.handoff import _read_draft_history
from reddit_search.engagement.review import (
    EngagementReview,
    ManualCheck,
    append_manual_check,
    append_review,
    read_manual_checks,
    read_reviews,
)
from reddit_search.ingest.state import file_sha256

# Align byte/row ceilings with context capture and drafting limits; allow ten
# full 100,000-message capture threads in one workspace bundle.
_MAX_CONTEXT_COMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_MAX_CONTEXT_DECOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_MAX_CONTEXT_MESSAGES = 1_000_000
_MAX_CONTEXT_ROW_BYTES = 2 * 1024 * 1024
_CONTEXT_READ_CHUNK_BYTES = 64 * 1024
_MAX_REQUEST_BYTES = 65_536
_ALLOWED_SCOPES = {
    "authorized_local_research",
    "user_provided_local_research_no_external_transmission",
}
_REVIEW_FIELDS = {
    "app_id",
    "thread_fullname",
    "reply_target_fullname",
    "source_message_evidence_ids",
    "product_fit",
    "supporting_claim_ids",
    "reviewer",
    "verdict",
    "reason",
    "author_need",
    "answered_in_available_context",
    "reason_codes",
}
_MANUAL_FIELDS = {
    "app_id",
    "thread_fullname",
    "reply_target_fullname",
    "reviewer",
    "live_status",
    "permalink",
    "observation",
    "rule_evidence",
    "final_human_decision",
}


def create_engagement_server(
    *,
    candidates_path: Path,
    shortlist_manifest_path: Path,
    context_path: Path,
    context_manifest_path: Path,
    registry_path: Path,
    review_journal_path: Path,
    manual_check_journal_path: Path,
    draft_history_path: Path | None = None,
    profile_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a loopback-only archive engagement review server.

    Missing journals are created privately; existing private journals are
    parsed and bound to this context before the server starts.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("engagement workspace host must be a loopback IP address") from error
    if not address.is_loopback:
        raise ValueError("engagement workspace must bind to a loopback IP address")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")

    workspace = _EngagementWorkspace(
        candidates_path=Path(candidates_path),
        shortlist_manifest_path=Path(shortlist_manifest_path),
        context_path=Path(context_path),
        context_manifest_path=Path(context_manifest_path),
        registry_path=Path(registry_path),
        review_journal_path=Path(review_journal_path),
        manual_check_journal_path=Path(manual_check_journal_path),
        draft_history_path=(Path(draft_history_path) if draft_history_path is not None else None),
        profile_path=Path(profile_path) if profile_path is not None else None,
    )

    class EngagementWorkspaceHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if not self._host_matches():
                self._send_json({"error": "host is not allowed"}, status=HTTPStatus.BAD_REQUEST)
                return
            if self.path == "/":
                self._send_bytes(_PAGE_HTML.encode(), content_type="text/html; charset=utf-8")
            elif self.path == "/api/bootstrap":
                self._send_json(workspace.snapshot())
            else:
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/api/review", "/api/manual-check"}:
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            if not self._host_matches() or not self._origin_matches():
                self._send_json(
                    {"error": "cross-origin request refused"}, status=HTTPStatus.FORBIDDEN
                )
                return
            if self.headers.get_content_type() != "application/json":
                self._send_json(
                    {"error": "Content-Type must be application/json"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > _MAX_REQUEST_BYTES:
                    raise ValueError("request body must be between 1 and 65536 bytes")
                payload = json.loads(self.rfile.read(length))
                result = (
                    workspace.save_review(payload)
                    if self.path == "/api/review"
                    else workspace.save_manual_check(payload)
                )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, ValidationError) as error:
                self._send_json(
                    {"saved": False, "error": str(error)}, status=HTTPStatus.BAD_REQUEST
                )
                return
            self._send_json({"saved": True, **result})

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _host_matches(self) -> bool:
            return self.headers.get("Host", "").lower() == workspace.authority.lower()

        def _origin_matches(self) -> bool:
            return self.headers.get("Origin") == workspace.origin

        def _send_json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send_bytes(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(),
                content_type="application/json; charset=utf-8",
                status=status,
            )

        def _send_bytes(
            self,
            payload: bytes,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; object-src 'none'; base-uri 'none'; "
                "frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

    loopback_server_type = type(
        "LoopbackThreadingHTTPServer",
        (ThreadingHTTPServer,),
        {"address_family": socket.AF_INET6 if address.version == 6 else socket.AF_INET},
    )
    server = loopback_server_type((str(address), port), EngagementWorkspaceHandler)
    server.daemon_threads = True
    server.engagement_workspace = workspace
    server.engagement_server_report = engagement_server_report(server)
    workspace.set_origin(server.server_address[0], server.server_address[1])
    return server


def engagement_server_report(server: ThreadingHTTPServer) -> dict[str, object]:
    host, port = server.server_address[:2]
    authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    return {
        "host": host,
        "port": port,
        "url": f"http://{authority}",
        "candidate_count": server.engagement_workspace.candidate_count,
    }


class _EngagementWorkspace:
    def __init__(
        self,
        *,
        candidates_path: Path,
        shortlist_manifest_path: Path,
        context_path: Path,
        context_manifest_path: Path,
        registry_path: Path,
        review_journal_path: Path,
        manual_check_journal_path: Path,
        draft_history_path: Path | None,
        profile_path: Path | None,
    ) -> None:
        for path in (
            candidates_path,
            shortlist_manifest_path,
            context_path,
            context_manifest_path,
            registry_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        raw_manifest = _read_json(shortlist_manifest_path, "shortlist manifest")
        try:
            self.shortlist_manifest = ShortlistManifest.model_validate(raw_manifest)
        except ValidationError as error:
            raise ValueError(f"invalid shortlist manifest: {error}") from error
        self.candidates_path = candidates_path
        self.shortlist_manifest_path = shortlist_manifest_path
        self.context_path = context_path
        self.context_manifest_path = context_manifest_path
        self.registry_path = registry_path
        self.context_sha256 = file_sha256(context_path)
        self.context_manifest_sha256 = file_sha256(context_manifest_path)
        self.shortlist_sha256 = file_sha256(shortlist_manifest_path)
        if file_sha256(candidates_path) != self.shortlist_manifest.candidates_sha256:
            raise ValueError("candidate JSONL hash does not match shortlist manifest")
        if self.shortlist_manifest.inputs.get("registry_sha256") != file_sha256(registry_path):
            raise ValueError("registry hash does not match shortlist manifest")
        self.context_manifest = _read_json(context_manifest_path, "context manifest")
        inputs = self.context_manifest.get("inputs")
        if (
            self.context_manifest.get("kind") != "engagement_thread_context_manifest"
            or self.context_manifest.get("schema_version") != 1
            or not isinstance(inputs, dict)
        ):
            raise ValueError("invalid context manifest schema")
        if self.context_manifest.get("output_sha256") != self.context_sha256:
            raise ValueError("context bundle hash does not match context manifest")
        if inputs.get("candidates_sha256") != self.shortlist_manifest.candidates_sha256:
            raise ValueError("context manifest candidates hash does not match shortlist")
        if inputs.get("shortlist_manifest_sha256") != self.shortlist_sha256:
            raise ValueError("context manifest shortlist hash does not match input manifest")
        if inputs.get("registry_sha256") != file_sha256(registry_path):
            raise ValueError("context manifest registry hash does not match registry")

        self.candidates = _read_candidates(candidates_path, self.shortlist_manifest)
        self.threads = _read_context(context_path, set(self.candidates_by_thread))
        self.thread_reports = _read_thread_reports(self.context_manifest)
        for thread in self.candidates_by_thread:
            if thread not in self.threads or thread not in self.thread_reports:
                raise ValueError(f"context bundle or manifest lacks candidate thread {thread}")
        self.source_scopes = _read_source_scopes(registry_path)
        source_ids = self.context_manifest.get("source_ids", [])
        if not isinstance(source_ids, list) or any(
            not isinstance(source_id, str) for source_id in source_ids
        ):
            raise ValueError("context manifest source_ids must be a list of strings")
        self.source_scope_rows = [
            {
                "source_id": source_id,
                "usage_scope": self.source_scopes.get(source_id),
                "scope_warning": (
                    None
                    if self.source_scopes.get(source_id) in _ALLOWED_SCOPES
                    else "usage_scope missing or unrecognized in source registry"
                ),
            }
            for source_id in source_ids
        ]
        self.profile = _load_profile(profile_path) if profile_path is not None else None
        self.review_journal_path = review_journal_path
        self.manual_check_journal_path = manual_check_journal_path
        self.draft_history_path = draft_history_path
        if review_journal_path.resolve() == manual_check_journal_path.resolve():
            raise ValueError("review and manual-check journals must use distinct paths")
        if draft_history_path is not None and draft_history_path.resolve() in {
            review_journal_path.resolve(),
            manual_check_journal_path.resolve(),
        }:
            raise ValueError("draft history must use a path distinct from review journals")
        if draft_history_path is not None:
            _read_draft_history(draft_history_path)
        self._lock = RLock()
        self.authority = ""
        self.origin = ""
        self._validate_journals(create_missing=True)

    @property
    def candidates_by_thread(self) -> dict[str, list[ShortlistCandidate]]:
        grouped: dict[str, list[ShortlistCandidate]] = {}
        for candidate in self.candidates:
            grouped.setdefault(candidate.thread_fullname, []).append(candidate)
        return grouped

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def messages_by_name(self, thread: str) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in self.threads.get(thread, []):
            grouped.setdefault(row["fullname"], []).append(row)
        return grouped

    def set_origin(self, host: str, port: int) -> None:
        authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        self.authority = authority
        self.origin = f"http://{authority}"

    def _validate_journals(
        self, *, create_missing: bool = False
    ) -> tuple[list[EngagementReview], list[ManualCheck]]:
        review_path = self.review_journal_path
        manual_path = self.manual_check_journal_path
        known_candidates = {
            (candidate.app_id, candidate.thread_fullname): candidate
            for candidate in self.candidates
        }
        present: dict[Path, bool] = {}
        reviews: list[EngagementReview] = []
        checks: list[ManualCheck] = []
        for path, label in (
            (review_path, "review"),
            (manual_path, "manual-check"),
        ):
            exists = path.exists() or path.is_symlink()
            present[path] = exists
            if not exists:
                continue
            try:
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
                    raise ValueError(f"existing {label} journal must be a regular mode-0600 file")
                if path == review_path:
                    rows = read_reviews(path, expected_bundle_sha256=self.context_sha256)
                    for row in rows:
                        candidate = known_candidates.get((row.app_id, row.thread_fullname))
                        if candidate is None:
                            raise ValueError("review journal contains a stale shortlist identity")
                        names = self.messages_by_name(row.thread_fullname)
                        if (
                            row.reply_target_fullname != candidate.reply_target_fullname
                            and len(names.get(row.reply_target_fullname, [])) != 1
                        ):
                            raise ValueError("review journal contains a stale reply target")
                        if len(row.source_message_evidence_ids) != len(
                            set(row.source_message_evidence_ids)
                        ) or any(
                            evidence not in names or len(names[evidence]) != 1
                            for evidence in row.source_message_evidence_ids
                        ):
                            raise ValueError("review journal contains stale or ambiguous evidence")
                    reviews = rows
                else:
                    rows = read_manual_checks(path, expected_bundle_sha256=self.context_sha256)
                    for row in rows:
                        if (row.app_id, row.thread_fullname) not in known_candidates:
                            raise ValueError(
                                "manual-check journal contains a stale shortlist identity"
                            )
                        names = self.messages_by_name(row.thread_fullname)
                        if len(names.get(row.reply_target_fullname, [])) != 1:
                            raise ValueError("manual-check journal contains a stale reply target")
                        if row.permalink is not None and row.permalink != names[
                            row.reply_target_fullname
                        ][0].get("permalink"):
                            raise ValueError(
                                "manual-check journal permalink does not match context"
                            )
                    checks = rows
            except (OSError, ValueError) as error:
                raise ValueError(f"invalid or stale {label} journal: {error}") from error
        if create_missing:
            created: list[Path] = []
            try:
                for path in (review_path, manual_path):
                    if present[path]:
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    created.append(path)
                    try:
                        os.fchmod(descriptor, 0o600)
                    finally:
                        os.close(descriptor)
            except OSError as error:
                for path in created:
                    path.unlink(missing_ok=True)
                raise ValueError("could not create private journals") from error
        return reviews, checks

    def _draft_history_for(
        self,
        candidate: ShortlistCandidate,
        records: list[ReplyDraft],
        latest_reviews: dict[tuple[str, str, str], EngagementReview],
        review_journal_sha256: str,
    ) -> list[dict[str, Any]]:
        matching = [
            record
            for record in records
            if record.app_id == candidate.app_id
            and record.thread_fullname == candidate.thread_fullname
        ]
        latest_by_target: dict[str, ReplyDraft] = {}
        for record in matching:
            latest_by_target[record.reply_target_fullname] = record
        names = self.messages_by_name(candidate.thread_fullname)
        results = []
        for target, record in latest_by_target.items():
            issues = []
            if len(names.get(target, [])) != 1:
                issues.append("reply target is not unique in the captured context")
            if record.context_bundle_sha256 != self.context_sha256:
                issues.append("draft context bundle hash is stale")
            if record.context_manifest_sha256 != self.context_manifest_sha256:
                issues.append("draft context manifest hash is stale")
            identity = (record.app_id, record.thread_fullname, record.reply_target_fullname)
            current_review = latest_reviews.get(identity)
            if current_review is None:
                issues.append("no current archive review exists for this target")
            else:
                review_sha256 = hashlib.sha256(
                    current_review.model_dump_json().encode("utf-8")
                ).hexdigest()
                if record.review_sha256 != review_sha256:
                    issues.append("draft does not reference the latest archive review")
                if record.created_at < current_review.reviewed_at:
                    issues.append("draft predates the latest archive review")
            if record.review_journal_sha256 != review_journal_sha256:
                issues.append("draft review-journal hash is stale")
            if not issues and record.record_type == "human_edit" and not record.human_edit_valid:
                issues.extend(record.human_edit_issues or ["human edit validation failed"])
            if issues:
                validation_status = (
                    "stale"
                    if any(
                        issue.startswith(
                            (
                                "reply target",
                                "draft context",
                                "draft does not",
                                "draft predates",
                                "draft review-journal",
                            )
                        )
                        or issue.startswith("no current")
                        for issue in issues
                    )
                    else "invalid"
                )
                validation_message = "; ".join(issues)
            elif record.outcome == "abstain":
                validation_status = "current_abstention"
                validation_message = "; ".join(record.abstention_reasons)
            else:
                validation_status = "current"
                validation_message = (
                    "Human-edited draft passed validation."
                    if record.record_type == "human_edit"
                    else "Machine draft matches the current review and context."
                )
            row = record.model_dump(mode="json")
            row.pop("text")
            row["validation_status"] = validation_status
            row["validation_message"] = validation_message
            row["display_text"] = record.text if validation_status == "current" else ""
            results.append(row)
        return results

    def snapshot(self) -> dict[str, object]:

        review_records, check_records = self._validate_journals()
        all_reviews = [entry.model_dump(mode="json") for entry in review_records]
        all_checks = [entry.model_dump(mode="json") for entry in check_records]
        if self.draft_history_path is not None:
            draft_records, _ = _read_draft_history(self.draft_history_path)
        else:
            draft_records = []
        latest_reviews = {
            (entry.app_id, entry.thread_fullname, entry.reply_target_fullname): entry
            for entry in review_records
        }
        review_journal_sha256 = file_sha256(self.review_journal_path)
        candidates = []
        for candidate in self.candidates:
            thread = candidate.thread_fullname
            messages = self.threads[thread]
            report = self.thread_reports[thread]
            messages_by_name = self.messages_by_name(thread)
            root = next((row for row in messages if row["fullname"] == thread), None)
            root_removed = root is not None and _historically_removed(root)
            target_rows = messages_by_name.get(candidate.reply_target_fullname, [])
            target_warning = (
                None
                if len(target_rows) == 1
                else "candidate reply target is missing from captured context"
                if not target_rows
                else (
                    "candidate reply target has multiple captured revisions; choose a unique "
                    "alternate target or record needs_context/reject"
                )
            )
            conflicting_ids = sorted(
                fullname for fullname, rows in messages_by_name.items() if len(rows) > 1
            )
            completeness = bool(
                self.context_manifest.get("scan_complete")
                and report.get("thread_context_complete_for_sources") is True
                and report.get("coverage_status") == "complete_for_scanned_sources"
                and report.get("root_missing") is False
                and root is not None
                and len(target_rows) == 1
                and not conflicting_ids
            )
            coverage_warnings = [
                reason for reason in report.get("coverage_reasons", []) if isinstance(reason, str)
            ]
            archive_content_notes = [
                note
                for note in report.get("archive_content_notes", [])
                if isinstance(note, str)
            ]
            if not self.context_manifest.get("scan_complete"):
                coverage_warnings.append("Context scan is incomplete.")
            if root is None:
                coverage_warnings.append("Submission root is missing from the captured context.")
            if report.get("coverage_status") == "needs_context" and not coverage_warnings:
                coverage_warnings.append("Coverage manifest reports needs_context.")
            if report.get("missing_parents"):
                coverage_warnings.extend(
                    f"Missing parent: {fullname}" for fullname in report["missing_parents"]
                )
            if report.get("missing_reply_targets"):
                coverage_warnings.extend(
                    f"Missing reply target: {fullname}"
                    for fullname in report["missing_reply_targets"]
                )
            if target_warning:
                coverage_warnings.append(target_warning)
            if conflicting_ids:
                coverage_warnings.append(
                    "Conflicting captured revisions: " + ", ".join(conflicting_ids)
                )
            identity = (candidate.app_id, thread, candidate.reply_target_fullname)
            history = [
                row
                for row in all_reviews
                if (row["app_id"], row["thread_fullname"], row["reply_target_fullname"]) == identity
            ]
            check_history = [
                row
                for row in all_checks
                if (row["app_id"], row["thread_fullname"], row["reply_target_fullname"]) == identity
            ]
            thread_history = [
                row
                for row in all_reviews
                if row["app_id"] == candidate.app_id and row["thread_fullname"] == thread
            ]
            thread_checks = [
                row
                for row in all_checks
                if row["app_id"] == candidate.app_id and row["thread_fullname"] == thread
            ]
            draft_history = self._draft_history_for(
                candidate,
                draft_records,
                latest_reviews,
                review_journal_sha256,
            )
            candidates.append(
                {
                    **candidate.model_dump(mode="json"),
                    "messages": messages,
                    "parent_chain": _parent_chain(
                        candidate.reply_target_fullname, self.messages_by_name(thread), thread
                    ),
                    "coverage": report,
                    "coverage_warnings": coverage_warnings,
                    "archive_content_notes": archive_content_notes,
                    "context_complete": completeness,
                    "target_warning": target_warning,
                    "root_removed": root_removed,
                    "archive_verdict": history[-1]["verdict"] if history else "unreviewed",
                    "live_status": check_history[-1]["live_status"]
                    if check_history
                    else "live_unverified",
                    "review_history": history,
                    "manual_check_history": check_history,
                    "thread_review_history": thread_history,
                    "thread_manual_check_history": thread_checks,
                    "target_permalink": target_rows[0].get("permalink")
                    if len(target_rows) == 1
                    else None,
                    "source_usage_scopes": self.source_scope_rows,
                    "profile_claims": _profile_claims(self.profile, candidate.app_id),
                    "draft_history": draft_history,
                    "draft_history_available": self.draft_history_path is not None,
                }
            )
        return {
            "candidates": candidates,
            "context_manifest": self.context_manifest,
            "profile": self.profile.model_dump(mode="json") if self.profile else None,
        }

    def save_review(self, payload: object) -> dict[str, str]:
        data = _object_fields(payload, _REVIEW_FIELDS, "review")
        candidate, thread, target = self._selected(data)
        names = self.messages_by_name(thread)
        evidence_ids = data.get("source_message_evidence_ids")
        if not isinstance(evidence_ids, list) or any(
            not isinstance(item, str) for item in evidence_ids
        ):
            raise ValueError("source_message_evidence_ids must be captured message IDs")
        if len(evidence_ids) != len(set(evidence_ids)) or any(
            item not in names or len(names[item]) != 1 for item in evidence_ids
        ):
            raise ValueError(
                "source_message_evidence_ids must identify unique captured messages in this thread"
            )
        report = self.thread_reports[thread]
        context_complete = bool(
            self.context_manifest.get("scan_complete")
            and report.get("thread_context_complete_for_sources") is True
            and report.get("coverage_status") == "complete_for_scanned_sources"
            and report.get("root_missing") is False
            and any(row["fullname"] == thread for row in self.threads[thread])
            and len(names.get(candidate.reply_target_fullname, [])) == 1
            and not any(len(rows) > 1 for rows in names.values())
        )
        root = next((row for row in self.threads[thread] if row["fullname"] == thread), None)
        root_removed = root is not None and _historically_removed(root)
        selected_target_is_unique = len(names.get(target, [])) == 1
        if data.get("verdict") == "provisional_candidate":
            if not selected_target_is_unique:
                raise ValueError("provisional_candidate requires a unique captured reply target")
            if not evidence_ids:
                raise ValueError("provisional_candidate requires captured message evidence")
            if not context_complete or root_removed:
                raise ValueError(
                    "provisional_candidate is blocked by incomplete coverage or historically "
                    "removed root; explicitly choose needs_context or reject"
                )
        record = EngagementReview.model_validate(
            {
                **data,
                "schema_version": 1,
                "app_id": candidate.app_id,
                "thread_fullname": thread,
                "reply_target_fullname": target,
                "bundle_sha256": self.context_sha256,
                "reviewed_at": datetime.now(UTC),
                "context_complete": context_complete,
            }
        )
        with self._lock:
            self._validate_journals()
            append_review(
                self.review_journal_path,
                record,
                profile=self.profile,
                expected_bundle_sha256=self.context_sha256,
            )
        return {
            "app_id": candidate.app_id,
            "thread_fullname": thread,
            "reply_target_fullname": target,
        }

    def save_manual_check(self, payload: object) -> dict[str, str]:
        data = _object_fields(payload, _MANUAL_FIELDS, "manual-check")
        if "live_status" not in data:
            raise ValueError("live_status must be an explicit human observation")
        candidate, thread, target = self._selected(data)
        names = self.messages_by_name(thread)
        if len(names.get(target, [])) != 1:
            raise ValueError("manual-check target must be a unique captured message in this thread")
        captured_permalink = names[target][0].get("permalink")
        supplied_permalink = data.get("permalink")
        if supplied_permalink is not None and supplied_permalink != captured_permalink:
            raise ValueError("manual-check permalink must match the selected captured target")
        record = ManualCheck.model_validate(
            {
                **data,
                "schema_version": 1,
                "app_id": candidate.app_id,
                "thread_fullname": thread,
                "reply_target_fullname": target,
                "bundle_sha256": self.context_sha256,
                "observed_at": datetime.now(UTC),
            }
        )
        with self._lock:
            self._validate_journals()
            append_manual_check(
                self.manual_check_journal_path,
                record,
                expected_bundle_sha256=self.context_sha256,
            )
        return {
            "app_id": candidate.app_id,
            "thread_fullname": thread,
            "reply_target_fullname": target,
        }

    def _selected(self, payload: dict[str, Any]) -> tuple[ShortlistCandidate, str, str]:
        thread = payload.get("thread_fullname")
        target = payload.get("reply_target_fullname")
        app_id = payload.get("app_id")
        if not isinstance(thread, str) or thread not in self.candidates_by_thread:
            raise ValueError("thread_fullname must identify a shortlisted thread")
        candidates = self.candidates_by_thread[thread]
        candidate = next((row for row in candidates if row.app_id == app_id), None)
        if candidate is None:
            raise ValueError("app_id must identify a shortlist candidate for this thread")
        if not isinstance(target, str):
            raise ValueError("reply_target_fullname must be a message fullname")
        target_matches = self.messages_by_name(thread).get(target, [])
        if target != candidate.reply_target_fullname and len(target_matches) != 1:
            raise ValueError(
                "an alternate reply target must identify one unique message in this captured thread"
            )
        return candidate, thread, target


def _object_fields(value: object, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} request must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} request has unknown fields: {', '.join(sorted(unknown))}")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_candidates(path: Path, manifest: ShortlistManifest) -> list[ShortlistCandidate]:
    records = []
    identities = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                candidate = ShortlistCandidate.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(
                    f"invalid shortlist candidate at line {line_number}: {error}"
                ) from error
            if candidate.app_id != manifest.app_id:
                raise ValueError("candidate app_id does not match shortlist manifest")
            identity = (
                candidate.app_id,
                candidate.thread_fullname,
                candidate.reply_target_fullname,
            )
            if identity in identities:
                raise ValueError("shortlist contains duplicate candidate identities")
            identities.add(identity)
            records.append(candidate)
    if not records or len(records) != manifest.candidate_count:
        raise ValueError("shortlist candidate count does not match manifest or is empty")
    return records


def _read_context(path: Path, expected_threads: set[str]) -> dict[str, list[dict[str, Any]]]:
    if path.stat().st_size > _MAX_CONTEXT_COMPRESSED_BYTES:
        raise ValueError("compressed context bundle exceeds configured limit")
    threads: dict[str, list[dict[str, Any]]] = {thread: [] for thread in expected_threads}
    total_bytes = 0
    message_count = 0
    buffer = bytearray()

    def consume(line: bytes, line_number: int) -> None:
        nonlocal message_count
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid context JSONL row {line_number}") from error
        thread = row.get("thread_fullname") if isinstance(row, dict) else None
        fullname = row.get("fullname") if isinstance(row, dict) else None
        if (
            not isinstance(thread, str)
            or thread not in expected_threads
            or not isinstance(fullname, str)
        ):
            raise ValueError(f"unexpected context identity at row {line_number}")
        message_count += 1
        if message_count > _MAX_CONTEXT_MESSAGES:
            raise ValueError("context bundle exceeds configured message limit")
        threads[thread].append(row)

    line_number = 0
    try:
        with path.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                while chunk := reader.read(_CONTEXT_READ_CHUNK_BYTES):
                    total_bytes += len(chunk)
                    if total_bytes > _MAX_CONTEXT_DECOMPRESSED_BYTES:
                        raise ValueError("decompressed context bundle exceeds configured limit")
                    buffer.extend(chunk)
                    while (newline := buffer.find(b"\n")) >= 0:
                        if newline > _MAX_CONTEXT_ROW_BYTES:
                            raise ValueError("context bundle contains an oversized JSONL row")
                        line_number += 1
                        consume(bytes(buffer[:newline]), line_number)
                        del buffer[: newline + 1]
                    if len(buffer) > _MAX_CONTEXT_ROW_BYTES:
                        raise ValueError("context bundle contains an oversized JSONL row")
    except zstandard.ZstdError as error:
        raise ValueError(f"invalid compressed context bundle: {error}") from error
    if buffer:
        line_number += 1
        consume(bytes(buffer), line_number)
    for thread, rows in threads.items():
        if len({(row["fullname"], row.get("source_revision_id")) for row in rows}) != len(rows):
            raise ValueError(f"duplicate context message revision for thread {thread}")
    return threads


def _read_thread_reports(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports = manifest.get("threads")
    if not isinstance(reports, list):
        raise ValueError("context manifest must contain a thread report list")
    indexed = {}
    for report in reports:
        if not isinstance(report, dict) or not isinstance(report.get("thread_fullname"), str):
            raise ValueError("context manifest contains malformed thread report")
        if report["thread_fullname"] in indexed:
            raise ValueError("context manifest contains duplicate thread reports")
        indexed[report["thread_fullname"]] = report
    return indexed


def _read_source_scopes(path: Path) -> dict[str, str | None]:
    data = _read_json(path, "source registry")
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source registry must contain a sources list")
    scopes: dict[str, str | None] = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
            raise ValueError("source registry contains malformed source")
        if source["source_id"] in scopes:
            raise ValueError("source registry contains duplicate source IDs")
        scopes[source["source_id"]] = source.get("usage_scope")
    return scopes


def _load_profile(path: Path) -> ProductProfile:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ProductProfile.model_validate(loaded)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"invalid product profile: {error}") from error


def _profile_claims(profile: ProductProfile | None, app_id: str) -> list[dict[str, Any]]:
    if profile is None or profile.app_id != app_id:
        return []
    return [claim.model_dump(mode="json") for claim in profile.claims]


def _historically_removed(message: dict[str, Any]) -> bool:
    status = message.get("archived_status")
    if not isinstance(status, dict):
        return False
    return any(
        status.get(field) not in (None, "", False)
        for field in (
            "removed_by_category",
            "removed_by",
            "banned_by",
            "deleted",
            "removed",
            "is_removed",
        )
    ) or message.get("raw_body", "").strip().casefold() in {"[removed]", "[deleted]"}


def _parent_chain(target: str, messages: dict[str, list[dict[str, Any]]], thread: str) -> list[str]:
    chain: list[str] = []
    current = target
    visited = set()
    while current not in visited:
        visited.add(current)
        chain.append(current)
        row = messages.get(current, [None])[0]
        if row is None:
            break
        parent = row.get("parent_fullname")
        if not parent or parent == thread:
            if thread not in chain:
                chain.append(thread)
            break
        current = parent
    return chain


def _create_private_journals(*paths: Path) -> None:
    created: list[Path] = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
            created.append(path)
    except OSError as error:
        for path in created:
            path.unlink(missing_ok=True)
        raise ValueError("journal paths must be fresh files and writable") from error


_PAGE_HTML = Path(__file__).with_name("workspace.html").read_text(encoding="utf-8")
