from __future__ import annotations

import hashlib
import json
import selectors
import signal
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import zstandard

from reddit_search.engagement.context import build_thread_context
from reddit_search.engagement.drafting import _append_record
from reddit_search.engagement.review import (
    EngagementReview,
    ManualCheck,
    append_manual_check,
    append_review,
)
from reddit_search.ingest.state import file_sha256


def test_engagement_serve_subprocess_bootstrap_review_and_origin_guard(
    tmp_path: Path,
) -> None:
    candidates, shortlist_manifest, context_bundle, context_manifest, registry = _inputs(tmp_path)
    decisions = tmp_path / "decisions.jsonl"
    manual_checks = tmp_path / "manual-checks.jsonl"
    process = subprocess.Popen(
        _serve_command(
            (candidates, shortlist_manifest, context_bundle, context_manifest, registry),
            decisions,
            manual_checks,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=15), "engagement serve did not print its address"
            launch = process.stdout.readline().strip()
        if not launch:
            assert process.stderr is not None
            raise AssertionError(process.stderr.read())

        parsed = urllib.parse.urlparse(launch)
        assert parsed.scheme == "http"
        assert parsed.hostname == "127.0.0.1"
        assert parsed.port is not None
        assert launch == f"http://127.0.0.1:{parsed.port}"
        with urllib.request.urlopen(f"{launch}/api/bootstrap", timeout=5) as response:
            bootstrap = json.loads(response.read())
        candidate = bootstrap["candidates"][0]
        assert candidate["app_id"] == "synthetic-fixture-app"
        assert candidate["thread_fullname"] == "t3_thread"

        review = {
            "app_id": "synthetic-fixture-app",
            "thread_fullname": "t3_thread",
            "reply_target_fullname": "t1_target",
            "source_message_evidence_ids": ["t1_target"],
            "product_fit": "not_evaluated",
            "supporting_claim_ids": [],
            "reviewer": "acceptance reviewer",
            "verdict": "needs_context",
            "reason": "Local subprocess HTTP review acceptance check.",
            "author_need": "unclear",
            "answered_in_available_context": "unknown",
            "reason_codes": ["context_incomplete"],
        }
        status, saved = _post(f"{launch}/api/review", review, origin=launch)
        assert status == 200
        assert json.loads(saved)["saved"] is True
        assert len(decisions.read_text(encoding="utf-8").splitlines()) == 1

        status, denied = _post(f"{launch}/api/review", review, origin="http://attacker.invalid")
        assert status == 403
        assert b"cross-origin request refused" in denied
        assert len(decisions.read_text(encoding="utf-8").splitlines()) == 1
        assert decisions.stat().st_mode & 0o777 == 0o600
        assert manual_checks.stat().st_mode & 0o777 == 0o600
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
    assert process.returncode == 0


def test_engagement_serve_resumes_journals_and_displays_verified_draft_versions(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    _, _, context_bundle, context_manifest, _ = inputs
    decisions = tmp_path / "decisions.jsonl"
    manual_checks = tmp_path / "manual-checks.jsonl"
    draft_history = tmp_path / "draft-history.jsonl"
    bundle_sha256 = file_sha256(context_bundle)
    manifest_sha256 = file_sha256(context_manifest)
    review = EngagementReview(
        app_id="synthetic-fixture-app",
        thread_fullname="t3_thread",
        reply_target_fullname="t1_target",
        bundle_sha256=bundle_sha256,
        source_message_evidence_ids=["t1_target"],
        product_fit="not_evaluated",
        supporting_claim_ids=[],
        reviewer="earlier reviewer",
        reviewed_at=datetime.now(UTC),
        verdict="needs_context",
        reason="Earlier archive review.",
        author_need="unclear",
        answered_in_available_context="unknown",
        context_complete=True,
        reason_codes=["context_incomplete"],
    )
    append_review(decisions, review, expected_bundle_sha256=bundle_sha256)
    manual = ManualCheck(
        app_id="synthetic-fixture-app",
        thread_fullname="t3_thread",
        reply_target_fullname="t1_target",
        bundle_sha256=bundle_sha256,
        reviewer="earlier reviewer",
        observed_at=datetime.now(UTC),
        live_status="human_checked_replyable",
        permalink="/r/personalfinance/comments/thread/_/target/",
        observation="Visible replyability: reply button present; relevant new answers: none.",
        rule_evidence=["Rule evidence: no unsolicited promotional links."],
        final_human_decision="defer",
    )
    append_manual_check(manual_checks, manual, expected_bundle_sha256=bundle_sha256)
    decisions.chmod(0o600)
    manual_checks.chmod(0o600)
    _append_record(
        draft_history,
        {
            "record_type": "machine",
            "created_at": datetime.now(UTC),
            "app_id": review.app_id,
            "thread_fullname": review.thread_fullname,
            "reply_target_fullname": review.reply_target_fullname,
            "outcome": "abstain",
            "context_bundle_sha256": bundle_sha256,
            "context_manifest_sha256": manifest_sha256,
            "review_sha256": hashlib.sha256(review.model_dump_json().encode()).hexdigest(),
            "review_journal_sha256": file_sha256(decisions),
            "product_profile_sha256": None,
            "product_profile_version": None,
            "qualification_prompt_version": "qualification-test-v4",
            "qualification_prompt_sha256": "a" * 64,
            "draft_prompt_version": "draft-test-v7",
            "draft_prompt_sha256": "b" * 64,
            "model_version": "local-test-model-v3",
            "model_sha256": None,
            "qualification": None,
            "used_message_ids": [],
            "used_claim_ids": [],
            "claim_statements": {},
            "operator_affiliation": "",
            "text": "",
            "abstention_reasons": ["Synthetic acceptance abstention."],
            "human_editor": None,
            "human_edit_valid": None,
            "human_edit_issues": [],
        },
    )
    process = subprocess.Popen(
        _serve_command(inputs, decisions, manual_checks, draft_history=draft_history),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=15)
            launch = process.stdout.readline().strip()
        with urllib.request.urlopen(f"{launch}/api/bootstrap", timeout=5) as response:
            candidate = json.loads(response.read())["candidates"][0]
        assert candidate["archive_verdict"] == "needs_context"
        assert candidate["live_status"] == "human_checked_replyable"
        assert len(candidate["manual_check_history"]) == 1
        manual_history = candidate["manual_check_history"][0]
        assert (
            manual_history["observed_at"]
            == manual.observed_at.isoformat().replace("+00:00", "Z")
        )
        assert manual_history["reviewer"] == "earlier reviewer"
        assert manual_history["live_status"] == "human_checked_replyable"
        assert manual_history["permalink"] == "/r/personalfinance/comments/thread/_/target/"
        assert (
            manual_history["observation"]
            == "Visible replyability: reply button present; relevant new answers: none."
        )
        assert manual_history["rule_evidence"] == [
            "Rule evidence: no unsolicited promotional links."
        ]
        assert manual_history["final_human_decision"] == "defer"
        latest_draft = candidate["draft_history"][0]
        assert latest_draft["qualification_prompt_version"] == "qualification-test-v4"
        assert latest_draft["draft_prompt_version"] == "draft-test-v7"
        assert latest_draft["model_version"] == "local-test-model-v3"
        assert latest_draft["validation_status"] == "current_abstention"
        assert latest_draft["validation_message"] == "Synthetic acceptance abstention."
        assert "text" not in latest_draft
        review_payload = {
            "app_id": "synthetic-fixture-app",
            "thread_fullname": "t3_thread",
            "reply_target_fullname": "t1_target",
            "source_message_evidence_ids": ["t1_target"],
            "product_fit": "not_evaluated",
            "supporting_claim_ids": [],
            "reviewer": "resuming reviewer",
            "verdict": "needs_context",
            "reason": "Resume the existing archive review.",
            "author_need": "unclear",
            "answered_in_available_context": "unknown",
            "reason_codes": ["context_incomplete"],
        }
        status, saved = _post(f"{launch}/api/review", review_payload, origin=launch)
        assert status == 200
        assert json.loads(saved)["saved"] is True
        with urllib.request.urlopen(f"{launch}/api/bootstrap", timeout=5) as response:
            refreshed = json.loads(response.read())["candidates"][0]
        assert refreshed["draft_history"][0]["validation_status"] == "stale"
        assert refreshed["draft_history"][0]["display_text"] == ""
        assert "text" not in refreshed["draft_history"][0]
        manual_payload = {
            "app_id": "synthetic-fixture-app",
            "thread_fullname": "t3_thread",
            "reply_target_fullname": "t1_target",
            "reviewer": "resuming reviewer",
            "live_status": "live_unverified",
            "permalink": "/r/personalfinance/comments/thread/_/target/",
            "observation": "No current-page check performed.",
            "rule_evidence": [],
            "final_human_decision": None,
        }
        status, saved = _post(f"{launch}/api/manual-check", manual_payload, origin=launch)
        assert status == 200
        assert json.loads(saved)["saved"] is True
        assert len(manual_checks.read_text(encoding="utf-8").splitlines()) == 2
        assert len(decisions.read_text(encoding="utf-8").splitlines()) == 2
        assert decisions.stat().st_mode & 0o777 == 0o600
        assert manual_checks.stat().st_mode & 0o777 == 0o600
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
    assert process.returncode == 0
    reviews_before = decisions.read_bytes()
    checks_before = manual_checks.read_bytes()
    tampered = json.loads(draft_history.read_text(encoding="utf-8"))
    tampered["model_version"] = "tampered-after-write"
    draft_history.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    result = subprocess.run(
        _serve_command(inputs, decisions, manual_checks, draft_history=draft_history),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert "draft history hash mismatch" in result.stderr
    assert decisions.read_bytes() == reviews_before
    assert manual_checks.read_bytes() == checks_before


def test_engagement_serve_rejects_stale_journal_without_modifying_it(
    tmp_path: Path,
) -> None:
    candidates, shortlist_manifest, context_bundle, context_manifest, registry = _inputs(tmp_path)
    decisions = tmp_path / "decisions.jsonl"
    manual_checks = tmp_path / "manual-checks.jsonl"
    bundle_sha256 = file_sha256(context_bundle)
    stale_bundle_sha256 = "0" * 64 if bundle_sha256 != "0" * 64 else "1" * 64
    append_review(
        decisions,
        EngagementReview(
            app_id="synthetic-fixture-app",
            thread_fullname="t3_thread",
            reply_target_fullname="t1_target",
            bundle_sha256=stale_bundle_sha256,
            source_message_evidence_ids=["t1_target"],
            product_fit="not_evaluated",
            supporting_claim_ids=[],
            reviewer="stale reviewer",
            reviewed_at=datetime.now(UTC),
            verdict="needs_context",
            reason="This review belongs to another bundle.",
            author_need="unclear",
            answered_in_available_context="unknown",
            context_complete=True,
            reason_codes=["context_incomplete"],
        ),
        expected_bundle_sha256=stale_bundle_sha256,
    )
    decisions.chmod(0o600)
    original = decisions.read_bytes()
    result = subprocess.run(
        _serve_command(
            (candidates, shortlist_manifest, context_bundle, context_manifest, registry),
            decisions,
            manual_checks,
        ),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert "bundle hash mismatch" in result.stderr
    assert decisions.read_bytes() == original
    assert not manual_checks.exists()


def _serve_command(
    inputs: tuple[Path, Path, Path, Path, Path],
    decisions: Path,
    manual_checks: Path,
    *,
    draft_history: Path | None = None,
) -> list[str]:
    candidates, shortlist_manifest, context_bundle, context_manifest, registry = inputs
    command = [
        sys.executable,
        "-m",
        "reddit_search",
        "engagement",
        "serve",
        "--candidates",
        str(candidates),
        "--shortlist-manifest",
        str(shortlist_manifest),
        "--context-bundle",
        str(context_bundle),
        "--context-manifest",
        str(context_manifest),
        "--registry",
        str(registry),
        "--decisions",
        str(decisions),
        "--manual-checks",
        str(manual_checks),
        "--port",
        "0",
    ]
    if draft_history is not None:
        command.extend(["--draft-history", str(draft_history)])
    return command


def _inputs(directory: Path) -> tuple[Path, Path, Path, Path, Path]:
    submissions = directory / "submissions.jsonl.zst"
    comments = directory / "comments.jsonl.zst"
    _write_archive(
        submissions,
        {
            "name": "t3_thread",
            "title": "Need a private offline budget tracker",
            "selftext": "I need an offline budget tool.",
            "subreddit": "personalfinance",
            "created_utc": 1,
            "permalink": "/r/personalfinance/comments/thread/",
        },
    )
    _write_archive(
        comments,
        {
            "name": "t1_target",
            "link_id": "t3_thread",
            "parent_id": "t3_thread",
            "body": "Have you tried a local spreadsheet?",
            "subreddit": "personalfinance",
            "created_utc": 2,
            "permalink": "/r/personalfinance/comments/thread/_/target/",
        },
    )
    registry = directory / "registry.json"
    sources = []
    for source_id, path, kind in (
        ("synthetic-submissions", submissions, "submission"),
        ("synthetic-comments", comments, "comment"),
    ):
        sources.append(
            {
                "source_id": source_id,
                "source_path": str(path),
                "source_kind": kind,
                "declared_month": "2026-08",
                "source_role": "context_only",
                "usage_scope": "user_provided_local_research_no_external_transmission",
                "input_size_bytes": path.stat().st_size,
                "verified_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "status": "complete",
                "invalid_records": 0,
            }
        )
    registry.write_text(json.dumps({"schema_version": 1, "sources": sources}), encoding="utf-8")

    candidates = directory / "candidates.jsonl"
    candidate = {
        "schema_version": 1,
        "app_id": "synthetic-fixture-app",
        "thread_fullname": "t3_thread",
        "reply_target_fullname": "t1_target",
        "source_message_fullname": "t1_target",
        "source_revision_id": "synthetic-target-revision",
        "permalink": "/r/personalfinance/comments/thread/_/target/",
        "subreddit": "personalfinance",
        "scenario_references": ["offline-budget-fixture"],
        "source_references": ["synthetic-comments:1:t1_target:revision-1"],
        "reason_for_inclusion": "Synthetic local subprocess acceptance fixture.",
    }
    candidates.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    shortlist_manifest = directory / "shortlist_manifest.json"
    shortlist_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "engagement_shortlist_manifest",
                "app_id": "synthetic-fixture-app",
                "inputs": {
                    "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
                    "tombstone_sha256": hashlib.sha256(b"").hexdigest(),
                },
                "candidate_count": 1,
                "candidates_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output = directory / "context"
    build_thread_context(candidates, shortlist_manifest, registry, output)
    return (
        candidates,
        shortlist_manifest,
        output / "threads.jsonl.zst",
        output / "context_manifest.json",
        registry,
    )


def _write_archive(path: Path, record: dict[str, object]) -> None:
    data = (json.dumps(record) + "\n").encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(data))


def _post(url: str, payload: dict[str, object], *, origin: str) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Origin": origin},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
