from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import zstandard

from reddit_search.engagement.review import EngagementReview


def _fixture(tmp_path: Path, *, verdict: str, app_id: str) -> dict[str, Path | str]:
    rows = [
        {
            "fullname": "t3_thread",
            "kind": "submission",
            "thread_fullname": "t3_thread",
            "parent_fullname": None,
            "raw_title": "Need a private tracker",
            "raw_body": "Looking for an offline way to organize my records.",
            "archived_status": {},
        },
        {
            "fullname": "t1_target",
            "kind": "comment",
            "thread_fullname": "t3_thread",
            "parent_fullname": "t3_thread",
            "raw_title": "",
            "raw_body": "I need a private way to keep these records organized.",
            "archived_status": {},
        },
    ]
    bundle = tmp_path / "threads.jsonl.zst"
    raw_bundle = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    bundle.write_bytes(zstandard.ZstdCompressor().compress(raw_bundle))
    bundle_sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()

    manifest = {
        "kind": "engagement_thread_context_manifest",
        "schema_version": 1,
        "output_sha256": bundle_sha256,
        "scan_complete": True,
        "sources": [{"source_id": "acceptance-source", "complete": True}],
        "live_status": "unverified",
        "threads": [
            {
                "thread_fullname": "t3_thread",
                "coverage_status": "complete_for_scanned_sources",
                "thread_context_complete_for_sources": True,
                "root_missing": False,
                "missing_parents": [],
                "missing_reply_targets": [],
                "coverage_reason_counts": {},
            }
        ],
    }
    manifest_path = tmp_path / "context_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    provisional = verdict == "provisional_candidate"
    review = EngagementReview(
        app_id=app_id,
        thread_fullname="t3_thread",
        reply_target_fullname="t1_target",
        bundle_sha256=bundle_sha256,
        source_message_evidence_ids=["t1_target"],
        product_fit="compatible" if provisional else "not_evaluated",
        supporting_claim_ids=["synthetic.claim"] if provisional else [],
        reviewer="acceptance-reviewer",
        reviewed_at=datetime(2026, 9, 25, tzinfo=UTC),
        verdict=verdict,
        reason="Acceptance fixture review.",
        author_need="first_person_relevant" if provisional else "no_need",
        answered_in_available_context="no" if provisional else "yes",
        context_complete=True,
        reason_codes=["author_has_current_need"] if provisional else ["no_useful_reply"],
    )
    review_path = tmp_path / "reviews.jsonl"
    review_path.write_text(review.model_dump_json() + "\n", encoding="utf-8")
    return {
        "bundle": bundle,
        "manifest": manifest_path,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "review": review_path,
        "history": tmp_path / "draft-history.jsonl",
        "app_id": app_id,
    }


def _invoke(*, fixture: dict[str, Path | str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "engagement",
            "draft",
            "--app-id",
            str(fixture["app_id"]),
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
            "--history",
            str(fixture["history"]),
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def test_draft_cli_appends_negative_sumlyst_abstention_without_profile_or_backend(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, verdict="reject", app_id="sumlyst")

    result = _invoke(fixture=fixture)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "abstain"
    assert report["reasons"]
    assert report["hashes"]["product_profile"] is None
    assert report["hashes"]["model"] is None
    assert "private way to keep these records organized" not in result.stdout
    assert "text" not in report and "draft" not in report
    record = json.loads(fixture["history"].read_text(encoding="utf-8").splitlines()[0])
    assert record["outcome"] == "abstain"
    assert record["model_version"] == "not_invoked"
    assert record["text"] == ""


def test_draft_cli_fails_closed_for_provisional_candidate_without_profile_or_backend(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, verdict="provisional_candidate", app_id="sumlyst")

    result = _invoke(fixture=fixture)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "abstain"
    assert report["reasons"]
    assert report["hashes"]["product_profile"] is None
    assert report["hashes"]["model"] is None
    assert "private way to keep these records organized" not in result.stdout
    assert "text" not in report and "draft" not in report
