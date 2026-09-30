from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import zstandard

import reddit_search.engagement.workspace as workspace
from reddit_search.contracts import ProductClaim, ProductProfile
from reddit_search.engagement.context import build_thread_context
from reddit_search.engagement.review import read_manual_checks, read_reviews
from reddit_search.engagement.workspace import create_engagement_server


def test_engagement_workspace_serves_local_context_and_appends_target_revisions(
    tmp_path: Path,
) -> None:
    candidates, shortlist_manifest, registry, context, context_manifest = _fixture(tmp_path)
    review_path = tmp_path / "private" / "reviews.jsonl"
    manual_path = tmp_path / "private" / "manual.jsonl"
    server = create_engagement_server(
        candidates_path=candidates,
        shortlist_manifest_path=shortlist_manifest,
        context_path=context,
        context_manifest_path=context_manifest,
        registry_path=registry,
        review_journal_path=review_path,
        manual_check_journal_path=manual_path,
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = server.engagement_server_report["url"]

        bootstrap = json.loads(_get(url + "/api/bootstrap"))
        candidate = bootstrap["candidates"][0]
        assert [row["fullname"] for row in candidate["messages"]] == ["t3_thread", "t1_target"]
        assert [row["provenance"] for row in candidate["messages"]] == [
            [{"source_id": "synthetic-submissions", "line_number": 1}],
            [{"source_id": "synthetic-comments", "line_number": 1}],
        ]
        assert candidate["parent_chain"] == ["t1_target", "t3_thread"]
        assert candidate["source_usage_scopes"] == [
            {
                "source_id": "synthetic-submissions",
                "usage_scope": "user_provided_local_research_no_external_transmission",
                "scope_warning": None,
            },
            {
                "source_id": "synthetic-comments",
                "usage_scope": "user_provided_local_research_no_external_transmission",
                "scope_warning": None,
            },
        ]
        assert candidate["archive_verdict"] == "unreviewed"
        assert candidate["live_status"] == "live_unverified"
        assert candidate["manual_check_history"] == []
        assert candidate["draft_history_available"] is False
        assert candidate["draft_history"] == []

        payload = {
            "app_id": "synthetic-fixture-app",
            "thread_fullname": "t3_thread",
            "reply_target_fullname": "t1_target",
            "source_message_evidence_ids": ["t3_thread", "t1_target"],
            "product_fit": "not_evaluated",
            "supporting_claim_ids": [],
            "reviewer": "fixture reviewer",
            "verdict": "needs_context",
            "reason": "Fixture decision one.",
            "author_need": "first_person_relevant",
            "answered_in_available_context": "yes",
            "reason_codes": ["need_already_answered"],
        }
        rejected = _post(url + "/api/review", payload, origin="http://evil.example")
        assert rejected[0] == 403
        assert review_path.read_text() == ""
        missing_answer = dict(payload)
        missing_answer.pop("answered_in_available_context")
        rejected_missing_answer = _post(url + "/api/review", missing_answer, origin=url)
        assert rejected_missing_answer[0] == 400
        assert b"answered_in_available_context" in rejected_missing_answer[1]
        invalid_provisional = {
            **payload,
            "verdict": "provisional_candidate",
            "author_need": "unclear",
            "answered_in_available_context": "no",
        }
        rejected_provisional = _post(url + "/api/review", invalid_provisional, origin=url)
        assert rejected_provisional[0] == 400
        assert b"first_person_relevant" in rejected_provisional[1]
        assert review_path.read_text() == ""

        assert _post(url + "/api/review", payload, origin=url)[0] == 200
        revised = {
            **payload,
            "verdict": "reject",
            "reason": "Fixture decision two.",
            "author_need": "tester_or_self_promotion",
            "answered_in_available_context": "unknown",
            "reason_codes": ["self_promotion_or_tester_request"],
        }
        assert _post(url + "/api/review", revised, origin=url)[0] == 200
        another_target = {
            **payload,
            "reply_target_fullname": "t3_thread",
            "source_message_evidence_ids": ["t3_thread"],
            "verdict": "needs_context",
            "reason": "Separate root-target decision.",
            "answered_in_available_context": "unknown",
            "reason_codes": ["context_incomplete"],
        }
        assert _post(url + "/api/review", another_target, origin=url)[0] == 200

        records = read_reviews(review_path)
        assert len(records) == 3
        assert [(record.reply_target_fullname, record.verdict) for record in records] == [
            ("t1_target", "needs_context"),
            ("t1_target", "reject"),
            ("t3_thread", "needs_context"),
        ]
        assert records[1].author_need == "tester_or_self_promotion"
        assert records[1].verdict == "reject"

        manual = {
            "app_id": "synthetic-fixture-app",
            "thread_fullname": "t3_thread",
            "reply_target_fullname": "t1_target",
            "reviewer": "fixture reviewer",
            "live_status": "live_unverified",
            "permalink": "/r/personalfinance/comments/thread/_/target/",
            "observation": "No current-page check performed.",
            "rule_evidence": [],
            "final_human_decision": None,
        }
        missing_live_status = dict(manual)
        missing_live_status.pop("live_status")
        assert _post(url + "/api/manual-check", missing_live_status, origin=url)[0] == 400
        assert manual_path.read_text() == ""
        assert _post(url + "/api/manual-check", manual, origin=url)[0] == 200
        check_records = read_manual_checks(manual_path)
        assert len(check_records) == 1
        assert check_records[0].live_status == "live_unverified"
        refreshed = json.loads(_get(url + "/api/bootstrap"))["candidates"][0]
        assert refreshed["archive_verdict"] == "reject"
        assert refreshed["live_status"] == "live_unverified"
        assert refreshed["manual_check_history"] == [
            {
                "schema_version": 1,
                "app_id": "synthetic-fixture-app",
                "thread_fullname": "t3_thread",
                "reply_target_fullname": "t1_target",
                "bundle_sha256": check_records[0].bundle_sha256,
                "reviewer": "fixture reviewer",
                "observed_at": check_records[0].observed_at.isoformat().replace("+00:00", "Z"),
                "live_status": "live_unverified",
                "permalink": "/r/personalfinance/comments/thread/_/target/",
                "observation": "No current-page check performed.",
                "rule_evidence": [],
                "final_human_decision": None,
            }
        ]
        assert review_path.stat().st_mode & 0o777 == 0o600
        assert manual_path.stat().st_mode & 0o777 == 0o600
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_archive_notes_render_payload_and_do_not_block_provisional_decisions(
    tmp_path: Path,
) -> None:
    for state in ("notes", "gap"):
        case_dir = tmp_path / state
        case_dir.mkdir()
        candidates, shortlist, registry, context, context_manifest = _fixture(case_dir)
        manifest = json.loads(context_manifest.read_text(encoding="utf-8"))
        report = manifest["threads"][0]
        if state == "notes":
            report["archive_content_notes"] = [
                "t1_other: message text is archived as removed/deleted"
            ]
            report["archive_content_counts"] = {"historical_removal": 1}
        else:
            report["coverage_status"] = "needs_context"
            report["thread_context_complete_for_sources"] = False
            report["coverage_reasons"] = ["missing_parent: t1_missing"]
            report["coverage_reason_counts"] = {"missing_parent": 1}
        context_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        profile_path = case_dir / "profile.json"
        profile_path.write_text(
            ProductProfile(
                app_id="synthetic-fixture-app",
                profile_version=1,
                verified_at="2026-09-30T00:00:00Z",
                verification_owner="synthetic fixture owner",
                verification_status="verified",
                synthetic=False,
                capabilities=[
                    ProductClaim(
                        claim_id="verified-claim",
                        status="verified",
                        evidence_ref="synthetic://fixture/evidence",
                        statement="Synthetic verified fixture claim.",
                        version_scope="synthetic fixture",
                        constraints=["not a real product claim"],
                    )
                ],
            ).model_dump_json(),
            encoding="utf-8",
        )

        server = create_engagement_server(
            candidates_path=candidates,
            shortlist_manifest_path=shortlist,
            context_path=context,
            context_manifest_path=context_manifest,
            registry_path=registry,
            review_journal_path=case_dir / "review.jsonl",
            manual_check_journal_path=case_dir / "manual.jsonl",
            port=0,
            profile_path=profile_path,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = server.engagement_server_report["url"]
            candidate = json.loads(_get(url + "/api/bootstrap"))["candidates"][0]
            if state == "notes":
                assert candidate["archive_content_notes"] == [
                    "t1_other: message text is archived as removed/deleted"
                ]
                assert candidate["coverage_warnings"] == []
            else:
                assert candidate["archive_content_notes"] == []
                assert candidate["coverage_warnings"]

            provisional = {
                "app_id": "synthetic-fixture-app",
                "thread_fullname": "t3_thread",
                "reply_target_fullname": "t1_target",
                "source_message_evidence_ids": ["t1_target"],
                "product_fit": "compatible",
                "supporting_claim_ids": ["verified-claim"],
                "reviewer": "fixture reviewer",
                "verdict": "provisional_candidate",
                "reason": "The need is current and unmet.",
                "author_need": "first_person_relevant",
                "answered_in_available_context": "no",
                "reason_codes": ["author_has_current_need"],
            }
            status, body = _post(url + "/api/review", provisional, origin=url)
            assert status == (200 if state == "notes" else 400), body
            if state == "notes":
                assert read_reviews(case_dir / "review.jsonl")[0].verdict == "provisional_candidate"
            else:
                assert b"explicitly choose needs_context" in body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_incomplete_or_removed_context_requires_explicit_needs_context(
    tmp_path: Path,
) -> None:
    for reason in ("coverage", "removed_root"):
        case_dir = tmp_path / reason
        case_dir.mkdir()
        candidates, shortlist, registry, context, context_manifest = _fixture(case_dir)
        manifest = json.loads(context_manifest.read_text(encoding="utf-8"))
        if reason == "coverage":
            manifest["threads"][0]["coverage_status"] = "needs_context"
            manifest["threads"][0]["thread_context_complete_for_sources"] = False
        else:
            rows = _read_compressed_jsonl(context)
            root = next(row for row in rows if row["fullname"] == "t3_thread")
            root["archived_status"] = {"deleted": True}
            _write_compressed_jsonl(context, rows)
            manifest["output_sha256"] = hashlib.sha256(context.read_bytes()).hexdigest()
        context_manifest.write_text(json.dumps(manifest), encoding="utf-8")

        server = create_engagement_server(
            candidates_path=candidates,
            shortlist_manifest_path=shortlist,
            context_path=context,
            context_manifest_path=context_manifest,
            registry_path=registry,
            review_journal_path=case_dir / "review.jsonl",
            manual_check_journal_path=case_dir / "manual.jsonl",
            port=0,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = server.engagement_server_report["url"]
            provisional = {
                "app_id": "synthetic-fixture-app",
                "thread_fullname": "t3_thread",
                "reply_target_fullname": "t1_target",
                "source_message_evidence_ids": ["t3_thread", "t1_target"],
                "product_fit": "not_evaluated",
                "supporting_claim_ids": [],
                "reviewer": "fixture reviewer",
                "verdict": "provisional_candidate",
                "reason": "Must not pass incomplete archive context.",
                "author_need": "first_person_relevant",
                "answered_in_available_context": "no",
                "reason_codes": ["author_has_current_need"],
            }
            blocked = _post(url + "/api/review", provisional, origin=url)
            assert blocked[0] == 400
            assert b"explicitly choose needs_context" in blocked[1]

            explicit = {**provisional, "verdict": "needs_context"}
            assert _post(url + "/api/review", explicit, origin=url)[0] == 200
            record = read_reviews(case_dir / "review.jsonl")[0]
            assert record.verdict == "needs_context"
            assert record.context_complete is (reason == "removed_root")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_incomplete_context_candidates_remain_reviewable(tmp_path: Path) -> None:
    for condition in ("missing_target", "ambiguous_target", "missing_root", "empty"):
        case_dir = tmp_path / condition
        case_dir.mkdir()
        candidates, shortlist, registry, context, context_manifest = _fixture(case_dir)
        rows = _read_compressed_jsonl(context)
        manifest = json.loads(context_manifest.read_text(encoding="utf-8"))
        report = manifest["threads"][0]
        if condition == "missing_target":
            rows = [row for row in rows if row["fullname"] != "t1_target"]
            report["missing_reply_targets"] = ["t1_target"]
        elif condition == "ambiguous_target":
            duplicate = dict(next(row for row in rows if row["fullname"] == "t1_target"))
            duplicate["source_revision_id"] = "conflicting-target-revision"
            rows.append(duplicate)
        elif condition == "missing_root":
            rows = [row for row in rows if row["fullname"] != "t3_thread"]
            report["root_missing"] = True
            report["missing_parents"] = ["t3_thread"]
        else:
            rows = []
            report["root_missing"] = True
            report["missing_reply_targets"] = ["t1_target"]
        _write_compressed_jsonl(context, rows)

        report["coverage_status"] = "needs_context"
        report["thread_context_complete_for_sources"] = False
        manifest["output_sha256"] = hashlib.sha256(context.read_bytes()).hexdigest()
        context_manifest.write_text(json.dumps(manifest), encoding="utf-8")

        server = create_engagement_server(
            candidates_path=candidates,
            shortlist_manifest_path=shortlist,
            context_path=context,
            context_manifest_path=context_manifest,
            registry_path=registry,
            review_journal_path=case_dir / "reviews.jsonl",
            manual_check_journal_path=case_dir / "manual.jsonl",
            port=0,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = server.engagement_server_report["url"]
            candidate = json.loads(_get(url + "/api/bootstrap"))["candidates"][0]
            assert candidate["context_complete"] is False
            assert candidate["coverage_warnings"]
            if condition in {"missing_target", "ambiguous_target", "empty"}:
                assert candidate["target_warning"]
            if condition == "ambiguous_target":
                assert any(
                    "Conflicting captured revisions" in warning
                    for warning in candidate["coverage_warnings"]
                )
            if condition in {"missing_root", "empty"}:
                assert any(
                    "Submission root is missing" in warning
                    for warning in candidate["coverage_warnings"]
                )

            names = [row["fullname"] for row in candidate["messages"]]
            evidence = next(
                (fullname for fullname in names if names.count(fullname) == 1),
                None,
            )
            provisional = {
                "app_id": "synthetic-fixture-app",
                "thread_fullname": "t3_thread",
                "reply_target_fullname": "t1_target",
                "source_message_evidence_ids": [evidence] if evidence else [],
                "product_fit": "not_evaluated",
                "supporting_claim_ids": [],
                "reviewer": "fixture reviewer",
                "verdict": "provisional_candidate",
                "reason": "Must remain blocked without complete identity/context.",
                "author_need": "first_person_relevant",
                "answered_in_available_context": "no",
                "reason_codes": ["context_incomplete"],
            }
            assert _post(url + "/api/review", provisional, origin=url)[0] == 400
            explicit = {**provisional, "verdict": "needs_context"}
            assert _post(url + "/api/review", explicit, origin=url)[0] == 200
            record = read_reviews(case_dir / "reviews.jsonl")[0]
            assert record.verdict == "needs_context"
            assert record.source_message_evidence_ids == ([evidence] if evidence else [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_workspace_context_reader_rejects_resource_limit_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "context.jsonl.zst"
    rows = [
        {"thread_fullname": "t3_thread", "fullname": f"t1_{index}", "raw_body": "x" * 128}
        for index in range(2)
    ]
    _write_compressed_jsonl(path, rows)

    monkeypatch.setattr(workspace, "_MAX_CONTEXT_COMPRESSED_BYTES", path.stat().st_size - 1)
    with pytest.raises(ValueError, match="compressed context bundle exceeds"):
        workspace._read_context(path, {"t3_thread"})

    monkeypatch.setattr(workspace, "_MAX_CONTEXT_COMPRESSED_BYTES", path.stat().st_size)
    monkeypatch.setattr(workspace, "_MAX_CONTEXT_DECOMPRESSED_BYTES", 128)
    with pytest.raises(ValueError, match="decompressed context bundle exceeds"):
        workspace._read_context(path, {"t3_thread"})

    monkeypatch.setattr(workspace, "_MAX_CONTEXT_DECOMPRESSED_BYTES", 4096)
    monkeypatch.setattr(workspace, "_MAX_CONTEXT_ROW_BYTES", 128)
    with pytest.raises(ValueError, match="oversized JSONL row"):
        workspace._read_context(path, {"t3_thread"})

    monkeypatch.setattr(workspace, "_MAX_CONTEXT_ROW_BYTES", 4096)
    monkeypatch.setattr(workspace, "_MAX_CONTEXT_MESSAGES", 1)
    with pytest.raises(ValueError, match="configured message limit"):
        workspace._read_context(path, {"t3_thread"})


def _read_compressed_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            return [json.loads(line) for line in reader.read().decode().splitlines()]


def _write_compressed_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(data))


def _fixture(directory: Path) -> tuple[Path, Path, Path, Path, Path]:
    submissions = directory / "submissions.jsonl.zst"
    comments = directory / "comments.jsonl.zst"
    _archive(
        submissions,
        {
            "name": "t3_thread",
            "title": "Need a private offline budget tracker",
            "selftext": "I need a way to plan a household budget without sending data online.",
            "subreddit": "personalfinance",
            "created_utc": 1,
            "permalink": "/r/personalfinance/comments/thread/",
        },
    )
    _archive(
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
        "reason_for_inclusion": "Synthetic local workspace acceptance fixture.",
    }
    candidates.write_text(json.dumps(candidate, separators=(",", ":")) + "\n", encoding="utf-8")
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
        registry,
        output / "threads.jsonl.zst",
        output / "context_manifest.json",
    )


def _archive(path: Path, record: dict[str, object]) -> None:
    path.write_bytes(zstandard.ZstdCompressor().compress((json.dumps(record) + "\n").encode()))


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"Host": url.split("//", 1)[1].split("/", 1)[0]})
    with urllib.request.urlopen(request) as response:
        return response.read()


def _post(url: str, payload: dict[str, object], *, origin: str) -> tuple[int, bytes]:
    headers = {
        "Content-Type": "application/json",
        "Origin": origin,
        "Host": url.split("//", 1)[1].split("/", 1)[0],
    }
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
