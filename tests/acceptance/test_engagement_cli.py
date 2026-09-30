import hashlib
import json
import subprocess
import sys
from pathlib import Path

import zstandard

from reddit_search.ingest.normalize import NormalizedMessage, SourceProvenance
from reddit_search.ingest.shard import serialize_message, write_compressed_jsonl


def test_engagement_shortlist_cli_builds_from_bounded_verified_selection(tmp_path: Path) -> None:
    selection, selection_manifest, registry = _fixture(tmp_path, ["t3_need"])
    output = tmp_path / "shortlist"
    result = _invoke(selection, selection_manifest, registry, output)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["built"] is True
    assert report["candidate_count"] == 1
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert rows[0]["reply_target_fullname"] == "t3_need"
    assert rows[0]["scenario_references"] == ["offline-budget"]
    assert rows[0]["source_references"] == ["local-source:1:t3_need:revision-1"]


def test_engagement_shortlist_cli_rejects_selection_hash_mismatch(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl.zst"
    selection.write_bytes(b"not the manifested selection")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "discovery_manifest",
                "schema_version": 2,
                "run_id": "bounded-test-run",
                "output_sha256": "0" * 64,
                "result": {"scan_complete": True},
                "budget_exhausted": False,
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"sources": []}), encoding="utf-8")
    result = _invoke(selection, manifest, registry, tmp_path / "shortlist")
    assert result.returncode == 2
    assert "selection shard hash does not match its manifest" in result.stderr
    assert not (tmp_path / "shortlist").exists()


def test_engagement_shortlist_cli_applies_target_allowlist(tmp_path: Path) -> None:
    selection, manifest, registry = _fixture(tmp_path, ["t3_one", "t3_two"])
    allowlist = tmp_path / "targets.json"
    allowlist.write_text(json.dumps(["t3_two"]), encoding="utf-8")
    output = tmp_path / "allowlisted"
    result = _invoke(selection, manifest, registry, output, target_allowlist=allowlist)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert [row["reply_target_fullname"] for row in rows] == ["t3_two"]

    unknown = tmp_path / "unknown-targets.json"
    unknown.write_text(json.dumps(["t3_missing"]), encoding="utf-8")
    failed = _invoke(
        selection, manifest, registry, tmp_path / "unknown-output", target_allowlist=unknown
    )
    assert failed.returncode == 2
    assert json.loads(failed.stderr)["built"] is False
    assert not (tmp_path / "unknown-output").exists()


def test_engagement_shortlist_cli_excludes_tombstones_and_rejects_deleted_allowlist(
    tmp_path: Path,
) -> None:
    selection, manifest, registry = _fixture(tmp_path, ["t3_one", "t3_two"])
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t3_one",
                "source_revision_id": "revision-1",
                "reason": "synthetic invalidation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "shortlist"
    result = _invoke(selection, manifest, registry, output, tombstones=tombstones)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert [row["reply_target_fullname"] for row in rows] == ["t3_two"]

    allowlist = tmp_path / "targets.json"
    allowlist.write_text('["t3_one"]', encoding="utf-8")
    rejected_output = tmp_path / "deleted-target"
    rejected = _invoke(
        selection,
        manifest,
        registry,
        rejected_output,
        tombstones=tombstones,
        target_allowlist=allowlist,
    )
    assert rejected.returncode == 2
    assert not rejected_output.exists()


def test_engagement_context_cli_builds_and_fails_closed_for_missing_parent_and_limit(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "complete"
    complete.mkdir()
    candidates, shortlist_manifest, registry = _context_fixture(complete, parent="t3_thread")
    output = complete / "context"
    result = _invoke_context(candidates, shortlist_manifest, registry, output)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["built"] is True
    context_manifest = json.loads((output / "context_manifest.json").read_text())
    assert context_manifest["scan_complete"] is True
    assert context_manifest["threads"][0]["thread_context_complete_for_sources"] is True
    assert (output / "threads.jsonl.zst").is_file()

    missing = tmp_path / "missing-parent"
    missing.mkdir()
    candidates, shortlist_manifest, registry = _context_fixture(missing, parent="t1_parent_missing")
    missing_output = missing / "context"
    missing_result = _invoke_context(candidates, shortlist_manifest, registry, missing_output)
    assert missing_result.returncode == 0, missing_result.stderr
    missing_report = json.loads((missing_output / "context_manifest.json").read_text())["threads"][
        0
    ]
    assert missing_report["thread_context_complete_for_sources"] is False
    assert missing_report["missing_parents"] == ["t1_parent_missing"]
    assert missing_report["coverage_status"] == "needs_context"

    limited = tmp_path / "limited"
    limited.mkdir()
    candidates, shortlist_manifest, registry = _context_fixture(limited, parent="t3_thread")
    limited_output = limited / "context"
    limited_result = _invoke_context(
        candidates,
        shortlist_manifest,
        registry,
        limited_output,
        "--max-messages-per-thread",
        "1",
    )
    assert limited_result.returncode == 0, limited_result.stderr
    limited_report = json.loads((limited_output / "context_manifest.json").read_text())["threads"][
        0
    ]
    assert limited_report["thread_context_complete_for_sources"] is False
    assert limited_report["omitted_messages"] > 0
    assert limited_report["coverage_status"] == "needs_context"


def _context_fixture(directory: Path, *, parent: str) -> tuple[Path, Path, Path]:
    submission = directory / "submissions.jsonl.zst"
    comment = directory / "comments.jsonl.zst"
    _write_archive(
        submission,
        {
            "name": "t3_thread",
            "title": "Need an offline budget tool",
            "selftext": "Looking for one.",
            "subreddit": "personalfinance",
            "created_utc": 1,
            "permalink": "/r/personalfinance/comments/thread/",
        },
    )
    _write_archive(
        comment,
        {
            "name": "t1_target",
            "link_id": "t3_thread",
            "parent_id": parent,
            "body": "Any offline recommendations?",
            "subreddit": "personalfinance",
            "created_utc": 2,
            "permalink": "/r/personalfinance/comments/thread/_/target/",
        },
    )
    registry = directory / "registry.json"
    sources = []
    for source_id, path, kind in (
        ("synthetic-submissions", submission, "submission"),
        ("synthetic-comments", comment, "comment"),
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
        "app_id": "synthetic-app",
        "thread_fullname": "t3_thread",
        "reply_target_fullname": "t1_target",
        "source_message_fullname": "t1_target",
        "source_revision_id": "synthetic-revision",
        "permalink": "/r/personalfinance/comments/thread/_/target/",
        "subreddit": "personalfinance",
        "scenario_references": ["offline-budget"],
        "source_references": ["synthetic-comments:1:t1_target:synthetic-revision"],
        "reason_for_inclusion": "Synthetic context API acceptance fixture.",
    }
    candidates.write_text(json.dumps(candidate, separators=(",", ":")) + "\n", encoding="utf-8")
    shortlist_manifest = directory / "shortlist_manifest.json"
    shortlist_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "engagement_shortlist_manifest",
                "app_id": "synthetic-app",
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
    return candidates, shortlist_manifest, registry


def _write_archive(path: Path, record: dict[str, object]) -> None:
    raw = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))


def _invoke_context(
    candidates: Path,
    shortlist_manifest: Path,
    registry: Path,
    output: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "engagement",
            "context",
            "--candidates",
            str(candidates),
            "--shortlist-manifest",
            str(shortlist_manifest),
            "--registry",
            str(registry),
            "--output",
            str(output),
            *extra_args,
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def _fixture(tmp_path: Path, target_ids: list[str]) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.jsonl"
    source.write_text("local archive source\n", encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "source_id": "local-source",
                        "source_path": str(source),
                        "verified_sha256": source_hash,
                        "input_size_bytes": source.stat().st_size,
                        "status": "complete",
                        "invalid_records": 0,
                        "usage_scope": "user_provided_local_research_no_external_transmission",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.jsonl.zst"
    messages = [
        serialize_message(
            NormalizedMessage(
                fullname=target_id,
                kind="submission",
                thread_fullname=target_id,
                parent_fullname=None,
                raw_title=f"Looking for a budget tool {target_id}",
                raw_body="Need offline tracking.",
                subreddit="personalfinance",
                created_utc=index,
                source_revision_id=f"revision-{index}",
                permalink=f"/r/personalfinance/comments/{target_id}/",
                provenance=(SourceProvenance("local-source", index),),
                archive_score=None,
                depth=None,
                selection_channels=("rule",),
                matched_rule_ids=("offline-budget",),
            )
        )
        for index, target_id in enumerate(target_ids, start=1)
    ]
    write_compressed_jsonl(selection, messages)
    manifest = tmp_path / "selection-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "discovery_manifest",
                "schema_version": 2,
                "run_id": "bounded-test-run",
                "output_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
                "result": {"scan_complete": True},
                "budget_exhausted": False,
            }
        ),
        encoding="utf-8",
    )
    return selection, manifest, registry


def _invoke(
    selection: Path,
    manifest: Path,
    registry: Path,
    output: Path,
    *,
    target_allowlist: Path | None = None,
    tombstones: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "engagement",
            "shortlist",
            "--selection",
            str(selection),
            "--selection-manifest",
            str(manifest),
            "--registry",
            str(registry),
            "--output",
            str(output),
            "--app-id",
            "example-app",
            *(["--target-allowlist", str(target_allowlist)] if target_allowlist else []),
            *(["--tombstones", str(tombstones)] if tombstones else []),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
