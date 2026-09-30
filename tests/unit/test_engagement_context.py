from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
import zstandard

import reddit_search.engagement.context as context_module
from reddit_search.engagement.context import build_thread_context
from reddit_search.engagement.contracts import ShortlistCandidate, ShortlistManifest
from reddit_search.ingest.state import file_sha256

SCOPE = "user_provided_local_research_no_external_transmission"


def _submission(name: str, **extra: object) -> dict[str, object]:
    return {
        "name": name,
        "id": name[3:],
        "title": "Title",
        "selftext": "Root text",
        "subreddit": "test",
        "permalink": "/root",
        "created_utc": 10,
        **extra,
    }


def _comment(
    name: str, parent: str, *, thread: str = "t3_root", **extra: object
) -> dict[str, object]:
    return {
        "name": name,
        "id": name[3:],
        "link_id": thread,
        "parent_id": parent,
        "body": "Comment text",
        "subreddit": "test",
        "permalink": "/comment",
        "created_utc": 11,
        **extra,
    }


def _setup(
    tmp_path: Path,
    records: list[dict[str, object]],
    *,
    max_messages: int = 100,
    max_bytes: int = 64 * 1024 * 1024,
    scope: str = SCOPE,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate = ShortlistCandidate(
        app_id="app",
        thread_fullname="t3_root",
        reply_target_fullname="t1_target",
        source_message_fullname="t1_target",
        source_revision_id="revision",
        permalink="/root",
        subreddit="test",
        scenario_references=["scenario"],
        source_references=["run:message"],
        reason_for_inclusion="matched",
    )
    payload = (candidate.model_dump_json() + "\n").encode()
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_bytes(payload)
    source_records = []
    for kind, prefix in (("submission", "t3_"), ("comment", "t1_")):
        selected = [row for row in records if str(row.get("name", "")).startswith(prefix)]
        if not selected:
            continue
        source = tmp_path / f"{kind}.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
        source_records.append(
            {
                "source_id": kind,
                "source_path": str(source),
                "source_kind": kind,
                "declared_month": "2026-01",
                "source_role": "discovery",
                "usage_scope": scope,
                "status": "complete",
                "verified_sha256": file_sha256(source),
                "invalid_records": 0,
            }
        )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"sources": source_records}), encoding="utf-8")
    shortlist_manifest = tmp_path / "shortlist_manifest.json"
    shortlist_manifest.write_text(
        ShortlistManifest(
            app_id="app",
            inputs={
                "registry_sha256": file_sha256(registry_path),
                "tombstone_sha256": hashlib.sha256(b"").hexdigest(),
            },
            candidate_count=1,
            candidates_sha256=hashlib.sha256(payload).hexdigest(),
        ).model_dump_json(),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    report = build_thread_context(
        candidates_path,
        shortlist_manifest,
        registry_path,
        output,
        max_messages_per_thread=max_messages,
        max_bytes_per_thread=max_bytes,
    )
    with (output / "threads.jsonl.zst").open("rb") as raw:
        decoded = zstandard.ZstdDecompressor().stream_reader(raw).read().decode()
    rows = [json.loads(line) for line in decoded.splitlines()]
    manifest = json.loads((output / "context_manifest.json").read_text())
    assert (output / "threads.jsonl.zst").is_file()
    assert manifest["output_sha256"] == file_sha256(output / "threads.jsonl.zst")
    return report, rows, manifest


def _rebind_registry(tmp_path: Path) -> None:
    manifest_path = tmp_path / "shortlist_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["registry_sha256"] = file_sha256(tmp_path / "registry.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_root_comments_parent_chain_and_unrelated_rows(tmp_path: Path) -> None:
    records = [
        _submission("t3_root", locked=True, archived=True, num_comments=9),
        _comment("t1_parent", "t3_root"),
        _comment("t1_target", "t1_parent"),
        _comment("t1_other", "t3_else", thread="t3_else"),
    ]
    report, rows, manifest = _setup(tmp_path, records)
    assert report["scan_complete"] is True
    assert {row["fullname"] for row in rows} == {"t3_root", "t1_parent", "t1_target"}
    root = next(row for row in rows if row["fullname"] == "t3_root")
    assert root["archived_status"] == {"locked": True, "archived": True, "num_comments": 9}
    assert (
        next(row for row in rows if row["fullname"] == "t1_target")["parent_fullname"]
        == "t1_parent"
    )
    assert "t1_other" not in {row["fullname"] for row in rows}
    assert manifest["threads"][0]["thread_context_complete_for_sources"] is True


def test_missing_root_parent_and_removed_comment_are_explicit(tmp_path: Path) -> None:
    report, rows, manifest = _setup(
        tmp_path,
        [
            _submission("t3_else"),
            _comment("t1_target", "t1_missing", removed_by_category="moderator"),
        ],
    )
    coverage = manifest["threads"][0]
    assert report["scan_complete"] is True
    assert coverage["root_missing"] is True
    assert coverage["missing_parents"] == ["t1_missing"]
    assert coverage["coverage_status"] == "needs_context"
    assert rows[0]["archived_status"]["removed_by_category"] == "moderator"


def test_removed_root_and_target_are_capture_gaps(tmp_path: Path) -> None:
    root = _submission("t3_root", removed_by_category="moderator")
    target = _comment("t1_target", "t3_root", is_removed=True)
    _, rows, manifest = _setup(tmp_path, [root, target])
    coverage = manifest["threads"][0]
    assert {row["fullname"] for row in rows} == {"t3_root", "t1_target"}
    assert coverage["root_missing"] is False
    assert coverage["coverage_status"] == "needs_context"
    assert coverage["coverage_reason_counts"] == {
        "root_text_removed": 1,
        "target_text_removed": 1,
    }
    assert any(
        "t3_root: historical removed_by_category" in reason
        for reason in coverage["coverage_reasons"]
    )
    assert any(
        "t1_target: historical removed/deleted status" in reason
        for reason in coverage["coverage_reasons"]
    )


def test_removed_non_target_comment_is_archive_content_note(tmp_path: Path) -> None:
    _, rows, manifest = _setup(
        tmp_path,
        [
            _submission("t3_root"),
            _comment("t1_target", "t3_root"),
            _comment("t1_other", "t3_root", body="[removed]"),
        ],
    )
    coverage = manifest["threads"][0]
    assert {row["fullname"] for row in rows} == {"t3_root", "t1_target", "t1_other"}
    assert coverage["coverage_status"] == "complete_for_scanned_sources"
    assert coverage["thread_context_complete_for_sources"] is True
    assert coverage["coverage_reasons"] == []
    assert coverage["coverage_reason_counts"] == {}
    assert coverage["archive_content_notes"] == [
        "t1_other: message text is archived as removed/deleted"
    ]
    assert coverage["archive_content_counts"] == {"historical_removal": 1}


def test_authorized_local_research_scope_is_accepted(tmp_path: Path) -> None:
    report, rows, _ = _setup(tmp_path, [_submission("t3_root")], scope="authorized_local_research")
    assert report["scan_complete"] is True
    assert rows[0]["fullname"] == "t3_root"


def test_changed_tombstone_ledger_is_rejected(tmp_path: Path) -> None:
    _setup(tmp_path, [_submission("t3_root")])
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps({"message_fullname": "t1_old", "source_revision_id": None, "reason": "operator"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tombstone ledger hash"):
        build_thread_context(
            tmp_path / "candidates.jsonl",
            tmp_path / "shortlist_manifest.json",
            tmp_path / "registry.json",
            tmp_path / "changed-ledger-output",
            tombstones_path=tombstones,
        )


def test_identical_revision_retains_all_provenance(tmp_path: Path) -> None:
    duplicate = _comment("t1_target", "t3_root")
    report, rows, _ = _setup(tmp_path, [_submission("t3_root"), duplicate, duplicate])
    target = next(row for row in rows if row["fullname"] == "t1_target")
    assert report["scan_complete"] is True
    assert len(target["provenance"]) == 2


def test_identical_message_across_registered_sources_retains_each_provenance(
    tmp_path: Path,
) -> None:
    duplicate = _comment("t1_target", "t3_root")
    _setup(tmp_path, [_submission("t3_root"), duplicate])

    registry_path = tmp_path / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    second_source = tmp_path / "comment-second.jsonl"
    second_source.write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
    comment_source = next(
        source for source in registry["sources"] if source["source_id"] == "comment"
    )
    comment_source["source_path"] = str(tmp_path / "comment.jsonl")
    (tmp_path / "comment.jsonl").write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
    registry["sources"].append(
        {
            **comment_source,
            "source_id": "comment-second",
            "source_path": str(second_source),
            "verified_sha256": file_sha256(second_source),
        }
    )
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    _rebind_registry(tmp_path)

    build_thread_context(
        tmp_path / "candidates.jsonl",
        tmp_path / "shortlist_manifest.json",
        registry_path,
        tmp_path / "multi-source-out",
    )
    with (tmp_path / "multi-source-out" / "threads.jsonl.zst").open("rb") as raw:
        decoded = zstandard.ZstdDecompressor().stream_reader(raw).read().decode()
    target = next(
        json.loads(line)
        for line in decoded.splitlines()
        if json.loads(line)["fullname"] == "t1_target"
    )

    assert target["provenance"] == [
        {"source_id": "comment", "line_number": 1},
        {"source_id": "comment-second", "line_number": 1},
    ]


def test_per_thread_cap_marks_needs_context(tmp_path: Path) -> None:
    report, rows, manifest = _setup(
        tmp_path, [_submission("t3_root"), _comment("t1_target", "t3_root")], max_messages=1
    )
    assert report["scan_complete"] is True
    assert manifest["threads"][0]["coverage_status"] == "needs_context"
    assert manifest["threads"][0]["missing_reply_targets"] == ["t1_target"]
    assert len(rows) == 1



def test_per_thread_byte_cap_marks_needs_context(tmp_path: Path) -> None:
    report, rows, manifest = _setup(
        tmp_path,
        [
            _submission("t3_root"),
            _comment("t1_target", "t3_root"),
        ],
        max_bytes=1,
    )
    thread = manifest["threads"][0]
    assert thread["coverage_status"] == "needs_context"
    assert thread["omitted_messages"] == 2
    assert any("thread_cap:" in reason for reason in thread["coverage_reasons"])
    assert thread["thread_context_complete_for_sources"] is False

def test_malformed_json_source_is_incomplete(tmp_path: Path) -> None:
    report, _, manifest = _setup(tmp_path, [_submission("t3_root")])
    assert report["scan_complete"] is True
    # A malformed row invalidates the input's previously registered verified hash.
    source = tmp_path / "submission.jsonl"
    source.write_text(json.dumps(_submission("t3_root")) + "\n{broken\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    record = json.loads(registry.read_text())
    record["sources"][0]["verified_sha256"] = file_sha256(source)
    registry.write_text(json.dumps(record), encoding="utf-8")
    _rebind_registry(tmp_path)
    output = tmp_path / "bad-output"
    result = build_thread_context(
        tmp_path / "candidates.jsonl", tmp_path / "shortlist_manifest.json", registry, output
    )
    bad_manifest = json.loads((output / "context_manifest.json").read_text())
    assert result["scan_complete"] is False
    assert bad_manifest["sources"][0]["invalid_records"] == 1
    assert bad_manifest["threads"][0]["coverage_status"] == "needs_context"


def test_different_revisions_mark_thread_incomplete(tmp_path: Path) -> None:
    first = _comment("t1_target", "t3_root")
    second = _comment("t1_target", "t3_root")
    second["body"] = "changed revision"
    _, _, manifest = _setup(tmp_path, [_submission("t3_root"), first, second])
    assert manifest["threads"][0]["coverage_status"] == "needs_context"
    assert manifest["scan_complete"] is True


def test_truncated_zstd_source_is_incomplete(tmp_path: Path) -> None:
    _setup(tmp_path, [_submission("t3_root")])
    source = tmp_path / "truncated.zst"
    source.write_bytes(b"\x28\xb5\x2f\xfd\x00")
    registry = tmp_path / "registry.json"
    payload = json.loads(registry.read_text())
    payload["sources"][0].update(source_path=str(source), verified_sha256=file_sha256(source))
    registry.write_text(json.dumps(payload), encoding="utf-8")
    _rebind_registry(tmp_path)
    # Fresh output and unchanged, hash-bound shortlist.
    output = tmp_path / "truncated-output"
    result = build_thread_context(
        tmp_path / "candidates.jsonl", tmp_path / "shortlist_manifest.json", registry, output
    )
    manifest = json.loads((output / "context_manifest.json").read_text())
    assert result["scan_complete"] is False
    assert manifest["sources"][0]["verified_sha256"] is None
    assert manifest["threads"][0]["coverage_status"] == "needs_context"


def test_context_artifacts_and_staging_database_are_private_under_umask_022(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_connect = context_module.sqlite3.connect
    staging_modes: list[int] = []

    def inspect_staging_database(path: str | Path, *args, **kwargs):
        staging_modes.append(stat.S_IMODE(Path(path).stat().st_mode))
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(context_module.sqlite3, "connect", inspect_staging_database)
    previous_umask = os.umask(0o022)
    try:
        _setup(tmp_path, [_submission("t3_root")])
    finally:
        os.umask(previous_umask)

    output = tmp_path / "out"
    assert staging_modes == [0o600]
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "threads.jsonl.zst").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "context_manifest.json").stat().st_mode) == 0o600
