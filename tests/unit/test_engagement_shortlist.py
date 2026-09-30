from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from reddit_search.engagement.shortlist import build_shortlist
from reddit_search.ingest.shard import write_compressed_jsonl
from reddit_search.ingest.state import file_sha256


def _row(
    fullname: str = "t3_post",
    *,
    revision: str = "rev-1",
    rules: list[str] | None = None,
    source_id: str = "source-a",
    line_number: int = 4,
) -> dict[str, object]:
    submission = fullname.startswith("t3_")
    return {
        "fullname": fullname,
        "kind": "submission" if submission else "comment",
        "thread_fullname": fullname if submission else "t3_post",
        "parent_fullname": None if submission else "t3_post",
        "raw_title": "Need a tool" if submission else "",
        "raw_body": "A concrete request" if submission else "Any suggestions?",
        "subreddit": "example",
        "created_utc": 1_700_000_000,
        "source_revision_id": revision,
        "permalink": "https://reddit.test/r/example/comments/post",
        "provenance": [{"source_id": source_id, "line_number": line_number}],
        "archive_score": None,
        "depth": None if submission else 1,
        "selection_channels": ["topic_rule"],
        "matched_rule_ids": rules or [],
    }


def _inputs(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    usage_scope: str = "user_provided_local_research_no_external_transmission",
) -> tuple[Path, Path, Path]:
    source_file = tmp_path / "source.zst"
    source_file.write_bytes(b"verified source bytes")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "source-a",
                        "source_path": str(source_file),
                        "usage_scope": usage_scope,
                        "status": "complete",
                        "invalid_records": 0,
                        "verified_sha256": file_sha256(source_file),
                        "input_size_bytes": source_file.stat().st_size,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.jsonl.zst"
    write_compressed_jsonl(selection, rows)
    manifest = tmp_path / "selection.jsonl.zst.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "discovery_manifest",
                "schema_version": 2,
                "run_id": "discovery-run-1",
                "output_sha256": file_sha256(selection),
                "result": {"scan_complete": True},
            }
        ),
        encoding="utf-8",
    )
    return selection, manifest, registry


def _read_candidates(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_discovery_selection_preserves_rule_provenance_and_deduplicates(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(
        tmp_path,
        [
            _row(rules=["rule.one"], line_number=4),
            _row(rules=["rule.two"], line_number=5),
        ],
    )
    output = tmp_path / "shortlist"

    report = build_shortlist(selection, manifest, registry, output, "explicit-app")
    rows = _read_candidates(Path(report["candidates_path"]))

    assert len(rows) == 1
    assert rows[0]["app_id"] == "explicit-app"
    assert rows[0]["thread_fullname"] == "t3_post"
    assert rows[0]["reply_target_fullname"] == "t3_post"
    assert rows[0]["scenario_references"] == ["rule.one", "rule.two"]
    assert rows[0]["source_references"] == [
        "source-a:4:t3_post:rev-1",
        "source-a:5:t3_post:rev-1",
    ]


def test_ranked_runs_filter_selection_by_exact_fullname_and_scenario(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(
        tmp_path,
        [_row(rules=["discovery.rule"]), _row("t1_comment", rules=["other.rule"])],
    )
    ranked_dir = tmp_path / "ranked"
    ranked_dir.mkdir()
    ranked_rows = [
        {
            "message_fullname": "t1_comment",
            "thread_fullname": "t3_post",
            "app_id": "app-explicit",
            "retrieval": {"scenario_id": "scenario-a"},
        },
        {
            "message_fullname": "t3_unselected",
            "thread_fullname": "t3_unselected",
            "app_id": "app-explicit",
            "retrieval": {"scenario_id": "scenario-a"},
        },
    ]
    ranked_file = ranked_dir / "scenario-a.jsonl"
    ranked_file.write_text("".join(json.dumps(row) + "\n" for row in ranked_rows), encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "ranked-run",
                "scenario_runs": {"scenario-a": {"sha256": file_sha256(ranked_file)}},
            }
        ),
        encoding="utf-8",
    )

    report = build_shortlist(
        selection,
        manifest,
        registry,
        tmp_path / "shortlist",
        "app-explicit",
        ranked_directory=ranked_dir,
        run_manifest_path=run_manifest,
    )
    rows = _read_candidates(Path(report["candidates_path"]))

    assert len(rows) == 1
    assert rows[0]["thread_fullname"] == "t3_post"
    assert rows[0]["reply_target_fullname"] == "t1_comment"
    assert rows[0]["scenario_references"] == ["scenario-a"]



def test_ranked_scenarios_collapse_target_and_preserve_all_references(
    tmp_path: Path,
) -> None:
    selection, manifest, registry = _inputs(
        tmp_path,
        [
            _row("t1_comment", source_id="source-a", line_number=4),
            _row("t1_comment", source_id="source-a", line_number=9),
        ],
    )
    ranked_dir = tmp_path / "ranked-multi"
    ranked_dir.mkdir()
    scenario_rows = {
        "scenario-z": [
            {
                "message_fullname": "t1_comment",
                "thread_fullname": "t3_post",
                "app_id": "app-explicit",
                "retrieval": {"scenario_id": "scenario-z"},
            }
        ],
        "scenario-a": [
            {
                "message_fullname": "t1_comment",
                "thread_fullname": "t3_post",
                "app_id": "app-explicit",
                "retrieval": {"scenario_id": "scenario-a"},
            }
        ],
    }
    scenario_runs = {}
    for scenario_id, rows in scenario_rows.items():
        scenario_file = ranked_dir / f"{scenario_id}.jsonl"
        scenario_file.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        scenario_runs[scenario_id] = {"sha256": file_sha256(scenario_file)}
    run_manifest = tmp_path / "multi_run_manifest.json"
    run_manifest.write_text(
        json.dumps({"snapshot_id": "ranked-run-multi", "scenario_runs": scenario_runs}),
        encoding="utf-8",
    )

    report = build_shortlist(
        selection,
        manifest,
        registry,
        tmp_path / "ranked-shortlist",
        "app-explicit",
        ranked_directory=ranked_dir,
        run_manifest_path=run_manifest,
    )
    rows = _read_candidates(Path(report["candidates_path"]))

    assert len(rows) == 1
    assert rows[0]["scenario_references"] == ["scenario-a", "scenario-z"]
    assert rows[0]["source_references"] == [
        "source-a:4:t1_comment:rev-1",
        "source-a:9:t1_comment:rev-1",
    ]

def test_selection_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(tmp_path, [_row(rules=["rule.one"])])
    selection.write_bytes(selection.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="hash does not match"):
        build_shortlist(selection, manifest, registry, tmp_path / "shortlist", "app")


def test_incomplete_or_unauthorized_registry_fails_closed(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(tmp_path, [_row(rules=["rule.one"])])
    records = json.loads(registry.read_text(encoding="utf-8"))
    records["sources"][0]["usage_scope"] = "unauthorized"
    registry.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="usage_scope"):
        build_shortlist(selection, manifest, registry, tmp_path / "shortlist", "app")


def test_authorized_local_research_scope_is_accepted(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(
        tmp_path,
        [_row(rules=["rule.local"])],
        usage_scope="authorized_local_research",
    )

    report = build_shortlist(selection, manifest, registry, tmp_path / "shortlist", "explicit-app")

    assert report["candidate_count"] == 1


def test_explicit_target_allowlist_selects_exact_submission_and_comment(
    tmp_path: Path,
) -> None:
    selection, manifest, registry = _inputs(
        tmp_path,
        [
            _row("t3_post", rules=["rule.post"]),
            _row("t1_comment", rules=["rule.comment"]),
            _row("t3_other", rules=["rule.other"]),
        ],
    )
    allowlist = tmp_path / "targets.json"
    allowlist_bytes = b'["t1_comment", "t3_post"]'
    allowlist.write_bytes(allowlist_bytes)

    report = build_shortlist(
        selection,
        manifest,
        registry,
        tmp_path / "shortlist",
        "app",
        target_allowlist_path=allowlist,
    )
    rows = _read_candidates(Path(report["candidates_path"]))
    manifest_data = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))

    assert {row["reply_target_fullname"] for row in rows} == {"t1_comment", "t3_post"}
    assert {row["reply_target_fullname"]: row["scenario_references"] for row in rows} == {
        "t1_comment": ["rule.comment"],
        "t3_post": ["rule.post"],
    }
    assert (
        manifest_data["inputs"]["target_allowlist_sha256"]
        == hashlib.sha256(allowlist_bytes).hexdigest()
    )


def test_target_allowlist_rejects_unknown_selected_id(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(
        tmp_path,
        [
            _row("t3_post", rules=["rule.post"]),
            _row("t1_comment", rules=["rule.comment"]),
            _row("t3_other", rules=["rule.other"]),
        ],
    )
    allowlist = tmp_path / "targets.json"
    allowlist.write_text('["t1_comment", "t3_missing"]', encoding="utf-8")

    with pytest.raises(ValueError, match="absent from the selected shard"):
        build_shortlist(
            selection,
            manifest,
            registry,
            tmp_path / "shortlist",
            "app",
            target_allowlist_path=allowlist,
        )


def test_ranked_file_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(tmp_path, [_row(rules=["rule.one"])])
    ranked_dir = tmp_path / "ranked"
    ranked_dir.mkdir()
    ranked_file = ranked_dir / "scenario-a.jsonl"
    ranked_file.write_text("{}\n", encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(
        json.dumps({"snapshot_id": "run", "scenario_runs": {"scenario-a": {"sha256": "0" * 64}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ranked scenario hash mismatch"):
        build_shortlist(
            selection,
            manifest,
            registry,
            tmp_path / "shortlist",
            "app",
            ranked_directory=ranked_dir,
            run_manifest_path=run_manifest,
        )


@pytest.mark.parametrize(
    ("ranked_app", "ranked_thread", "expected_error"),
    [
        ("different-app", "t3_post", "invalid identity/app/scenario"),
        ("app", "t3_wrong", "ranked thread ID contradicts selected"),
    ],
)
def test_ranked_rows_with_wrong_app_or_thread_are_rejected(
    tmp_path: Path, ranked_app: str, ranked_thread: str, expected_error: str
) -> None:
    selection, manifest, registry = _inputs(tmp_path, [_row("t1_comment", rules=["rule"])])
    ranked_dir = tmp_path / "ranked"
    ranked_dir.mkdir()
    ranked_file = ranked_dir / "scenario-a.jsonl"
    ranked_file.write_text(
        json.dumps(
            {
                "message_fullname": "t1_comment",
                "thread_fullname": ranked_thread,
                "app_id": ranked_app,
                "retrieval": {"scenario_id": "scenario-a"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "run",
                "scenario_runs": {"scenario-a": {"sha256": file_sha256(ranked_file)}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_error):
        build_shortlist(
            selection,
            manifest,
            registry,
            tmp_path / "shortlist",
            "app",
            ranked_directory=ranked_dir,
            run_manifest_path=run_manifest,
        )


def test_conflicting_revisions_for_same_message_are_rejected(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(
        tmp_path, [_row(rules=["rule.one"]), _row(revision="rev-2", rules=["rule.two"])]
    )
    output = tmp_path / "shortlist"

    with pytest.raises(ValueError):
        build_shortlist(selection, manifest, registry, output, "app")

    assert not output.exists()


def test_target_allowlist_rejects_selected_row_without_scenario(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(tmp_path, [_row(rules=[])])
    allowlist = tmp_path / "targets.json"
    allowlist.write_text('["t3_post"]', encoding="utf-8")
    output = tmp_path / "shortlist"

    with pytest.raises(ValueError, match="no discovery or ranked scenario"):
        build_shortlist(
            selection,
            manifest,
            registry,
            output,
            "app",
            target_allowlist_path=allowlist,
        )

    assert not output.exists()


def test_tombstoned_selection_is_excluded_and_hash_is_recorded(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(tmp_path, [_row(rules=["rule.one"])])
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t3_post",
                "source_revision_id": "rev-1",
                "reason": "operator invalidation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_shortlist(
        selection,
        manifest,
        registry,
        tmp_path / "shortlist",
        "app",
        tombstones_path=tombstones,
    )

    assert report["candidate_count"] == 0
    shortlist_manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
    assert shortlist_manifest["inputs"]["tombstone_sha256"] == file_sha256(tombstones)

    allowlist = tmp_path / "targets.json"
    allowlist.write_text('["t3_post"]', encoding="utf-8")
    with pytest.raises(ValueError, match="tombstoned message t3_post"):
        build_shortlist(
            selection,
            manifest,
            registry,
            tmp_path / "allowlisted-shortlist",
            "app",
            tombstones_path=tombstones,
            target_allowlist_path=allowlist,
        )


def test_shortlist_artifacts_are_private_under_permissive_umask(tmp_path: Path) -> None:
    selection, manifest, registry = _inputs(tmp_path, [_row(rules=["rule.one"])])
    output = tmp_path / "private-shortlist"
    previous_umask = os.umask(0o022)
    try:
        report = build_shortlist(selection, manifest, registry, output, "explicit-app")
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE(Path(report["candidates_path"]).stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(report["manifest_path"]).stat().st_mode) == 0o600
