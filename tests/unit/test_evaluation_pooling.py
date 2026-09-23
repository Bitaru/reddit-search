"""Unit tests for deterministic blinded pooling (execution step 4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import zstandard

from reddit_search.evaluation.pooling import (
    _enrich_retained_sources,
    _ranked_source,
    build_blinded_pool,
)


def test_retained_source_aliases_unit_id_to_message_fullname() -> None:
    source = {"msg-full": {"raw_body": "canonical body"}}
    corpus = {"unit-1": {"message_fullname": "msg-full"}}
    assert _enrich_retained_sources(source, corpus)["unit-1"] == source["msg-full"]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _write_zstd_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(zstandard.ZstdCompressor().compress(payload))
    return path


def _ranked(
    candidate_id: str,
    scenario: str,
    rank: int,
    thread: str,
    *,
    app_id: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "rank": rank,
        "score": -10.0 - rank,
        "message_fullname": candidate_id,
        "thread_fullname": thread,
        "focus_text": f"focus text {candidate_id}",
        "permalink": f"/r/test/comments/x/{candidate_id}/",
        "subreddit": "test",
        "created_utc": 1780000000,
        "context_complete": True,
        "retrieval": {"branch": "lexical", "scenario_id": scenario},
        "matched_rule_ids": ["rule.a"] if rank == 1 else [],
        "selection_channels": ["topic_rule"],
    }
    if app_id is not None:
        row["app_id"] = app_id
    return row


def _control(fullname: str, thread: str) -> dict[str, object]:
    return {
        "fullname": fullname,
        "kind": "comment",
        "thread_fullname": thread,
        "parent_fullname": None,
        "raw_title": "",
        "raw_body": f"control body {fullname}",
        "subreddit": "test",
        "created_utc": 1780000100,
        "source_revision_id": f"rev-{fullname}",
        "permalink": f"/r/test/comments/x/{fullname}/",
        "provenance": [{"source_id": "june-comments", "line_number": 1}],
        "archive_score": 4,
        "depth": 1,
        "selection_channels": [],
        "matched_rule_ids": [],
    }


@pytest.fixture()
def environment(tmp_path: Path) -> dict[str, Path]:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_jsonl(
        runs / "scenario-a.jsonl",
        [
            _ranked("c1", "scenario-a", 1, "t3_a"),
            _ranked("c2", "scenario-a", 2, "t3_a"),
            _ranked("c3", "scenario-a", 4, "t3_b"),
            _ranked("c1", "scenario-b", 7, "t3_a"),
        ],
    )
    _write_jsonl(
        runs / "scenario-b.jsonl",
        [
            _ranked("c4", "scenario-b", 1, "t3_c"),
            _ranked("c5", "scenario-b", 9, "t3_d"),
            _ranked("c6", "scenario-b", 2, "t3_e"),
        ],
    )
    controls = _write_zstd_jsonl(
        tmp_path / "rejected-controls.jsonl.zst",
        [_control("k1", "t3_k1"), _control("k2", "t3_k2"), _control("k3", "t3_k3")],
    )
    return {"runs": runs, "controls": controls, "output": tmp_path / "pool"}


_FORBIDDEN = {
    "rank",
    "score",
    "retrieval",
    "matched_rule_ids",
    "selection_channels",
    "stratum",
}


def test_pool_is_deterministic_and_blinded(environment: dict[str, Path]) -> None:
    first = build_blinded_pool(
        environment["runs"], environment["controls"], environment["output"] / "one", target_count=6
    )
    second = build_blinded_pool(
        environment["runs"], environment["controls"], environment["output"] / "two", target_count=6
    )
    assert first == second
    assert (environment["output"] / "one" / "blinded_pool.jsonl").read_bytes() == (
        environment["output"] / "two" / "blinded_pool.jsonl"
    ).read_bytes()

    rows = [
        json.loads(line)
        for line in (environment["output"] / "one" / "blinded_pool.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 6
    for row in rows:
        assert _FORBIDDEN.isdisjoint(row)
        source = row["source"]
        assert _FORBIDDEN.isdisjoint(source)
        assert "focus_text" in source or "raw_body" in source
    control_rows = [row for row in rows if row["scenario_id"] is None]
    for row in control_rows:
        assert "provenance" in row["source"] and "source_revision_id" in row["source"]

    manifest_strata = {s for s, count in first["stratum_counts"].items() if count}
    key = json.loads((environment["output"] / "one" / "pool_key.json").read_text())
    assert {entry["stratum"] for entry in key.values()} == manifest_strata
    # Reserve policy: target 6 with 3 controls reserves 2 slots for controls.
    assert first["stratum_counts"]["unknown"] == 2
    assert first["stratum_counts"]["strong"] + first["stratum_counts"]["adjacent"] == 4


def test_pool_deduplicates_and_targets(environment: dict[str, Path]) -> None:
    manifest = build_blinded_pool(
        environment["runs"], environment["controls"], environment["output"], target_count=4
    )
    rows = [
        json.loads(line)
        for line in (environment["output"] / "blinded_pool.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["pool_count"] == len(rows) == 4
    keys = [(row["candidate_id"], row["scenario_id"]) for row in rows]
    assert len(keys) == len(set(keys))
    # Controls are included even when the target is already met by ranked rows.
    assert manifest["stratum_counts"]["unknown"] >= 1


def test_pool_manifest_counts_cross_scenario_tasks_separately(environment: dict[str, Path]) -> None:
    manifest = build_blinded_pool(
        environment["runs"], environment["controls"], environment["output"], target_count=4
    )

    assert manifest["ranked_task_count"] == 7
    assert manifest["duplicate_row_count"] == 0
    assert manifest["duplicate_cross_scenario_count"] == 1


def test_pool_controls_standardized_source(environment: dict[str, Path]) -> None:
    build_blinded_pool(
        environment["runs"], environment["controls"], environment["output"], target_count=9
    )
    rows = [
        json.loads(line)
        for line in (environment["output"] / "blinded_pool.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    control_rows = [row for row in rows if row["scenario_id"] is None]
    assert control_rows
    for row in control_rows:
        source = row["source"]
        assert source["raw_body"] == f"control body {row['candidate_id']}"
        assert source["kind"] == "comment"
        assert _FORBIDDEN.isdisjoint(source)

def test_ranked_source_rejects_focus_slice_without_canonical_text() -> None:
    unit = {
        "message_fullname": "c1",
        "source_revision_id": "rev",
        "focus_field": "body",
        "focus_start": 5,
        "focus_end": 10,
        "focus_text": "short",
    }
    with pytest.raises(ValueError, match="canonical full field text"):
        _ranked_source({"message_fullname": "c1"}, unit)

def test_ranked_source_accepts_focus_when_it_is_the_full_zero_span() -> None:
    unit = {
        "message_fullname": "c1",
        "source_revision_id": "rev",
        "focus_field": "body",
        "focus_start": 0,
        "focus_end": 5,
        "focus_text": "short",
    }
    source = _ranked_source({"message_fullname": "c1"}, unit)
    assert source["body"] == "short"
    assert source["focus_text"] == "short"
 
def test_ranked_source_uses_raw_body_for_nonzero_selftext_span() -> None:
    canonical = "prefix canonical body suffix"
    unit = {
        "message_fullname": "c1",
        "source_revision_id": "rev",
        "focus_field": "selftext",
        "focus_start": 7,
        "focus_end": 15,
        "focus_text": canonical[7:15],
    }
    source = _ranked_source(
        {"message_fullname": "c1", "selftext": "focus slice", "raw_body": canonical},
        unit,
    )
    assert source["selftext"] == canonical
    assert source["selftext"][source["start"]:source["end"]] == unit["focus_text"]


def test_pool_binds_product_profile_versions_and_hash(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_jsonl(
        runs / "scenario.jsonl",
        [_ranked("c1", "scenario-a", 1, "t3_a", app_id="scenario-a")],
    )
    controls = _write_zstd_jsonl(tmp_path / "controls.jsonl.zst", [])
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    profile = profiles / "scenario-a.yaml"
    profile.write_text(
        "app_id: scenario-a\nprofile_version: 2\nverification_status: unknown\n",
        encoding="utf-8",
    )

    first = build_blinded_pool(
        runs,
        controls,
        tmp_path / "first",
        target_count=1,
        profiles_directory=profiles,
    )
    first_row = json.loads((tmp_path / "first" / "blinded_pool.jsonl").read_text(encoding="utf-8"))
    assert first_row["app_profile_version"] == 2
    assert first_row["app_profile_sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
    assert first["profiles"]["versions"] == {"scenario-a": 2}
    first_digest = first["profiles"]["sha256"]

    profile.write_text(
        "app_id: scenario-a\nprofile_version: 3\nverification_status: unknown\n",
        encoding="utf-8",
    )
    second = build_blinded_pool(
        runs,
        controls,
        tmp_path / "second",
        target_count=1,
        profiles_directory=profiles,
    )
    second_row = json.loads(
        (tmp_path / "second" / "blinded_pool.jsonl").read_text(encoding="utf-8")
    )
    assert second_row["app_profile_version"] == 3
    assert second["profiles"]["versions"] == {"scenario-a": 3}
    assert second["profiles"]["sha256"] != first_digest


def test_pool_topic_only_scenario_does_not_guess_profile_binding(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_jsonl(
        runs / "scenario.jsonl",
        [_ranked("c1", "expense_tracking.no_bank_link", 1, "t3_a")],
    )
    controls = _write_zstd_jsonl(tmp_path / "controls.jsonl.zst", [])
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "known.yaml").write_text(
        "app_id: known\nprofile_version: 2\nverification_status: unknown\n",
        encoding="utf-8",
    )

    build_blinded_pool(
        runs,
        controls,
        tmp_path / "pool",
        target_count=1,
        profiles_directory=profiles,
    )
    row = json.loads((tmp_path / "pool" / "blinded_pool.jsonl").read_text(encoding="utf-8"))
    assert row["app_id"] is None
    assert row["app_profile_version"] is None
    assert row["app_profile_sha256"] is None
def test_pool_scenario_namespace_matching_profile_stays_unbound(tmp_path: Path) -> None:
    # Decisive case: the scenario namespace prefix matches an existing profile,
    # yet without explicit app_id the row must stay unbound (no inference).
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_jsonl(
        runs / "scenario.jsonl",
        [_ranked("c1", "known.need", 1, "t3_a")],
    )
    controls = _write_zstd_jsonl(tmp_path / "controls.jsonl.zst", [])
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "known.yaml").write_text(
        "app_id: known\nprofile_version: 2\nverification_status: unknown\n",
        encoding="utf-8",
    )

    build_blinded_pool(
        runs,
        controls,
        tmp_path / "pool",
        target_count=1,
        profiles_directory=profiles,
    )
    row = json.loads((tmp_path / "pool" / "blinded_pool.jsonl").read_text(encoding="utf-8"))
    assert row["app_id"] is None
    assert row["app_profile_version"] is None
    assert row["app_profile_sha256"] is None


def test_pool_rejects_unknown_explicit_app_binding(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_jsonl(
        runs / "scenario.jsonl",
        [_ranked("c1", "expense_tracking.no_bank_link", 1, "t3_a", app_id="missing")],
    )
    controls = _write_zstd_jsonl(tmp_path / "controls.jsonl.zst", [])
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "known.yaml").write_text(
        "app_id: known\nprofile_version: 2\nverification_status: unknown\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no product profile found for app_id: missing"):
        build_blinded_pool(
            runs,
            controls,
            tmp_path / "pool",
            target_count=1,
            profiles_directory=profiles,
        )
