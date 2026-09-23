"""Unit tests for the persistent lexical baseline (execution step 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard

from reddit_search.corpus.baseline import (
    BudgetError,
    build_corpus_index,
    run_lexical_scenarios,
)


def _write_shard(path: Path, payloads: list[dict[str, object]]) -> Path:
    with path.open("wb") as output:
        with zstandard.ZstdCompressor().stream_writer(output, closefd=False) as compressed:
            for payload in payloads:
                compressed.write(json.dumps(payload).encode() + b"\n")
    return path


def _submission(fullname: str, title: str, body: str, created: int) -> dict[str, object]:
    return {
        "fullname": fullname,
        "kind": "submission",
        "thread_fullname": fullname,
        "parent_fullname": None,
        "raw_title": title,
        "raw_body": body,
        "subreddit": "personalfinance",
        "created_utc": created,
        "source_revision_id": f"rev-{fullname}",
        "permalink": f"/r/personalfinance/comments/{fullname}/",
        "provenance": [{"source_id": "june-submissions", "line_number": created}],
        "archive_score": None,
        "depth": None,
        "selection_channels": ["topic_rule"],
        "matched_rule_ids": ["mieru.no_bank_link"],
    }


def _comment(fullname: str, parent: str, body: str, created: int) -> dict[str, object]:
    return {
        "fullname": fullname,
        "kind": "comment",
        "thread_fullname": "t3_root",
        "parent_fullname": parent,
        "raw_title": "",
        "raw_body": body,
        "subreddit": "personalfinance",
        "created_utc": created,
        "source_revision_id": f"rev-{fullname}",
        "permalink": f"/r/personalfinance/comments/root/comment/{fullname}/",
        "provenance": [{"source_id": "june-comments", "line_number": created}],
        "archive_score": 4,
        "depth": 1,
        "selection_channels": ["topic_rule"],
        "matched_rule_ids": ["mieru.no_bank_link"],
    }


@pytest.fixture()
def shard(tmp_path: Path) -> Path:
    payloads = [
        _submission(
            "t3_root",
            "Bank sync frustrations",
            "I wish expense tracking did not require linking my bank.",
            1,
        ),
        _comment("t1_need", "t3_root", "I need an expense tracker without bank sync.", 2),
        _submission(
            "t3_invoice",
            "Invoicing occasionally",
            "I only invoice a few clients per month and hate subscriptions.",
            3,
        ),
    ]
    return _write_shard(tmp_path / "selected.jsonl.zst", payloads)


def test_build_is_deterministic_and_rebuild_yields_equivalent_units(
    shard: Path, tmp_path: Path
) -> None:
    first = build_corpus_index(
        shard, tmp_path / "a.db", snapshot_id="snap", minimum_free_disk_bytes=1
    )
    build_corpus_index(shard, tmp_path / "b.db", snapshot_id="snap", minimum_free_disk_bytes=1)

    assert first["indexed_unit_count"] == 3
    assert first["unit_ids_stable"] is True
    assert first["context_recipe_version"] == "v2"

    import sqlite3

    def unit_ids(db: Path) -> list[tuple[str, str, str]]:
        connection = sqlite3.connect(db)
        try:
            return [
                (row[0], row[1], row[2])
                for row in connection.execute(
                    "SELECT unit_id, message_fullname, context_recipe_version "
                    "FROM search_units ORDER BY message_fullname"
                )
            ]
        finally:
            connection.close()

    assert unit_ids(tmp_path / "a.db") == unit_ids(tmp_path / "b.db")


def test_build_refuses_existing_output_when_context_recipe_changes(
    shard: Path, tmp_path: Path
) -> None:
    output = tmp_path / "corpus.db"
    build_corpus_index(
        shard, output, snapshot_id="snap", context_recipe_version="v2", minimum_free_disk_bytes=1
    )

    with pytest.raises(ValueError, match="does not match current inputs"):
        build_corpus_index(
            shard,
            output,
            snapshot_id="snap",
            context_recipe_version="v3",
            minimum_free_disk_bytes=1,
        )


def test_build_is_idempotent_and_rejects_mismatched_existing_output(
    shard: Path, tmp_path: Path
) -> None:
    output = tmp_path / "corpus.db"
    first = build_corpus_index(shard, output, snapshot_id="snap", minimum_free_disk_bytes=1)
    second = build_corpus_index(shard, output, snapshot_id="snap", minimum_free_disk_bytes=1)
    assert second == first
    with pytest.raises(ValueError, match="does not match current inputs"):
        build_corpus_index(
            shard, output, snapshot_id="different-snapshot", minimum_free_disk_bytes=1
        )

    duplicated = _write_shard(
        tmp_path / "duplicate.jsonl.zst",
        [
            _submission("t3_root", "One", "body one", 1),
            _submission("t3_root", "Two", "body two", 2),
        ],
    )
    with pytest.raises(ValueError, match="duplicate message fullnames"):
        build_corpus_index(
            duplicated, tmp_path / "dup.db", snapshot_id="snap", minimum_free_disk_bytes=1
        )


def test_build_refuses_when_free_disk_budget_is_violated(
    shard: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reddit_search.resources as resources

    def fake_disk_usage(_path: object):
        usage = type("Usage", (), {"free": 0, "total": 1, "used": 1})
        return usage()

    monkeypatch.setattr(resources.shutil, "disk_usage", fake_disk_usage)
    with pytest.raises(BudgetError, match="insufficient free disk"):
        build_corpus_index(shard, tmp_path / "corpus.db", snapshot_id="snap")
    assert not (tmp_path / "corpus.db").exists()
    assert not (tmp_path / "corpus.db.tmp").exists()


def test_build_refuses_when_staging_budget_is_violated(shard: Path, tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match="staging budget exceeded"):
        build_corpus_index(
            shard,
            tmp_path / "corpus.db",
            snapshot_id="snap",
            max_staging_bytes=1,
            minimum_free_disk_bytes=0,
        )
    assert not (tmp_path / "corpus.db").exists()
    assert not (tmp_path / "corpus.db.tmp").exists()


def test_search_runs_are_deterministic_traceable_and_merged(shard: Path, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.db"
    build_corpus_index(shard, corpus, snapshot_id="snap", minimum_free_disk_bytes=1)

    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    (scenarios_dir / "mieru.yaml").write_text(
        """
scenarios:
  - schema_version: 1
    scenario_id: mieru.no_bank_link
    app_id: mieru
    scenario_version: 1
    description: Needs to track expenses without bank links.
    lexical_queries: ["expense tracker bank", "expense tracker bank sync", "bank sync"]
    semantic_queries: ["I need expense tracking without bank sync.",
      "Track spending with no bank link."]
""",
        encoding="utf-8",
    )

    output = tmp_path / "runs"
    result = run_lexical_scenarios(
        corpus,
        scenarios_dir,
        output,
        snapshot_id="snap",
        candidates_per_scenario=10,
        output_limit=20,
    )
    assert result["scenario_count"] == 1
    assert result["total_output_hits"] >= 1

    scenario_id = "mieru.no_bank_link"
    first_bytes = (output / f"{scenario_id}.jsonl").read_bytes()

    ranks = [json.loads(line) for line in first_bytes.decode().splitlines()]
    assert ranks == sorted(ranks, key=lambda row: (row["score"], row["candidate_id"]))
    assert all(row["retrieval"]["branch"] == "lexical" for row in ranks)
    assert all(row["retrieval"]["scenario_id"] == scenario_id for row in ranks)
    # Both lexical queries share a hit; the merge keeps one row per unit.
    assert len({row["candidate_id"] for row in ranks}) == len(ranks)
    assert result["scenario_sha256"][scenario_id] is not None

    run_lexical_scenarios(
        corpus,
        scenarios_dir,
        output,
        snapshot_id="snap",
        candidates_per_scenario=10,
        output_limit=20,
    )
    assert (output / f"{scenario_id}.jsonl").read_bytes() == first_bytes


def test_search_rejects_empty_scenario_directory(shard: Path, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.db"
    build_corpus_index(shard, corpus, snapshot_id="snap", minimum_free_disk_bytes=1)
    empty = tmp_path / "empty-scenarios"
    empty.mkdir()
    with pytest.raises(ValueError, match="no scenarios"):
        run_lexical_scenarios(
            corpus,
            empty,
            tmp_path / "runs",
            snapshot_id="snap",
            candidates_per_scenario=10,
            output_limit=20,
        )


def test_build_excludes_tombstoned_selection(shard: Path, tmp_path: Path) -> None:
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t1_need",
                "source_revision_id": None,
                "reason": "removed by operator",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_corpus_index(
        shard,
        tmp_path / "corpus.db",
        snapshot_id="snap",
        tombstones_path=tombstones,
        minimum_free_disk_bytes=1,
    )

    assert result["indexed_unit_count"] == 2
    assert result["tombstoned_count"] == 1
