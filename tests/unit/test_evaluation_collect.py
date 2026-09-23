"""CLI command for step-5 human gold-label annotation (regression tests only)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from reddit_search.cli import app

runner = CliRunner()


def _pool_row(candidate_id: str, scenario_id: str | None) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "snapshot_id": "snap-1",
        "scenario_id": scenario_id,
        "app_id": "app-1",
        "app_profile_version": 1,
        "source": {
            "message_fullname": f"t1_{candidate_id}",
            "source_revision_id": "rev-1",
            "field": "body",
            "text": "body text",
            "subreddit": "personalfinance",
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _worksheet_row(
    candidate_id: str, order: int, *, status: str = "pending", fit: str = "unreviewed"
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "review_queue": {"queue_order": order, "stratum_rule_id": None},
        "selection": {"matched_rule_ids": []},
        "annotation": {
            "review_status": status,
            "topic_fit": fit,
            "need_clarity": "clear" if status == "complete" else "unreviewed",
            "duplicate_of": None,
            "reviewer_note": "note" if status == "complete" else None,
        },
    }


def test_collect_completed_converts_annotations(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import _identity_digest
    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [_pool_row("c1", "scenario-1"), _pool_row("c2", None), _pool_row("c3", "scenario-1")],
    )
    worksheet = _write_jsonl(
        tmp_path / "worksheet.jsonl",
        [
            _worksheet_row("c1", 1, status="complete", fit="relevant"),
            _worksheet_row("c2", 2, status="complete", fit="not_relevant"),
            _worksheet_row("c3", 3),
        ],
    )
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "labels",
            "collect",
            "--pool",
            str(pool),
            "--worksheet",
            str(worksheet),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["collected"] is True
    assert payload["collected_count"] == 2
    assert payload["pending_count"] == 1
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == 2
    by_id = {row["candidate_id"]: row for row in labels}
    assert by_id["c1"]["topic_fit"] == "yes"
    assert by_id["c1"]["scenario_id"] == "scenario-1"
    assert by_id["c1"]["evaluator_kind"] == "human"
    assert by_id["c2"]["topic_fit"] == "no"
    assert by_id["c2"]["scenario_id"] is None
    # Uncertain maps to unknown, not to a negative.
    assert all(row["product_fit"] == "not_evaluated" for row in labels)
    assert all(row["validation_status"] == "unknown" for row in labels)
    report = json.loads((output / "collection_report.json").read_text())
    assert report["dependency_identity_sha256"] == _identity_digest(
        [_pool_row("c1", "scenario-1"), _pool_row("c2", None), _pool_row("c3", "scenario-1")]
    )
    assert report["identity_coverage"]["candidate_id"] == 3
    assert report["identity_coverage"]["source_revision_id"] == 3
    assert report["identity_coverage"]["snapshot_id"] == 3


def test_collect_rejects_unknown_candidate(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", None)])
    worksheet = _write_jsonl(
        tmp_path / "worksheet.jsonl",
        [_worksheet_row("cX", 1, status="complete", fit="relevant")],
    )
    result = runner.invoke(
        app,
        [
            "labels",
            "collect",
            "--pool",
            str(tmp_path / "pool.jsonl"),
            "--worksheet",
            str(worksheet),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["collected"] is False
    assert "not in the blinded pool" in payload["error"]


def test_labels_import_binds_verified_claims_from_profiles(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "app-1.yaml").write_text(
        "app_id: app-1\n"
        "profile_version: 1\n"
        "capabilities:\n"
        "  - claim_id: app-1.capability\n"
        "    status: verified\n"
        "    evidence_ref: https://example.invalid/listing\n"
        "limitations: []\n",
        encoding="utf-8",
    )
    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "scenario-1")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-1",
                "topic_fit": "yes",
                "evaluator_kind": "human",
                "product_fit": "compatible",
                "supported_claim_ids": ["app-1.capability"],
            }
        ],
    )
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "labels",
            "import",
            "--pool",
            str(pool),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--profiles-dir",
            str(profile_dir),
            "--allow-incomplete",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["imported"] is True
    assert payload["valid_count"] == 1
    row = json.loads((output / "labels.jsonl").read_text().strip())
    assert row["product_fit"] == "compatible"


def test_labels_import_rejects_unverified_claim_without_profiles(tmp_path: Path) -> None:
    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "scenario-1")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-1",
                "topic_fit": "yes",
                "evaluator_kind": "human",
                "product_fit": "compatible",
                "supported_claim_ids": ["app-1.capability"],
            }
        ],
    )
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "labels",
            "import",
            "--pool",
            str(pool),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--allow-incomplete",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["valid_count"] == 0
