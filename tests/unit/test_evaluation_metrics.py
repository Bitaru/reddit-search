import hashlib
import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _run_row(
    candidate_id: str,
    rank: int,
    *,
    complete: bool = True,
    scenario_id: str = "scenario-1",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "rank": rank,
        "context_complete": complete,
        "retrieval": {"branch": "lexical", "scenario_id": scenario_id},
    }


def _pool_row(candidate_id: str, *, scenario_id: str = "scenario-1") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "snapshot_id": "snap-1",
        "scenario_id": scenario_id,
        "app_id": "app-1",
        "app_profile_version": 1,
        "source": {"message_fullname": f"t1_{candidate_id}", "text": "body"},
    }


def _label(
    candidate_id: str,
    topic_fit: str,
    *,
    kind: str = "human",
    scenario_id: str | None = None,
) -> dict[str, object]:
    row = {"candidate_id": candidate_id, "topic_fit": topic_fit, "evaluator_kind": kind}
    if scenario_id is not None:
        row["scenario_id"] = scenario_id
    return row


def _write_run(directory: Path, scenario_id: str, rows: list[dict[str, object]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    run_file = directory / f"{scenario_id}.jsonl"
    _write_jsonl(run_file, rows)
    manifest = {
        "kind": "lexical_baseline_run_manifest",
        "schema_version": 1,
        "scenario_runs": {
            scenario_id: {
                "hit_count": len(rows),
                "sha256": __import__("hashlib").sha256(run_file.read_bytes()).hexdigest(),
            }
        },
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def test_precision_recall_and_mrr_over_judged_rows(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report

    pool_ids = [f"c{i}" for i in range(1, 7)]
    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row(cid) for cid in pool_ids])
    run = _write_run(
        tmp_path / "runs",
        "scenario-1",
        [_run_row(f"c{i}", i) for i in range(1, 7)],
    )
    # c2 relevant, c4 relevant, c5 negative, c1/c3/c6 unjudged.
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            _label("c2", "yes"),
            _label("c4", "yes"),
            _label("c5", "no"),
            _label("c1", "unknown"),
        ],
    )

    report = compute_run_report(run, pool, labels)
    scenario = report["scenarios"]["scenario-1"]
    assert scenario["returned_count"] == 6
    assert scenario["judged_count"] == 3
    assert scenario["relevant_count"] == 2
    assert scenario["precision_at_judged_count"] == pytest.approx(2 / 3)
    assert scenario["precision_at_returned_count"] is None
    assert scenario["judgment_coverage_at_returned_count"] == pytest.approx(3 / 6)
    assert scenario["pool_limited_recall"] == pytest.approx(2 / 2)
    assert scenario["mean_reciprocal_rank"] == pytest.approx(1 / 2)
    assert scenario["unjudged_count"] == 3
    # Unknown topic fit counts as coverage but never as a precision denominator.
    coverage = report["judgment_coverage"]
    assert coverage["yes_no_topic_fit_decisions"] == 3
    assert coverage["unknown_topic_fit_decisions"] == 1


def test_unjudged_rows_are_not_negatives(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row(cid) for cid in ("c1", "c2")])
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1), _run_row("c2", 2)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c1", "yes")])

    report = compute_run_report(run, pool, labels)
    scenario = report["scenarios"]["scenario-1"]
    assert scenario["relevant_count"] == 1
    assert scenario["unjudged_count"] == 1
    # Unjudged rows are excluded from judged precision, and returned precision
    # remains undefined until every returned slot has a judgment.
    assert scenario["precision_at_judged_count"] == 1.0
    assert scenario["precision_at_returned_count"] is None
    assert scenario["unjudged_candidate_ids"] == ["c2"]


def test_recall_is_none_without_relevant_pool_labels(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row(cid) for cid in ("c1", "c2")])
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1), _run_row("c2", 2)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c1", "no")])

    report = compute_run_report(run, pool, labels)
    scenario = report["scenarios"]["scenario-1"]
    assert scenario["pool_limited_recall"] is None
    assert scenario["mean_reciprocal_rank"] is None


def test_non_human_and_mock_labels_are_excluded(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row(cid) for cid in ("c1", "c2")])
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1), _run_row("c2", 2)])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [_label("c1", "yes", kind="human"), _label("c2", "no", kind="model")],
    )

    report = compute_run_report(run, pool, labels)
    scenario = report["scenarios"]["scenario-1"]
    assert scenario["judged_count"] == 1
    assert scenario["unjudged_candidate_ids"] == ["c2"]


def test_split_restriction_limits_measured_rows(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row(cid) for cid in ("c1", "c2")])
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1), _run_row("c2", 2)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c2", "yes")])
    splits = _write_jsonl(
        tmp_path / "splits.jsonl",
        [{"candidate_id": "c2", "group_id": "g1", "split": "dev"}],
    )

    report = compute_run_report(run, pool, labels, splits)
    scenario = report["scenarios"]["scenario-1"]
    assert scenario["returned_count"] == 1
    assert scenario["relevant_count"] == 1
    assert report["inputs"]["split_provenance"] == "unknown"

def test_manifest_rejects_stale_pool_and_tampered_splits(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report
    from reddit_search.evaluation.splits import build_group_split

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1"), _pool_row("c2")])
    split_dir = tmp_path / "split"
    build_group_split(pool, split_dir)
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c1", "yes")])
    splits = split_dir / "candidate_splits.jsonl"
    splits.write_text(splits.read_text() + '{"candidate_id":"x","split":"dev"}\n')
    with pytest.raises(ValueError, match="candidate split"):
        compute_run_report(run, pool, labels, splits)

    build_group_split(pool, split_dir)
    splits = split_dir / "candidate_splits.jsonl"
    stale = _write_jsonl(tmp_path / "stale.jsonl", [_pool_row("c1"), _pool_row("c3")])
    with pytest.raises(ValueError, match="pool sha256"):
        compute_run_report(run, stale, labels, splits)


def test_run_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1")])
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c1", "yes")])
    # Tamper with the ranked file after the manifest froze its hash.
    run_file = run / "scenario-1.jsonl"
    run_file.write_text(run_file.read_text() + json.dumps(_run_row("c1", 2)) + "\n")

    with pytest.raises(ValueError, match="does not match manifest"):
        compute_run_report(run, pool, labels)


def test_run_rows_outside_pool_remain_unjudged(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1")])
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("cX", 1)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [])

    report = compute_run_report(run, pool, labels)
    scenario = report["scenarios"]["scenario-1"]
    assert scenario["outside_pool_count"] == 1
    assert scenario["unjudged_task_keys"] == [
        {"candidate_id": "cX", "scenario_id": "scenario-1"}
    ]


def test_labels_join_by_candidate_and_scenario(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report

    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [_pool_row("c1", scenario_id="scenario-1"), _pool_row("c1", scenario_id="scenario-2")],
    )
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1)])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [_label("c1", "yes", scenario_id="scenario-2")],
    )

    report = compute_run_report(run, pool, labels)
    scenario = report["scenarios"]["scenario-1"]
    assert scenario["judged_count"] == 0
    assert scenario["unjudged_count"] == 1
    assert scenario["pool_relevant_count"] == 0


def test_write_run_report_is_atomic_json(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import write_run_report

    destination = write_run_report({"kind": "lexical_run_metrics_report"}, tmp_path)
    assert destination.name == "run_metrics_report.json"
    assert json.loads(destination.read_text())["kind"] == "lexical_run_metrics_report"
    assert not destination.with_suffix(".json.tmp").exists()

def test_comparison_provenance_accepts_matching_artifacts_and_rejects_ranked_or_labels_tamper(
    tmp_path: Path,
) -> None:
    from reddit_search.evaluation.metrics import validate_comparison_provenance
    from reddit_search.retrieval.comparison import canonical_hash
    run_dir = tmp_path / "runs-a-abc"
    run_dir.mkdir()
    ranked = run_dir / "scenario.jsonl"
    ranked.write_text('{"candidate_id":"c1","rank":1}\n')
    ranked_sha = hashlib.sha256(ranked.read_bytes()).hexdigest()
    run_manifest = {"scenario_runs": {"scenario": {"sha256": ranked_sha}}}
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest))
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"candidate_id":"c1","topic_fit":"yes"}\n')
    identity = {"labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest()}
    parent = canonical_hash(identity)
    run_manifest["parent_comparison_identity"] = parent
    run_manifest_path.write_text(json.dumps(run_manifest))
    run_manifest_sha = hashlib.sha256(run_manifest_path.read_bytes()).hexdigest()
    comparison = {
        "kind": "retrieval_comparison_manifest",
        "identity": identity,
        "comparison_identity": parent,
        "artifacts": {"A": {"run_manifest_sha256": run_manifest_sha}},
    }
    (tmp_path / "comparison_manifest.json").write_text(json.dumps(comparison))
    validate_comparison_provenance(tmp_path / "comparison_manifest.json", run_dir, labels)
    ranked.write_text(ranked.read_text() + "tampered\n")
    with pytest.raises(ValueError):
        validate_comparison_provenance(tmp_path / "comparison_manifest.json", run_dir, labels)
    ranked.write_text('{"candidate_id":"c1","rank":1}\n')
    labels.write_text("tampered\n")
    with pytest.raises(ValueError):
        validate_comparison_provenance(tmp_path / "comparison_manifest.json", run_dir, labels)


def test_legacy_metrics_report_marks_comparison_provenance_unknown(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report
    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1")])
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c1", "yes")])
    report = compute_run_report(run, pool, labels)
    assert report["inputs"]["comparison_provenance"] == "unknown"


@pytest.mark.parametrize(
    ("fullname", "revision"),
    [("t1_c1", "rev-1"), ("t1_c1", None)],
)
def test_tombstoned_pool_source_is_excluded_from_metrics(
    tmp_path: Path, fullname: str, revision: str | None
) -> None:
    from reddit_search.evaluation.metrics import compute_run_report
    from reddit_search.ingest.invalidation import load_tombstone_ledger

    pool_row = _pool_row("c1")
    pool_row["source"] = {
        "message_fullname": fullname,
        "source_revision_id": "rev-1",
        "text": "body",
    }
    pool = _write_jsonl(tmp_path / "pool.jsonl", [pool_row, _pool_row("c2")])
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1), _run_row("c2", 2)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c1", "yes"), _label("c2", "yes")])
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps({"message_fullname": fullname, "source_revision_id": revision, "reason": "gone"})
        + "\n",
        encoding="utf-8",
    )
    report = compute_run_report(
        run, pool, labels, tombstone_ledger=load_tombstone_ledger(tombstones)
    )
    assert report["scenarios"]["scenario-1"]["returned_count"] == 1
    assert report["scenarios"]["scenario-1"]["relevant_count"] == 1


def test_metrics_report_records_tombstone_ledger_honestly(tmp_path: Path) -> None:
    from reddit_search.evaluation.metrics import compute_run_report
    from reddit_search.ingest.invalidation import load_tombstone_ledger

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1")])
    run = _write_run(tmp_path / "runs", "scenario-1", [_run_row("c1", 1)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c1", "yes")])

    without_ledger = compute_run_report(run, pool, labels)
    assert without_ledger["inputs"]["tombstone_ledger"] == "not_checked"

    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t1_missing",
                "source_revision_id": "rev-1",
                "reason": "gone",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with_ledger = compute_run_report(
        run, pool, labels, tombstone_ledger=load_tombstone_ledger(tombstones)
    )
    recorded = with_ledger["inputs"]["tombstone_ledger"]
    assert recorded["count"] == 1
    assert recorded["sha256"]
