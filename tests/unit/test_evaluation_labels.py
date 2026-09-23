import json
from pathlib import Path

import pytest


def _pool_row(candidate_id: str, text: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "snapshot_id": "snap-1",
        "scenario_id": "scenario-1",
        "app_id": "app-1",
        "app_profile_version": 1,
        "source": {
            "message_fullname": f"t1_{candidate_id}",
            "source_revision_id": "rev-1",
            "field": "body",
            "text": text,
            "title": "A title",
            "subreddit": "personalfinance",
        },
    }


def _evidence(text: str, **overrides: object) -> dict[str, object]:
    evidence = {
        "message_fullname": "t1_c1",
        "source_revision_id": "rev-1",
        "field": "body",
        "start": 0,
        "end": len(text),
        "quote": text,
        "supports": "topic_fit",
        "attribution": "focal_author",
        "quoted": False,
    }
    evidence.update(overrides)
    return evidence


def _label(candidate_id: str, **overrides: object) -> dict[str, object]:
    label = {
        "candidate_id": candidate_id,
        "topic_fit": "yes",
        "evaluator_kind": "human",
    }
    label.update(overrides)
    return label


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_export_hides_retrieval_and_defaults_label(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import export_label_worksheet

    pool_path = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            _pool_row("c1", "I need a budgeting tool."),
            dict(_pool_row("c2", "Misc."), rank=3, score=0.9, matched_rule_ids=["r1"]),
        ],
    )

    report = export_label_worksheet(pool_path, tmp_path / "worksheet")

    rows = [
        json.loads(line)
        for line in (tmp_path / "worksheet" / "label_worksheet.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert report["row_count"] == 2
    assert rows[0]["candidate_id"] == "c1"
    assert rows[0]["source"]["message_fullname"] == "t1_c1"
    assert rows[0]["label"]["topic_fit"] == "unknown"
    assert rows[0]["label"]["product_fit"] == "not_evaluated"
    assert rows[0]["label"]["live_status"] == "unverified"
    assert rows[0]["label"]["policy_status"] == "unverified"
    for row in rows:
        assert "rank" not in row
        assert "score" not in row
        assert "matched_rule_ids" not in row
        assert "retrieval" not in row
        assert "selection" not in row
        assert "review_queue" not in row


def test_nested_provenance_is_exported_and_protected(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import (
        _identity_digest,
        export_label_worksheet,
        import_label_rows,
    )

    row = _pool_row("c1", "text")
    row["source"]["context_recipe_version"] = "recipe-2"
    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [row])
    export_label_worksheet(pool_path, tmp_path / "worksheet")
    worksheet_path = tmp_path / "worksheet" / "label_worksheet.jsonl"
    exported = json.loads(worksheet_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (tmp_path / "worksheet" / "label_worksheet_manifest.json").read_text(encoding="utf-8")
    )

    assert exported["source"]["source_revision_id"] == "rev-1"
    assert exported["source"]["context_recipe_version"] == "recipe-2"
    assert manifest["dependency_identity_sha256"] == _identity_digest([row])
    assert manifest["identity_coverage"]["source_revision_id"] == 1
    assert manifest["identity_coverage"]["context_recipe_version"] == 1

    changed = dict(row)
    changed["source"] = dict(row["source"], context_recipe_version="recipe-3")
    assert _identity_digest([changed]) != _identity_digest([row])

    tampered = dict(exported)
    tampered["source_revision_id"] = "rev-tampered"
    tampered["source"] = dict(exported["source"], source_revision_id="rev-tampered")
    tampered_path = _write_jsonl(tmp_path / "tampered.jsonl", [tampered])
    report = import_label_rows(
        pool_path, tampered_path, tmp_path / "imported", require_complete=False
    )
    assert report["invalid_count"] >= 1
    assert any(
        "source_revision_id does not match pool" in failure
        for row in report["rows"]
        for failure in row["failures"]
    )


def test_import_roundtrip_accepts_complete_valid_labels(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import export_label_worksheet, import_label_rows

    text = "I need a budgeting tool."
    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", text)])
    export_label_worksheet(pool_path, tmp_path / "worksheet")
    labels_path = tmp_path / "worksheet" / "label_worksheet.jsonl"

    report = import_label_rows(pool_path, labels_path, tmp_path / "imported")

    assert report["valid_count"] == 1
    assert report["invalid_count"] == 0
    labels = [
        json.loads(line)
        for line in (tmp_path / "imported" / "labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert labels[0]["candidate_id"] == "c1"
    assert labels[0]["topic_fit"] == "unknown"


def test_import_rejects_unknown_fields_and_bad_enums(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import export_label_worksheet, import_label_rows

    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "text")])
    export_label_worksheet(pool_path, tmp_path / "worksheet")
    labels_path = tmp_path / "worksheet" / "label_worksheet.jsonl"
    rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    bad_enum = dict(rows[0])
    bad_enum["label"] = dict(rows[0]["label"], topic_fit="maybe")
    unknown_field = dict(rows[0])
    unknown_field["candidate_id"] = "c1"
    unknown_field["label"] = dict(rows[0]["label"], score=0.9)

    labels_in = _write_jsonl(tmp_path / "labels_in.jsonl", [bad_enum, unknown_field])
    report = import_label_rows(pool_path, labels_in, tmp_path / "imported", require_complete=False)

    assert report["valid_count"] == 0
    assert report["invalid_count"] == 2


def test_import_rejects_label_task_absent_from_pool_without_evidence(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_label_rows

    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "text")])
    labels_path = _write_jsonl(
        tmp_path / "labels.jsonl",
        [_label("not-in-pool", evidence=[])],
    )

    report = import_label_rows(
        pool_path, labels_path, tmp_path / "imported", require_complete=False
    )

    assert report["valid_count"] == 0
    assert report["invalid_count"] >= 1
    assert "candidate/scenario identity does not match pool" in report["rows"][0]["failures"]


def test_import_rejects_evidence_bounds_and_quote_mismatch(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import export_label_worksheet, import_label_rows

    text = "I need a budgeting tool."
    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", text)])
    export_label_worksheet(pool_path, tmp_path / "worksheet")
    rows = [
        json.loads(line)
        for line in (tmp_path / "worksheet" / "label_worksheet.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    out_of_bounds = dict(rows[0])
    out_of_bounds["label"] = dict(
        rows[0]["label"],
        evidence=[_evidence(text, start=0, end=len(text) + 5)],
    )
    quote_mismatch = dict(rows[0])
    quote_mismatch["label"] = dict(
        rows[0]["label"],
        evidence=[_evidence(text, start=0, end=len(text), quote="different text")],
    )
    labels_in = _write_jsonl(tmp_path / "labels_in.jsonl", [out_of_bounds, quote_mismatch])

    report = import_label_rows(pool_path, labels_in, tmp_path / "imported", require_complete=False)

    assert report["invalid_count"] == 2


def test_import_rejects_author_requirement_from_other_author(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import export_label_worksheet, import_label_rows

    text = "I need a budgeting tool."
    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", text)])
    export_label_worksheet(pool_path, tmp_path / "worksheet")
    rows = [
        json.loads(line)
        for line in (tmp_path / "worksheet" / "label_worksheet.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    misattributed = dict(rows[0])
    misattributed["label"] = dict(
        rows[0]["label"],
        evidence=[_evidence(text, supports="author_requirement", attribution="quoted_other")],
    )
    labels_in = _write_jsonl(tmp_path / "labels_in.jsonl", [misattributed])

    report = import_label_rows(pool_path, labels_in, tmp_path / "imported", require_complete=False)

    assert report["invalid_count"] >= 1


def test_compatible_product_fit_requires_verified_claim_ids(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_label_rows

    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "text")])
    labels_in = _write_jsonl(
        tmp_path / "labels_in.jsonl",
        [_label("c1", product_fit="compatible", supported_claim_ids=["claim-9"])],
    )

    report = import_label_rows(
        pool_path,
        labels_in,
        tmp_path / "imported",
        verified_claim_ids=("claim-1",),
        require_complete=False,
    )

    assert report["invalid_count"] >= 1
    assert any("verified" in failure for row in report["rows"] for failure in row["failures"])


def test_mock_evaluator_kind_is_rejected(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_label_rows

    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "text")])
    labels_in = _write_jsonl(tmp_path / "labels_in.jsonl", [_label("c1", evaluator_kind="mock")])

    report = import_label_rows(pool_path, labels_in, tmp_path / "imported", require_complete=False)

    assert report["invalid_count"] >= 1


def test_require_complete_flags_missing_rows(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import export_label_worksheet, import_label_rows

    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "a"), _pool_row("c2", "b")])
    export_label_worksheet(pool_path, tmp_path / "worksheet")

    report = import_label_rows(
        pool_path,
        tmp_path / "worksheet" / "label_worksheet.jsonl",
        tmp_path / "imported",
    )

    assert report["invalid_count"] == 0


def test_require_complete_false_tolerates_missing_rows(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import export_label_worksheet, import_label_rows

    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "a"), _pool_row("c2", "b")])
    export_label_worksheet(pool_path, tmp_path / "worksheet")

    report = import_label_rows(
        pool_path,
        tmp_path / "worksheet" / "label_worksheet.jsonl",
        tmp_path / "imported",
        require_complete=False,
    )

    assert report["valid_count"] == 2
    assert report["invalid_count"] == 0


def test_legacy_import_maps_relevance_and_keeps_sidecar_and_duplicates(
    tmp_path: Path,
) -> None:
    from reddit_search.evaluation.labels import import_legacy_worksheet

    legacy_path = _write_jsonl(
        tmp_path / "review_worksheet.jsonl",
        [
            {
                "candidate_id": "c1",
                "annotation": {
                    "topic_fit": "relevant",
                    "need_clarity": "clear",
                    "duplicate_of": "c2",
                    "reviewer_note": "old note",
                },
            },
            {
                "candidate_id": "c2",
                "annotation": {"topic_fit": "not_relevant"},
            },
            {
                "candidate_id": "c3",
                "annotation": {"topic_fit": "uncertain"},
            },
        ],
    )

    report = import_legacy_worksheet(legacy_path, tmp_path / "imported")

    assert report["row_count"] == 3
    labels = [
        json.loads(line)
        for line in (tmp_path / "imported" / "labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["topic_fit"] for row in labels] == ["yes", "no", "unknown"]
    sidecar = [
        json.loads(line)
        for line in (tmp_path / "imported" / "legacy_sidecar.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sidecar[0]["legacy"]["annotation"]["need_clarity"] == "clear"
    assert sidecar[0]["legacy"]["annotation"]["reviewer_note"] == "old note"
    duplicates = [
        json.loads(line)
        for line in (tmp_path / "imported" / "duplicate_links.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert duplicates == [{"candidate_id": "c1", "duplicate_of": "c2"}]


def test_legacy_import_does_not_invent_richer_fields(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_legacy_worksheet

    legacy_path = _write_jsonl(
        tmp_path / "review_worksheet.jsonl",
        [{"candidate_id": "c1", "annotation": {"topic_fit": "relevant"}}],
    )

    import_legacy_worksheet(legacy_path, tmp_path / "imported")

    labels = [
        json.loads(line)
        for line in (tmp_path / "imported" / "labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert labels[0]["speaker_intent"] == "unknown"
    assert labels[0]["product_fit"] == "not_evaluated"
    assert labels[0]["live_status"] == "unverified"
    assert labels[0]["policy_status"] == "unverified"
    assert labels[0]["supported_claim_ids"] == []


def test_export_roundtrip_deterministic(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import export_label_worksheet

    pool_path = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "text")])
    export_label_worksheet(pool_path, tmp_path / "a")
    export_label_worksheet(pool_path, tmp_path / "b")

    first = (tmp_path / "a" / "label_worksheet.jsonl").read_bytes()
    second = (tmp_path / "b" / "label_worksheet.jsonl").read_bytes()
    assert first == second


def test_legacy_pending_rows_stay_in_sidecar_not_labels(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_legacy_worksheet

    legacy_path = _write_jsonl(
        tmp_path / "review_worksheet.jsonl",
        [
            {
                "candidate_id": "pending-1",
                "annotation": {
                    "review_status": "pending",
                    "topic_fit": "unreviewed",
                    "reviewer_note": "not a judgment",
                },
            }
        ],
    )

    report = import_legacy_worksheet(legacy_path, tmp_path / "imported")

    assert report["row_count"] == 0
    assert report["skipped_unreviewed_count"] == 1
    assert report["input_row_count"] == 1
    assert len((tmp_path / "imported" / "legacy_sidecar.jsonl").read_text().splitlines()) == 1


def test_reuse_labels_requires_same_task_source_and_rubric(tmp_path: Path) -> None:
    from reddit_search.evaluation.reuse import reuse_labels

    source_pool = _write_jsonl(tmp_path / "source-pool.jsonl", [_pool_row("c1", "same")])
    target_pool = _write_jsonl(tmp_path / "target-pool.jsonl", [_pool_row("c1", "same")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-1",
                "topic_fit": "yes",
                "evaluator_kind": "human",
                "rubric_version": "topic-fit-v1",
            }
        ],
    )

    report = reuse_labels(
        source_pool,
        labels,
        target_pool,
        tmp_path / "reused",
        rubric_version="topic-fit-v1",
    )

    assert report["reused_label_count"] == 1
    reused = [
        json.loads(line) for line in (tmp_path / "reused" / "labels.jsonl").read_text().splitlines()
    ]
    assert reused[0]["candidate_id"] == "c1"
    assert reused[0]["scenario_id"] == "scenario-1"

    changed_target = _write_jsonl(tmp_path / "changed-pool.jsonl", [_pool_row("c1", "changed")])
    mismatch = reuse_labels(
        source_pool,
        labels,
        changed_target,
        tmp_path / "mismatch",
        rubric_version="topic-fit-v1",
    )
    assert mismatch["reused_label_count"] == 0
    assert mismatch["skipped"] == {"source_fingerprint_mismatch": 1}


def test_reuse_rejects_changed_parent_context_identity(tmp_path: Path) -> None:
    from reddit_search.evaluation.reuse import reuse_labels

    source = _pool_row("c1", "same")
    source["source"] = dict(
        source["source"],
        context_text="parent\nsame",
        context_dependency_identity={"parents": ["t1_parent"]},
    )
    target = _pool_row("c1", "same")
    target["source"] = dict(
        target["source"],
        context_text="parent\nchanged",
        context_dependency_identity={"parents": ["t1_other"]},
    )
    source_pool = _write_jsonl(tmp_path / "source-pool.jsonl", [source])
    target_pool = _write_jsonl(tmp_path / "target-pool.jsonl", [target])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-1",
                "topic_fit": "yes",
                "evaluator_kind": "human",
                "rubric_version": "topic-fit-v1",
            }
        ],
    )

    report = reuse_labels(
        source_pool, labels, target_pool, tmp_path / "reused", rubric_version="topic-fit-v1"
    )

    assert report["reused_label_count"] == 0
    assert report["skipped"] == {"context_dependency_identity_mismatch": 1}


def test_reuse_rejects_same_profile_version_with_changed_profile_hash(tmp_path: Path) -> None:
    from reddit_search.evaluation.reuse import reuse_labels

    source = _pool_row("c1", "same")
    source["app_profile_sha256"] = "a" * 64
    target = _pool_row("c1", "same")
    target["app_profile_sha256"] = "b" * 64
    source_pool = _write_jsonl(tmp_path / "source-pool.jsonl", [source])
    target_pool = _write_jsonl(tmp_path / "target-pool.jsonl", [target])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-1",
                "topic_fit": "yes",
                "evaluator_kind": "human",
                "rubric_version": "topic-fit-v1",
            }
        ],
    )

    report = reuse_labels(
        source_pool,
        labels,
        target_pool,
        tmp_path / "reused",
        rubric_version="topic-fit-v1",
    )

    assert report["reused_label_count"] == 0
    assert report["skipped"] == {"source_fingerprint_mismatch": 1}


def test_reuse_preserves_profile_hash_on_legacy_compatible_row(tmp_path: Path) -> None:
    from reddit_search.evaluation.reuse import reuse_labels

    source = _pool_row("c1", "same")
    source["app_profile_sha256"] = "a" * 64
    target = _pool_row("c1", "same")
    target["app_profile_sha256"] = "a" * 64
    source_pool = _write_jsonl(tmp_path / "source-pool.jsonl", [source])
    target_pool = _write_jsonl(tmp_path / "target-pool.jsonl", [target])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-1",
                "topic_fit": "yes",
                "evaluator_kind": "human",
                "rubric_version": "topic-fit-v1",
            }
        ],
    )

    reuse_labels(
        source_pool,
        labels,
        target_pool,
        tmp_path / "reused",
        rubric_version="topic-fit-v1",
    )
    reused = json.loads((tmp_path / "reused" / "labels.jsonl").read_text().strip())
    assert reused["app_profile_sha256"] == "a" * 64


def test_collect_worksheet_labels_keeps_duplicate_candidate_scenarios(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import collect_worksheet_labels

    pool = _write_jsonl(
        tmp_path / "pool.jsonl",
        [_pool_row("c1", "same"), dict(_pool_row("c1", "same"), scenario_id="scenario-2")],
    )
    worksheet = _write_jsonl(
        tmp_path / "worksheet.jsonl",
        [
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-1",
                "annotation": {
                    "review_status": "complete",
                    "topic_fit": "relevant",
                    "need_clarity": "clear",
                    "speaker_intent": "unreviewed",
                    "resolution_in_available_context": "unreviewed",
                },
            },
            {
                "candidate_id": "c1",
                "scenario_id": "scenario-2",
                "annotation": {
                    "review_status": "complete",
                    "topic_fit": "not_relevant",
                    "need_clarity": "unclear",
                },
            },
        ],
    )

    result = collect_worksheet_labels(pool, worksheet, tmp_path / "collected")
    labels = [
        json.loads(line)
        for line in (tmp_path / "collected" / "labels.jsonl").read_text().splitlines()
    ]

    assert result["collected_count"] == 2
    assert [(row["candidate_id"], row["scenario_id"]) for row in labels] == [
        ("c1", "scenario-1"),
        ("c1", "scenario-2"),
    ]
    assert labels[0]["speaker_intent"] == "unknown"
    assert labels[0]["resolution_in_available_context"] == "unknown"

def test_collect_compatible_product_fit_requires_supported_claim_ids(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import collect_worksheet_labels

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "hello")])
    worksheet = _write_jsonl(
        tmp_path / "worksheet.jsonl",
        [
            {
                "candidate_id": "c1",
                "annotation": {
                    "review_status": "complete",
                    "topic_fit": "relevant",
                    "need_clarity": "clear",
                    "product_fit": "compatible",
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="compatible product_fit requires supported_claim_ids"):
        collect_worksheet_labels(pool, worksheet, tmp_path / "collected")


def test_collect_carries_supported_claim_ids(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import collect_worksheet_labels

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "hello")])
    worksheet = _write_jsonl(
        tmp_path / "worksheet.jsonl",
        [
            {
                "candidate_id": "c1",
                "annotation": {
                    "review_status": "complete",
                    "topic_fit": "relevant",
                    "need_clarity": "clear",
                    "product_fit": "compatible",
                    "supported_claim_ids": ["app.capability"],
                },
            }
        ],
    )

    result = collect_worksheet_labels(pool, worksheet, tmp_path / "collected")
    labels = [
        json.loads(line)
        for line in (tmp_path / "collected" / "labels.jsonl").read_text().splitlines()
    ]
    assert result["collected_count"] == 1
    assert labels[0]["product_fit"] == "compatible"
    assert labels[0]["supported_claim_ids"] == ["app.capability"]


def test_collect_rejects_claim_ids_without_compatible_fit(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import collect_worksheet_labels

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "hello")])
    worksheet = _write_jsonl(
        tmp_path / "worksheet.jsonl",
        [
            {
                "candidate_id": "c1",
                "annotation": {
                    "review_status": "complete",
                    "topic_fit": "relevant",
                    "need_clarity": "clear",
                    "product_fit": "needs_clarification",
                    "supported_claim_ids": ["app.capability"],
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="supported_claim_ids requires compatible product_fit"):
        collect_worksheet_labels(pool, worksheet, tmp_path / "collected")


def test_import_rejects_invalidated_sibling_context(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_label_rows
    from reddit_search.ingest.invalidation import load_tombstone_ledger

    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c1", scenario_id="scenario-1")])
    pool_row = _pool_row("c1", "text")
    pool_row["context"] = {"message_fullnames": ["t1_context"]}
    pool = _write_jsonl(tmp_path / "pool.jsonl", [pool_row])
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        '{"message_fullname":"t1_context","source_revision_id":null,"reason":"gone"}\n',
        encoding="utf-8",
    )
    report = import_label_rows(
        pool, labels, tmp_path / "out", tombstone_ledger=load_tombstone_ledger(tombstones)
    )
    assert report["valid_count"] == 0
    assert any("invalidated" in failure for row in report["rows"] for failure in row["failures"])


def test_legacy_label_import_defaults_new_fields(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_label_rows

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "hello")])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label("c1", scenario_id="scenario-1")])
    report = import_label_rows(pool, labels, tmp_path / "out")
    assert report["valid_count"] == 1
    row = json.loads((tmp_path / "out" / "labels.jsonl").read_text().strip())
    assert row["speaker_intent"] == "unknown"
    assert row["resolution_in_available_context"] == "unknown"
    assert row["product_fit"] == "not_evaluated"


def test_compatible_product_fit_without_supporting_claim_fails_closed(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_label_rows

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "hello")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [_label("c1", scenario_id="scenario-1", product_fit="compatible")],
    )
    report = import_label_rows(pool, labels, tmp_path / "out")
    assert report["invalid_count"] >= 1


def test_compatible_product_fit_with_verified_claim_imports(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_label_rows

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "hello")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            _label(
                "c1",
                scenario_id="scenario-1",
                product_fit="compatible",
                supported_claim_ids=["app.capability"],
            )
        ],
    )
    report = import_label_rows(
        pool, labels, tmp_path / "out", verified_claim_ids=("app.capability",)
    )
    assert report["valid_count"] == 1
    row = json.loads((tmp_path / "out" / "labels.jsonl").read_text().strip())
    assert row["product_fit"] == "compatible"
    assert row["supported_claim_ids"] == ["app.capability"]


def test_compatible_product_fit_unverified_claim_fails_closed(tmp_path: Path) -> None:
    from reddit_search.evaluation.labels import import_label_rows

    pool = _write_jsonl(tmp_path / "pool.jsonl", [_pool_row("c1", "hello")])
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [
            _label(
                "c1",
                scenario_id="scenario-1",
                product_fit="compatible",
                supported_claim_ids=["app.capability"],
            )
        ],
    )
    report = import_label_rows(
        pool, labels, tmp_path / "out", verified_claim_ids=("app.other",)
    )
    assert report["valid_count"] == 0
    assert any(
        "not verified" in failure for row in report["rows"] for failure in row["failures"]
    )
