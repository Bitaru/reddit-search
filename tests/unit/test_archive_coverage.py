from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reddit_search.evaluation.archive_coverage import derive_archive_coverage


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _audit(root: Path, *, snapshot: str = "snap") -> Path:
    audit = root / "audit"
    audit.mkdir()
    registry = root / "registry.json"
    corpus = root / "corpus.sqlite"
    exposure = root / "exposure.jsonl"
    registry.write_text("{}", encoding="utf-8")
    corpus.write_bytes(b"synthetic corpus")
    _jsonl(
        exposure,
        [
            {
                "candidate_id": "c1",
                "message_fullname": "t3_hit",
                "source_revision_id": "rev1",
                "thread_fullname": "t3_thread",
                "focus_text": "body",
                "status": "matched",
                "archive_matches": [{"source_id": "src", "line": 1}],
                "snapshot_id": snapshot,
                "line": 1,
            },
            {
                "candidate_id": "c2",
                "message_fullname": "t3_missing",
                "source_revision_id": "rev2",
                "thread_fullname": "t3_thread2",
                "focus_text": "missing",
                "status": "unresolved",
                "archive_matches": [],
                "snapshot_id": snapshot,
                "line": 2,
            },
        ],
    )
    outputs = {
        "archive_exposure_matches.jsonl": exposure.read_bytes(),
        "archive_index_coverage.jsonl": b'{"source_id":"src","matched_records":1}\n',
        "archive_identity_collisions.jsonl": b"",
    }
    output_meta: dict[str, dict[str, object]] = {}
    for name, data in outputs.items():
        path = audit / name
        path.write_bytes(data)
        output_meta[name] = {"sha256": _sha(path), "rows": data.count(b"\n")}
    inputs = {
        "registry": {"path": str(registry), "sha256": _sha(registry)},
        "corpus": {"path": str(corpus), "sha256": _sha(corpus)},
        "exposure": [{"path": str(exposure), "sha256": _sha(exposure)}],
    }
    identity = {
        "schema_version": 3,
        "algorithm_version": "archive-audit-exact-match-v1",
        "snapshot_id": snapshot,
        "input_hashes": inputs,
    }
    manifest = {
        "kind": "registered_archive_audit", "schema_version": 3, "complete": True,
        "snapshot_id": snapshot, "identity": identity,
        "algorithm": {"retained_exact_match_only": True, "near_text": False},
        "counts": {"matched": 1, "unresolved": 1, "collisions": 0, "exposure_errors": 0},
        "input_hashes": inputs, "outputs": output_meta,
    }
    (audit / "archive_audit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return audit


def test_derives_exact_accounting_and_is_deterministic_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    audit = _audit(tmp_path)
    before = {path: path.read_bytes() for path in audit.iterdir()}
    first = derive_archive_coverage(audit, tmp_path / "out-a")
    second = derive_archive_coverage(audit, tmp_path / "out-b")
    assert first["counts"] == {
        "exposure_rows": 2,
        "matched": 1,
        "unresolved": 1,
        "collisions": 0,
        "exposure_errors": 0,
    }
    assert first["outputs"] == second["outputs"]
    assert (
        (tmp_path / "out-a/derived_coverage_manifest.json").read_bytes()
        == (tmp_path / "out-b/derived_coverage_manifest.json").read_bytes()
    )
    assert {path: path.read_bytes() for path in audit.iterdir()} == before
    unresolved = json.loads(
        (tmp_path / "out-a/unresolved_archive_exclusions.jsonl").read_text()
    )
    assert unresolved["coverage_status"] == "unresolved"
    assert unresolved["archive_matches"] == []
    assert "not a tombstone" in first["derivation_policy"]["unresolved_semantics"]


@pytest.mark.parametrize("tamper", ["hash", "rows", "incomplete", "snapshot"])
def test_rejects_untrusted_or_tampered_audit(tmp_path: Path, tamper: str) -> None:
    audit = _audit(tmp_path)
    manifest_path = audit / "archive_audit_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if tamper == "hash":
        (audit / "archive_exposure_matches.jsonl").write_text("{}\n", encoding="utf-8")
    elif tamper == "rows":
        manifest["outputs"]["archive_exposure_matches.jsonl"]["rows"] = 99
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "incomplete":
        manifest["complete"] = False
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        manifest["snapshot_id"] = "wrong"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        derive_archive_coverage(audit, tmp_path / "out")


def _rematch(root: Path, audit: Path, *, flip: bool = False) -> Path:
    """Build a rematch directory over the synthetic audit, optionally recovering c2."""
    import hashlib

    audit_manifest = json.loads((audit / "archive_audit_manifest.json").read_text())
    exposure_rows = [
        json.loads(line)
        for line in (audit / "archive_exposure_matches.jsonl").read_text().splitlines()
    ]
    if flip:
        for row in exposure_rows:
            if row["candidate_id"] == "c2":
                row["status"] = "matched"
                row["archive_matches"] = [{"source_id": "src", "line": 2}]
    matched = sum(1 for row in exposure_rows if row["status"] == "matched")
    unresolved = len(exposure_rows) - matched
    rematch = root / "rematch"
    rematch.mkdir()
    outputs: dict[str, bytes] = {
        "archive_exposure_matches.jsonl": b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode() for row in exposure_rows
        ),
        "archive_index_coverage.jsonl": (audit / "archive_index_coverage.jsonl").read_bytes(),
        "archive_identity_collisions.jsonl": b"",
    }
    output_meta: dict[str, dict[str, object]] = {}
    for name, data in outputs.items():
        path = rematch / name
        path.write_bytes(data)
        output_meta[name] = {"sha256": hashlib.sha256(data).hexdigest(), "rows": data.count(b"\n")}
    manifest = {
        "kind": "archive_audit_rematch",
        "schema_version": 1,
        "provenance": "recomputed_from_index_artifacts",
        "fresh_archive_scan": False,
        "supersedes_counts": True,
        "supersession_note": "test supersession note",
        "snapshot_id": audit_manifest["snapshot_id"],
        "audit_manifest_sha256": _sha(audit / "archive_audit_manifest.json"),
        "audit_directory": str(audit),
        "audit_identity": audit_manifest["identity"],
        "audit_counts": audit_manifest["counts"],
        "input_hashes": {"corpus": {"path": "corpus", "sha256": "x"}, "exposure": []},
        "index_partition_count": 1,
        "algorithm": {"identity": ["fullname"], "near_text": False, "partition_count": 64},
        "outputs": output_meta,
        "counts": {
            "matched": matched,
            "unresolved": unresolved,
            "collisions": 0,
            "exposure_errors": 0,
        },
        "per_source_matched": {"src": matched},
        "run_identity": "a" * 64,
    }
    (rematch / "rematch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rematch


def test_rematch_derives_corrected_counts_and_records_supersession(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    rematch = _rematch(tmp_path, audit, flip=True)
    result = derive_archive_coverage(audit, tmp_path / "out", rematch_directory=rematch)
    assert result["counts"] == {
        "exposure_rows": 2,
        "matched": 2,
        "unresolved": 0,
        "collisions": 0,
        "exposure_errors": 0,
    }
    policy = result["derivation_policy"]
    assert policy["provenance"] == "recomputed_from_index_artifacts"
    assert policy["fresh_archive_scan"] is False
    assert policy["supersedes_counts"] is True
    assert policy["superseded_audit_counts"]["matched"] == 1
    assert policy["superseded_audit_counts"]["unresolved"] == 1
    assert policy["rematch_manifest_sha256"] == _sha(rematch / "rematch_manifest.json")
    matched_rows = [
        json.loads(line)
        for line in (tmp_path / "out/matched_archive_coverage.jsonl").read_text().splitlines()
    ]
    assert len(matched_rows) == 2
    assert all(row["coverage_status"] == "matched" for row in matched_rows)
    assert not (tmp_path / "out/unresolved_archive_exclusions.jsonl").read_text()


def test_rematch_without_recovery_matches_audit_semantics_with_supersession(
    tmp_path: Path,
) -> None:
    audit = _audit(tmp_path)
    rematch = _rematch(tmp_path, audit)
    result = derive_archive_coverage(audit, tmp_path / "out", rematch_directory=rematch)
    assert result["counts"]["matched"] == 1
    assert result["counts"]["unresolved"] == 1


@pytest.mark.parametrize(
    "tamper",
    [
        "kind",
        "provenance",
        "scan",
        "supersede",
        "snapshot",
        "audit_binding",
        "audit_counts",
        "hash",
        "collisions",
    ],
)
def test_rejects_untrusted_or_tampered_rematch(tmp_path: Path, tamper: str) -> None:
    audit = _audit(tmp_path)
    rematch = _rematch(tmp_path, audit, flip=True)
    manifest_path = rematch / "rematch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if tamper == "kind":
        manifest["kind"] = "something_else"
    elif tamper == "provenance":
        manifest["provenance"] = "fresh_archive_scan"
    elif tamper == "scan":
        manifest["fresh_archive_scan"] = True
    elif tamper == "supersede":
        manifest["supersedes_counts"] = False
    elif tamper == "snapshot":
        manifest["snapshot_id"] = "wrong"
    elif tamper == "audit_binding":
        manifest["audit_manifest_sha256"] = "0" * 64
    elif tamper == "audit_counts":
        manifest["audit_counts"] = {
            "matched": 99,
            "unresolved": 0,
            "collisions": 0,
            "exposure_errors": 0,
        }
    elif tamper == "hash":
        (rematch / "archive_exposure_matches.jsonl").write_text("{}\n", encoding="utf-8")
    else:
        manifest["counts"]["collisions"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        derive_archive_coverage(audit, tmp_path / "out", rematch_directory=rematch)


def test_rematch_total_mismatch_rejected(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    rematch = _rematch(tmp_path, audit, flip=True)
    manifest_path = rematch / "rematch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # keep artifacts consistent (2 rows) but claim a total that contradicts the audit
    manifest["counts"] = {"matched": 1, "unresolved": 0, "collisions": 0, "exposure_errors": 0}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exposure total does not match"):
        derive_archive_coverage(audit, tmp_path / "out", rematch_directory=rematch)


def test_rematch_tolerates_drifted_corpus_and_discloses_it(tmp_path: Path) -> None:
    """A present-but-drifted corpus input is disclosed, not silently accepted or fatal."""
    audit = _audit(tmp_path)
    # rewrite the audit's bound corpus file with different bytes, rebind the manifest hash
    audit_manifest_path = audit / "archive_audit_manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text())
    corpus_path = Path(audit_manifest["input_hashes"]["corpus"]["path"])
    corpus_path.write_bytes(b"drifted corpus bytes")
    # the audit manifest itself still binds the old corpus sha; derivation records the drift
    rematch = _rematch(tmp_path, audit, flip=True)
    result = derive_archive_coverage(audit, tmp_path / "out", rematch_directory=rematch)
    drift = result["derivation_policy"]["input_drift"]
    assert len(drift) == 1
    assert drift[0]["label"] == "corpus input"
    assert drift[0]["observed_sha256"] == _sha(corpus_path)
    assert drift[0]["expected_sha256"] == audit_manifest["input_hashes"]["corpus"]["sha256"]
    assert result["counts"]["matched"] == 2


def test_rematch_missing_corpus_input_still_fails_closed(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    audit_manifest_path = audit / "archive_audit_manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text())
    corpus_path = Path(audit_manifest["input_hashes"]["corpus"]["path"])
    corpus_path.unlink()
    rematch = _rematch(tmp_path, audit, flip=True)
    with pytest.raises(ValueError, match="corpus input is missing"):
        derive_archive_coverage(audit, tmp_path / "out", rematch_directory=rematch)
