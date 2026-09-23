"""Re-match a completed archive audit's index artifacts with the fixed matcher.

The published r2 audit predates the (source_id, bucket) connection-pool fix in
``archive_partitioned``: its matcher probed only one source per bucket, so
submission identities held only by later sources were reported unresolved. This
module re-runs the match phase against the audit's own retained index
artifacts — no archive rescan — after binding every input byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from reddit_search.evaluation.archive_partitioned import (
    ALGORITHM_VERSION,
    PARTITION_COUNT,
    SCHEMA_VERSION,
    _load_corpus_identities,
    _load_exposure_targets,
    _matches_for,
    _open_partition_connections,
    _scan_index_identities,
    _source_dir,
)
from reddit_search.ingest.state import (
    atomic_write_json,
    file_sha256,
    json_sha256,
    load_json,
)

_AUDIT_KIND = "registered_archive_audit"
_AUDIT_SCHEMA_VERSION = 3
_REMATCH_KIND = "archive_audit_rematch"
_REMATCH_SCHEMA_VERSION = 1

_OUTPUT_NAMES = (
    "archive_exposure_matches.jsonl",
    "archive_index_coverage.jsonl",
    "archive_identity_collisions.jsonl",
)


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> tuple[str, int]:
    """Write canonical JSONL atomically without publishing on failure."""
    temp = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    try:
        with temp.open("wb") as stream:
            for row in rows:
                payload = (
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode()
                stream.write(payload)
                digest.update(payload)
                count += 1
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), count


def _validate_audit(audit_directory: Path) -> dict[str, Any]:
    """Validate the audit manifest before any output is produced."""
    manifest_path = audit_directory / "archive_audit_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"archive audit manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("kind") != _AUDIT_KIND:
        raise ValueError("invalid archive audit manifest kind")
    if manifest.get("schema_version") != _AUDIT_SCHEMA_VERSION:
        raise ValueError("archive audit must use schema version 3")
    if manifest.get("complete") is not True:
        raise ValueError("archive audit is not complete")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("archive audit lacks identity")
    if identity.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("unsupported archive audit algorithm")
    if manifest.get("snapshot_id") != identity.get("snapshot_id"):
        raise ValueError("archive audit snapshot identity mismatch")
    inputs = manifest.get("input_hashes")
    if not isinstance(inputs, dict) or identity.get("input_hashes") != inputs:
        raise ValueError("archive audit input identity mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(_OUTPUT_NAMES):
        raise ValueError("archive audit output manifest is incomplete")
    for name in _OUTPUT_NAMES:
        item = outputs[name]
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"archive audit output metadata is malformed: {name}")
        _require_hash(audit_directory / name, item["sha256"], f"{name} output")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("archive audit counts are missing")
    exposure_path = audit_directory / "archive_exposure_matches.jsonl"
    exposure_rows = sum(1 for _ in exposure_path.open(encoding="utf-8"))
    if exposure_rows != counts["matched"] + counts["unresolved"]:
        raise ValueError("archive audit counts do not match exposure rows")
    return manifest


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 mismatch: expected {expected}, got {actual}")


def _verify_index_artifacts(
    audit_directory: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, int | str]]:
    """Bind every bucket file against the audit's recorded partition metadata.

    The audit itself does not embed per-bucket hashes in its manifest; it
    records them in the checkpoint's ``partition_metadata``, bound to an
    identical ``identity``. That checkpoint is the audit's own index manifest,
    so it is the honest binding source; the computed hashes are republished in
    the rematch manifest.
    """
    checkpoint_path = manifest.get("operational", {}).get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("archive audit manifest lacks a checkpoint path")
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        # The manifest records the path relative to the repository root that
        # produced the audit; fall back to the run directory beside the audit.
        candidate = audit_directory.parent / "checkpoint.json"
        if candidate.is_file():
            checkpoint_file = candidate
        else:
            raise ValueError(f"archive audit checkpoint is missing: {checkpoint_path}")
    checkpoint = load_json(checkpoint_file)
    if checkpoint.get("kind") != "registered_archive_audit_checkpoint":
        raise ValueError("audit checkpoint kind mismatch")
    if checkpoint.get("identity") != manifest.get("identity"):
        raise ValueError("audit checkpoint identity does not match audit manifest")
    if checkpoint.get("complete") is not True:
        raise ValueError("audit checkpoint is not complete")
    expected = checkpoint.get("partition_metadata")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("audit checkpoint lacks partition metadata")

    index_directory = Path(manifest["operational"]["index_directory"])
    if not index_directory.is_dir():
        raise ValueError(f"archive index directory is missing: {index_directory}")
    observed: dict[str, dict[str, int | str]] = {}
    for path in sorted(index_directory.rglob("bucket-*.sqlite")):
        relative = path.relative_to(index_directory).as_posix()
        observed[relative] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    if observed != expected:
        mismatched = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise ValueError(
            f"archive index artifacts do not match audit partition metadata "
            f"({len(mismatched)} mismatched, first: {mismatched[0]})"
        )
    return observed


def _resolve_corpus_path(
    inputs: dict[str, Any],
    corpus_path: Path | None,
) -> tuple[Path, str]:
    bound = inputs.get("corpus")
    if not isinstance(bound, dict) or not isinstance(bound.get("sha256"), str):
        raise ValueError("archive audit corpus input hash is malformed")
    expected = bound["sha256"]
    if corpus_path is not None:
        resolved = Path(corpus_path)
    else:
        bound_path = bound.get("path")
        if not isinstance(bound_path, str) or not bound_path:
            raise ValueError("archive audit corpus input path is malformed")
        resolved = Path(bound_path)
        if not resolved.is_file():
            raise ValueError(
                "audit-bound corpus path is not accessible; pass corpus_path explicitly"
            )
    _require_hash(resolved, expected, "corpus input")
    return resolved, expected


def rematch_archive_audit(
    audit_directory: Path,
    output_directory: Path,
    *,
    corpus_path: Path | None = None,
) -> dict[str, Any]:
    """Re-match a completed audit's index artifacts and publish byte-stable outputs.

    Validates the audit manifest, its checkpoint-bound index hashes, the
    exposure inputs, and the corpus (an explicit override must hash-match the
    audit's bound corpus sha256) before producing any output. The match phase
    probes every source's buckets — the defect fixed in
    ``archive_partitioned`` after the published audit ran.
    """
    audit_directory = Path(audit_directory)
    output_directory = Path(output_directory)
    manifest = _validate_audit(audit_directory)
    inputs = manifest["input_hashes"]
    index_hashes = _verify_index_artifacts(audit_directory, manifest)

    exposure_paths: list[Path] = []
    for index, item in enumerate(inputs.get("exposure", [])):
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"archive audit exposure hash is malformed at index {index}")
        exposure_path = Path(item.get("path", ""))
        if not exposure_path.is_file():
            raise ValueError(f"exposure input {index} is missing: {exposure_path}")
        _require_hash(exposure_path, item["sha256"], f"exposure input {index}")
        exposure_paths.append(exposure_path)
    if not exposure_paths:
        raise ValueError("archive audit exposure inputs are missing")

    resolved_corpus, corpus_sha256 = _resolve_corpus_path(inputs, corpus_path)
    snapshot_id = manifest.get("snapshot_id")

    corpus_ids, corpus_counts = _load_corpus_identities(resolved_corpus, snapshot_id)
    targets, exposure_errors, exposures = _load_exposure_targets(exposure_paths, corpus_ids)
    if exposure_errors:
        raise ValueError(f"exposure inputs contain {len(exposure_errors)} error rows")
    if exposures and not targets:
        raise ValueError(
            f"exposure target resolution produced no corpus identities for snapshot "
            f"{snapshot_id!r}"
        )
    completed = {item["source_id"] for item in manifest.get("sources", [])}
    index_directory = Path(manifest["operational"]["index_directory"])
    for item in manifest.get("sources", []):
        if not item.get("reader_stats", {}).get("complete"):
            raise ValueError(f"audit source is incomplete: {item['source_id']}")
        if not _source_dir(index_directory, item["source_id"]).is_dir():
            raise ValueError(f"archive index source directory is missing: {item['source_id']}")

    collision_matches, coverage_by_source = _scan_index_identities(
        index_directory, completed
    )
    partition_connections = _open_partition_connections(index_directory, completed)
    try:
        match_count = unresolved_count = 0

        def output_rows() -> Iterator[dict[str, Any]]:
            nonlocal match_count, unresolved_count
            for event in exposures:
                path, line, row, identities = event
                matches = _matches_for(identities, partition_connections, completed)
                if matches:
                    match_count += 1
                else:
                    unresolved_count += 1
                result = dict(row)
                result.update(
                    {
                        "path": str(path),
                        "line": line,
                        "status": "matched" if matches else "unresolved",
                        "corpus_matches": len(identities),
                        "archive_matches": matches,
                    }
                )
                yield result

        output_directory.mkdir(parents=True, exist_ok=True)
        output_paths = {name: output_directory / name for name in _OUTPUT_NAMES}
        exposure_hash, exposure_count = _write_jsonl(
            output_paths["archive_exposure_matches.jsonl"], output_rows()
        )
        coverage = [
            {**item, "matched_records": coverage_by_source.get(item["source_id"], 0)}
            for item in manifest.get("sources", [])
        ]
        coverage_hash, coverage_count = _write_jsonl(
            output_paths["archive_index_coverage.jsonl"], iter(coverage)
        )
        collisions = (
            {
                "identity": list(ident),
                "archive_count": len(collision_matches[ident]),
                "corpus_count": corpus_counts.get(ident, 0),
            }
            for ident in sorted(collision_matches)
            if len(collision_matches[ident]) > 1 or corpus_counts.get(ident, 0) > 1
        )
        collision_hash, collision_count = _write_jsonl(
            output_paths["archive_identity_collisions.jsonl"], collisions
        )
    finally:
        for connection in partition_connections.values():
            connection.close()

    per_source = {
        item["source_id"]: coverage_by_source.get(item["source_id"], 0)
        for item in manifest.get("sources", [])
    }
    rematch_manifest = {
        "kind": _REMATCH_KIND,
        "schema_version": _REMATCH_SCHEMA_VERSION,
        "provenance": "recomputed_from_index_artifacts",
        "fresh_archive_scan": False,
        "supersedes_counts": True,
        "supersession_note": (
            "the source audit's matcher probed one source per bucket; this rematch probes "
            "every source's buckets, so previously-unmatched rows recovered by the fixed "
            "matcher are included and the source audit's counts are superseded"
        ),
        "snapshot_id": snapshot_id,
        "audit_manifest_sha256": file_sha256(audit_directory / "archive_audit_manifest.json"),
        "audit_directory": str(audit_directory),
        "audit_identity": manifest.get("identity"),
        "audit_counts": manifest.get("counts"),
        "input_hashes": {
            "corpus": {"path": str(resolved_corpus), "sha256": corpus_sha256},
            "exposure": inputs.get("exposure"),
            "index_partitions": index_hashes,
        },
        "index_partition_count": len(index_hashes),
        "algorithm": {
            "identity": [
                "fullname",
                "source_revision_id",
                "thread_fullname",
                "NFKC-casefold focus_text",
            ],
            "near_text": False,
            "retained_exact_match_only": True,
            "partition_count": PARTITION_COUNT,
            "rematch_algorithm": (
                "archive-audit-exact-match-v1 with per-(source_id, bucket) connection "
                "pooling probing every source's buckets"
            ),
        },
        "outputs": {
            "archive_exposure_matches.jsonl": {"sha256": exposure_hash, "rows": exposure_count},
            "archive_index_coverage.jsonl": {"sha256": coverage_hash, "rows": coverage_count},
            "archive_identity_collisions.jsonl": {
                "sha256": collision_hash,
                "rows": collision_count,
            },
        },
        "counts": {
            "matched": match_count,
            "unresolved": unresolved_count,
            "collisions": collision_count,
            "exposure_errors": 0,
        },
        "per_source_matched": dict(sorted(per_source.items())),
        "run_identity": json_sha256(
            {
                "audit_manifest_sha256": file_sha256(
                    audit_directory / "archive_audit_manifest.json"
                ),
                "corpus_sha256": corpus_sha256,
                "index_hashes": index_hashes,
                "exposure_hashes": inputs.get("exposure"),
                "snapshot_id": snapshot_id,
                "schema_version": SCHEMA_VERSION,
            }
        ),
    }
    atomic_write_json(output_directory / "rematch_manifest.json", rematch_manifest)
    return rematch_manifest
