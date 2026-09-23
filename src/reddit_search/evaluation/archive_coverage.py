"""Derive auditable coverage artifacts from a completed archive audit."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from reddit_search.ingest.state import atomic_write_json, file_sha256, json_sha256

_AUDIT_KIND = "registered_archive_audit"
_AUDIT_SCHEMA_VERSION = 3
_AUDIT_ALGORITHM_VERSION = "archive-audit-exact-match-v1"
_DERIVED_SCHEMA_VERSION = 1
_REMATCH_KIND = "archive_audit_rematch"
_REMATCH_PROVENANCE = "recomputed_from_index_artifacts"
_OUTPUT_NAMES = (
    "archive_exposure_matches.jsonl",
    "archive_index_coverage.jsonl",
    "archive_identity_collisions.jsonl",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"row at {path}:{line_number} must be an object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
                stream.write("\n")
                count += 1
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path), count


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 mismatch: expected {expected}, got {actual}")


def _require_hash_or_drift(
    path: Path, expected: str, label: str, drifted: list[dict[str, str]]
) -> None:
    """Hash-check an input, recording drift instead of failing.

    Coverage derivation never reads large source inputs such as the corpus
    database — it re-sorts hash-validated match rows. A drifted (but present)
    input therefore cannot change the derived rows; it is disclosed in the
    manifest rather than silently blocking publication. Missing inputs still
    fail closed.
    """
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = file_sha256(path)
    if actual != expected:
        drifted.append(
            {
                "label": label,
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": actual,
            }
        )


def _input_hashes(
    manifest: dict[str, Any], drifted: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    inputs = manifest.get("input_hashes")
    identity = manifest.get("identity")
    if not isinstance(inputs, dict) or not isinstance(identity, dict):
        raise ValueError("archive audit lacks input identity")
    if identity.get("input_hashes") != inputs:
        raise ValueError("archive audit identity input hashes mismatch")
    for key in ("registry", "corpus"):
        item = inputs.get(key)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(
            item.get("sha256"), str
        ):
            raise ValueError(f"archive audit input hash is malformed: {key}")
        if drifted is None:
            _require_hash(Path(item["path"]), item["sha256"], f"{key} input")
        else:
            _require_hash_or_drift(Path(item["path"]), item["sha256"], f"{key} input", drifted)
    exposures = inputs.get("exposure")
    if not isinstance(exposures, list) or not exposures:
        raise ValueError("archive audit exposure inputs are missing")
    for index, item in enumerate(exposures):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(
            item.get("sha256"), str
        ):
            raise ValueError(f"archive audit exposure hash is malformed at index {index}")
        _require_hash(Path(item["path"]), item["sha256"], f"exposure input {index}")
    return inputs


def _identity_fields(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source")
    source = source if isinstance(source, dict) else {}
    values = {
        "candidate_id": row.get("candidate_id"),
        "message_fullname": row.get("message_fullname", source.get("message_fullname")),
        "source_revision_id": row.get("source_revision_id", source.get("source_revision_id")),
        "thread_fullname": row.get("thread_fullname", source.get("thread_fullname")),
        "focus_text": row.get("focus_text", source.get("text")),
    }
    if not isinstance(values["message_fullname"], str) or not isinstance(
        values["source_revision_id"], str
    ):
        raise ValueError("archive exposure row lacks canonical message identity")
    return values


def _validate_audit(
    audit_directory: Path, drifted: list[dict[str, str]] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = audit_directory / "archive_audit_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"archive audit manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("kind") != _AUDIT_KIND:
        raise ValueError("invalid archive audit manifest kind")
    if manifest.get("schema_version") != _AUDIT_SCHEMA_VERSION or not manifest.get("complete"):
        raise ValueError("archive audit must be complete and use schema version 3")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("schema_version") != _AUDIT_SCHEMA_VERSION:
        raise ValueError("archive audit identity schema mismatch")
    if identity.get("algorithm_version") != _AUDIT_ALGORITHM_VERSION:
        raise ValueError("unsupported archive audit algorithm")
    if manifest.get("snapshot_id") != identity.get("snapshot_id"):
        raise ValueError("archive audit snapshot identity mismatch")
    algorithm = manifest.get("algorithm")
    if not isinstance(algorithm, dict) or algorithm.get("retained_exact_match_only") is not True:
        raise ValueError("derived coverage requires retained exact-match audit")
    if algorithm.get("near_text") is not False:
        raise ValueError("derived coverage does not accept near-text audits")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("archive audit counts are missing")
    for key in ("matched", "unresolved", "collisions", "exposure_errors"):
        if not isinstance(counts.get(key), int) or counts[key] < 0:
            raise ValueError(f"archive audit count is malformed: {key}")
    if counts["collisions"] != 0 or counts["exposure_errors"] != 0:
        raise ValueError("archive audit contains collisions or exposure errors")
    inputs = _input_hashes(manifest, drifted=drifted)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(_OUTPUT_NAMES):
        raise ValueError("archive audit output manifest is incomplete")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in _OUTPUT_NAMES:
        item = outputs[name]
        path = audit_directory / name
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or not isinstance(item.get("rows"), int)
            or item["rows"] < 0
        ):
            raise ValueError(f"archive audit output metadata is malformed: {name}")
        _require_hash(path, item["sha256"], f"{name} output")
        rows = _read_rows(path)
        if len(rows) != item["rows"]:
            raise ValueError(f"archive audit output row count mismatch: {name}")
        loaded[name] = rows
    collision_rows = loaded["archive_identity_collisions.jsonl"]
    if collision_rows:
        raise ValueError("archive audit collision artifact is non-empty")
    exposure_rows = loaded["archive_exposure_matches.jsonl"]
    if len(exposure_rows) != counts["matched"] + counts["unresolved"]:
        raise ValueError("archive audit exposure count does not match output rows")
    for row in exposure_rows:
        if row.get("snapshot_id", manifest["snapshot_id"]) != manifest["snapshot_id"]:
            raise ValueError("archive exposure row snapshot mismatch")
        if row.get("status") not in {"matched", "unresolved"}:
            raise ValueError("archive exposure row status is invalid")
    return manifest, exposure_rows, inputs


def _validate_rematch(
    audit_directory: Path, rematch_directory: Path, drifted: list[dict[str, str]] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate a rematch directory against its source audit before deriving coverage.

    The rematch must be the byte-stable product of ``archive_audit_rematch``
    over the same audit: manifest kind/provenance verified, every output
    hash-bound, row statuses verified, and the source audit re-validated so
    the supersession is anchored to a real audit artifact. Large source
    inputs (corpus) that drifted are recorded in ``drifted`` rather than
    failing, because derivation never reads them.
    """
    audit_manifest, _audit_rows, _audit_inputs = _validate_audit(
        audit_directory, drifted=drifted
    )
    manifest_path = rematch_directory / "rematch_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"rematch manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("kind") != _REMATCH_KIND:
        raise ValueError("invalid rematch manifest kind")
    if manifest.get("schema_version") != 1:
        raise ValueError("rematch manifest must use schema version 1")
    if manifest.get("provenance") != _REMATCH_PROVENANCE:
        raise ValueError("rematch provenance is not index-artifact recomputation")
    if manifest.get("fresh_archive_scan") is not False:
        raise ValueError("rematch must not claim a fresh archive scan")
    if manifest.get("supersedes_counts") is not True:
        raise ValueError("rematch does not supersede the source audit counts")
    if manifest.get("snapshot_id") != audit_manifest["snapshot_id"]:
        raise ValueError("rematch snapshot identity mismatch")
    audit_sha = file_sha256(audit_directory / "archive_audit_manifest.json")
    if manifest.get("audit_manifest_sha256") != audit_sha:
        raise ValueError("rematch audit binding does not match the source audit manifest")
    audit_counts = manifest.get("audit_counts")
    if not isinstance(audit_counts, dict) or audit_counts != audit_manifest["counts"]:
        raise ValueError("rematch audit counts do not match the source audit manifest")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("rematch counts are missing")
    for key in ("matched", "unresolved", "collisions", "exposure_errors"):
        if not isinstance(counts.get(key), int) or counts[key] < 0:
            raise ValueError(f"rematch count is malformed: {key}")
    if counts["collisions"] != 0 or counts["exposure_errors"] != 0:
        raise ValueError("rematch contains collisions or exposure errors")
    if counts["matched"] + counts["unresolved"] != audit_counts["matched"] + audit_counts[
        "unresolved"
    ]:
        raise ValueError("rematch exposure total does not match the source audit exposure")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(_OUTPUT_NAMES):
        raise ValueError("rematch output manifest is incomplete")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in _OUTPUT_NAMES:
        item = outputs[name]
        path = rematch_directory / name
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or not isinstance(item.get("rows"), int)
            or item["rows"] < 0
        ):
            raise ValueError(f"rematch output metadata is malformed: {name}")
        _require_hash(path, item["sha256"], f"{name} rematch output")
        rows = _read_rows(path)
        if len(rows) != item["rows"]:
            raise ValueError(f"rematch output row count mismatch: {name}")
        loaded[name] = rows
    collision_rows = loaded["archive_identity_collisions.jsonl"]
    if collision_rows:
        raise ValueError("rematch collision artifact is non-empty")
    exposure_rows = loaded["archive_exposure_matches.jsonl"]
    if len(exposure_rows) != counts["matched"] + counts["unresolved"]:
        raise ValueError("rematch exposure count does not match output rows")
    for row in exposure_rows:
        if row.get("status") not in {"matched", "unresolved"}:
            raise ValueError("rematch exposure row status is invalid")
    return manifest, exposure_rows, audit_manifest


def derive_archive_coverage(
    audit_directory: Path,
    output_directory: Path,
    *,
    rematch_directory: Path | None = None,
) -> dict[str, Any]:
    """Validate a completed audit and publish deterministic derived coverage artifacts.

    With ``rematch_directory`` the coverage is derived from the corrected
    per-(source_id, bucket) rematch rows instead of the source audit's
    superseded counts; the manifest records the supersession explicitly.
    """
    audit_directory = Path(audit_directory)
    output_directory = Path(output_directory)
    drifted: list[dict[str, str]] = []
    if rematch_directory is not None:
        manifest, exposure_rows, audit_manifest = _validate_rematch(
            audit_directory, Path(rematch_directory), drifted=drifted
        )
        source_counts = manifest["counts"]
        source_manifest_sha = file_sha256(Path(rematch_directory) / "rematch_manifest.json")
        source_run_identity = manifest["run_identity"]
        inputs = manifest.get("input_hashes")
        provenance = {
            "algorithm": "archive-audit-exact-match-v1 with per-(source_id, bucket) "
            "connection pooling probing every source's buckets",
            "near_text": False,
            "unresolved_semantics": (
                "unresolved archive coverage; not a tombstone or deletion proof"
            ),
            "source_artifacts_unchanged": True,
            "provenance": _REMATCH_PROVENANCE,
            "fresh_archive_scan": False,
            "supersedes_counts": True,
            "superseded_audit_counts": audit_manifest["counts"],
            "supersession_note": manifest.get("supersession_note"),
            "rematch_manifest_sha256": source_manifest_sha,
            "rematch_run_identity": source_run_identity,
            "input_drift": drifted,
        }
    else:
        manifest, exposure_rows, inputs = _validate_audit(audit_directory)
        source_counts = manifest["counts"]
        provenance = {
            "algorithm": "archive-audit-exact-match-v1",
            "near_text": False,
            "unresolved_semantics": (
                "unresolved archive coverage; not a tombstone or deletion proof"
            ),
            "source_artifacts_unchanged": True,
        }
    matched = [row for row in exposure_rows if row.get("status") == "matched"]
    unresolved = [row for row in exposure_rows if row.get("status") == "unresolved"]
    if (
        len(matched) != source_counts["matched"]
        or len(unresolved) != source_counts["unresolved"]
    ):
        raise ValueError("archive audit status counts do not match exposure rows")
    if any(row.get("archive_matches") for row in unresolved):
        raise ValueError("unresolved exposure row contains archive matches")
    if any(not row.get("archive_matches") for row in matched):
        raise ValueError("matched exposure row lacks archive matches")

    def derived_rows(rows: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in rows:
            identity = _identity_fields(row)
            yield {
                **identity,
                "archive_matches": row.get("archive_matches", []),
                "audit_row": row.get("line"),
                "coverage_status": row["status"],
            }

    output_directory.mkdir(parents=True, exist_ok=True)
    matched_path = output_directory / "matched_archive_coverage.jsonl"
    unresolved_path = output_directory / "unresolved_archive_exclusions.jsonl"
    matched_hash, matched_count = _write_jsonl(matched_path, derived_rows(matched))
    unresolved_hash, unresolved_count = _write_jsonl(unresolved_path, derived_rows(unresolved))
    derived_manifest = {
        "kind": "derived_archive_coverage",
        "schema_version": _DERIVED_SCHEMA_VERSION,
        "snapshot_id": manifest["snapshot_id"],
        "audit_run_identity": json_sha256(
            manifest["audit_identity"] if rematch_directory is not None else manifest["identity"]
        ),
        "audit_manifest_sha256": file_sha256(audit_directory / "archive_audit_manifest.json"),
        "audit_identity": (
            manifest["audit_identity"] if rematch_directory is not None else manifest["identity"]
        ),
        "inputs": inputs,
        "derivation_policy": provenance,
        "counts": {
            "exposure_rows": len(exposure_rows),
            "matched": matched_count,
            "unresolved": unresolved_count,
            "collisions": 0,
            "exposure_errors": 0,
        },
        "outputs": {
            matched_path.name: {"sha256": matched_hash, "rows": matched_count},
            unresolved_path.name: {"sha256": unresolved_hash, "rows": unresolved_count},
        },
    }
    atomic_write_json(output_directory / "derived_coverage_manifest.json", derived_manifest)
    return {
        **derived_manifest,
        "manifest_path": str(output_directory / "derived_coverage_manifest.json"),
    }
