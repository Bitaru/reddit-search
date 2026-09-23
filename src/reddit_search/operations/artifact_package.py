"""Additive packaging of the canonical hydrated corpus as a managed artifact."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from reddit_search.ingest.state import atomic_write_json, file_sha256


class ArtifactPackageError(ValueError):
    """Packaging was refused or the packaged database failed verification."""


def _copy_sqlite_backup(source: Path, destination: Path) -> None:
    """Copy a SQLite database through its backup API, including committed WAL pages."""
    staged = destination.with_name(destination.name + ".tmp")
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(staged)
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()
    staged.replace(destination)


def _counts(path: Path) -> tuple[int, int]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        search_units = connection.execute("SELECT COUNT(*) FROM search_units").fetchone()[0]
        unit_fts = connection.execute("SELECT COUNT(*) FROM unit_fts").fetchone()[0]
        return int(search_units), int(unit_fts)
    finally:
        connection.close()


def package_artifact(
    source_db: Path,
    artifact_root: Path,
    *,
    snapshot_id: str,
    corpus_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish a verified, additive managed-artifact copy of ``source_db``."""
    source_db = Path(source_db)
    artifact_root = Path(artifact_root)
    if not source_db.is_file():
        raise ArtifactPackageError(f"source db does not exist: {source_db}")
    if artifact_root.exists() and (not artifact_root.is_dir() or any(artifact_root.iterdir())):
        raise ArtifactPackageError(f"artifact root exists and is non-empty: {artifact_root}")

    source_sha256 = file_sha256(source_db)
    if corpus_sha256 is not None and source_sha256 != corpus_sha256:
        raise ArtifactPackageError(
            f"source db sha256 mismatch: expected {corpus_sha256}, got {source_sha256}"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    packaged_db = artifact_root / source_db.name
    _copy_sqlite_backup(source_db, packaged_db)
    packaged_sha256 = file_sha256(packaged_db)
    byte_identical = packaged_sha256 == source_sha256
    source_counts = _counts(source_db)
    packaged_counts = _counts(packaged_db)
    semantic_verified = packaged_counts == source_counts
    if not byte_identical and not semantic_verified:
        raise ArtifactPackageError("packaged database differs semantically from source")

    connection = sqlite3.connect(f"file:{packaged_db.resolve()}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()
    if integrity != "ok":
        raise ArtifactPackageError(f"packaged database integrity check failed: {integrity}")

    manifest: dict[str, Any] = {
        "kind": "managed_artifact_manifest",
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "source_path": str(source_db),
        "source_sha256": source_sha256,
        "packaged_sha256": packaged_sha256,
        "output_sha256": packaged_sha256,
        "byte_identical": byte_identical,
        "unit_count": packaged_counts[0],
        "integrity": "ok",
    }
    if not byte_identical:
        manifest["semantic_verified"] = True
    atomic_write_json(packaged_db.with_name(packaged_db.name + ".manifest.json"), manifest)
    return manifest
