"""Explicit, non-destructive registration for operator-authorized source files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, model_validator

from reddit_search.contracts import StrictModel

from .reader import ArchiveReader, ReaderStats, SourceReadError, SourceSpec

_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_CHECKSUM_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SourceDeclaration(StrictModel):
    source_id: Annotated[str, Field(min_length=1)]
    path: Annotated[str, Field(min_length=1)]
    source_kind: Literal["submission", "comment"]
    declared_month: Annotated[str, Field(min_length=7, max_length=7)]
    source_role: Literal["discovery", "context_only"]
    usage_scope: Annotated[str, Field(min_length=1)]
    expected_checksum: str | None = None

    @model_validator(mode="after")
    def validates_declared_values(self) -> SourceDeclaration:
        if not _MONTH_PATTERN.fullmatch(self.declared_month):
            raise ValueError("declared_month must use YYYY-MM")
        if not self.usage_scope.strip():
            raise ValueError("usage_scope must not be blank")
        if self.expected_checksum and not _CHECKSUM_PATTERN.fullmatch(self.expected_checksum):
            raise ValueError("expected_checksum must be a lowercase SHA-256 hex digest")
        return self


class SourceSet(StrictModel):
    schema_version: Literal[1] = 1
    sources: list[SourceDeclaration] = Field(min_length=1)

    @model_validator(mode="after")
    def source_ids_are_unique(self) -> SourceSet:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique")
        return self


def register_source_set(source_config: Path, registry_path: Path) -> dict[str, Any]:
    """Validate declared files and atomically record metadata without reading archive content."""
    source_set = _load_source_set(source_config)
    registry_sources = _declarations_to_records(source_set, source_config)
    registry = {"schema_version": 1, "sources": registry_sources}
    persist_source_registry(registry_path, registry)
    return {"registered_count": len(registry_sources), "registry_file": str(registry_path)}


def register_source_set_additive(source_config: Path, registry_path: Path) -> dict[str, Any]:
    """Add new source identities while rejecting changes to registered identities.

    Existing rows are retained byte-for-byte, including validation metadata.
    """
    source_set = _load_source_set(source_config)
    additions = _declarations_to_records(source_set, source_config)
    existing = load_source_registry(registry_path) if registry_path.exists() else {
        "schema_version": 1, "sources": []
    }
    existing_rows = existing["sources"]
    by_id = {str(row.get("source_id")): row for row in existing_rows}
    conflicts: list[str] = []
    for candidate in additions:
        prior = by_id.get(candidate["source_id"])
        if prior is None:
            continue
        identity = ("source_path", "declared_month", "expected_checksum", "usage_scope")
        changed = [field for field in identity if prior.get(field) != candidate.get(field)]
        if changed:
            conflicts.append(f"{candidate['source_id']}: changed {', '.join(changed)}")
    if conflicts:
        raise ValueError("source registry identity conflict: " + "; ".join(conflicts))
    added = [candidate for candidate in additions if candidate["source_id"] not in by_id]
    if added:
        updated = {**existing, "schema_version": 1, "sources": [*existing_rows, *added]}
        persist_source_registry(registry_path, updated)
    return {
        "added_count": len(added),
        "idempotent_count": len(additions) - len(added),
        "registry_file": str(registry_path),
    }


def _declarations_to_records(source_set: SourceSet, source_config: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for declaration in source_set.sources:
        resolved_path = _resolve_path(source_config.parent, declaration.path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"registered source does not exist: {resolved_path}")
        records.append(
            {
                "source_id": declaration.source_id,
                "source_path": str(resolved_path),
                "source_kind": declaration.source_kind,
                "declared_month": declaration.declared_month,
                "source_role": declaration.source_role,
                "usage_scope": declaration.usage_scope,
                "input_size_bytes": resolved_path.stat().st_size,
                "expected_checksum": declaration.expected_checksum,
                "verified_sha256": None,
                "status": "registered",
            }
        )
    return records


def validate_registered_sources(registry_path: Path) -> dict[str, int]:
    """Fully read each registered source, then persist counts and verified checksums."""
    registry = load_source_registry(registry_path)
    validated_count = 0
    failed_count = 0
    for record in registry["sources"]:
        source = SourceSpec(
            source_id=str(record["source_id"]),
            path=Path(str(record["source_path"])),
            source_kind=str(record["source_kind"]),  # type: ignore[arg-type]
            declared_month=str(record["declared_month"]),
            source_role=str(record["source_role"]),  # type: ignore[arg-type]
            usage_scope=str(record["usage_scope"]),
        )
        reader = ArchiveReader()
        try:
            for _ in reader.iter_records(source):
                pass
        except SourceReadError as error:
            record_validation_failure(record, error)
            failed_count += 1
            continue

        if record_reader_validation(record, reader.stats):
            validated_count += 1
        else:
            failed_count += 1

    persist_source_registry(registry_path, registry)
    return {"validated_count": validated_count, "failed_count": failed_count}


def record_reader_validation(record: dict[str, Any], stats: ReaderStats) -> bool:
    """Update one registry row from a fully consumed reader result."""
    record["compressed_bytes_read"] = stats.compressed_bytes_read
    record["decompressed_bytes_read"] = stats.decompressed_bytes_read
    record["lines_seen"] = stats.lines_seen
    record["valid_records"] = stats.valid_records
    record["invalid_records"] = stats.invalid_records
    record["verified_sha256"] = stats.verified_sha256
    expected_checksum = record.get("expected_checksum")
    if expected_checksum and expected_checksum != stats.verified_sha256:
        record["status"] = "failed"
        record["validation_error"] = "verified SHA-256 does not match expected_checksum"
        return False
    if stats.complete:
        record["status"] = "complete"
        record.pop("validation_error", None)
        return True
    record["status"] = "complete_with_errors"
    record["validation_error"] = f"invalid JSONL rows: {stats.invalid_records}"
    return False


def record_validation_failure(record: dict[str, Any], error: SourceReadError) -> None:
    """Persist an unreadable source as failed; no downstream stage may treat it as complete."""
    record["status"] = "failed"
    record["validation_error"] = str(error)


def load_source_registry(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("sources"), list):
        raise ValueError(f"{path} must contain a source registry")
    return loaded


def persist_source_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _load_source_set(path: Path) -> SourceSet:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return SourceSet.model_validate(loaded)


def _resolve_path(base_directory: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (base_directory / candidate).resolve()
    )
