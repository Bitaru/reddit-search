"""Source-read manifests that make incomplete inputs explicit."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .reader import ReaderStats, SourceSpec


@dataclass(frozen=True, slots=True)
class SourceManifest:
    source_id: str
    source_path: str
    source_kind: str
    declared_month: str
    usage_scope: str
    source_role: str
    input_size_bytes: int
    run_id: str
    configuration_hash: str
    started_at: datetime
    completed_at: datetime | None
    status: str
    compressed_bytes_read: int
    decompressed_bytes_read: int
    lines_seen: int
    valid_records: int
    invalid_records: int
    warnings: list[str] = field(default_factory=list)
    interruption_reason: str | None = None


def manifest_from_reader(
    source: SourceSpec,
    stats: ReaderStats,
    *,
    run_id: str,
    configuration_hash: str,
    started_at: datetime,
) -> SourceManifest:
    """Derive a truthful manifest status from a completed reader attempt."""
    if stats.decompression_error:
        status = "failed"
        completed_at = None
        interruption_reason = stats.decompression_error
    elif stats.complete:
        status = "complete"
        completed_at = datetime.now(started_at.tzinfo)
        interruption_reason = None
    elif stats.invalid_records:
        status = "complete_with_errors"
        completed_at = datetime.now(started_at.tzinfo)
        interruption_reason = None
    else:
        status = "partial"
        completed_at = None
        interruption_reason = "reader did not reach a complete state"

    warnings = [f"invalid JSONL rows: {stats.invalid_records}"] if stats.invalid_records else []
    return SourceManifest(
        source_id=source.source_id,
        source_path=str(source.path),
        source_kind=source.source_kind,
        declared_month=source.declared_month,
        usage_scope=source.usage_scope,
        source_role=source.source_role,
        input_size_bytes=source.path.stat().st_size,
        run_id=run_id,
        configuration_hash=configuration_hash,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        compressed_bytes_read=stats.compressed_bytes_read,
        decompressed_bytes_read=stats.decompressed_bytes_read,
        lines_seen=stats.lines_seen,
        valid_records=stats.valid_records,
        invalid_records=stats.invalid_records,
        warnings=warnings,
        interruption_reason=interruption_reason,
    )
