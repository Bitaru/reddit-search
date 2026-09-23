"""Persist deterministic operator-side review session metadata."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reddit_search.ingest.state import atomic_write_json


def write_review_session_metadata(
    output_path: Path,
    *,
    session_id: str,
    started_at_utc: str,
    ended_at_utc: str,
    active_minutes: int | float,
    completed_task_count: int,
    pool_sha256: str,
    worksheet_sha256: str,
    notes: str | None = None,
) -> None:
    """Validate and write session metadata without inferring unavailable claims."""
    if active_minutes < 0:
        raise ValueError("active_minutes must be non-negative")
    if completed_task_count < 0:
        raise ValueError("completed_task_count must be non-negative")
    try:
        started = datetime.fromisoformat(started_at_utc.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(ended_at_utc.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamps must be ISO-8601") from error
    if ended < started:
        raise ValueError("ended_at_utc must not precede started_at_utc")
    metadata = {
        "active_minutes": active_minutes,
        "completed_task_count": completed_task_count,
        "ended_at_utc": ended_at_utc,
        "notes": notes,
        "pool_sha256": pool_sha256,
        "session_id": session_id,
        "started_at_utc": started_at_utc,
        "worksheet_sha256": worksheet_sha256,
    }
    atomic_write_json(output_path, metadata)
