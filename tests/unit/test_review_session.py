import json
from pathlib import Path

import pytest

from reddit_search.review.session import write_review_session_metadata


def test_write_review_session_metadata_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    write_review_session_metadata(
        path,
        session_id="session-1",
        started_at_utc="2026-09-20T10:00:00Z",
        ended_at_utc="2026-09-20T10:30:00Z",
        active_minutes=25,
        completed_task_count=3,
        pool_sha256="p" * 64,
        worksheet_sha256="w" * 64,
        notes="manual review",
    )
    expected = {
        "active_minutes": 25,
        "completed_task_count": 3,
        "ended_at_utc": "2026-09-20T10:30:00Z",
        "notes": "manual review",
        "pool_sha256": "p" * 64,
        "session_id": "session-1",
        "started_at_utc": "2026-09-20T10:00:00Z",
        "worksheet_sha256": "w" * 64,
    }
    assert json.loads(path.read_text()) == expected
    first = path.read_bytes()
    write_review_session_metadata(
        path,
        session_id="session-1",
        started_at_utc="2026-09-20T10:00:00Z",
        ended_at_utc="2026-09-20T10:30:00Z",
        active_minutes=25,
        completed_task_count=3,
        pool_sha256="p" * 64,
        worksheet_sha256="w" * 64,
        notes="manual review",
    )
    assert path.read_bytes() == first


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"active_minutes": -1, "completed_task_count": 0}, "active_minutes"),
        ({"active_minutes": 0, "completed_task_count": -1}, "completed_task_count"),
        (
            {
                "active_minutes": 0,
                "completed_task_count": 0,
                "ended_at_utc": "2026-09-20T09:59:59Z",
            },
            "ended_at_utc",
        ),
    ],
)
def test_write_review_session_metadata_rejects_invalid_values(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    values = {
        "session_id": "session-1",
        "started_at_utc": "2026-09-20T10:00:00Z",
        "ended_at_utc": "2026-09-20T10:30:00Z",
        "active_minutes": 0,
        "completed_task_count": 0,
        "pool_sha256": "p" * 64,
        "worksheet_sha256": "w" * 64,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        write_review_session_metadata(tmp_path / "session.json", **values)
