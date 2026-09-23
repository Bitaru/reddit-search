"""Deterministic operational and topic reports over stored review artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .telemetry import load_stage_telemetry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _card_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.jsonl"))


def topic_report(
    input_path: Path, *, snapshot_id: str | None = None, app: str | None = None
) -> dict[str, Any]:
    files = _card_files(input_path)
    rows = [row for path in files for row in _rows(path)]
    if app is not None:
        # App binding is explicit-only: cards must carry an ``app`` or
        # ``app_id`` field to match; scenario IDs never imply an app.
        filtered = [row for row in rows if row.get("app") == app or row.get("app_id") == app]
        rows = filtered
    topics: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    dates: list[float] = []
    for row in rows:
        topic = row.get("topic_fit", "unknown")
        topics[str(topic)] += 1
        statuses[str(row.get("review_status", "unknown"))] += 1
        source = row.get("source")
        created = source.get("created_utc") if isinstance(source, dict) else None
        if isinstance(created, (int, float)):
            dates.append(float(created))
    return {
        "kind": "topic_report",
        "snapshot_id": snapshot_id,
        "app": app,
        "inputs": {"files": [{"path": str(path), "sha256": _sha256(path)} for path in files]},
        "counts": {
            "cards": len(rows),
            "topic_fit": dict(sorted(topics.items())),
            "review_status": dict(sorted(statuses.items())),
        },
        "observed_date_coverage": {
            "min_created_utc": min(dates) if dates else None,
            "max_created_utc": max(dates) if dates else None,
            "count": len(dates),
        },
        "limitations": [
            "Unknown and unavailable values are counted separately; "
            "absence of a decision is not a negative.",
        ],
    }


def operational_report(
    input_path: Path,
    *,
    snapshot_id: str | None = None,
    telemetry_path: Path | None = None,
) -> dict[str, Any]:
    files = _card_files(input_path)
    input_files = [{"path": str(path), "sha256": _sha256(path)} for path in files]
    rows = [row for path in files for row in _rows(path)]
    context = Counter()
    review = Counter()
    errors: list[str] = []
    for row in rows:
        ctx = row.get("context") if isinstance(row.get("context"), dict) else {}
        context["complete" if ctx.get("context_complete") is True else "incomplete"] += 1
        review[str(row.get("review_status", "unknown"))] += 1
        if row.get("error") is not None:
            errors.append(str(row["error"]))
    resource_usage: dict[str, Any] = {
        "elapsed_seconds": None,
        "rss_bytes": None,
        "disk_bytes": None,
        "cache": "unknown",
        "cleanup_state": "unknown",
    }
    telemetry_identity = None
    if telemetry_path is not None:
        telemetry = load_stage_telemetry(telemetry_path)
        expected = {"files": input_files}
        if telemetry.get("input_hashes") != expected:
            raise ValueError("telemetry input hashes do not match report inputs")
        observations = telemetry["observations"]
        resource_usage = {
            "elapsed_seconds": observations.get("elapsed_seconds"),
            "rss_bytes": observations.get("rss_bytes"),
            "disk_bytes": observations.get("free_disk_bytes"),
            "cache": observations.get("cache", "unknown"),
            "cleanup_state": observations.get("cleanup", "unknown"),
        }
        telemetry_identity = {
            "path": str(telemetry_path),
            "sha256": _sha256(telemetry_path),
            "status": telemetry.get("status"),
        }
    return {
        "kind": "operational_report",
        "snapshot_id": snapshot_id,
        "inputs": {"files": input_files},
        "telemetry": telemetry_identity,
        "processed": {"cards": len(rows), "files": len(files)},
        "reviewable_questions": sum(
            value for key, value in review.items() if key in {"pending", "complete"}
        ),
        "missing_context": dict(sorted(context.items())),
        "review_status": dict(sorted(review.items())),
        "errors": sorted(errors),
        "resource_usage": resource_usage,
        "limitations": [
            "Resource and cleanup values remain unavailable unless durably "
            "recorded by the input artifacts.",
        ],
    }


def write_report(report: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
