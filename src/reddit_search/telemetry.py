"""Deterministic, injected stage telemetry for local operational reports."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .ingest.state import atomic_write_json, file_sha256

_SCHEMA_VERSION = 1
_CACHE_STATES = {"hit", "miss", "partial", "not_used", "not_attempted", "unknown"}
_CLEANUP_STATES = {
    "complete",
    "partial",
    "incomplete",
    "failed",
    "not_attempted",
    "not_applicable",
    "unknown",
}
_STATUSES = {"complete", "incomplete", "failed"}


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def capture_stage_telemetry(
    *,
    stage: str,
    input_hashes: Mapping[str, Any] | None = None,
    run_identity: str | None = None,
    limits: Mapping[str, Any] | None = None,
    path: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    started_at: float | None = None,
    rss_provider: Callable[[], int] | None = None,
    disk_provider: Callable[[Path], int] | None = None,
    cache_provider: Callable[[], str] | None = None,
    cleanup_provider: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Capture only explicitly supplied/injected observations; never infer values."""
    if not stage.strip():
        raise ValueError("stage must not be empty")
    started = clock() if started_at is None else started_at
    observations: dict[str, Any] = {
        "elapsed_seconds": None,
        "rss_bytes": None,
        "free_disk_bytes": None,
        "sampled_path": str(path) if path is not None else None,
        "cache": "unknown",
        "cleanup": "unknown",
    }
    errors: list[dict[str, str]] = []
    try:
        ended = clock()
        elapsed = float(ended - started)
        if elapsed < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        observations["elapsed_seconds"] = elapsed
    except Exception as error:
        errors.append({"field": "elapsed_seconds", "error": _error_text(error)})
    if rss_provider is not None:
        try:
            value = int(rss_provider())
            if value < 0:
                raise ValueError("rss_bytes must be non-negative")
            observations["rss_bytes"] = value
        except Exception as error:
            errors.append({"field": "rss_bytes", "error": _error_text(error)})
    if disk_provider is not None and path is not None:
        try:
            value = int(disk_provider(path))
            if value < 0:
                raise ValueError("free_disk_bytes must be non-negative")
            observations["free_disk_bytes"] = value
        except Exception as error:
            errors.append({"field": "free_disk_bytes", "error": _error_text(error)})
    for field, provider, allowed in (
        ("cache", cache_provider, _CACHE_STATES),
        ("cleanup", cleanup_provider, _CLEANUP_STATES),
    ):
        if provider is None:
            continue
        try:
            value = str(provider())
            if value not in allowed:
                raise ValueError(f"unsupported {field} state: {value}")
            observations[field] = value
        except Exception as error:
            errors.append({"field": field, "error": _error_text(error)})
    status = "complete" if not errors else "incomplete"
    return {
        "kind": "stage_telemetry",
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "stage": stage,
        "run_identity": run_identity,
        "input_hashes": dict(input_hashes or {}),
        "limits": dict(limits or {}),
        "observations": observations,
        "errors": errors,
        "limitations": [
            "Values are sampled from injected providers and are not exact heap attribution.",
            "Unknown values remain unknown when no provider is supplied.",
        ],
    }


def write_stage_telemetry(path: Path, telemetry: Mapping[str, Any]) -> Path:
    """Atomically write a stage telemetry manifest."""
    atomic_write_json(path, dict(telemetry))
    return path


def load_stage_telemetry(path: Path) -> dict[str, Any]:
    """Load and validate a telemetry manifest without filling missing values."""
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "stage_telemetry":
        raise ValueError("telemetry manifest has an invalid kind")
    if value.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported telemetry schema version")
    if value.get("status") not in _STATUSES:
        raise ValueError("telemetry manifest has an invalid status")
    if not isinstance(value.get("stage"), str) or not value["stage"].strip():
        raise ValueError("telemetry manifest has an invalid stage")
    if not isinstance(value.get("input_hashes"), dict) or not isinstance(
        value.get("observations"), dict
    ):
        raise ValueError("telemetry manifest has malformed identity or observations")
    observations = value["observations"]
    for field in ("elapsed_seconds", "rss_bytes", "free_disk_bytes"):
        observed = observations.get(field)
        if observed is not None and (
            isinstance(observed, bool) or not isinstance(observed, (int, float)) or observed < 0
        ):
            raise ValueError(f"telemetry observation {field} is invalid")
    if observations.get("cache", "unknown") not in _CACHE_STATES:
        raise ValueError("telemetry observation cache is invalid")
    if observations.get("cleanup", "unknown") not in _CLEANUP_STATES:
        raise ValueError("telemetry observation cleanup is invalid")
    return value


def telemetry_file_identity(path: Path) -> dict[str, str]:
    """Return provenance for an explicitly supplied telemetry artifact."""
    return {"path": str(path), "sha256": file_sha256(path)}
