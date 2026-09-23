"""Shared staging, RSS, and free-disk budget checks for bounded pipeline stages."""

from __future__ import annotations

import json
import platform
import resource
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_STAGING_BYTES = 10_737_418_240
DEFAULT_MINIMUM_FREE_DISK_BYTES = 5_368_709_120
DEFAULT_MAX_PROCESS_RSS_BYTES = 4 * 1024**3
MEMORY_POLICY_VERSION = "rss-v1"


class BudgetError(ValueError):
    """Raised when a stage would violate its configured resource budget."""

    def __init__(self, message: str, *, metadata: dict[str, int | str] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Disk and process RSS limits shared by bounded pipeline stages."""

    max_staging_bytes: int = DEFAULT_MAX_STAGING_BYTES
    minimum_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES
    max_process_rss_bytes: int = DEFAULT_MAX_PROCESS_RSS_BYTES

    def validate(self) -> None:
        if self.max_staging_bytes <= 0:
            raise ValueError("max_staging_bytes must be positive")
        if self.minimum_free_disk_bytes < 0:
            raise ValueError("minimum_free_disk_bytes must not be negative")
        if self.max_process_rss_bytes <= 0:
            raise ValueError("max_process_rss_bytes must be positive")


def process_rss_bytes() -> int:
    """Return process peak RSS in bytes, normalized on macOS and Linux."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    system = platform.system()
    if system == "Darwin":
        return value  # macOS reports bytes.
    if system == "Linux":
        return value * 1024  # Linux reports KiB.
    raise OSError(f"unsupported platform for process RSS sampling: {system}")


def check_rss_budget(
    *, limits: ResourceLimits, sampler: Callable[[], int] = process_rss_bytes,
    stage: str, reason: str,
) -> dict[str, int | str]:
    """Sample RSS and raise a BudgetError carrying structured breach metadata."""
    limits.validate()
    observed = int(sampler())
    if observed < 0:
        raise ValueError("RSS sampler returned a negative measurement")
    metadata: dict[str, int | str] = {
        "memory_policy_version": MEMORY_POLICY_VERSION,
        "max_process_rss_bytes": limits.max_process_rss_bytes,
        "observed_process_rss_bytes": observed,
        "stage": stage,
        "reason": reason,
    }
    if observed > limits.max_process_rss_bytes:
        raise BudgetError(
            f"process RSS budget exceeded during {stage}: observed {observed} bytes, "
            f"limit {limits.max_process_rss_bytes} ({reason})",
            metadata=metadata,
        )
    return metadata


def check_output_budget(path: Path, *, estimated_bytes: int, limits: ResourceLimits) -> None:
    """Reject output that exceeds staging or leaves too little free disk."""
    limits.validate()
    if estimated_bytes < 0:
        raise ValueError("estimated_bytes must not be negative")
    if estimated_bytes > limits.max_staging_bytes:
        raise BudgetError(
            f"staging budget exceeded: estimated {estimated_bytes} bytes, "
            f"limit {limits.max_staging_bytes}"
        )
    usage = shutil.disk_usage(path)
    required_free = estimated_bytes + limits.minimum_free_disk_bytes
    if usage.free < required_free:
        raise BudgetError(
            f"insufficient free disk: {usage.free} bytes free, need {estimated_bytes} plus the "
            f"{limits.minimum_free_disk_bytes}-byte reserve"
        )


def serialized_json_bytes(payload: object) -> int:
    """Return a conservative UTF-8 JSONL byte estimate for one payload."""
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()) + 1
