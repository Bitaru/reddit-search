"""Local, non-invasive environment diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sqlite3
import sys
from pathlib import Path
from typing import Any


def _fts5_available() -> bool:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(content)")
    except sqlite3.OperationalError:
        return False
    finally:
        connection.close()
    return True


def _physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.1):
            return True
    except OSError:
        return False


def _accelerator_report() -> dict[str, str]:
    try:
        import torch
    except ModuleNotFoundError:
        return {"status": "not_installed", "runtime": "torch"}

    if torch.cuda.is_available():
        return {"status": "available", "runtime": "torch", "device": "cuda"}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"status": "available", "runtime": "torch", "device": "mps"}
    return {"status": "unavailable", "runtime": "torch"}


def _cached_model_artifacts(models_dir: Path) -> list[str]:
    if not models_dir.is_dir():
        return []
    return sorted(path.name for path in models_dir.iterdir())


def collect_doctor_report(workspace: Path | None = None) -> dict[str, Any]:
    """Collect only local capability data; never read or expose credentials."""
    root = (workspace or Path.cwd()).resolve()
    disk = shutil.disk_usage(root)

    return {
        "schema_version": 1,
        "workspace": str(root),
        "interpreter": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "sqlite": {
            "version": sqlite3.sqlite_version,
            "fts5_available": _fts5_available(),
        },
        "resources": {
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "physical_memory_bytes": _physical_memory_bytes(),
        },
        "optional_backends": {
            "docker_on_path": shutil.which("docker") is not None,
            "qdrant_reachable_on_loopback": _port_available("127.0.0.1", 6333),
            "cached_model_artifacts": _cached_model_artifacts(root / "data" / "models"),
            "accelerator": _accelerator_report(),
        },
        "network": {
            "external_backends_configured": False,
            "note": "No configuration is loaded by the bootstrap doctor.",
        },
    }
