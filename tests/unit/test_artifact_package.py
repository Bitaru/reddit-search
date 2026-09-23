import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from reddit_search.operations.artifact_package import ArtifactPackageError, package_artifact
from reddit_search.operations.preflight import (
    PREFLIGHT_REPORT_KIND,
    PreflightConfig,
    run_preflight,
)


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "source.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE search_units (unit_id TEXT PRIMARY KEY, body TEXT)")
    connection.execute("CREATE TABLE unit_fts (unit_id TEXT, body TEXT)")
    connection.execute("INSERT INTO search_units VALUES ('one', 'one')")
    connection.execute("INSERT INTO unit_fts VALUES ('one', 'one')")
    connection.commit()
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("INSERT INTO search_units VALUES ('wal', 'wal')")
    connection.execute("INSERT INTO unit_fts VALUES ('wal', 'wal')")
    connection.commit()
    connection.close()
    return path


def test_package_happy_path_and_wal(tmp_path: Path) -> None:
    source = _source(tmp_path)
    result = package_artifact(source, tmp_path / "artifact", snapshot_id="snap")
    packaged = tmp_path / "artifact" / source.name
    manifest = json.loads(packaged.with_name(packaged.name + ".manifest.json").read_text())
    assert manifest == result
    assert result["packaged_sha256"] == result["output_sha256"]
    assert result["unit_count"] == 2
    assert result["integrity"] == "ok"
    with sqlite3.connect(packaged) as connection:
        assert connection.execute("SELECT COUNT(*) FROM search_units").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM unit_fts").fetchone()[0] == 2


def test_package_refuses_nonempty_root(tmp_path: Path) -> None:
    source = _source(tmp_path)
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "existing").write_text("x")
    with pytest.raises(ArtifactPackageError):
        package_artifact(source, root, snapshot_id="snap")


def test_package_refuses_sha_mismatch(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(ArtifactPackageError):
        package_artifact(source, tmp_path / "artifact", snapshot_id="snap", corpus_sha256="bad")


def test_run_preflight_ready_and_required_missing(tmp_path: Path) -> None:
    source = _source(tmp_path)
    artifact_root = tmp_path / "artifact"
    root = package_artifact(source, artifact_root, snapshot_id="snap")
    output = tmp_path / "preflight.json"
    report = run_preflight(
        PreflightConfig(
            artifact_root=artifact_root,
            output_path=output,
            required=("artifact", "ledger"),
        )
    )
    assert report["kind"] == PREFLIGHT_REPORT_KIND
    assert report["ready"] is False
    assert report["required"] == ["artifact", "ledger"]
    assert report["checks"]["artifact"]["status"] == "ok"
    assert report["checks"]["artifact"]["recorded_sha256"] == root["packaged_sha256"]
    assert report["checks"]["ledger"]["status"] == "missing"
    assert report["blockers"] == [
        {"check": "ledger", "reason": "required check 'ledger' is missing"}
    ]
    stored = json.loads(output.read_text())
    assert stored["ready"] is False


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reddit_search", *args],
        capture_output=True,
        check=False,
        text=True,
    )


def test_preflight_cli_ready_and_exit_3(tmp_path: Path) -> None:
    source = _source(tmp_path)
    artifact_root = tmp_path / "artifact"
    package_artifact(source, artifact_root, snapshot_id="snap")
    ready = _cli(
        "tombstones",
        "preflight",
        "--artifact-root",
        str(artifact_root),
        "--disk-path",
        str(tmp_path),
    )
    assert ready.returncode == 0, ready.stderr
    payload = json.loads(ready.stdout)
    assert payload["ready"] is True
    assert payload["kind"] == PREFLIGHT_REPORT_KIND
    assert payload["schema_version"] == 1
    assert payload["checks"]["ledger"]["status"] == "missing"
    blocked = _cli(
        "tombstones",
        "preflight",
        "--artifact-root",
        str(artifact_root),
        "--disk-path",
        str(tmp_path),
        "--required",
        "ledger",
    )
    assert blocked.returncode == 3
    assert json.loads(blocked.stdout)["ready"] is False


def test_artifacts_package_cli(tmp_path: Path) -> None:
    source = _source(tmp_path)
    artifact_root = tmp_path / "artifact"
    result = _cli(
        "artifacts",
        "package",
        "--source",
        str(source),
        "--artifact-root",
        str(artifact_root),
        "--snapshot-id",
        "snap",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["packaged"] is True
    assert payload["unit_count"] == 2
    conflict = _cli(
        "artifacts",
        "package",
        "--source",
        str(source),
        "--artifact-root",
        str(artifact_root),
        "--snapshot-id",
        "snap2",
    )
    assert conflict.returncode == 3
    assert json.loads(conflict.stderr)["packaged"] is False
