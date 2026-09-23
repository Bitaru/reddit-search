import json
import subprocess
import sys
from pathlib import Path


def test_sources_register_writes_local_registry(tmp_path: Path) -> None:
    source = tmp_path / "RS_2026-06.zst"
    source.write_bytes(b"test source")
    config = tmp_path / "sources.yaml"
    config.write_text(
        """schema_version: 1
sources:
  - source_id: june-submissions
    path: ./RS_2026-06.zst
    source_kind: submission
    declared_month: 2026-06
    source_role: discovery
    usage_scope: authorized_test_fixture
"""
    )
    registry = tmp_path / "registry.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "sources",
            "register",
            "--file",
            str(config),
            "--registry",
            str(registry),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["registered_count"] == 1
    assert registry.exists()


def test_sources_register_additive_monthly_idempotence_and_conflict(tmp_path: Path) -> None:
    old_source = tmp_path / "RS_2026-06.zst"
    new_source = tmp_path / "RS_2026-07.zst"
    old_source.write_bytes(b"old")
    new_source.write_bytes(b"new")
    config = tmp_path / "sources.yaml"
    config.write_text(
        """schema_version: 1
sources:
  - source_id: june-submissions
    path: ./RS_2026-06.zst
    source_kind: submission
    declared_month: 2026-06
    source_role: discovery
    usage_scope: authorized_test_fixture
"""
    )
    registry = tmp_path / "registry.json"
    base = [
        sys.executable,
        "-m",
        "reddit_search",
        "sources",
        "register",
        "--file",
        str(config),
        "--registry",
        str(registry),
    ]
    assert subprocess.run(base, capture_output=True, text=True).returncode == 0
    config.write_text(
        config.read_text()
        .replace("june-submissions", "july-submissions")
        .replace("2026-06", "2026-07")
        .replace("RS_2026-06", "RS_2026-07")
    )
    additive = base + ["--additive"]
    assert subprocess.run(additive, capture_output=True, text=True).returncode == 0
    assert len(json.loads(registry.read_text())["sources"]) == 2
    assert subprocess.run(additive, capture_output=True, text=True).returncode == 0
    assert len(json.loads(registry.read_text())["sources"]) == 2
    before = registry.read_bytes()
    config.write_text(config.read_text().replace("july-submissions", "june-submissions"))
    conflict = subprocess.run(additive, capture_output=True, text=True)
    assert conflict.returncode == 2
    assert registry.read_bytes() == before
