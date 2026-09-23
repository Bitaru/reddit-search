import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.operations import TombstoneOutbox


def _make_outbox(tmp_path: Path) -> Path:
    outbox = TombstoneOutbox(tmp_path / "state.db")
    outbox.register(
        project_tombstone_identities([{"message_fullname": "t3_x", "source_revision_id": None}]),
        scope={"snapshot_id": "s"},
    )
    return outbox.path


def _run_cli(outbox: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "tombstones",
            "inventory",
            "--outbox",
            str(outbox),
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def test_tombstones_inventory_cli_exit_3_on_missing_outbox(tmp_path: Path) -> None:
    result = _run_cli(tmp_path / "missing.db", tmp_path / "inventory.json")
    assert result.returncode == 3
    error = json.loads(result.stderr)["error"]
    assert "does not exist" in error
    assert not (tmp_path / "inventory.json").exists()


def test_tombstones_inventory_cli_success(tmp_path: Path) -> None:
    outbox = _make_outbox(tmp_path)
    output = tmp_path / "inventory.json"
    result = _run_cli(outbox, output)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["inventoried"] is True
    assert payload["row_count"] == 1
    inventory = json.loads(output.read_text())
    assert inventory["kind"] == "tombstone_outbox_purge_inventory"
    assert inventory["executed"] is False
    assert inventory["not_a_deletion_claim"] is True


def test_tombstones_inventory_cli_exit_3_on_malformed_scope(tmp_path: Path) -> None:
    path = tmp_path / "bad.db"
    with sqlite3.connect(path) as db:
        db.execute(
            """CREATE TABLE tombstone_outbox (
            identity TEXT PRIMARY KEY, scope TEXT NOT NULL, records TEXT NOT NULL,
            status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT, requested_count INTEGER NOT NULL DEFAULT 0,
            observed_count INTEGER NOT NULL DEFAULT 0, deleted_count INTEGER NOT NULL DEFAULT 0,
            requested_at TEXT NOT NULL, claimed_at TEXT, observed_at TEXT,
            backend_results TEXT
        )"""
        )
        db.execute(
            """INSERT INTO tombstone_outbox
            (identity, scope, records, status, requested_count, requested_at)
            VALUES ('i', '{}', '[]', 'pending', 0, '2026-01-01T00:00:00+00:00')"""
        )
    output = tmp_path / "inventory.json"
    result = _run_cli(path, output)
    assert result.returncode == 3
    assert "snapshot_id" in json.loads(result.stderr)["error"]
    assert not output.exists()


def test_tombstones_inventory_cli_help(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reddit_search", "tombstones", "inventory", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0
    assert "--outbox" in result.stdout
    assert "--output" in result.stdout
    assert "without deleting" in result.stdout
