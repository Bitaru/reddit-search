# ruff: noqa: E501
import json
import subprocess
import sys
from pathlib import Path

from reddit_search.corpus.sqlite_store import LexicalStore
from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.operations import TombstoneOutbox


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reddit_search", *arguments],
        capture_output=True,
        check=False,
        text=True,
    )


def _unit(unit_id: str, snapshot: str, fullname: str, revision: str):
    from reddit_search.corpus.units import SearchUnit

    return SearchUnit(
        unit_id=unit_id,
        snapshot_id=snapshot,
        message_fullname=fullname,
        source_revision_id=revision,
        thread_fullname="t3_thread",
        focus_field="body",
        focus_start=0,
        focus_end=4,
        focus_text="text",
        context_only_text="",
        context_text="text",
        missing_context_ids=(),
        permalink="/x",
        subreddit="test",
        created_utc=1,
        synthetic=True,
    )


def _seed(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    source = tmp_path / "source.db"
    with LexicalStore(source) as store:
        store.index_units([_unit("drop", "snap", "t1_x", "rev")])
    outbox = tmp_path / "outbox.db"
    store = TombstoneOutbox(outbox)
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_x", "source_revision_id": "rev"}]
    )
    row = store.register(projection, scope={"snapshot_id": "snap"})
    return outbox, source, tmp_path / "output.db", row["identity"]


def test_replay_help_documents_local_only_flags() -> None:
    result = run_cli("tombstones", "replay", "--help")

    assert result.returncode == 0, result.stderr
    for flag in (
        "--outbox",
        "--identity",
        "--snapshot-id",
        "--sqlite-input",
        "--sqlite-output",
        "--sqlite-manifest",
    ):
        assert flag in result.stdout
    for flag in ("--qdrant", "--allow-live", "--live"):
        assert flag not in result.stdout


def test_replay_success_reports_row_status_and_derivative_result(tmp_path: Path) -> None:
    outbox, source, output, identity = _seed(tmp_path)
    store = TombstoneOutbox(outbox)
    store.complete(
        identity,
        observed_count=1,
        deleted_count=1,
        backend_results=[{"backend": "sqlite", "observed_count": 1, "deleted_count": 1}],
    )
    result = run_cli(
        "tombstones",
        "replay",
        "--outbox",
        str(outbox),
        "--identity",
        identity,
        "--snapshot-id",
        "snap",
        "--sqlite-input",
        str(source),
        "--sqlite-output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["replayed"] is True
    assert payload["row_status"] == "applied"
    assert not output.exists()


def test_replay_qdrant_refusal_fails_with_exit_three_and_json_error(tmp_path: Path) -> None:
    outbox, source, output, identity = _seed(tmp_path)
    result = run_cli(
        "tombstones",
        "replay",
        "--outbox",
        str(outbox),
        "--identity",
        identity,
        "--snapshot-id",
        "snap",
        "--sqlite-input",
        str(source),
        "--sqlite-output",
        str(output),
    )
    assert result.returncode == 3
    payload = json.loads(result.stderr.strip().splitlines()[-1])
    assert payload["replayed"] is False


def test_replay_reports_missing_paths_as_exit_three_json_errors(tmp_path: Path) -> None:
    outbox, source, output, identity = _seed(tmp_path)
    missing_outbox = run_cli(
        "tombstones",
        "replay",
        "--outbox",
        str(tmp_path / "missing.db"),
        "--identity",
        identity,
        "--snapshot-id",
        "snap",
        "--sqlite-input",
        str(source),
        "--sqlite-output",
        str(output),
    )
    assert missing_outbox.returncode == 3
    payload = json.loads(missing_outbox.stderr.strip().splitlines()[-1])
    assert payload["replayed"] is False
    assert "does not exist" in payload["error"]
    missing_input = run_cli(
        "tombstones",
        "replay",
        "--outbox",
        str(outbox),
        "--identity",
        identity,
        "--snapshot-id",
        "snap",
        "--sqlite-input",
        str(tmp_path / "missing-source.db"),
        "--sqlite-output",
        str(output),
    )
    assert missing_input.returncode == 3
    payload = json.loads(missing_input.stderr.strip().splitlines()[-1])
    assert "does not exist" in payload["error"]


def test_replay_rejects_missing_identity_and_snapshot_mismatch(tmp_path: Path) -> None:
    outbox, source, output, identity = _seed(tmp_path)
    missing = run_cli(
        "tombstones",
        "replay",
        "--outbox",
        str(outbox),
        "--identity",
        "deadbeef",
        "--snapshot-id",
        "snap",
        "--sqlite-input",
        str(source),
        "--sqlite-output",
        str(output),
    )
    assert missing.returncode == 3
    snapshot = run_cli(
        "tombstones",
        "replay",
        "--outbox",
        str(outbox),
        "--identity",
        identity,
        "--snapshot-id",
        "other",
        "--sqlite-input",
        str(source),
        "--sqlite-output",
        str(output),
    )
    assert snapshot.returncode == 3
    payload = json.loads(snapshot.stderr.strip().splitlines()[-1])
    assert "snapshot" in payload["error"]


def test_replay_rejects_alias_output_path(tmp_path: Path) -> None:
    outbox, source, _, identity = _seed(tmp_path)
    result = run_cli(
        "tombstones",
        "replay",
        "--outbox",
        str(outbox),
        "--identity",
        identity,
        "--snapshot-id",
        "snap",
        "--sqlite-input",
        str(source),
        "--sqlite-output",
        str(source),
    )
    assert result.returncode == 3
    payload = json.loads(result.stderr.strip().splitlines()[-1])
    assert "differ" in payload["error"]
