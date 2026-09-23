# ruff: noqa: E501
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from reddit_search.corpus.sqlite_store import LexicalStore, propagate_tombstone_projection
from reddit_search.corpus.units import SearchUnit
from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.ingest.state import file_sha256


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reddit_search", *arguments],
        capture_output=True,
        check=False,
        text=True,
    )


def _unit(unit_id: str, snapshot: str, fullname: str, revision: str, text: str) -> SearchUnit:
    return SearchUnit(
        unit_id=unit_id,
        snapshot_id=snapshot,
        message_fullname=fullname,
        source_revision_id=revision,
        thread_fullname="t3_thread",
        focus_field="body",
        focus_start=0,
        focus_end=4,
        focus_text=text,
        context_only_text="",
        context_text=text,
        missing_context_ids=(),
        permalink="/x",
        subreddit="test",
        created_utc=1,
        synthetic=True,
    )


def _seed(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a small LexicalStore corpus and its first published generation.

    The published generation holds ``t1_keep`` and ``t1_x`` (an unrelated
    ``t1_seed`` unit was tombstoned to produce it), so a purge of ``t1_x``
    has real work to do and the FTS index must be resynced.
    """
    source = tmp_path / "source.db"
    with LexicalStore(source) as store:
        store.index_units(
            [
                _unit("seed", "snap", "t1_seed", "r_seed", "seed text"),
                _unit("keep", "snap", "t1_keep", "r_keep", "keep me searchable"),
                _unit("drop", "snap", "t1_x", "r_x", "drop me searchable"),
            ]
        )
    artifact_root = tmp_path / "artifact"
    published = artifact_root / "published.db"
    manifest = published.with_name(published.name + ".manifest.json")
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_seed", "source_revision_id": "r_seed", "reason": "seed"}],
        source_artifacts={"sqlite": (source, file_sha256(source))},
    )
    propagate_tombstone_projection(
        source, published, projection, snapshot_id="snap", manifest_path=manifest
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"message_fullname": "t1_x", "source_revision_id": "r_x", "reason": "cli test"})
        + "\n",
        encoding="utf-8",
    )
    return artifact_root, published, manifest, ledger


def _purge_arguments(tmp_path: Path, artifact_root: Path, ledger: Path) -> list[str]:
    return [
        "tombstones",
        "purge",
        "--artifact-root",
        str(artifact_root),
        "--outbox",
        str(tmp_path / "outbox" / "outbox.db"),
        "--ledger",
        str(ledger),
        "--snapshot-id",
        "snap",
        "--backup-dir",
        str(tmp_path / "backups"),
        "--reconciliation",
        str(tmp_path / "reconciliation" / "purge.json"),
    ]


def _fts_hits(db: Path, query: str) -> list[str]:
    connection = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT unit_id FROM unit_fts WHERE unit_fts MATCH ? ORDER BY unit_id", (query,)
        ).fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]


# --- happy path ---------------------------------------------------------------


def test_purge_help_documents_flags() -> None:
    result = run_cli("tombstones", "purge", "--help")
    assert result.returncode == 0
    for flag in (
        "--artifact-root",
        "--outbox",
        "--ledger",
        "--snapshot-id",
        "--backup-dir",
        "--reconciliation",
        "--artifact-db",
        "--json",
    ):
        assert flag in result.stdout


def test_purge_happy_path_scrubs_generation_resyncs_fts_and_backup_restorable(
    tmp_path: Path,
) -> None:
    artifact_root, published, manifest, ledger = _seed(tmp_path)
    prior_db_sha = file_sha256(published)
    prior_manifest_bytes = manifest.read_bytes()
    with LexicalStore(published) as store:
        prior_rows = [
            (unit.unit_id, unit.message_fullname, unit.focus_text)
            for unit in store.units(snapshot_id="snap")
        ]
    assert sorted(unit_id for unit_id, _fn, _text in prior_rows) == ["drop", "keep"]

    result = run_cli(*_purge_arguments(tmp_path, artifact_root, ledger))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["purged"] is True
    assert payload["destructive"] is True
    assert payload["status"] == "applied"
    assert payload["idempotent"] is False

    # The new published generation is scrubbed: t1_x rows are gone, t1_keep
    # survives, and the FTS index no longer matches the dropped text.
    with LexicalStore(published) as store:
        units = store.units(snapshot_id="snap")
        assert [unit.unit_id for unit in units] == ["keep"]
        assert units[0].focus_text == "keep me searchable"
    assert "drop" not in _fts_hits(published, "searchable")
    assert "keep" in _fts_hits(published, "searchable")
    rows = sqlite3.connect(published).execute(
        "SELECT COUNT(*) FROM search_units WHERE message_fullname = 't1_x'"
    ).fetchone()[0]
    assert rows == 0

    # The manifest binding is fresh for the new generation.
    reloaded = json.loads(manifest.read_text(encoding="utf-8"))
    assert reloaded["output_sha256"] == file_sha256(published)

    # The backup is restorable: actually restore it into a temp directory and
    # assert it equals the prior generation. Backup dbs are produced through
    # the SQLite backup API, so equality is semantic (identical canonical rows
    # and a working store), while the content-addressed name binds the backup
    # to the prior published file's exact sha256.
    reconciliation = json.loads(
        (tmp_path / "reconciliation" / "purge.json").read_text(encoding="utf-8")
    )
    assert reconciliation["destructive"] is True
    assert reconciliation["archives_touched"] is False
    assert reconciliation["executed"] == {
        "backup": True,
        "propagation": True,
        "publication": True,
        "outbox_claim": True,
    }
    backup_dir = Path(reconciliation["backup"]["directory"])
    assert reconciliation["backup"]["db"]["name"] == f"{prior_db_sha}.db"
    restored = tmp_path / "restored" / "published.db"
    restored.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_dir / reconciliation["backup"]["db"]["name"], restored)
    restored_manifest = tmp_path / "restored" / "published.db.manifest.json"
    shutil.copy2(
        backup_dir / reconciliation["backup"]["manifest"]["name"], restored_manifest
    )
    assert restored_manifest.read_bytes() == prior_manifest_bytes
    # And the restored generation is a working, intact LexicalStore holding the
    # pre-purge rows, byte-equivalent in content to the prior generation.
    with LexicalStore(restored) as store:
        assert sorted(unit.unit_id for unit in store.units(snapshot_id="snap")) == [
            "drop",
            "keep",
        ]
    connection = sqlite3.connect(restored)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()
    assert integrity == "ok"
    # The restored rows equal the pre-purge generation's rows, captured before
    # the purge ran.
    with LexicalStore(restored) as store:
        assert [
            (unit.unit_id, unit.message_fullname, unit.focus_text)
            for unit in store.units(snapshot_id="snap")
        ] == prior_rows


def test_purge_rerun_is_no_op(tmp_path: Path) -> None:
    artifact_root, published, manifest, ledger = _seed(tmp_path)
    first = run_cli(*_purge_arguments(tmp_path, artifact_root, ledger))
    assert first.returncode == 0, first.stderr
    before_db = published.read_bytes()
    before_manifest = manifest.read_bytes()

    second = run_cli(*_purge_arguments(tmp_path, artifact_root, ledger))
    assert second.returncode == 0, second.stderr
    payload = json.loads(second.stdout)
    assert payload["purged"] is True
    assert payload["status"] == "no_op"
    assert payload["idempotent"] is True
    # Nothing churned.
    assert published.read_bytes() == before_db
    assert manifest.read_bytes() == before_manifest
    reconciliation = json.loads(
        (tmp_path / "reconciliation" / "purge.json").read_text(encoding="utf-8")
    )
    assert reconciliation["executed"] == {
        "backup": False,
        "propagation": False,
        "publication": False,
        "outbox_claim": False,
    }
    assert reconciliation["generation"]["changed"] is False


def test_purge_json_output_prints_reconciliation(tmp_path: Path) -> None:
    artifact_root, published, manifest, ledger = _seed(tmp_path)
    prior_db_sha = file_sha256(published)
    arguments = _purge_arguments(tmp_path, artifact_root, ledger) + ["--json"]
    result = run_cli(*arguments)
    assert result.returncode == 0, result.stderr
    reconciliation = json.loads(result.stdout)
    assert reconciliation["kind"] == "tombstone_artifact_purge_reconciliation"
    assert reconciliation["snapshot_id"] == "snap"
    assert reconciliation["ledger"]["count"] == 1
    assert reconciliation["outbox"]["status"] == "applied"
    assert reconciliation["outbox"]["mode"] == "destructive_artifact_purge"
    # The prior generation hash was captured before publication and the new
    # generation differs from it; the published manifest binding is fresh.
    assert reconciliation["generation"]["prior"]["db_sha256"] == prior_db_sha
    assert reconciliation["generation"]["new"]["db_sha256"] != prior_db_sha
    assert reconciliation["generation"]["new"]["db_sha256"] == file_sha256(published)


# --- exit codes ----------------------------------------------------------------


def test_purge_validation_failures_exit_three(tmp_path: Path) -> None:
    artifact_root, _published, _manifest, ledger = _seed(tmp_path)
    # Missing ledger.
    result = run_cli(
        "tombstones",
        "purge",
        "--artifact-root",
        str(artifact_root),
        "--outbox",
        str(tmp_path / "outbox.db"),
        "--ledger",
        str(tmp_path / "missing-ledger.jsonl"),
        "--snapshot-id",
        "snap",
        "--backup-dir",
        str(tmp_path / "backups"),
        "--reconciliation",
        str(tmp_path / "reconciliation.json"),
    )
    assert result.returncode == 3
    assert json.loads(result.stderr)["purged"] is False
    # Backup dir inside the artifact root.
    result = run_cli(
        "tombstones",
        "purge",
        "--artifact-root",
        str(artifact_root),
        "--outbox",
        str(tmp_path / "outbox.db"),
        "--ledger",
        str(ledger),
        "--snapshot-id",
        "snap",
        "--backup-dir",
        str(artifact_root / "backups"),
        "--reconciliation",
        str(tmp_path / "reconciliation.json"),
    )
    assert result.returncode == 3
    assert "outside" in result.stderr
    # Cross-snapshot refusal.
    result = run_cli(
        "tombstones",
        "purge",
        "--artifact-root",
        str(artifact_root),
        "--outbox",
        str(tmp_path / "outbox.db"),
        "--ledger",
        str(ledger),
        "--snapshot-id",
        "other-snap",
        "--backup-dir",
        str(tmp_path / "backups"),
        "--reconciliation",
        str(tmp_path / "reconciliation.json"),
    )
    assert result.returncode == 3
    assert "snapshot" in result.stderr


def test_purge_boundary_failure_exits_four(tmp_path: Path) -> None:
    artifact_root, published, manifest, ledger = _seed(tmp_path)
    prior_manifest_bytes = manifest.read_bytes()
    with LexicalStore(published) as store:
        prior_units = sorted(unit.unit_id for unit in store.units(snapshot_id="snap"))
    assert prior_units == ["drop", "keep"]
    # Inject a boundary failure inside the CLI subprocess: a wrapper module
    # patches the propagate boundary to explode, then runs the real CLI main.
    wrapper = tmp_path / "purge_fault_wrapper.py"
    wrapper.write_text(
        "import sys\n"
        "import reddit_search.operations.artifact_purge as module\n"
        "def failing(input_path, output_path, manifest_path, projection, *, snapshot_id):\n"
        "    raise RuntimeError('staging exploded')\n"
        "module._propagate_boundary = failing\n"
        "from reddit_search.cli import main\n"
        "main()\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        # _purge_arguments already starts with the "tombstones purge"
        # subcommand; the wrapper's main() consumes it from argv[1:].
        [sys.executable, str(wrapper), *_purge_arguments(tmp_path, artifact_root, ledger)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 4
    payload = json.loads(result.stderr)
    assert payload["purged"] is False
    assert payload["destructive"] is True
    assert "staging exploded" in payload["error"]
    # The prior generation is still published: same manifest bytes, same rows.
    assert manifest.read_bytes() == prior_manifest_bytes
    with LexicalStore(published) as store:
        assert sorted(unit.unit_id for unit in store.units(snapshot_id="snap")) == prior_units
    # The outbox row honestly records the failure and remains retryable.
    outbox = sqlite3.connect(tmp_path / "outbox" / "outbox.db")
    status, last_error = outbox.execute(
        "SELECT status, last_error FROM tombstone_outbox"
    ).fetchone()
    outbox.close()
    assert status == "failed"
    assert "staging exploded" in last_error
    # No reconciliation was written on failure.
    assert not (tmp_path / "reconciliation" / "purge.json").exists()
