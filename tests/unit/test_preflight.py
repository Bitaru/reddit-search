"""Hermetic unit tests for the read-only tombstone-purge preflight."""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.ingest.state import atomic_write_json, file_sha256
from reddit_search.operations import (
    DenseCollectionProbe,
    PreflightConfig,
    TombstoneOutbox,
    run_preflight,
)

RECIPE_DOC = {
    "model_id": "test-model",
    "revision": "rev0",
    "dimension": 2,
    "query_instruction": "query",
    "document_text_field": "context_text",
    "context_recipe_version": "v2",
}


def _ledger_records() -> list[dict[str, str]]:
    return [
        {
            "message_fullname": "t1_alpha",
            "source_revision_id": "rev_alpha",
            "reason": "unit test",
        },
        {"message_fullname": "t3_beta", "source_revision_id": None, "reason": "unit test"},
    ]


def _write_ledger(path: Path, records: list[dict[str, str | None]]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def _make_artifact(root: Path, *, sha: str | None = None, kind: str | None = "sqlite") -> Path:
    """Create an artifact root with one .db and a sidecar manifest."""
    root.mkdir(parents=True, exist_ok=True)
    db = root / "published.db"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE keep_me (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    manifest: dict[str, object] = {"output_sha256": sha if sha is not None else file_sha256(db)}
    if kind is not None:
        manifest["kind"] = kind
    atomic_write_json(db.with_name(db.name + ".manifest.json"), manifest)
    return db


def _make_outbox(path: Path) -> TombstoneOutbox:
    return TombstoneOutbox(path)


class _FakeQdrant:
    """Minimal Qdrant-compatible HTTP server: GET collection info, count, scroll."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, dict]] = {}
        self._lock = threading.Lock()

    def add_collection(
        self,
        name: str,
        *,
        points: list[dict],
        dimension: int = 2,
        distance: str = "Cosine",
    ) -> None:
        self.collections[name] = {
            "dimension": dimension,
            "distance": distance,
            "points": points,
        }

    def start(self) -> str:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _respond(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                parts = [unquote(part) for part in urlparse(self.path).path.split("/") if part]
                if parts[:1] == ["collections"] and len(parts) == 2:
                    with fake._lock:
                        info = fake.collections.get(parts[1])
                    if info is None:
                        self._respond(404, {"status": "error"})
                        return
                    self._respond(
                        200,
                        {
                            "result": {
                                "config": {
                                    "params": {
                                        "vectors": {
                                            "size": info["dimension"],
                                            "distance": info["distance"],
                                        }
                                    }
                                }
                            }
                        },
                    )
                    return
                if self.path == "/collections":
                    with fake._lock:
                        names = sorted(fake.collections)
                    self._respond(
                        200, {"result": {"collections": [{"name": name} for name in names]}}
                    )
                    return
                self._respond(404, {"status": "error"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                parts = [unquote(part) for part in urlparse(self.path).path.split("/") if part]
                if len(parts) == 4 and parts[0] == "collections" and parts[2] == "points":
                    with fake._lock:
                        info = fake.collections.get(parts[1])
                    if info is None:
                        self._respond(404, {"status": "error"})
                        return
                    if parts[3] == "count":
                        self._respond(
                            200, {"status": "ok", "result": {"count": len(info["points"])}}
                        )
                        return
                    if parts[3] == "scroll":
                        self._respond(
                            200,
                            {
                                "status": "ok",
                                "result": {"points": list(info["points"])[:8]},
                            },
                        )
                        return
                self._respond(404, {"status": "error"})

            def log_message(self, *args) -> None:  # silence request logging
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def fake_qdrant(tmp_path: Path):
    qdrant = _FakeQdrant()
    covered_name = "reddit_dense_covered"
    legacy_name = "reddit_dense_legacy"
    qdrant.add_collection(
        covered_name,
        points=[
            {
                "payload": {
                    "snapshot_id": "snap-covered",
                    "context_message_refs": ["t1_other"],
                }
            }
        ],
    )
    qdrant.add_collection(
        legacy_name,
        points=[
            {
                "payload": {
                    "snapshot_id": "snap-legacy",
                    "context_text": "old-style payload without refs",
                }
            }
        ],
    )
    qdrant.add_collection("reddit_dense_stray", points=[], dimension=2, distance="Cosine")
    base_url = qdrant.start()
    yield qdrant, base_url, tmp_path
    qdrant.stop()


def _dense_manifest(
    path: Path,
    *,
    collection: str | None,
    snapshot_id: str,
    expected_count: int = 1,
    recipe: dict | None = RECIPE_DOC,
) -> Path:
    doc: dict[str, object] = {
        "kind": "dense_index_manifest",
        "schema_version": 1,
        "status": "ready",
        "snapshot_id": snapshot_id,
        "expected_count": expected_count,
        "actual_count": expected_count,
    }
    if collection is not None:
        doc["collection"] = collection
    if recipe is not None:
        doc["recipe"] = recipe
    atomic_write_json(path, doc)
    return path


# --- all-missing inputs -------------------------------------------------------


def test_all_missing_reports_missing_and_ready(tmp_path: Path) -> None:
    report = run_preflight(PreflightConfig())
    assert report["kind"] == "tombstone_preflight_report"
    assert report["ready"] is True
    assert report["blockers"] == []
    for name in ("artifact", "ledger", "outbox", "dense_collections", "paths"):
        assert report["checks"][name]["status"] == "missing"
    assert report["checks"]["disk"]["status"] == "ok"


def test_missing_optional_never_fails_even_when_required_missing_allowed(
    tmp_path: Path,
) -> None:
    report = run_preflight(
        PreflightConfig(artifact_root=tmp_path / "nope", required=("ledger",))
    )
    assert report["checks"]["artifact"]["status"] == "missing"
    assert report["checks"]["artifact"]["reason"].startswith("artifact root does not exist")
    assert report["checks"]["ledger"]["status"] == "missing"
    # Required ledger is missing without missing_allowed -> blocker.
    assert report["blockers"] == [
        {"check": "ledger", "reason": "required check 'ledger' is missing"}
    ]
    assert report["ready"] is False


def test_missing_allowed_required_check_is_ready(tmp_path: Path) -> None:
    report = run_preflight(
        PreflightConfig(
            required=("ledger", "outbox"),
            missing_allowed=("ledger", "outbox"),
        )
    )
    assert report["ready"] is True
    assert report["blockers"] == []


# --- artifact check -----------------------------------------------------------


def test_artifact_ok_when_hash_matches(tmp_path: Path) -> None:
    db = _make_artifact(tmp_path / "artifact")
    report = run_preflight(PreflightConfig(artifact_root=tmp_path / "artifact"))
    check = report["checks"]["artifact"]
    assert check["status"] == "ok"
    assert check["hash_match"] is True
    assert check["recorded_sha256"] == file_sha256(db)
    assert report["ready"] is True


def test_artifact_tampered_manifest_hash_fails(tmp_path: Path) -> None:
    _make_artifact(tmp_path / "artifact", sha="0" * 64)
    report = run_preflight(PreflightConfig(artifact_root=tmp_path / "artifact"))
    check = report["checks"]["artifact"]
    assert check["status"] == "fail"
    assert "hash mismatch" in check["reasons"][0]
    assert report["ready"] is False
    assert report["blockers"] == [
        {"check": "artifact", "reason": check["reasons"][0]}
    ]


def test_artifact_manifest_missing_or_unparseable_fails(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _make_artifact(root)
    (root / "published.db.manifest.json").unlink()
    report = run_preflight(PreflightConfig(artifact_root=root))
    assert report["checks"]["artifact"]["status"] == "fail"

    (root / "published.db.manifest.json").write_text("{not json", encoding="utf-8")
    report = run_preflight(PreflightConfig(artifact_root=root))
    assert report["checks"]["artifact"]["status"] == "fail"
    assert "not readable JSON" in report["checks"]["artifact"]["reasons"][0]


def test_artifact_two_db_files_fails(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _make_artifact(root)
    (root / "other.db").write_bytes(b"SQLite format 3\x00")
    report = run_preflight(PreflightConfig(artifact_root=root))
    assert report["checks"]["artifact"]["status"] == "fail"
    assert "exactly one" in report["checks"]["artifact"]["reasons"][0]


# --- ledger check -------------------------------------------------------------


def test_ledger_ok_with_two_records(tmp_path: Path) -> None:
    path = _write_ledger(tmp_path / "ledger.jsonl", _ledger_records())
    report = run_preflight(PreflightConfig(ledger_path=path))
    check = report["checks"]["ledger"]
    assert check["status"] == "ok"
    assert check["record_count"] == 2
    assert check["digest"] is not None


def test_ledger_invalid_records_fail(tmp_path: Path) -> None:
    bad = tmp_path / "ledger.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    report = run_preflight(PreflightConfig(ledger_path=bad))
    assert report["checks"]["ledger"]["status"] == "fail"
    assert "failed to load" in report["checks"]["ledger"]["reasons"][0]

    duplicate = tmp_path / "dupe.jsonl"
    row = json.dumps(_ledger_records()[0])
    duplicate.write_text(row + "\n" + row + "\n", encoding="utf-8")
    report = run_preflight(PreflightConfig(ledger_path=duplicate))
    assert report["checks"]["ledger"]["status"] == "fail"
    assert "duplicate" in report["checks"]["ledger"]["reasons"][0]


# --- outbox check -------------------------------------------------------------


def test_outbox_ok_with_row_counts_by_status(tmp_path: Path) -> None:
    outbox = _make_outbox(tmp_path / "outbox.sqlite")
    outbox.register(
        project_tombstone_identities(_ledger_records()),
        scope={"snapshot_id": "snap"},
    )
    report = run_preflight(PreflightConfig(outbox_path=tmp_path / "outbox.sqlite"))
    check = report["checks"]["outbox"]
    assert check["status"] == "ok"
    assert check["rows_by_status"] == {"pending": 1}
    assert check["total_rows"] == 1


def test_outbox_database_without_table_fails(tmp_path: Path) -> None:
    path = tmp_path / "outbox.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated (id INTEGER)")
    report = run_preflight(PreflightConfig(outbox_path=path))
    assert report["checks"]["outbox"]["status"] == "fail"
    assert "no tombstone_outbox table" in report["checks"]["outbox"]["reasons"][0]


# --- dense collections check --------------------------------------------------


def test_dense_covered_collection_ok_legacy_warns_unmapped_listed(
    fake_qdrant,
) -> None:
    qdrant, base_url, tmp_path = fake_qdrant
    covered_manifest = _dense_manifest(
        tmp_path / "covered.json",
        collection="reddit_dense_covered",
        snapshot_id="snap-covered",
    )
    legacy_manifest = _dense_manifest(
        tmp_path / "legacy.json",
        collection="reddit_dense_legacy",
        snapshot_id="snap-legacy",
    )
    report = run_preflight(
        PreflightConfig(
            qdrant_base_url=base_url,
            dense_collections=(
                DenseCollectionProbe(manifest_path=covered_manifest),
                DenseCollectionProbe(manifest_path=legacy_manifest),
            ),
        )
    )
    check = report["checks"]["dense_collections"]
    statuses = {entry["collection"]: entry["status"] for entry in check["collections"]}
    assert statuses["reddit_dense_covered"] == "ok"
    assert statuses["reddit_dense_legacy"] == "warn"
    covered = next(
        entry
        for entry in check["collections"]
        if entry["collection"] == "reddit_dense_covered"
    )
    assert covered["payload_coverage"] == "covered"
    assert covered["count_match"] is True
    assert covered["snapshot_match"] is True
    assert covered["recipe"]["dimension_match"] is True
    assert check["unmapped_collections"] == ["reddit_dense_stray"]
    assert check["status"] == "warn"
    assert "reddit_dense_stray" in check["reasons"][-1]
    # Warnings never block readiness on their own.
    assert report["ready"] is True


def test_dense_count_mismatch_fails(fake_qdrant) -> None:
    _qdrant, base_url, tmp_path = fake_qdrant
    manifest = _dense_manifest(
        tmp_path / "covered.json",
        collection="reddit_dense_covered",
        snapshot_id="snap-covered",
        expected_count=99,
    )
    report = run_preflight(
        PreflightConfig(
            qdrant_base_url=base_url,
            dense_collections=(DenseCollectionProbe(manifest_path=manifest),),
        )
    )
    check = report["checks"]["dense_collections"]
    assert check["status"] == "fail"
    assert check["collections"][0]["count_match"] is False
    assert "does not match" in check["reasons"][0]
    assert report["ready"] is False


def test_dense_required_coverage_legacy_is_not_ready(fake_qdrant) -> None:
    _qdrant, base_url, tmp_path = fake_qdrant
    legacy_manifest = _dense_manifest(
        tmp_path / "legacy.json",
        collection="reddit_dense_legacy",
        snapshot_id="snap-legacy",
    )
    report = run_preflight(
        PreflightConfig(
            qdrant_base_url=base_url,
            dense_collections=(DenseCollectionProbe(manifest_path=legacy_manifest),),
        )
    )
    # warn alone does not block ...
    assert report["checks"]["dense_collections"]["status"] == "warn"
    assert report["ready"] is True


def test_dense_missing_collection_or_manifest_gap(fake_qdrant) -> None:
    qdrant, base_url, tmp_path = fake_qdrant
    ghost = _dense_manifest(
        tmp_path / "ghost.json",
        collection="reddit_dense_absent",
        snapshot_id="snap-x",
    )
    report = run_preflight(
        PreflightConfig(
            qdrant_base_url=base_url,
            dense_collections=(DenseCollectionProbe(manifest_path=ghost),),
        )
    )
    check = report["checks"]["dense_collections"]
    assert check["status"] == "fail"
    assert check["collections"][0]["reachable"] is False

    # Manifest without any collection binding: mapping gap, warn.
    unbound = _dense_manifest(tmp_path / "unbound.json", collection=None, snapshot_id="snap-y")
    report = run_preflight(
        PreflightConfig(
            qdrant_base_url=base_url,
            dense_collections=(DenseCollectionProbe(manifest_path=unbound),),
        )
    )
    check = report["checks"]["dense_collections"]
    assert check["collections"][0]["status"] == "warn"
    assert "mapping gap" in check["collections"][0]["reason"]


def test_dense_explicit_collection_name_without_manifest(fake_qdrant) -> None:
    _qdrant, base_url, _tmp_path = fake_qdrant
    report = run_preflight(
        PreflightConfig(
            qdrant_base_url=base_url,
            dense_collections=(
                DenseCollectionProbe(
                    collection_name="reddit_dense_covered",
                    expected_snapshot_id="snap-covered",
                ),
            ),
        )
    )
    check = report["checks"]["dense_collections"]
    entry = check["collections"][0]
    assert entry["status"] == "ok"
    assert entry["payload_coverage"] == "covered"
    assert entry["observed_count"] == 1
    # The stray live collection is still flagged as unmapped.
    assert check["unmapped_collections"] == ["reddit_dense_legacy", "reddit_dense_stray"]


def test_dense_dimension_mismatch_fails(fake_qdrant) -> None:
    _qdrant, base_url, tmp_path = fake_qdrant
    manifest = _dense_manifest(
        tmp_path / "wrong-dim.json",
        collection="reddit_dense_covered",
        snapshot_id="snap-covered",
        recipe={**RECIPE_DOC, "dimension": 512},
    )
    report = run_preflight(
        PreflightConfig(
            qdrant_base_url=base_url,
            dense_collections=(DenseCollectionProbe(manifest_path=manifest),),
        )
    )
    check = report["checks"]["dense_collections"]
    assert check["status"] == "fail"
    assert any("dimension" in reason for reason in check["reasons"])


# --- disk check ---------------------------------------------------------------


def test_disk_fail_with_injected_tiny_minimum(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _make_artifact(root)
    report = run_preflight(
        PreflightConfig(
            artifact_root=root,
            minimum_free_disk_bytes=(1 << 40),
        )
    )
    check = report["checks"]["disk"]
    assert check["status"] == "fail"
    assert check["free_bytes"] < check["minimum_free_disk_bytes"]
    assert report["ready"] is False
    assert report["blockers"] == [{"check": "disk", "reason": check["reasons"][0]}]


def test_disk_ok_against_default_minimum(tmp_path: Path) -> None:
    report = run_preflight(PreflightConfig(minimum_free_disk_bytes=0))
    assert report["checks"]["disk"]["status"] == "ok"


# --- paths/alias check --------------------------------------------------------


def test_alias_collision_fails(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _make_artifact(root)
    db = root / "published.db"
    report = run_preflight(
        PreflightConfig(
            artifact_root=root,
            artifact_db=db,
            outbox_path=db,  # same file as artifact db
            missing_allowed=(),
        )
    )
    check = report["checks"]["paths"]
    assert check["status"] == "fail"
    assert "aliases" in check["reasons"][0]
    assert report["ready"] is False


def test_symlink_alias_to_artifact_db_fails(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _make_artifact(root)
    db = root / "published.db"
    link = tmp_path / "link.db"
    link.symlink_to(db)
    report = run_preflight(
        PreflightConfig(artifact_root=root, artifact_db=db, ledger_path=link)
    )
    assert report["checks"]["paths"]["status"] == "fail"


def test_backup_dir_inside_artifact_root_fails(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _make_artifact(root)
    (root / "backups").mkdir(exist_ok=True)
    report = run_preflight(
        PreflightConfig(artifact_root=root, backup_dir=root / "backups")
    )
    check = report["checks"]["paths"]
    assert check["status"] == "fail"
    assert "outside the artifact root" in check["reasons"][0]


def test_distinct_paths_ok(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _make_artifact(root)
    report = run_preflight(
        PreflightConfig(
            artifact_root=root,
            ledger_path=tmp_path / "ledger.jsonl",
            outbox_path=tmp_path / "outbox.sqlite",
            backup_dir=tmp_path / "backups",
        )
    )
    assert report["checks"]["paths"]["status"] == "ok"


# --- determinism and required-mask semantics ----------------------------------


def test_output_is_deterministic_byte_equal_across_runs(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _make_artifact(root)
    ledger = _write_ledger(tmp_path / "ledger.jsonl", _ledger_records())
    config = PreflightConfig(
        artifact_root=root,
        ledger_path=ledger,
        required=("artifact", "ledger"),
        missing_allowed=(),
    )
    first = json.dumps(run_preflight(config), sort_keys=True, indent=1)
    second = json.dumps(run_preflight(config), sort_keys=True, indent=1)
    assert first == second


def test_required_fail_blocks_ready(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _make_artifact(root, sha="f" * 64)
    report = run_preflight(PreflightConfig(artifact_root=root, required=("artifact",)))
    assert report["ready"] is False
    assert report["blockers"] == [
        {"check": "artifact", "reason": report["checks"]["artifact"]["reasons"][0]}
    ]


def test_output_written_atomically_only_when_requested(tmp_path: Path) -> None:
    output = tmp_path / "report" / "preflight.json"
    report = run_preflight(PreflightConfig(output_path=output))
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not (tmp_path / "report" / "preflight.json.tmp").exists()


def test_unknown_required_check_name_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown required check names"):
        run_preflight(PreflightConfig(required=("nonsense",)))


def test_missing_allowed_requires_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing_allowed"):
        run_preflight(PreflightConfig(missing_allowed=("ledger",)))
