# ruff: noqa: E501
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from reddit_search.corpus.sqlite_store import LexicalStore
from reddit_search.ingest.state import file_sha256


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


def _seed(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.db"
    with LexicalStore(source) as store:
        store.index_units([_unit("drop", "snap", "t1_x", "rev")])
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"message_fullname": "t1_x", "source_revision_id": "rev", "reason": "test"})
        + "\n",
        encoding="utf-8",
    )
    return source, ledger


# --- fake Qdrant (request-capture, in-process, loopback only) ----------------


class _FakeQdrant:
    """Minimal Qdrant-compatible HTTP server with in-memory points."""

    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension
        self.points: dict[str, dict] = {}
        self.requests: list[tuple[str, str]] = []
        self._lock = threading.Lock()

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
                with fake._lock:
                    fake.requests.append(("GET", self.path))
                if self.path == "/healthz":
                    self._respond(200, {"status": "ok"})
                    return
                if self.path.startswith("/collections/"):
                    name = unquote(self.path.split("/")[2])
                    if name not in fake._collections:
                        self._respond(404, {"status": "error"})
                        return
                    self._respond(
                        200,
                        {
                            "result": {
                                "config": {
                                    "params": {
                                        "vectors": {
                                            "size": fake.dimension,
                                            "distance": "Cosine",
                                        }
                                    }
                                }
                            }
                        },
                    )
                    return
                self._respond(404, {"status": "error"})

            def do_PUT(self) -> None:
                with fake._lock:
                    fake.requests.append(("PUT", self.path))
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                if self.path.startswith("/collections/"):
                    fake._collections.add(unquote(self.path.split("/")[2]))
                    self._respond(200, {"status": "ok"})
                    return
                self._respond(404, {"status": "error"})

            def do_POST(self) -> None:
                with fake._lock:
                    fake.requests.append(("POST", self.path))
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                parsed = urlparse(self.path)
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) >= 4 and parts[2] == "points" and parts[3] == "count":
                    self._respond(200, {"status": "ok", "result": {"count": fake._count(body.get("filter"))}})
                    return
                if len(parts) >= 4 and parts[2] == "points" and parts[3] == "delete":
                    fake._delete(body.get("filter"))
                    self._respond(200, {"status": "ok", "result": {"status": "completed"}})
                    return
                if len(parts) >= 4 and parts[2] == "points" and parts[3] == "scroll":
                    snapshot = fake._single_match_value(body.get("filter"), "snapshot_id")
                    selected = [p for p in fake.points.values() if p["payload"].get("snapshot_id") == snapshot]
                    self._respond(
                        200,
                        {"status": "ok", "result": {"points": selected[: body.get("limit", 8)]}},
                    )
                    return
                self._respond(404, {"status": "error"})

            def log_message(self, *args) -> None:  # silence request logging
                pass

        self._collections: set[str] = set()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    # --- matching helpers (top-level must/should/must_not) ---

    def _single_match_value(self, node: dict, key: str):
        if not isinstance(node, dict):
            return None
        for clause in node.get("must", []):
            if clause.get("key") == key:
                return clause.get("match", {}).get("value")
        return None

    def _matches(self, point: dict, node: dict) -> bool:
        payload = point["payload"]
        if "must" in node and not all(self._matches(point, c) for c in node["must"]):
            return False
        if "should" in node and not any(self._matches(point, c) for c in node["should"]):
            return False
        if "must_not" in node and any(self._matches(point, c) for c in node["must_not"]):
            return False
        if "key" in node:
            expected = node.get("match", {}).get("value")
            actual = payload.get(node["key"])
            if isinstance(actual, list):
                return expected in actual
            if isinstance(expected, list):
                return actual in expected
            return actual == expected
        return True

    def _count(self, selector: dict) -> int:
        return sum(1 for point in self.points.values() if self._matches(point, selector))

    def _delete(self, selector: dict) -> None:
        doomed = [pid for pid, point in self.points.items() if self._matches(point, selector)]
        for pid in doomed:
            del self.points[pid]

    def seed_point(
        self,
        unit_id: str,
        snapshot_id: str,
        fullname: str,
        revision: str,
        *,
        context_refs: list[str] | None = None,
    ) -> None:
        self.points[unit_id] = {
            "id": unit_id,
            "payload": {
                "unit_id": unit_id,
                "snapshot_id": snapshot_id,
                "message_fullname": fullname,
                "source_revision_id": revision,
                "context_message_refs": context_refs or [],
            },
        }


# --- helpers -----------------------------------------------------------------


def _operate_arguments(
    tmp_path: Path,
    source: Path,
    ledger: Path,
    *,
    reconciliation: Path | None = None,
    dense_flags: list[str] | None = None,
    snapshot_id: str = "snap",
) -> list[str]:
    arguments = [
        "tombstones",
        "operate",
        "--outbox",
        str(tmp_path / "outbox.db"),
        "--ledger",
        str(ledger),
        "--snapshot-id",
        snapshot_id,
        "--sqlite-input",
        str(source),
        "--sqlite-output",
        str(tmp_path / "output.db"),
        "--reconciliation",
        str(reconciliation or (tmp_path / "reconciliation.json")),
    ]
    return arguments + (dense_flags or [])


def _dense_flags(base_url: str) -> list[str]:
    return [
        "--dense-mode",
        "loopback",
        "--dense-url",
        base_url,
        "--dense-collection",
        "synthetic-review",
        "--dense-model-id",
        "qwen",
        "--dense-revision",
        "rev-1",
        "--dense-dimension",
        "2",
        "--dense-query-instruction",
        "Represent the query.",
    ]


# --- tests -------------------------------------------------------------------


def test_operate_help_documents_explicit_flags() -> None:
    result = run_cli("tombstones", "operate", "--help")

    assert result.returncode == 0, result.stderr
    for flag in (
        "--outbox",
        "--ledger",
        "--snapshot-id",
        "--sqlite-input",
        "--sqlite-output",
        "--sqlite-manifest",
        "--reconciliation",
        "--dense-mode",
        "--dense-url",
        "--dense-collection",
        "--dense-model-id",
        "--dense-revision",
        "--dense-dimension",
        "--json",
    ):
        assert flag in result.stdout
    # long flag names may wrap across lines in narrow help panels
    assert "--dense-query-instruction" in result.stdout.replace("\n", "").replace(" ", "") or "--dense-query-instruc" in result.stdout
    for flag in ("--qdrant", "--allow-live", "--live"):
        assert flag not in result.stdout


def test_operate_happy_path_with_loopback_dense_and_deterministic_rerun(
    tmp_path: Path,
) -> None:
    source, ledger = _seed(tmp_path)
    fake = _FakeQdrant()
    base_url = fake.start()
    try:
        # two snapshot points: one direct identity match, one untouched
        fake.seed_point("drop", "snap", "t1_x", "rev")
        fake.seed_point("keep", "snap", "t1_other", "rev2")
        # one context contributor point that references t1_x without matching it
        fake.seed_point(
            "contributor", "snap", "t1_unrelated", "rev3", context_refs=["t1_x"]
        )
        fake._collections.add("synthetic-review")

        result = run_cli(
            *_operate_arguments(tmp_path, source, ledger, dense_flags=_dense_flags(base_url))
        )
    finally:
        fake.stop()
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["operated"] is True and summary["status"] == "applied"

    output = tmp_path / "output.db"
    assert output.exists()
    with LexicalStore(output) as derived:
        assert derived.units(snapshot_id="snap") == []

    manifest = json.loads((tmp_path / "reconciliation.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "tombstone_operator_reconciliation"
    assert manifest["status"] == "applied" and manifest["applied"] is True
    assert manifest["executed"] == {"sqlite": True, "dense": True}
    assert manifest["suppression_precheck"]["already_suppressed"] == 0
    assert manifest["outbox"]["deleted_count"] == 2  # direct + context contributor
    assert manifest["dense"]["result"]["requested_identities"][0][
        "message_fullname"
    ] == "t1_x"

    # deterministic re-run: same inputs -> byte-stable core, no further deletes
    fake._collections.add("synthetic-review")
    fake.requests.clear()
    base_url = fake.start()
    try:
        rerun = run_cli(
            *_operate_arguments(
                tmp_path,
                source,
                ledger,
                dense_flags=_dense_flags(base_url),
                reconciliation=tmp_path / "reconciliation2.json",
            )
        )
    finally:
        fake.stop()
    assert rerun.returncode == 0, rerun.stderr

    first = json.loads((tmp_path / "reconciliation.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "reconciliation2.json").read_text(encoding="utf-8"))

    def _core(document: dict) -> dict:
        stable = {key: value for key, value in document.items() if key != "observation"}
        # the rerun targets a fresh fake server (new port) and a distinct
        # reconciliation output path; both are recorded scope, not flow state
        stable["scope"] = {
            key: value
            for key, value in stable["scope"].items()
            if key != "dense"
        }
        stable["scope"] = dict(stable["scope"])
        stable["scope"]["reconciliation"] = None
        return stable

    assert _core(first) == _core(second)
    # only the untouched non-matching point remains after both runs
    remaining = [point["payload"]["unit_id"] for point in fake.points.values()]
    assert remaining == ["keep"]


def test_operate_dense_disabled_refusal_is_honest_and_retryable(tmp_path: Path) -> None:
    source, ledger = _seed(tmp_path)
    source_hash = file_sha256(source)

    failed = run_cli(*_operate_arguments(tmp_path, source, ledger))
    assert failed.returncode == 4, failed.stdout + failed.stderr
    payload = json.loads(failed.stderr)
    assert payload["operated"] is False
    assert "not permitted" in payload["error"]
    assert not (tmp_path / "reconciliation.json").exists()
    assert file_sha256(source) == source_hash

    # retry with dense loopback enabled succeeds via the same command
    fake = _FakeQdrant()
    base_url = fake.start()
    try:
        fake.seed_point("drop", "snap", "t1_x", "rev")
        fake._collections.add("synthetic-review")
        retry = run_cli(
            *_operate_arguments(tmp_path, source, ledger, dense_flags=_dense_flags(base_url))
        )
    finally:
        fake.stop()
    assert retry.returncode == 0, retry.stdout + retry.stderr
    manifest = json.loads((tmp_path / "reconciliation.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "applied"
    assert manifest["outbox"]["attempts"] == 2
    with LexicalStore(tmp_path / "output.db") as derived:
        assert derived.units(snapshot_id="snap") == []


def test_operate_dense_disabled_with_dense_flags_is_validation_failure(
    tmp_path: Path,
) -> None:
    source, ledger = _seed(tmp_path)
    result = run_cli(
        *_operate_arguments(
            tmp_path,
            source,
            ledger,
            dense_flags=["--dense-url", "http://127.0.0.1:6333"],
        )
    )
    assert result.returncode == 3
    assert "loopback" in result.stderr


def test_operate_loopback_requires_all_dense_flags(tmp_path: Path) -> None:
    source, ledger = _seed(tmp_path)
    partial = ["--dense-mode", "loopback", "--dense-url", "http://127.0.0.1:6333"]
    result = run_cli(*_operate_arguments(tmp_path, source, ledger, dense_flags=partial))
    assert result.returncode == 3
    assert "missing" in result.stderr


def test_operate_validation_failures_exit_three(tmp_path: Path) -> None:
    source, ledger = _seed(tmp_path)

    # missing ledger
    result = run_cli(
        *_operate_arguments(tmp_path, source, tmp_path / "missing.jsonl")
    )
    assert result.returncode == 3
    assert "does not exist" in result.stderr

    # aliasing output onto input
    result = run_cli(
        "tombstones",
        "operate",
        "--outbox",
        str(tmp_path / "outbox.db"),
        "--ledger",
        str(ledger),
        "--snapshot-id",
        "snap",
        "--sqlite-input",
        str(source),
        "--sqlite-output",
        str(source),
        "--reconciliation",
        str(tmp_path / "reconciliation.json"),
    )
    assert result.returncode == 3
    assert "alias" in result.stderr

    # snapshot mismatch against an already-registered row
    outbox = tmp_path / "outbox.db"
    from reddit_search.ingest.invalidation import project_tombstone_identities
    from reddit_search.operations import TombstoneOutbox

    store = TombstoneOutbox(outbox)
    store.register(
        project_tombstone_identities(
            [{"message_fullname": "t1_x", "source_revision_id": "rev"}]
        ),
        scope={"snapshot_id": "snap"},
    )
    result = run_cli(
        *_operate_arguments(tmp_path, source, ledger, snapshot_id="other-snap")
    )
    assert result.returncode == 3
    assert "snapshot" in result.stderr
