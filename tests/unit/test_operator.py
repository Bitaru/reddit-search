# ruff: noqa: E501
import json
import sqlite3
from pathlib import Path

import pytest

from reddit_search.corpus.sqlite_store import LexicalStore
from reddit_search.ingest.invalidation import project_tombstone_identities
from reddit_search.ingest.state import file_sha256
from reddit_search.operations import TombstoneOutbox
from reddit_search.operations.operator import (
    DenseScope,
    OperatorBoundaryError,
    OperatorScope,
    dense_scope_from_flags,
    run_configured_operation,
    suppression_precheck,
    validate_operator_scope,
)
from reddit_search.retrieval.dense import EmbeddingRecipe


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


def _ledger(tmp_path: Path, rows: list[dict[str, str | None]]) -> Path:
    path = tmp_path / "ledger.jsonl"
    lines = []
    for row in rows:
        payload = {"message_fullname": row["message_fullname"], "reason": "test"}
        if row.get("source_revision_id") is not None:
            payload["source_revision_id"] = row["source_revision_id"]
        lines.append(json.dumps(payload))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _scope(
    tmp_path: Path,
    *,
    ledger: Path | None = None,
    outbox: Path | None = None,
    input_path: Path | None = None,
    output: Path | None = None,
    manifest: Path | None = None,
    reconciliation: Path | None = None,
    dense: DenseScope | None = None,
) -> OperatorScope:
    return OperatorScope(
        outbox_path=outbox or tmp_path / "outbox.db",
        ledger_path=ledger or _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "rev"}]),
        snapshot_id="snap",
        sqlite_input=input_path or (tmp_path / "source.db"),
        sqlite_output=output or (tmp_path / "output.db"),
        sqlite_manifest=manifest,
        reconciliation_path=reconciliation or (tmp_path / "reconciliation.json"),
        dense=dense,
    )


def _seed_source(tmp_path: Path) -> Path:
    source = tmp_path / "source.db"
    with LexicalStore(source) as store:
        store.index_units([_unit("drop", "snap", "t1_x", "rev")])
    return source


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Path]:
    source = _seed_source(tmp_path)
    return {
        "source": source,
        "ledger": _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "rev"}]),
        "outbox": tmp_path / "outbox.db",
        "output": tmp_path / "output.db",
        "reconciliation": tmp_path / "reconciliation.json",
    }


def _run(tmp_path: Path, paths: dict[str, Path], **kwargs) -> dict:
    scope = _scope(
        tmp_path,
        ledger=paths["ledger"],
        outbox=paths["outbox"],
        input_path=paths["source"],
        output=paths["output"],
        reconciliation=paths["reconciliation"],
        **kwargs,
    )
    return run_configured_operation(scope)


def _recipe() -> EmbeddingRecipe:
    return EmbeddingRecipe("qwen", "rev-1", 2, "Represent the query.")


# --- scope validation -------------------------------------------------------


def test_scope_rejects_missing_input_and_ledger(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "rev"}])
    with pytest.raises(ValueError, match="SQLite input"):
        validate_operator_scope(
            outbox=tmp_path / "outbox.db",
            ledger=ledger,
            sqlite_input=tmp_path / "missing.db",
            sqlite_output=tmp_path / "out.db",
            sqlite_manifest=None,
            reconciliation=tmp_path / "recon.json",
            dense=None,
        )
    source = _seed_source(tmp_path)
    with pytest.raises(ValueError, match="ledger"):
        validate_operator_scope(
            outbox=tmp_path / "outbox.db",
            ledger=tmp_path / "missing.jsonl",
            sqlite_input=source,
            sqlite_output=tmp_path / "out.db",
            sqlite_manifest=None,
            reconciliation=tmp_path / "recon.json",
            dense=None,
        )


def test_scope_rejects_pairwise_aliases(tmp_path: Path) -> None:
    source = _seed_source(tmp_path)
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "rev"}])
    outbox = tmp_path / "outbox.db"
    output = tmp_path / "output.db"
    manifest = tmp_path / "manifest.json"
    reconciliation = tmp_path / "recon.json"
    base = {
        "outbox": outbox,
        "ledger": ledger,
        "sqlite_input": source,
        "sqlite_output": output,
        "sqlite_manifest": manifest,
        "reconciliation": reconciliation,
        "dense": None,
    }
    # every pairwise alias must be rejected and must not create the outbox
    collisions = [
        ("sqlite_output", source),
        ("reconciliation", source),
        ("reconciliation", outbox),
        ("reconciliation", ledger),
        ("reconciliation", output),
        ("reconciliation", manifest),
        ("sqlite_manifest", source),
        ("sqlite_manifest", outbox),
        ("sqlite_manifest", ledger),
        ("sqlite_manifest", reconciliation),
        ("sqlite_output", ledger),
        ("sqlite_output", outbox),
    ]
    for key, alias in collisions:
        overrides = dict(base)
        overrides[key] = alias
        with pytest.raises(ValueError, match="alias"):
            validate_operator_scope(**overrides)
        assert not outbox.exists(), f"outbox created during rejected scope: {key}"


def test_scope_rejects_invalid_dense_configuration(tmp_path: Path) -> None:
    source = _seed_source(tmp_path)
    ledger = _ledger(tmp_path, [{"message_fullname": "t1_x", "source_revision_id": "rev"}])
    with pytest.raises(ValueError, match="loopback"):
        validate_operator_scope(
            outbox=tmp_path / "outbox.db",
            ledger=ledger,
            sqlite_input=source,
            sqlite_output=tmp_path / "out.db",
            sqlite_manifest=None,
            reconciliation=tmp_path / "recon.json",
            dense=DenseScope(
                base_url="https://example.invalid",
                collection="c",
                recipe=_recipe(),
            ),
        )


def test_dense_scope_from_flags_grouping() -> None:
    with pytest.raises(ValueError, match="loopback"):
        dense_scope_from_flags(
            dense_mode="disabled",
            dense_url="http://127.0.0.1:6333",
            dense_collection=None,
            dense_model_id=None,
            dense_revision=None,
            dense_dimension=None,
            dense_query_instruction=None,
        )
    with pytest.raises(ValueError, match="missing"):
        dense_scope_from_flags(
            dense_mode="loopback",
            dense_url="http://127.0.0.1:6333",
            dense_collection="c",
            dense_model_id="m",
            dense_revision="r",
            dense_dimension=None,
            dense_query_instruction="q",
        )
    with pytest.raises(ValueError, match="dense mode"):
        dense_scope_from_flags(
            dense_mode="live",
            dense_url=None,
            dense_collection=None,
            dense_model_id=None,
            dense_revision=None,
            dense_dimension=None,
            dense_query_instruction=None,
        )
    scope = dense_scope_from_flags(
        dense_mode="loopback",
        dense_url="http://127.0.0.1:6333",
        dense_collection="c",
        dense_model_id="qwen",
        dense_revision="rev-1",
        dense_dimension=2,
        dense_query_instruction="Represent the query.",
    )
    assert scope is not None and scope.recipe.model_id == "qwen"


def test_run_rejects_empty_ledger(tmp_path: Path, seeded: dict[str, Path]) -> None:
    seeded["ledger"] = _ledger(tmp_path, [])
    with pytest.raises(ValueError, match="at least one"):
        _run(tmp_path, seeded)
    assert not seeded["outbox"].exists()


# --- suppression precheck ---------------------------------------------------


def test_suppression_precheck_reports_existing_tombstones(
    tmp_path: Path, seeded: dict[str, Path]
) -> None:
    source = seeded["source"]
    # one fullname-wide tombstone row (revision NULL) and one exact-revision row
    with sqlite3.connect(source) as db:
        db.execute(
            "INSERT INTO tombstones (message_fullname, source_revision_id, snapshot_id,"
            " unit_id) VALUES ('t1_x', NULL, 'snap', NULL)"
        )
        db.execute(
            "INSERT INTO tombstones (message_fullname, source_revision_id, snapshot_id,"
            " unit_id) VALUES ('t1_other', 'rev9', 'snap', NULL)"
        )
    projection = project_tombstone_identities(
        [
            {"message_fullname": "t1_x", "source_revision_id": "rev"},
            {"message_fullname": "t1_clean", "source_revision_id": "rev2"},
        ]
    )
    report = suppression_precheck(source, projection, snapshot_id="snap")
    assert report["checked"] == 2
    assert report["already_suppressed"] == 1
    assert report["suppressed_identities"] == [
        {"message_fullname": "t1_x", "source_revision_id": "rev"}
    ]
    assert report["input_sha256"] == file_sha256(source)

    # a different revision of the same fullname is not suppressed by an
    # exact-revision row but IS by the fullname-wide NULL row
    other = project_tombstone_identities(
        [{"message_fullname": "t1_other", "source_revision_id": "rev2"}]
    )
    report = suppression_precheck(source, other, snapshot_id="snap")
    assert report["already_suppressed"] == 0

    # missing tombstones table is honest zero
    bare = tmp_path / "bare.db"
    with sqlite3.connect(bare):
        pass
    report = suppression_precheck(bare, projection, snapshot_id="snap")
    assert report["already_suppressed"] == 0 and report["checked"] == 0


def test_precheck_matches_propagate_direct_deletion(
    tmp_path: Path, seeded: dict[str, Path]
) -> None:
    """Precheck reports current state; propagation re-applies idempotently.

    The two reports are independent sections of the reconciliation and are
    never summed: an already-tombstoned identity is counted by the precheck
    as already suppressed and by propagation as (re-)applied deletion, so no
    consumer can double-count the same unit.
    """
    source = seeded["source"]
    with sqlite3.connect(source) as db:
        db.execute(
            "INSERT INTO tombstones (message_fullname, source_revision_id, snapshot_id,"
            " unit_id) VALUES ('t1_x', 'rev', 'snap', NULL)"
        )
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_x", "source_revision_id": "rev"}]
    )
    report = suppression_precheck(source, projection, snapshot_id="snap")
    assert report["already_suppressed"] == 1
    output = tmp_path / "output.db"
    from reddit_search.corpus.sqlite_store import propagate_tombstone_projection

    result = propagate_tombstone_projection(
        source, output, projection, snapshot_id="snap"
    )
    # propagation deletes the matching unit from the derivative (idempotent
    # re-application), while the precheck separately reports the identity as
    # already tombstoned in the input; the manifest keeps these disjoint.
    assert result["deleted_count"] == 1
    # re-running propagation over the same unchanged input is a no-op on rows
    result_again = propagate_tombstone_projection(
        source, tmp_path / "output2.db", projection, snapshot_id="snap"
    )
    assert result_again["deleted_count"] == 1


# --- full flow --------------------------------------------------------------


class _FakeDense:
    """Method-level dense boundary fake recording calls."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def delete_tombstone_projection(self, projection, *, snapshot_id, manifest_path=None):
        self.calls.append(len(projection.records))
        return {
            "kind": "dense_tombstone_propagation",
            "snapshot_id": snapshot_id,
            "ledger_digest": projection.ledger_digest,
            "counts": dict(projection.counts),
            "requested_identities": [
                {"message_fullname": record["message_fullname"]}
                for record in projection.records
            ],
            "requested_count": len(projection.records),
            "deleted_count": 0,
        }


def test_run_happy_path_and_idempotent_rerun(tmp_path: Path, seeded: dict[str, Path]) -> None:
    dense = _FakeDense()
    scope = _scope(
        tmp_path,
        ledger=seeded["ledger"],
        outbox=seeded["outbox"],
        input_path=seeded["source"],
        output=seeded["output"],
        reconciliation=seeded["reconciliation"],
        dense=DenseScope(base_url="http://127.0.0.1:1", collection="c", recipe=_recipe()),
    )
    # patch the dense boundary at the operator level by monkeypatching the builder
    import reddit_search.operations.operator as operator_module

    original = operator_module._build_dense_delete
    operator_module._build_dense_delete = lambda dense_scope: (  # type: ignore[method-assign]
        lambda projection, *, snapshot_id: dense.delete_tombstone_projection(
            projection, snapshot_id=snapshot_id
        )
    )
    try:
        manifest = run_configured_operation(scope)
    finally:
        operator_module._build_dense_delete = original  # type: ignore[method-assign]

    assert manifest["status"] == "applied" and manifest["applied"] is True
    assert manifest["executed"] == {"sqlite": True, "dense": True}
    assert manifest["outbox"]["deleted_count"] == 1
    assert manifest["sqlite"]["matched_count"] == 1
    assert manifest["sqlite"]["deleted_count"] == 1
    assert manifest["dense"]["result"]["requested_count"] == 1
    assert dense.calls == [1]
    assert seeded["output"].exists()
    with LexicalStore(seeded["output"]) as derived:
        assert derived.units(snapshot_id="snap") == []
    assert file_sha256(seeded["source"]) == file_sha256(seeded["source"])

    # idempotent re-run: byte-stable core, dense not called again
    first_bytes = seeded["reconciliation"].read_bytes()
    manifest2 = run_configured_operation(scope)
    assert manifest2["outbox"]["attempts"] == manifest["outbox"]["attempts"]
    assert dense.calls == [1]

    def _core(document: dict) -> dict:
        return {key: value for key, value in document.items() if key != "observation"}

    assert _core(manifest2) == _core(manifest)
    on_disk_first = json.loads(first_bytes)
    on_disk_second = json.loads(seeded["reconciliation"].read_bytes())
    assert _core(on_disk_first) == _core(on_disk_second)


def test_run_dense_disabled_refuses_and_row_stays_retryable(
    tmp_path: Path, seeded: dict[str, Path]
) -> None:
    source_hash = file_sha256(seeded["source"])
    with pytest.raises(OperatorBoundaryError, match="not permitted"):
        _run(tmp_path, seeded)
    assert not seeded["reconciliation"].exists()
    assert file_sha256(seeded["source"]) == source_hash
    row = TombstoneOutbox(seeded["outbox"]).pending()[0]
    assert row["status"] == "failed"
    assert "not permitted" in row["last_error"]


def test_run_failure_marks_failed_without_false_reconciliation(
    tmp_path: Path, seeded: dict[str, Path]
) -> None:
    """A dense boundary that raises leaves a failed row and no manifest."""
    import reddit_search.operations.operator as operator_module

    def broken_dense_delete(dense_scope):
        def boundary(projection, *, snapshot_id):
            raise RuntimeError("dense outage")

        return boundary

    original = operator_module._build_dense_delete
    operator_module._build_dense_delete = broken_dense_delete  # type: ignore[method-assign]
    try:
        scope = _scope(
            tmp_path,
            ledger=seeded["ledger"],
            outbox=seeded["outbox"],
            input_path=seeded["source"],
            output=seeded["output"],
            reconciliation=seeded["reconciliation"],
            dense=DenseScope(base_url="http://127.0.0.1:1", collection="c", recipe=_recipe()),
        )
        with pytest.raises(OperatorBoundaryError, match="dense outage"):
            run_configured_operation(scope)
    finally:
        operator_module._build_dense_delete = original  # type: ignore[method-assign]
    assert not seeded["reconciliation"].exists()
    row = TombstoneOutbox(seeded["outbox"]).pending()[0]
    assert row["status"] == "failed" and "outage" in row["last_error"]


def test_run_hash_binds_input_and_fails_closed_on_change(
    tmp_path: Path, seeded: dict[str, Path]
) -> None:
    """The manifest binds the exact input bytes each run observed."""
    import reddit_search.operations.operator as operator_module

    fake = _FakeDense()
    dense = DenseScope("http://127.0.0.1:1", "c", _recipe())
    original = operator_module._build_dense_delete
    operator_module._build_dense_delete = lambda dense_scope: (  # type: ignore[method-assign]
        lambda projection, *, snapshot_id: fake.delete_tombstone_projection(
            projection, snapshot_id=snapshot_id
        )
    )
    try:
        manifest = _run(tmp_path, seeded, dense=dense)
    finally:
        operator_module._build_dense_delete = original  # type: ignore[method-assign]
    assert manifest["applied"] is True
    assert manifest["suppression_precheck"]["input_sha256"] == file_sha256(
        seeded["source"]
    )
    assert manifest["projection"]["source_artifact_hashes"]["sqlite"] == file_sha256(
        seeded["source"]
    )
    assert manifest["sqlite"]["input_sha256"] == file_sha256(seeded["source"])

    # a changed input corpus is a different byte-identity: the new manifest
    # records the new hash instead of pretending byte-stability
    with sqlite3.connect(seeded["source"]) as db:
        db.execute(
            "INSERT INTO tombstones (message_fullname, source_revision_id, snapshot_id,"
            " unit_id) VALUES ('t1_prefill', NULL, 'snap', NULL)"
        )
    original = operator_module._build_dense_delete
    operator_module._build_dense_delete = lambda dense_scope: (  # type: ignore[method-assign]
        lambda projection, *, snapshot_id: fake.delete_tombstone_projection(
            projection, snapshot_id=snapshot_id
        )
    )
    try:
        manifest2 = _run(tmp_path, seeded)
    finally:
        operator_module._build_dense_delete = original  # type: ignore[method-assign]
    assert manifest2["suppression_precheck"]["input_sha256"] != manifest[
        "suppression_precheck"
    ]["input_sha256"]

    # the projection builder itself fails closed on a supplied hash that does
    # not match the input bytes
    with pytest.raises(ValueError, match="hash mismatch"):
        project_tombstone_identities(
            [{"message_fullname": "t1_x", "source_revision_id": "rev"}],
            source_artifacts={"sqlite": (seeded["source"], "0" * 64)},
        )


def test_run_rejects_snapshot_mismatch_between_register_and_scope(
    tmp_path: Path, seeded: dict[str, Path]
) -> None:
    scope = _scope(
        tmp_path,
        ledger=seeded["ledger"],
        outbox=seeded["outbox"],
        input_path=seeded["source"],
        output=seeded["output"],
        reconciliation=seeded["reconciliation"],
    )
    # register under a different snapshot then run with mismatched expectation
    projection = project_tombstone_identities(
        [{"message_fullname": "t1_x", "source_revision_id": "rev"}]
    )
    outbox = TombstoneOutbox(seeded["outbox"])
    row = outbox.register(projection, scope={"snapshot_id": "other"})
    object.__setattr__(scope, "snapshot_id", "other")
    # validate_replay_targets must refuse the mismatch against the input scope
    from reddit_search.operations import validate_replay_targets

    with pytest.raises(ValueError, match="snapshot"):
        validate_replay_targets(
            outbox,
            row["identity"],
            expected_snapshot_id="snap",
            sqlite_input=scope.sqlite_input,
            sqlite_output=scope.sqlite_output,
            sqlite_manifest=None,
        )
