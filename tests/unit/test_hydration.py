"""Unit tests for bounded second-pass hydration (execution step 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard

from reddit_search.ingest.hydration import (
    HydrationLimits,
    freeze_thread_ids,
    run_hydration,
)
from reddit_search.ingest.shard import read_selected_messages


def _write_shard(path: Path, payloads: list[dict[str, object]]) -> Path:
    with path.open("wb") as output:
        with zstandard.ZstdCompressor().stream_writer(output, closefd=False) as compressed:
            for payload in payloads:
                compressed.write(json.dumps(payload).encode() + b"\n")
    return path


def _archive(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _submission(
    fullname: str, title: str, body: str, created: int, *, subreddit: str = "personalfinance"
) -> dict[str, object]:
    return {
        "id": fullname.removeprefix("t3_"),
        "name": fullname,
        "title": title,
        "selftext": body,
        "subreddit": subreddit,
        "created_utc": created,
        "permalink": f"/r/{subreddit}/comments/{fullname}/{fullname}_slug/",
        "score": 5,
    }


def _comment(
    fullname: str,
    thread: str,
    parent: str | None,
    body: str,
    created: int,
    *,
    subreddit: str = "personalfinance",
) -> dict[str, object]:
    return {
        "id": fullname.removeprefix("t1_"),
        "name": fullname,
        "link_id": thread,
        "parent_id": parent,
        "body": body,
        "subreddit": subreddit,
        "created_utc": created,
        "permalink": f"/r/{subreddit}/comments/{thread}/slug/{fullname}/",
        "score": 2,
    }


def _selection_row(
    fullname: str, thread: str, parent: str | None, created: int
) -> dict[str, object]:
    kind = "submission" if fullname.startswith("t3_") else "comment"
    return {
        "fullname": fullname,
        "kind": kind,
        "thread_fullname": thread,
        "parent_fullname": parent,
        "raw_title": "Bank sync frustrations" if kind == "submission" else "",
        "raw_body": "I wish expense tracking did not require linking my bank.",
        "subreddit": "personalfinance",
        "created_utc": created,
        "source_revision_id": f"rev-{fullname}",
        "permalink": f"/r/personalfinance/comments/{thread}/slug/{fullname}/",
        "provenance": [{"source_id": "june-comments", "line_number": created}],
        "archive_score": 4,
        "depth": 1,
        "selection_channels": ["topic_rule"],
        "matched_rule_ids": ["mieru.no_bank_link"],
    }


def _registry(path: Path, sources: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "sources": sources}, indent=2), encoding="utf-8"
    )
    return path


@pytest.fixture()
def workspace(tmp_path: Path) -> dict[str, Path]:
    """Registry with one discovery source; selected shard roots thread t3_root."""
    archive = _archive(
        tmp_path / "RS_2026-06.jsonl",
        [
            # Selected focus: t1_focus comment in thread t3_root.
            _comment("t1_focus", "t3_root", "t3_root", "Focus body.", 100),
            # Unselected but available comment parent in the same thread.
            _comment("t1_parent", "t3_root", "t3_root", "Available parent body.", 110),
            _comment("t1_sibling", "t3_root", "t3_root", "Sibling body.", 120),
            # Rejected records in unrelated threads (control candidates).
            _comment("t1_control", "t3_other", None, "Unrelated comment.", 130),
            _comment("t1_control2", "t3_another", None, "Another unrelated comment.", 140),
        ],
    )
    registry = _registry(
        tmp_path / "registry.json",
        [
            {
                "source_id": "june-comments",
                "source_path": str(archive),
                "source_kind": "comment",
                "declared_month": "2026-06",
                "source_role": "discovery",
                "usage_scope": "synthetic",
                "status": "complete",
            }
        ],
    )
    selection = _write_shard(
        tmp_path / "selected.jsonl.zst",
        [_selection_row("t1_focus", "t3_root", "t3_root", 100)],
    )
    return {
        "registry": registry,
        "selection": selection,
        "archive": archive,
        "output": tmp_path / "hydration",
    }


def test_freeze_thread_ids_counts_selected_and_threads(workspace: dict[str, Path]) -> None:
    report = freeze_thread_ids(workspace["selection"])
    assert report == {"selected_count": 1, "thread_count": 1}


def test_hydration_recovers_unselected_parent_with_provenance(
    workspace: dict[str, Path],
) -> None:
    result = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
    )
    assert result["complete"] is True
    assert result["selected_count"] == 1
    assert result["control_count"] == 2  # t1_control and t1_control2
    context_path = workspace["output"] / "hydrated-context.jsonl.zst"
    messages = {message.fullname: message for message in read_selected_messages(context_path)}
    assert set(messages) == {"t1_focus", "t1_parent", "t1_sibling"}
    parent = messages["t1_parent"]
    assert parent.raw_body == "Available parent body."
    assert parent.provenance[0].source_id == "june-comments"
    assert parent.provenance[0].line_number == 2  # exact archive line

    controls = {
        message.fullname: message
        for message in read_selected_messages(workspace["output"] / "rejected-controls.jsonl.zst")
    }
    assert set(controls) == {"t1_control", "t1_control2"}
    report = json.loads((workspace["output"] / "hydration_manifest.json").read_text())
    assert report["complete"] is True
    assert report["sources"][0]["processed"] == 5
    assert report["sources"][0]["thread_matched"] == 3
    assert report["sources"][0]["rejected"] == 2
    telemetry = json.loads(
        (workspace["output"] / "stage_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "complete"
    assert telemetry["run_identity"] == result["run_id"]
    assert result["telemetry"]["sha256"]


def test_hydration_context_capacity_bounds_retention(
    workspace: dict[str, Path],
) -> None:
    result = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=1, seed=7, control_target=0
        ),
    )
    assert result["complete"] is True
    assert result["context_count"] == 1
    assert result["context_candidate_count"] == 2
    report = json.loads((workspace["output"] / "hydration_manifest.json").read_text())
    assert report["context_count"] == 1


def test_hydration_staging_byte_budget_marks_incomplete_and_withholds_shards(
    workspace: dict[str, Path],
) -> None:
    result = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1,
            max_context_messages=100, seed=7, control_target=0, max_staging_bytes=1
        ),
    )
    assert result["complete"] is False
    assert result["interruption_reason"] == "staging budget exceeded"
    assert result["context_path"] is None
    assert not (workspace["output"] / "hydrated-context.jsonl.zst").exists()
    report = json.loads((workspace["output"] / "hydration_manifest.json").read_text())
    assert report["complete"] is False
    assert report["budget_exhausted"] is True


def test_hydration_refuses_when_free_disk_reserve_is_unavailable(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import reddit_search.resources as resources

    usage = type("Usage", (), {"free": 0, "total": 1, "used": 1})
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _path: usage())
    with pytest.raises(ValueError, match="insufficient free disk"):
        run_hydration(
            workspace["registry"],
            workspace["selection"],
            workspace["output"],
            limits=HydrationLimits(
                max_context_messages=100,
                seed=7,
                control_target=0,
                minimum_free_disk_bytes=1,
            ),
        )
    assert not (workspace["output"] / "hydrated-context.jsonl.zst").exists()
    assert not (workspace["output"] / "rejected-controls.jsonl.zst").exists()


def test_hydration_publishes_both_shards_atomically(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import reddit_search.ingest.hydration as hydration

    original = hydration.write_compressed_jsonl
    calls = 0

    def fail_on_controls(path: Path, messages: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic controls write failure")
        original(path, messages)  # type: ignore[arg-type]

    monkeypatch.setattr(hydration, "write_compressed_jsonl", fail_on_controls)
    with pytest.raises(OSError, match="synthetic controls write failure"):
        run_hydration(
            workspace["registry"],
            workspace["selection"],
            workspace["output"],
            limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
        )
    assert not (workspace["output"] / "hydrated-context.jsonl.zst").exists()
    assert not (workspace["output"] / "rejected-controls.jsonl.zst").exists()
    assert not list(workspace["output"].glob("*.staging"))


def test_hydration_deterministic_across_reruns(workspace: dict[str, Path]) -> None:
    first = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"] / "one",
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
    )
    second = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"] / "two",
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
    )
    assert first["complete"] and second["complete"]
    for name in ("hydrated-context.jsonl.zst", "rejected-controls.jsonl.zst"):
        left = (workspace["output"] / "one" / name).read_bytes()
        right = (workspace["output"] / "two" / name).read_bytes()
        assert left == right


def test_hydration_resumes_from_completed_source_checkpoint(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import reddit_search.ingest.hydration as hydration

    original = hydration._write_hydration_checkpoint
    interrupted = False

    def write_then_interrupt(*args: object, **kwargs: object) -> None:
        nonlocal interrupted
        original(*args, **kwargs)  # type: ignore[arg-type]
        if not interrupted:
            interrupted = True
            raise RuntimeError("synthetic interruption after checkpoint")

    monkeypatch.setattr(hydration, "_write_hydration_checkpoint", write_then_interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_hydration(
            workspace["registry"],
            workspace["selection"],
            workspace["output"],
            limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
        )

    checkpoint = workspace["output"] / "hydration_checkpoint.json"
    assert checkpoint.exists()
    with pytest.raises(ValueError, match="does not match current inputs or limits"):
        run_hydration(
            workspace["registry"],
            workspace["selection"],
            workspace["output"],
            limits=HydrationLimits(
                minimum_free_disk_bytes=1, max_context_messages=100, seed=8, control_target=2
            ),
        )

    def fail_if_source_is_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("completed source was reread")

    monkeypatch.setattr(hydration.ArchiveReader, "iter_records", fail_if_source_is_read)

    result = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
    )
    assert result["complete"] is True
    assert result["resumed_from_checkpoint"] is True
    assert not checkpoint.exists()
    repeated = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
    )
    assert repeated == result
    (workspace["output"] / "hydrated-context.jsonl.zst").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="outputs do not match their manifest"):
        run_hydration(
            workspace["registry"],
            workspace["selection"],
            workspace["output"],
            limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
        )
def test_hydration_rss_budget_withholds_shards_and_records_manifest(
    workspace: dict[str, Path],
) -> None:
    result = run_hydration(
        workspace["registry"], workspace["selection"], workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1,
            max_context_messages=100,
            seed=7,
            control_target=0,
            max_process_rss_bytes=1,
        ),
        rss_sampler=lambda: 2,
    )
    report = json.loads((workspace["output"] / "hydration_manifest.json").read_text())
    assert result["complete"] is False
    assert result["budget_exhausted"] is True
    assert not (workspace["output"] / "hydrated-context.jsonl.zst").exists()
    assert not (workspace["output"] / "rejected-controls.jsonl.zst").exists()
    assert report["complete"] is False and report["budget_exhausted"] is True
    assert report["memory_budget"]["max_process_rss_bytes"] == 1


def test_hydration_scan_limit_marks_incomplete(workspace: dict[str, Path]) -> None:
    result = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=0
        ),
        max_records=2,
    )
    assert result["complete"] is False
    assert result["scanned_record_count"] == 2
    telemetry = json.loads(
        (workspace["output"] / "stage_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "incomplete"
    assert telemetry["errors"][-1]["field"] == "hydration"


def test_hydration_resumes_within_source_checkpoint(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import reddit_search.ingest.hydration as hydration

    original_checkpoint = hydration._write_hydration_checkpoint
    original_normalize = hydration.normalize_record
    normalized_records: list[str] = []
    interrupted = False
    tombstones = workspace["output"].parent / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t1_parent",
                "source_revision_id": None,
                "reason": "removed by operator",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def track_normalize(envelope: object) -> object:
        normalized_records.append(str(envelope.payload["id"]))  # type: ignore[attr-defined]
        return original_normalize(envelope)  # type: ignore[arg-type]

    def write_then_interrupt(*args: object, **kwargs: object) -> None:
        nonlocal interrupted
        original_checkpoint(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs.get("active_source") is not None and not interrupted:
            interrupted = True
            raise RuntimeError("synthetic mid-source interruption")

    monkeypatch.setattr(hydration, "normalize_record", track_normalize)
    monkeypatch.setattr(hydration, "_write_hydration_checkpoint", write_then_interrupt)
    with pytest.raises(RuntimeError, match="synthetic mid-source interruption"):
        run_hydration(
            workspace["registry"],
            workspace["selection"],
            workspace["output"],
            limits=HydrationLimits(
                minimum_free_disk_bytes=1,
                max_context_messages=100,
                seed=7,
                control_target=2,
                checkpoint_interval_records=2,
            ),
            tombstones_path=tombstones,
        )

    checkpoint = json.loads(
        (workspace["output"] / "hydration_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["active_source"]["processed_records"] == 2
    assert checkpoint["active_source"]["tombstoned"] == 1

    result = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1,
            max_context_messages=100,
            seed=7,
            control_target=2,
            checkpoint_interval_records=2,
        ),
        tombstones_path=tombstones,
    )
    assert result["complete"] is True
    assert result["scanned_record_count"] == 5
    assert result["tombstoned_count"] == 1
    assert len(normalized_records) == 5
    assert not (workspace["output"] / "hydration_checkpoint.json").exists()


def test_hydration_missing_source_marks_incomplete_and_fails_registry_row(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    missing = tmp_path / "RS_missing.jsonl"
    registry_path = _registry(
        tmp_path / "registry_missing.json",
        [
            {
                "source_id": "june-missing",
                "source_path": str(missing),
                "source_kind": "comment",
                "declared_month": "2026-06",
                "source_role": "discovery",
                "usage_scope": "synthetic",
                "status": "complete",
            }
        ],
    )
    result = run_hydration(
        registry_path,
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=10, seed=7, control_target=0
        ),
    )
    assert result["complete"] is False
    assert "source read failed" in result["interruption_reason"]
    registry = json.loads(registry_path.read_text())
    assert registry["sources"][0]["status"] == "failed"
    report = json.loads((workspace["output"] / "hydration_manifest.json").read_text())
    assert report["complete"] is False


def test_hydration_rejects_negative_limits(workspace: dict[str, Path]) -> None:
    with pytest.raises(ValueError):
        run_hydration(
            workspace["registry"],
            workspace["selection"],
            workspace["output"],
            limits=HydrationLimits(
                minimum_free_disk_bytes=1, max_context_messages=-1, seed=7, control_target=0
            ),
        )
    with pytest.raises(ValueError, match="checkpoint_interval_records"):
        run_hydration(
            workspace["registry"],
            workspace["selection"],
            workspace["output"],
            limits=HydrationLimits(
                minimum_free_disk_bytes=1,
                max_context_messages=10,
                seed=7,
                control_target=0,
                checkpoint_interval_records=0,
            ),
        )


def test_hydration_excludes_tombstoned_context(workspace: dict[str, Path]) -> None:
    tombstones = workspace["output"].parent / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t1_parent",
                "source_revision_id": None,
                "reason": "removed by operator",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
        tombstones_path=tombstones,
    )

    assert result["selected_count"] == 1
    assert result["tombstoned_count"] == 1
    context = {
        message.fullname
        for message in read_selected_messages(workspace["output"] / "hydrated-context.jsonl.zst")
    }
    assert context == {"t1_focus", "t1_sibling"}


def test_hydration_publishes_empty_outputs_when_selection_is_tombstoned(
    workspace: dict[str, Path],
) -> None:
    tombstones = workspace["output"].parent / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t1_focus",
                "source_revision_id": None,
                "reason": "removed by operator",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_hydration(
        workspace["registry"],
        workspace["selection"],
        workspace["output"],
        limits=HydrationLimits(
            minimum_free_disk_bytes=1, max_context_messages=100, seed=7, control_target=2
        ),
        tombstones_path=tombstones,
    )

    assert result["complete"] is True
    assert result["selected_count"] == 0
    assert result["context_count"] == 0
    assert result["control_count"] == 0
    assert result["tombstoned_count"] == 1
