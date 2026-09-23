import json
from pathlib import Path

import pytest

from reddit_search.reporting import operational_report
from reddit_search.telemetry import (
    capture_stage_telemetry,
    load_stage_telemetry,
    write_stage_telemetry,
)


def cards(path: Path) -> None:
    path.write_text(
        json.dumps({"review_status": "complete", "context": {"context_complete": True}})
        + "\n"
    )


def telemetry_for(cards_path: Path, **kwargs: object) -> dict:
    from reddit_search.reporting import _sha256

    return capture_stage_telemetry(
        stage="review",
        input_hashes={"files": [{"path": str(cards_path), "sha256": _sha256(cards_path)}]},
        **kwargs,
    )


def test_fixed_providers_are_captured_and_reported_deterministically(tmp_path: Path) -> None:
    card_path = tmp_path / "cards.jsonl"
    cards(card_path)
    telemetry = telemetry_for(
        card_path,
        clock=iter([10.0, 12.5]).__next__,
        rss_provider=lambda: 123,
        disk_provider=lambda _: 456,
        path=tmp_path,
        cache_provider=lambda: "hit",
        cleanup_provider=lambda: "complete",
    )
    path = tmp_path / "telemetry.json"
    write_stage_telemetry(path, telemetry)
    report = operational_report(card_path, telemetry_path=path)
    assert report["resource_usage"] == {
        "elapsed_seconds": 2.5,
        "rss_bytes": 123,
        "disk_bytes": 456,
        "cache": "hit",
        "cleanup_state": "complete",
    }
    assert report["telemetry"]["status"] == "complete"


def test_telemetry_input_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    card_path = tmp_path / "cards.jsonl"
    cards(card_path)
    telemetry = telemetry_for(card_path)
    telemetry["input_hashes"]["files"][0]["sha256"] = "0" * 64
    telemetry_path = tmp_path / "telemetry.json"
    write_stage_telemetry(telemetry_path, telemetry)
    with pytest.raises(ValueError, match="input hashes"):
        operational_report(card_path, telemetry_path=telemetry_path)


def test_legacy_report_without_telemetry_preserves_unknown_values(tmp_path: Path) -> None:
    card_path = tmp_path / "cards.jsonl"
    cards(card_path)
    report = operational_report(card_path)
    assert report["resource_usage"] == {
        "elapsed_seconds": None,
        "rss_bytes": None,
        "disk_bytes": None,
        "cache": "unknown",
        "cleanup_state": "unknown",
    }
    assert report["telemetry"] is None

def test_provider_failures_are_incomplete_and_preserve_unknowns(tmp_path: Path) -> None:
    card_path = tmp_path / "cards.jsonl"
    cards(card_path)
    telemetry = telemetry_for(
        card_path,
        path=tmp_path,
        rss_provider=lambda: (_ for _ in ()).throw(OSError("rss unavailable")),
        disk_provider=lambda _: (_ for _ in ()).throw(OSError("disk unavailable")),
        cache_provider=lambda: "not_used",
        cleanup_provider=lambda: "incomplete",
    )
    assert telemetry["status"] == "incomplete"
    assert {item["field"] for item in telemetry["errors"]} == {"rss_bytes", "free_disk_bytes"}
    observations = telemetry["observations"]
    assert observations["rss_bytes"] is None
    assert observations["free_disk_bytes"] is None
    assert observations["cache"] == "not_used"
    assert observations["cleanup"] == "incomplete"
def test_negative_provider_value_is_incomplete_and_unknown(tmp_path: Path) -> None:
    telemetry = capture_stage_telemetry(
        stage="review",
        path=tmp_path,
        rss_provider=lambda: -1,
        disk_provider=lambda _: -2,
    )
    assert telemetry["status"] == "incomplete"
    assert telemetry["observations"]["rss_bytes"] is None
    assert telemetry["observations"]["free_disk_bytes"] is None


def test_loader_rejects_malformed_status_and_observations(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.json"
    path.write_text(
        json.dumps(
            {
                "kind": "stage_telemetry",
                "schema_version": 1,
                "status": "bogus",
                "stage": "review",
                "input_hashes": {},
                "observations": {"elapsed_seconds": -1},
            }
        )
    )
    with pytest.raises(ValueError, match="invalid status"):
        load_stage_telemetry(path)
    payload = json.loads(path.read_text())
    payload["status"] = "complete"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="elapsed_seconds"):
        load_stage_telemetry(path)


def test_partial_cache_state_is_valid() -> None:
    telemetry = capture_stage_telemetry(stage="review", cache_provider=lambda: "partial")
    assert telemetry["status"] == "complete"
    assert telemetry["observations"]["cache"] == "partial"

