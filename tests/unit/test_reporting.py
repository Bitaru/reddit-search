import json
from pathlib import Path

from reddit_search.reporting import operational_report, topic_report, write_report


def _cards(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "app": "app",
                "topic_fit": "yes",
                "review_status": "complete",
                "source": {"created_utc": 10},
                "context": {"context_complete": True},
            }
        )
        + "\n"
        + json.dumps(
            {
                "app": "app",
                "topic_fit": "unknown",
                "review_status": "pending",
                "source": {"created_utc": 20},
                "context": {"context_complete": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_topic_report_handles_legacy_missing_mappings(tmp_path: Path) -> None:
    cards = tmp_path / "legacy.jsonl"
    cards.write_text(
        json.dumps(
            {
                "topic_fit": "unknown",
                "review_status": "pending",
                "source": None,
                "retrieval": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = topic_report(cards, app="app")
    assert report["counts"] == {
        "cards": 0,
        "review_status": {},
        "topic_fit": {},
    }
    report = topic_report(cards)
    assert report["counts"]["cards"] == 1
    assert report["observed_date_coverage"]["count"] == 0


def test_topic_report_counts_unknown_and_date_coverage(tmp_path: Path) -> None:
    cards = tmp_path / "cards.jsonl"
    _cards(cards)
    report = topic_report(cards, snapshot_id="snap", app="app")
    assert report["counts"]["cards"] == 2
    assert report["counts"]["topic_fit"] == {"unknown": 1, "yes": 1}
    assert report["observed_date_coverage"] == {
        "min_created_utc": 10.0,
        "max_created_utc": 20.0,
        "count": 2,
    }


def test_topic_report_app_filter_is_explicit_only(tmp_path: Path) -> None:
    cards = tmp_path / "cards.jsonl"
    cards.write_text(
        json.dumps(
            {
                "topic_fit": "yes",
                "review_status": "pending",
                "retrieval": {"scenario_id": "something.legacy_prefix"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "app_id": "something",
                "topic_fit": "unknown",
                "review_status": "pending",
                "retrieval": {"scenario_id": "unrelated.topic"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "app": "something",
                "topic_fit": "yes",
                "review_status": "complete",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = topic_report(cards, app="something")
    assert report["counts"]["cards"] == 2
    assert report["counts"]["topic_fit"] == {"unknown": 1, "yes": 1}
    unfiltered = topic_report(cards)
    assert unfiltered["counts"]["cards"] == 3


def test_operational_report_marks_unavailable_resources_and_is_deterministic(
    tmp_path: Path,
) -> None:
    cards = tmp_path / "cards.jsonl"
    _cards(cards)
    first = operational_report(cards)
    second = operational_report(cards)
    assert first == second
    assert first["missing_context"] == {"complete": 1, "incomplete": 1}
    assert first["resource_usage"]["rss_bytes"] is None
    output = write_report(first, tmp_path / "out.json")
    assert json.loads(output.read_text(encoding="utf-8")) == first
