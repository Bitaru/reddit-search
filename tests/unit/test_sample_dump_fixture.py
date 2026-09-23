"""Real-format tracked fixture: submission and comment normalization via the
production reader path, plus neutral-rule discovery over the same fixture."""

import hashlib
from pathlib import Path

from reddit_search.ingest.discovery import DiscoverySelector, load_discovery_rules
from reddit_search.ingest.normalize import normalize_record
from reddit_search.ingest.reader import ArchiveReader, SourceSpec

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sample_dump"


def _read(kind: str, name: str):
    reader = ArchiveReader()
    source = SourceSpec(f"sample-{kind}", FIXTURES / name, kind, "2026-01")
    messages = [normalize_record(envelope) for envelope in reader.iter_records(source)]
    assert reader.stats.complete, reader.stats
    assert reader.stats.valid_records == len(messages)
    return messages


def test_fixture_submissions_normalize_with_real_fields() -> None:
    messages = _read("submission", "RS_2026-01.jsonl")

    assert len(messages) == 3
    by_fullname = {message.fullname: message for message in messages}
    assert set(by_fullname) == {"t3_fx1", "t3_fx2", "t3_fx3"}
    for message in messages:
        assert message.kind == "submission"
        assert message.thread_fullname == message.fullname
        assert message.parent_fullname is None
        assert message.source_revision_id == hashlib.sha256(
            f"{message.raw_title}\0{message.raw_body}".encode()
        ).hexdigest()
    assert by_fullname["t3_fx1"].raw_title == (
        "How do I track spending without linking my bank?"
    )
    assert by_fullname["t3_fx1"].subreddit == "personalfinance"
    assert by_fullname["t3_fx1"].created_utc == 1_767_225_600


def test_fixture_comments_normalize_with_thread_linkage() -> None:
    messages = _read("comment", "RC_2026-01.jsonl")

    assert len(messages) == 2
    by_fullname = {message.fullname: message for message in messages}
    assert set(by_fullname) == {"t1_fxc1", "t1_fxc2"}
    for message in messages:
        assert message.kind == "comment"
        assert message.fullname.startswith("t1_")
        assert message.thread_fullname.startswith("t3_")
        assert message.thread_fullname != message.fullname
    assert by_fullname["t1_fxc1"].thread_fullname == "t3_fx1"
    assert by_fullname["t1_fxc1"].parent_fullname == "t3_fx1"
    assert by_fullname["t1_fxc2"].thread_fullname == "t3_fx2"


def test_fixture_discovery_selects_neutral_expense_rows(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """rules:
  - rule_id: expense.no_bank_link
    required_term_groups:
      - [expense, spending, budget]
      - [bank, sync, link, privacy]
""",
        encoding="utf-8",
    )
    selector = DiscoverySelector(load_discovery_rules(rules_path))

    submissions = _read("submission", "RS_2026-01.jsonl")
    comments = _read("comment", "RC_2026-01.jsonl")

    selected = []
    for message in [*submissions, *comments]:
        decision = selector.select(message)
        if decision.selected:
            selected.append((message, decision))
    # The bank-privacy submission and its agreement comment match; the weekly
    # chat thread and the mobile-invoicing submission do not.
    assert [message.fullname for message, _ in selected] == ["t3_fx1", "t1_fxc1"]
    for _message, decision in selected:
        assert "expense.no_bank_link" in decision.matched_rule_ids
