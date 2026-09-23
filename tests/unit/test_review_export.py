import json
from pathlib import Path
from types import SimpleNamespace


def test_lexical_export_omits_invalidated_hit(tmp_path: Path) -> None:
    from reddit_search.corpus.sqlite_store import LexicalHit
    from reddit_search.ingest.invalidation import load_tombstone_ledger
    from reddit_search.review.export import write_lexical_review_cards

    def unit(fullname: str) -> SimpleNamespace:
        return SimpleNamespace(
            unit_id=fullname,
            synthetic=False,
            message_fullname=fullname,
            source_revision_id="rev-1",
            chunking_version="v2",
            context_recipe_version="v2",
            focus_field="body",
            focus_start=0,
            focus_end=4,
            focus_text="text",
            permalink="/",
            subreddit="x",
            created_utc=1,
            context_text="text",
            context_message_refs=[],
            missing_context_ids=[],
            ancestors_truncated=False,
            context_missing=False,
        )

    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        '{"message_fullname":"t1_stale","source_revision_id":null,"reason":"gone"}\n',
        encoding="utf-8",
    )
    hits = [
        LexicalHit(unit("t1_stale"), 1.0, 1, "stale"),
        LexicalHit(unit("t1_live"), 0.5, 2, "live"),
    ]
    write_lexical_review_cards(
        tmp_path / "out",
        hits,
        scenario_id="s1",
        query="q",
        tombstone_ledger=load_tombstone_ledger(tombstones),
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "review_cards.jsonl").read_text().splitlines()
    ]
    assert [row["source"]["message_fullname"] for row in rows] == ["t1_live"]
    assert rows[0]["source"]["chunking_version"] == "v2"
    assert rows[0]["source"]["context_recipe_version"] == "v2"


def test_lexical_cards_default_structured_judgments_are_explicit(tmp_path: Path) -> None:
    from reddit_search.corpus.sqlite_store import LexicalHit
    from reddit_search.review.export import write_lexical_review_cards

    unit = SimpleNamespace(
        unit_id="c1",
        synthetic=False,
        message_fullname="t1_c1",
        source_revision_id="r",
        chunking_version="v2",
        context_recipe_version="v2",
        focus_field="body",
        focus_start=0,
        focus_end=1,
        focus_text="x",
        permalink="/",
        subreddit="x",
        created_utc=1,
        context_text="x",
        context_message_refs=[],
        missing_context_ids=[],
        ancestors_truncated=False,
        context_missing=False,
    )
    write_lexical_review_cards(
        tmp_path, [LexicalHit(unit, 1.0, 1, "x")], scenario_id="s", query="q"
    )
    row = json.loads((tmp_path / "review_cards.jsonl").read_text().strip())
    assert row["speaker_intent"] == "unknown"
    assert row["product_fit"] == "not_evaluated"
    assert row["resolution_in_available_context"] == "unknown"
