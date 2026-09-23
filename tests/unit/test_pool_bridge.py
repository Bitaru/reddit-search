import json
from pathlib import Path

from reddit_search.ingest.state import file_sha256
from reddit_search.review.pool_bridge import convert_blinded_pool


def _row(candidate_id: str, scenario_id: str | None) -> dict[str, object]:
    return {
        "pool_id": "p",
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "app_id": "app" if scenario_id else None,
        "app_profile_version": "1" if scenario_id else None,
        "app_profile_sha256": "profile" if scenario_id else None,
        "thread_fullname": "t_" + candidate_id,
        "snapshot_id": "snap",
        "context_complete": True,
        "source": {"kind": "test"},
        "rank": 1,
        "score": 0.5,
        "variant_id": "v",
        "retrieval": {"x": 1},
        "stratum": "x",
    }


def test_conversion_is_deterministic_and_binds_pool(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    rows = [_row("r1", "s1"), _row("c1", None), _row("r2", "s2")]
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out1, out2 = tmp_path / "one", tmp_path / "two"
    manifest = convert_blinded_pool(pool, out1)
    manifest2 = convert_blinded_pool(pool, out2)
    assert (out1 / "review_cards.jsonl").read_bytes() == (out2 / "review_cards.jsonl").read_bytes()
    assert manifest == manifest2
    assert manifest["pool_sha256"] == file_sha256(pool)
    cards = [json.loads(line) for line in (out1 / "review_cards.jsonl").read_text().splitlines()]
    assert [card["review_queue"]["queue_order"] for card in cards] == [1, 2, 3]
    assert cards[1]["scenario_id"] is None
    assert cards[1]["review_queue"]["stratum_rule_id"] == "operational_control"
    forbidden_keys = [{"rank", "score", "variant_id", "retrieval", "stratum"}]
    assert all(not forbidden.intersection(card) for card in cards for forbidden in forbidden_keys)
    assert all(
        card["selection"]["matched_rule_ids"] == [card["review_queue"]["stratum_rule_id"]]
        for card in cards
    )

def test_conversion_accepts_distinct_nonempty_pool_ids(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    rows = [_row("r1", "s1"), _row("r2", "s2")]
    rows[1]["pool_id"] = "p2"
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = convert_blinded_pool(pool, tmp_path / "out")
    assert manifest["pool_row_count"] == 2
