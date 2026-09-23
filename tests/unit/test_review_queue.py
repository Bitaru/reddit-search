import hashlib
import json
from pathlib import Path


def test_balanced_review_queue_assigns_multi_rule_card_to_rarest_stratum(tmp_path: Path) -> None:
    from reddit_search.review.queue import build_balanced_review_queue

    cards_path = tmp_path / "review_cards.jsonl"
    cards = [
        _card("shared", 1, ["rule.common", "rule.rare"]),
        _card("common-one", 2, ["rule.common"]),
        _card("common-two", 3, ["rule.common"]),
        _card("common-later", 4, ["rule.common"]),
        _card("rare-only", 5, ["rule.rare"]),
        _card("rare-later", 6, ["rule.rare"]),
    ]
    cards_path.write_text("".join(json.dumps(card) + "\n" for card in cards))

    report = build_balanced_review_queue(cards_path, tmp_path / "queue", limit=4)

    queue_path = tmp_path / "queue" / "review_cards.jsonl"
    queued_cards = [json.loads(line) for line in queue_path.read_text().splitlines()]
    manifest = json.loads((tmp_path / "queue" / "queue_manifest.json").read_text())
    assert report == {
        "input_card_count": 6,
        "queued_card_count": 4,
        "cards_file": str(queue_path),
        "manifest_file": str(tmp_path / "queue" / "queue_manifest.json"),
    }
    assert [card["candidate_id"] for card in queued_cards] == [
        "shared",
        "rare-only",
        "common-one",
        "common-two",
    ]
    assert [card["review_queue"]["stratum_rule_id"] for card in queued_cards] == [
        "rule.rare",
        "rule.rare",
        "rule.common",
        "rule.common",
    ]
    assert manifest["strata"] == [
        {"available_card_count": 4, "queued_card_count": 2, "rule_id": "rule.common"},
        {"available_card_count": 3, "queued_card_count": 2, "rule_id": "rule.rare"},
    ]


def test_queue_manifest_hashes_raw_input_and_dependency_identity(tmp_path: Path) -> None:
    from reddit_search.review.queue import build_balanced_review_queue

    card = _card("candidate", 1, ["rule"])
    card["snapshot_id"] = "snapshot-1"
    cards_path = tmp_path / "cards.jsonl"
    raw = (json.dumps(card, separators=(",", ":")) + "\n").encode()
    cards_path.write_bytes(raw)
    first = build_balanced_review_queue(cards_path, tmp_path / "first", limit=1)
    manifest1 = json.loads(Path(first["manifest_file"]).read_text())
    assert manifest1["input_cards_sha256"] == hashlib.sha256(raw).hexdigest()

    card["snapshot_id"] = "snapshot-2"
    changed = (json.dumps(card, separators=(",", ":")) + "\n").encode()
    cards_path.write_bytes(changed)
    second = build_balanced_review_queue(cards_path, tmp_path / "second", limit=1)
    manifest2 = json.loads(Path(second["manifest_file"]).read_text())
    assert manifest2["input_cards_sha256"] == hashlib.sha256(changed).hexdigest()
    assert manifest2["input_cards_sha256"] != manifest1["input_cards_sha256"]
    assert manifest2["dependency_identity_sha256"] != manifest1["dependency_identity_sha256"]


def _card(candidate_id: str, selection_order: int, rule_ids: list[str]) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "selection": {
            "matched_rule_ids": rule_ids,
            "selection_order": selection_order,
        },
    }
