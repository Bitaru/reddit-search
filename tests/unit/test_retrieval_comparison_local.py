from types import SimpleNamespace

from pytest import raises

from reddit_search.ingest.invalidation import load_tombstone_ledger
from reddit_search.retrieval.comparison_local import (
    VERIFIED_REGISTRY_COLLECTION,
    _read_positive_seeds,
    _resolve_dense_collection,
)


def test_dense_collection_override_uses_verified_registry_collection() -> None:
    assert _resolve_dense_collection(VERIFIED_REGISTRY_COLLECTION) == VERIFIED_REGISTRY_COLLECTION


def test_dense_collection_override_defaults_to_derived_behavior() -> None:
    assert _resolve_dense_collection(None) is None


def test_dense_collection_override_rejects_unverified_collection() -> None:
    with raises(ValueError, match="not the verified registry collection"):
        _resolve_dense_collection("reddit_dense_49bd031cf94bed88a581b2b5")




def test_positive_seed_can_use_exact_external_development_card(tmp_path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"candidate_id":"seed-1","scenario_id":"scenario","topic_fit":"yes"}\n',
        encoding="utf-8",
    )
    cards = tmp_path / "cards.jsonl"
    cards.write_text(
        '{"candidate_id":"seed-1","scenario_id":"scenario",'
        '"source":{"context_text":"preserved development evidence"}}\n',
        encoding="utf-8",
    )

    seeds, tombstoned, unverifiable = _read_positive_seeds(
        labels,
        {"corpus-1": SimpleNamespace(context_text="corpus text")},
        cards,
    )

    assert seeds == {"scenario": [("seed-1", "preserved development evidence")]}
    assert tombstoned == 0


def test_tombstoned_seed_card_is_skipped_with_counted_reason(tmp_path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"candidate_id":"seed-1","scenario_id":"scenario","topic_fit":"yes"}\n',
        encoding="utf-8",
    )
    cards = tmp_path / "cards.jsonl"
    cards.write_text(
        '{"candidate_id":"seed-1","scenario_id":"scenario",'
        '"source":{"message_fullname":"t1_seed","source_revision_id":"rev-1",'
        '"context_text":"tombstoned development evidence"}}\n',
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        '{"message_fullname":"t1_seed","source_revision_id":"rev-1",'
        '"reason":"operator deletion"}\n',
        encoding="utf-8",
    )
    ledger = load_tombstone_ledger(ledger_path)

    seeds, tombstoned, unverifiable = _read_positive_seeds(
        labels,
        {},
        cards,
        tombstone_ledger=ledger,
    )

    assert seeds == {}
    assert tombstoned == 1


def test_seed_card_without_structured_context_identity_is_skipped_with_ledger(tmp_path) -> None:
    """Free-text context cannot be verified against an identity-only ledger: fail closed."""
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"candidate_id":"seed-1","scenario_id":"scenario","topic_fit":"yes"}\n',
        encoding="utf-8",
    )
    cards = tmp_path / "cards.jsonl"
    cards.write_text(
        '{"candidate_id":"seed-1","scenario_id":"scenario",'
        '"source":{"message_fullname":"t1_seed","source_revision_id":"rev-1",'
        '"context_text":"[COMMENT t1_gone]\\ntombstoned contributor text\\n\\n'
        '[FOCUS COMMENT t1_seed]\\nseed text"}}\n',
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        '{"message_fullname":"t1_gone","source_revision_id":null,"reason":"user deletion"}\n',
        encoding="utf-8",
    )
    ledger = load_tombstone_ledger(ledger_path)

    seeds, tombstoned, unverifiable = _read_positive_seeds(
        labels, {}, cards, tombstone_ledger=ledger
    )
    assert seeds == {}
    assert tombstoned == 0
    assert unverifiable == 1

    # Without a ledger, behavior is unchanged: the card is embedded.
    seeds, tombstoned, unverifiable = _read_positive_seeds(labels, {}, cards)
    assert seeds == {
        "scenario": [(
            "seed-1",
            "[COMMENT t1_gone]\ntombstoned contributor text\n\n"
            "[FOCUS COMMENT t1_seed]\nseed text",
        )]
    }
    assert tombstoned == 0
    assert unverifiable == 0


def test_seed_card_with_structured_context_identity_is_kept_with_ledger(tmp_path) -> None:
    """Cards carrying structured context identity remain verifiable and embeddable."""
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"candidate_id":"seed-1","scenario_id":"scenario","topic_fit":"yes"}\n',
        encoding="utf-8",
    )
    base = (
        '{"candidate_id":"seed-1","scenario_id":"scenario",'
        '"source":{"message_fullname":"t1_seed","source_revision_id":"rev-1",'
    )
    cards = tmp_path / "cards.jsonl"
    cards.write_text(
        base + '"context_text":"seed text","context_message_refs":["t3_thread"]}}\n',
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        '{"message_fullname":"t1_gone","source_revision_id":null,"reason":"user deletion"}\n',
        encoding="utf-8",
    )
    ledger = load_tombstone_ledger(ledger_path)

    seeds, tombstoned, unverifiable = _read_positive_seeds(
        labels, {}, cards, tombstone_ledger=ledger
    )
    assert seeds == {"scenario": [("seed-1", "seed text")]}
    assert tombstoned == 0
    assert unverifiable == 0


def test_seed_card_with_unrelated_tombstoned_ref_is_counted_as_tombstoned(tmp_path) -> None:
    """Structured refs still route to the tombstoned counter when they match the ledger."""
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"candidate_id":"seed-1","scenario_id":"scenario","topic_fit":"yes"}\n',
        encoding="utf-8",
    )
    cards = tmp_path / "cards.jsonl"
    cards.write_text(
        '{"candidate_id":"seed-1","scenario_id":"scenario",'
        '"source":{"message_fullname":"t1_seed","source_revision_id":"rev-1",'
        '"context_text":"seed text","context":{"message_fullnames":["t1_gone"]}}}\n',
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        '{"message_fullname":"t1_gone","source_revision_id":null,"reason":"user deletion"}\n',
        encoding="utf-8",
    )
    ledger = load_tombstone_ledger(ledger_path)

    seeds, tombstoned, unverifiable = _read_positive_seeds(
        labels, {}, cards, tombstone_ledger=ledger
    )
    assert seeds == {}
    assert tombstoned == 1
    assert unverifiable == 0
