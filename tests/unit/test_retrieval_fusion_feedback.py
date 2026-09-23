import pytest

from reddit_search.retrieval.feedback import (
    FeedbackExample,
    build_average_vector,
    union_feedback_candidates,
)
from reddit_search.retrieval.fusion import (
    RankedCandidate,
    merge_ranked_variants,
    reciprocal_rank_fusion,
)


def candidate(candidate_id: str, rank: int, score: float) -> RankedCandidate:
    return RankedCandidate(candidate_id=candidate_id, rank=rank, raw_score=score)


def test_variant_merge_counts_each_unit_once_using_best_rank() -> None:
    merged = merge_ranked_variants(
        [
            [candidate("a", 1, 100), candidate("b", 2, 1)],
            [candidate("b", 1, 999), candidate("c", 2, 2)],
        ],
        limit=3,
    )

    assert [(item.candidate_id, item.rank) for item in merged] == [
        ("a", 1),
        ("b", 2),
        ("c", 3),
    ]
    assert merged[1].raw_score == 999


def test_rrf_uses_ranks_not_incomparable_raw_scores_and_is_deterministic() -> None:
    fused = reciprocal_rank_fusion(
        {
            "lexical": [candidate("a", 1, 0.01), candidate("b", 2, 1000)],
            "dense": [candidate("b", 1, 0.02), candidate("a", 2, 2000)],
        },
        k=60,
        limit=2,
    )

    assert [item.candidate_id for item in fused] == ["a", "b"]
    assert fused[0].branch_ranks == {"dense": 2, "lexical": 1}
    assert fused[0].branch_scores == {"dense": 2000.0, "lexical": 0.01}
    assert fused[0].rrf_score == fused[1].rrf_score
    assert [item.rank for item in fused] == [1, 2]


def test_feedback_requires_positive_same_scenario_and_preserves_union_audit() -> None:
    examples = [
        FeedbackExample("yes-1", "scenario", "yes", (1.0, 0.0)),
        FeedbackExample("no-1", "scenario", "no", (0.0, 1.0)),
    ]
    plan = build_average_vector(examples, dimension=2)
    assert plan.vector == (1.0, 0.0)
    assert plan.positive_ids == ("yes-1",)
    assert plan.negative_ids == ("no-1",)

    union = union_feedback_candidates(
        [candidate("seed", 1, 10.0), candidate("seen", 2, 9.0)],
        [candidate("seen", 1, 100.0), candidate("new", 2, 8.0)],
        limit=3,
    )
    assert [item.candidate_id for item in union.candidates] == ["seed", "seen", "new"]
    assert union.new_ids == ("new",)
    assert union.already_seen_ids == ("seen",)
    assert union.seed_ids == ("seed", "seen")


def test_feedback_rejects_unknown_or_cross_scenario_examples() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        build_average_vector(
            [FeedbackExample("no", "scenario", "no", (1.0, 0.0))],
            dimension=2,
        )
    with pytest.raises(ValueError, match="one scenario"):
        build_average_vector(
            [
                FeedbackExample("a", "one", "yes", (1.0, 0.0)),
                FeedbackExample("b", "two", "yes", (0.0, 1.0)),
            ],
            dimension=2,
        )
