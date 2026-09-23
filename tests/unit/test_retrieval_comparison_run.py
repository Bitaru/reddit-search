from collections.abc import Mapping, Sequence

import pytest

from reddit_search.retrieval.comparison import ComparisonConfig, ComparisonVariant
from reddit_search.retrieval.comparison_run import (
    ComparisonPrerequisiteError,
    ComparisonScenario,
    run_comparison,
    run_heldout_comparison,
)
from reddit_search.retrieval.dense import DenseHit, EmbeddingRecipe
from reddit_search.retrieval.feedback import FeedbackExample
from reddit_search.retrieval.fusion import RankedCandidate


class Adapter:
    recipe = EmbeddingRecipe("qwen", "rev-1", 2, "Represent the query.")

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [(1.0, 0.0) for _ in texts]

    def encode_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [(1.0, 0.0) for _ in texts]


class Dense:
    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        payload_filter: Mapping[str, str] | None = None,
    ) -> Sequence[DenseHit]:
        return [DenseHit("dense", 1, 0.8, {"context_text": "dense text"})]


class Reranker:
    model_id = "reranker"
    revision = "rev-1"

    def score(
        self,
        *,
        scenario_query: str,
        candidate_texts: Sequence[str],
        instruction: str,
    ) -> Sequence[float]:
        return [float(len(text)) for text in candidate_texts]


def comparison_config() -> ComparisonConfig:
    return ComparisonConfig(
        snapshot_id="snapshot",
        split="dev",
        output_limit=2,
        review_budget=2,
        lexical_candidates=5,
        dense_candidates=5,
        rerank_candidates=5,
        feedback_candidates=5,
        rrf_k=60,
        variants=(
            ComparisonVariant("A", True, False, False, False),
            ComparisonVariant("B", True, True, True, False),
            ComparisonVariant("C", True, True, True, True),
        ),
    )


def lexical_search(query: str, limit: int) -> Sequence[RankedCandidate]:
    return [RankedCandidate("lexical", 1, -1.0)]


def test_dev_comparison_runs_all_variants_and_reports_feedback_cost() -> None:
    scenario = ComparisonScenario(
        scenario_id="scenario",
        lexical_queries=("need tool",),
        semantic_queries=("need a tool",),
        candidate_texts={
            "lexical": "lexical text",
            "dense": "long dense text",
            "new": "feedback text",
        },
        feedback_examples=(FeedbackExample("dense", "scenario", "yes", (1.0, 0.0)),),
    )
    run = run_comparison(
        comparison_config(),
        [scenario],
        lexical_search=lexical_search,
        dense_adapter=Adapter(),
        dense_backend=Dense(),
        reranker=Reranker(),
        feedback_search=lambda vector, limit, payload_filter: [RankedCandidate("new", 1, 0.5)],
    )

    assert [result.variant_id for result in run.variants] == ["A", "B", "C"]
    assert run.variants[0].candidate_ids == ("lexical",)
    assert run.variants[2].new_feedback_ids == ("new",)
    assert run.variants[2].processing_cost["feedback_queries"] == 1
    assert run.review_budget == 2


def test_comparison_refuses_test_split_and_missing_dense_backends() -> None:
    scenario = ComparisonScenario(
        scenario_id="scenario",
        lexical_queries=("need tool",),
        semantic_queries=("need a tool",),
        candidate_texts={"lexical": "text"},
        feedback_examples=(FeedbackExample("lexical", "scenario", "yes", (1.0, 0.0)),),
    )
    test_config = ComparisonConfig(
        snapshot_id="snapshot",
        split="test",
        output_limit=2,
        review_budget=2,
        lexical_candidates=5,
        dense_candidates=5,
        rerank_candidates=5,
        feedback_candidates=5,
        rrf_k=60,
        variants=comparison_config().variants,
    )
    with pytest.raises(ValueError, match="development split"):
        run_comparison(test_config, [scenario], lexical_search=lexical_search)
    with pytest.raises(ComparisonPrerequisiteError, match="embedding adapter"):
        run_comparison(comparison_config(), [scenario], lexical_search=lexical_search)


def test_heldout_comparison_requires_explicit_test_wrapper() -> None:
    scenario = ComparisonScenario(
        scenario_id="scenario",
        lexical_queries=("need tool",),
        semantic_queries=("need a tool",),
        candidate_texts={
            "lexical": "lexical text",
            "dense": "long dense text",
            "new": "feedback text",
        },
        feedback_examples=(FeedbackExample("dense", "scenario", "yes", (1.0, 0.0)),),
    )
    config = ComparisonConfig(
        snapshot_id="snapshot",
        split="test",
        output_limit=2,
        review_budget=2,
        lexical_candidates=5,
        dense_candidates=5,
        rerank_candidates=5,
        feedback_candidates=5,
        rrf_k=60,
        variants=comparison_config().variants,
    )

    run = run_heldout_comparison(
        config,
        [scenario],
        lexical_search=lexical_search,
        dense_adapter=Adapter(),
        dense_backend=Dense(),
        reranker=Reranker(),
        feedback_search=lambda vector, limit, payload_filter: [
            RankedCandidate("new", 1, 0.5)
        ],
    )

    assert run.split == "test"
