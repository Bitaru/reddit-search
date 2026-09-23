from pathlib import Path

import pytest

from reddit_search.retrieval.comparison import (
    ComparisonConfig,
    ComparisonVariant,
    load_comparison_config,
)

ROOT = Path(__file__).parents[2]


def test_checked_in_comparison_config_keeps_shared_budget_and_order() -> None:
    config = load_comparison_config(ROOT / "configs" / "retrieval_comparison.yaml")

    assert [variant.variant_id for variant in config.variants] == ["A", "B", "C"]
    assert config.output_limit == config.review_budget == 20
    assert config.variants[0].dense is False
    assert config.variants[1].dense is True
    assert config.variants[2].feedback is True
    assert len(config.configuration_hash()) == 64


def test_comparison_variant_rejects_hidden_pipeline_changes() -> None:
    with pytest.raises(ValueError, match="flags"):
        ComparisonVariant(
            variant_id="A",
            lexical=True,
            dense=True,
            reranker=False,
            feedback=False,
        )


def test_comparison_config_rejects_output_budget_mismatch() -> None:
    with pytest.raises(ValueError, match="output_limit"):
        ComparisonConfig(
            snapshot_id="snapshot",
            split="test",
            output_limit=21,
            review_budget=20,
            lexical_candidates=100,
            dense_candidates=100,
            rerank_candidates=100,
            feedback_candidates=100,
            rrf_k=60,
            variants=(
                ComparisonVariant("A", True, False, False, False),
                ComparisonVariant("B", True, True, True, False),
                ComparisonVariant("C", True, True, True, True),
            ),
        )
