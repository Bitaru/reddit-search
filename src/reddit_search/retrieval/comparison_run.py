"""Development-only A/B/C comparison execution over injected local backends."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reddit_search.ingest.invalidation import TombstoneLedger

from .comparison import ComparisonConfig
from .dense import EmbeddingAdapter
from .feedback import FeedbackExample, build_average_vector
from .fusion import FusedCandidate, RankedCandidate
from .pipeline import (
    DenseSearchBackend,
    FeedbackSearch,
    LexicalSearch,
    generate_candidates,
    run_feedback_pass,
)
from .rerank import RerankedCandidate, RerankerAdapter, rerank_candidates


class ComparisonPrerequisiteError(RuntimeError):
    """A configured comparison branch lacks an explicit local backend."""


@dataclass(frozen=True, slots=True)
class ComparisonScenario:
    scenario_id: str
    lexical_queries: tuple[str, ...]
    semantic_queries: tuple[str, ...]
    candidate_texts: Mapping[str, str]
    feedback_examples: tuple[FeedbackExample, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("comparison scenario_id must not be empty")
        if not self.lexical_queries or any(not query.strip() for query in self.lexical_queries):
            raise ValueError("comparison scenarios require non-empty lexical queries")
        if not self.semantic_queries or any(not query.strip() for query in self.semantic_queries):
            raise ValueError("comparison scenarios require non-empty semantic queries")
        if any(example.scenario_id != self.scenario_id for example in self.feedback_examples):
            raise ValueError("feedback examples must match the comparison scenario")


@dataclass(frozen=True, slots=True)
class ComparisonVariantResult:
    variant_id: str
    scenario_id: str
    candidate_ids: tuple[str, ...]
    new_feedback_ids: tuple[str, ...] = ()
    already_seen_ids: tuple[str, ...] = ()
    processing_cost: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "scenario_id": self.scenario_id,
            "candidate_ids": list(self.candidate_ids),
            "new_feedback_ids": list(self.new_feedback_ids),
            "already_seen_ids": list(self.already_seen_ids),
            "processing_cost": dict(sorted(self.processing_cost.items())),
        }


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    config_hash: str
    snapshot_id: str
    split: str
    review_budget: int
    variants: tuple[ComparisonVariantResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "retrieval_comparison_run",
            "schema_version": 1,
            "config_hash": self.config_hash,
            "snapshot_id": self.snapshot_id,
            "split": self.split,
            "review_budget": self.review_budget,
            "variants": [variant.as_dict() for variant in self.variants],
        }

    def write(self, path: Path) -> None:
        """Write one atomic, inspectable comparison report."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
def run_comparison(
    config: ComparisonConfig,
    scenarios: Sequence[ComparisonScenario],
    *,
    lexical_search: LexicalSearch,
    dense_adapter: EmbeddingAdapter | None = None,
    dense_backend: DenseSearchBackend | None = None,
    reranker: RerankerAdapter | None = None,
    feedback_search: FeedbackSearch | None = None,
    rerank_cache: MutableMapping[str, float] | None = None,
    tombstone_ledger: TombstoneLedger | None = None,
    allow_test: bool = False,
) -> ComparisonRun:
    """Run one local A/B/C comparison with an explicit split guard.

    Development runs are the default. Held-out runs must call
    ``run_heldout_comparison`` so the test-split escape hatch is explicit at
    the call site.
    """
    if config.split != "dev" and not allow_test:
        raise ValueError(
            "A/B/C comparison must run on the development split unless "
            "run_heldout_comparison explicitly enables the test split"
        )
    if config.split not in {"dev", "test"}:
        raise ValueError(f"unsupported comparison split: {config.split}")
    if not scenarios:
        raise ValueError("comparison requires at least one scenario")
    if dense_adapter is None or dense_backend is None or reranker is None:
        raise ComparisonPrerequisiteError(
            "B/C require an embedding adapter, dense backend, and reranker"
        )

    variant_results: list[ComparisonVariantResult] = []
    for scenario in scenarios:
        lexical_only = generate_candidates(
            scenario_id=scenario.scenario_id,
            lexical_queries=scenario.lexical_queries,
            lexical_search=lexical_search,
            lexical_limit=config.lexical_candidates,
            output_limit=config.output_limit,
            tombstone_ledger=tombstone_ledger,
            rrf_k=config.rrf_k,
        )
        variant_results.append(
            ComparisonVariantResult(
                variant_id="A",
                scenario_id=scenario.scenario_id,
                candidate_ids=tuple(candidate.candidate_id for candidate in lexical_only.fused),
                processing_cost={"lexical_queries": len(scenario.lexical_queries)},
            )
        )

        hybrid = generate_candidates(
            scenario_id=scenario.scenario_id,
            lexical_queries=scenario.lexical_queries,
            lexical_search=lexical_search,
            semantic_queries=scenario.semantic_queries,
            dense_adapter=dense_adapter,
            dense_backend=dense_backend,
            lexical_limit=config.lexical_candidates,
            dense_limit=config.dense_candidates,
            output_limit=config.rerank_candidates,
            tombstone_ledger=tombstone_ledger,
            rrf_k=config.rrf_k,
        )
        reranked = _rerank(
            hybrid.fused,
            scenario=scenario,
            reranker=reranker,
            cache=rerank_cache,
            limit=config.output_limit,
            tombstone_ledger=tombstone_ledger,
        )
        variant_results.append(
            ComparisonVariantResult(
                variant_id="B",
                scenario_id=scenario.scenario_id,
                candidate_ids=tuple(item.candidate.candidate_id for item in reranked),
                processing_cost={
                    "lexical_queries": len(scenario.lexical_queries),
                    "dense_queries": len(scenario.semantic_queries),
                    "reranker_candidates": len(hybrid.fused),
                },
            )
        )

        if feedback_search is None:
            raise ComparisonPrerequisiteError("C requires a full-index feedback search function")
        plan = build_average_vector(
            scenario.feedback_examples,
            dimension=dense_adapter.recipe.dimension,
            max_positive=5,
            max_negative=5,
        )
        feedback_union = run_feedback_pass(
            plan=plan,
            original=[
                RankedCandidate(
                    candidate_id=candidate.candidate_id,
                    rank=candidate.rank,
                    raw_score=candidate.rrf_score,
                    payload=candidate.payload,
                )
                for candidate in hybrid.fused
            ],
            search=feedback_search,
            limit=config.feedback_candidates,
            tombstone_ledger=tombstone_ledger,
        )
        c_candidates = _feedback_fused_candidates(feedback_union.candidates, hybrid.fused)
        c_reranked = _rerank(
            c_candidates,
            scenario=scenario,
            reranker=reranker,
            cache=rerank_cache,
            limit=config.output_limit,
            tombstone_ledger=tombstone_ledger,
        )
        variant_results.append(
            ComparisonVariantResult(
                variant_id="C",
                scenario_id=scenario.scenario_id,
                candidate_ids=tuple(item.candidate.candidate_id for item in c_reranked),
                new_feedback_ids=feedback_union.new_ids,
                already_seen_ids=feedback_union.already_seen_ids,
                processing_cost={
                    "lexical_queries": len(scenario.lexical_queries),
                    "dense_queries": len(scenario.semantic_queries),
                    "reranker_candidates": len(c_candidates),
                    "feedback_queries": 1,
                },
            )
        )

    return ComparisonRun(
        config_hash=config.configuration_hash(),
        snapshot_id=config.snapshot_id,
        split=config.split,
        review_budget=config.review_budget,
        variants=tuple(variant_results),
    )


def run_heldout_comparison(
    config: ComparisonConfig,
    scenarios: Sequence[ComparisonScenario],
    *,
    lexical_search: LexicalSearch,
    dense_adapter: EmbeddingAdapter | None = None,
    dense_backend: DenseSearchBackend | None = None,
    reranker: RerankerAdapter | None = None,
    feedback_search: FeedbackSearch | None = None,
    rerank_cache: MutableMapping[str, float] | None = None,
    tombstone_ledger: TombstoneLedger | None = None,
) -> ComparisonRun:
    """Run A/B/C against a test corpus using development-only seeds."""
    if config.split != "test":
        raise ValueError("held-out comparison requires the test split")
    return run_comparison(
        config,
        scenarios,
        lexical_search=lexical_search,
        dense_adapter=dense_adapter,
        dense_backend=dense_backend,
        reranker=reranker,
        feedback_search=feedback_search,
        rerank_cache=rerank_cache,
        tombstone_ledger=tombstone_ledger,
        allow_test=True,
    )
def _rerank(
    candidates: Sequence[FusedCandidate],
    *,
    scenario: ComparisonScenario,
    reranker: RerankerAdapter,
    cache: MutableMapping[str, float] | None,
    limit: int,
    tombstone_ledger: TombstoneLedger | None = None,
) -> list[RerankedCandidate]:
    return rerank_candidates(
        candidates,
        scenario_id=scenario.scenario_id,
        scenario_query=scenario.semantic_queries[0],
        candidate_texts=scenario.candidate_texts,
        adapter=reranker,
        cache=cache,
        limit=limit,
        tombstone_ledger=tombstone_ledger,
    )


def _feedback_fused_candidates(
    candidates: Sequence[RankedCandidate],
    original: Sequence[FusedCandidate],
) -> list[FusedCandidate]:
    original_by_id = {candidate.candidate_id: candidate for candidate in original}
    return [
        original_by_id.get(
            candidate.candidate_id,
            FusedCandidate(
                candidate_id=candidate.candidate_id,
                rank=candidate.rank,
                rrf_score=0.0,
                branch_ranks={"feedback": candidate.rank},
                branch_scores={"feedback": candidate.raw_score},
                payload=candidate.payload,
            ),
        )
        for candidate in candidates
    ]
