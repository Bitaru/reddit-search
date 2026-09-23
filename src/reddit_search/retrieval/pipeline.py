"""Candidate generation orchestration for lexical, dense, and fused branches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from reddit_search.ingest.invalidation import TombstoneLedger, tombstone_blocks_identity

from .dense import DenseBackendError, DenseHit, EmbeddingAdapter, validate_vectors
from .feedback import FeedbackPlan, FeedbackUnion, union_feedback_candidates
from .fusion import FusedCandidate, RankedCandidate, merge_ranked_variants, reciprocal_rank_fusion


class DenseSearchBackend(Protocol):
    """Minimal dense search surface used by the pipeline and local Qdrant client."""

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        payload_filter: Mapping[str, str] | None = None,
    ) -> Sequence[DenseHit]:
        """Search the already-indexed snapshot."""


LexicalSearch = Callable[[str, int], Sequence[RankedCandidate]]
FeedbackSearch = Callable[
    [Sequence[float], int, Mapping[str, str] | None], Sequence[RankedCandidate]
]


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    scenario_id: str
    lexical: tuple[RankedCandidate, ...]
    dense: tuple[RankedCandidate, ...]
    fused: tuple[FusedCandidate, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "lexical_candidate_ids": [candidate.candidate_id for candidate in self.lexical],
            "dense_candidate_ids": [candidate.candidate_id for candidate in self.dense],
            "fused_candidate_ids": [candidate.candidate_id for candidate in self.fused],
            "fused_branch_ranks": {
                candidate.candidate_id: dict(candidate.branch_ranks) for candidate in self.fused
            },
        }

def generate_candidates(
    *,
    scenario_id: str,
    lexical_queries: Sequence[str],
    lexical_search: LexicalSearch,
    semantic_queries: Sequence[str] = (),
    dense_adapter: EmbeddingAdapter | None = None,
    dense_backend: DenseSearchBackend | None = None,
    payload_filter: Mapping[str, str] | None = None,
    tombstone_ledger: TombstoneLedger | None = None,
    lexical_limit: int = 100,
    dense_limit: int = 100,
    rrf_k: int = 60,
    output_limit: int = 100,
) -> CandidateGenerationResult:
    """Generate inspectable lexical/dense branches and fuse by rank.

    Dense mode is opt-in and requires both the installed adapter and a working
    backend. Missing prerequisites fail explicitly rather than producing a
    lexical-only result labeled as hybrid.
    """
    if not scenario_id.strip():
        raise ValueError("scenario_id must not be empty")
    if not lexical_queries or any(not query.strip() for query in lexical_queries):
        raise ValueError("lexical_queries must contain non-empty queries")
    if lexical_limit <= 0 or dense_limit <= 0 or output_limit <= 0:
        raise ValueError("candidate and output limits must be positive")
    def active(candidates: Sequence[RankedCandidate]) -> list[RankedCandidate]:
        if tombstone_ledger is None:
            return list(candidates)
        return [
            candidate
            for candidate in candidates
            if not tombstone_blocks_identity(
                tombstone_ledger,
                str(candidate.payload.get("message_fullname", "")),
                candidate.payload.get("source_revision_id"),
                candidate.payload.get("context_message_refs", ()),
            )
        ]

    lexical_variants = [active(lexical_search(query, lexical_limit)) for query in lexical_queries]
    lexical_branch = merge_ranked_variants(lexical_variants, limit=lexical_limit)

    dense_branch: list[RankedCandidate] = []
    dense_requested = bool(semantic_queries)
    if dense_requested:
        if dense_adapter is None or dense_backend is None:
            raise DenseBackendError(
                "dense branch requested but embedding adapter and backend are both required"
            )
        if any(not query.strip() for query in semantic_queries):
            raise ValueError("semantic_queries must contain non-empty queries")
        vectors = validate_vectors(
            dense_adapter.encode_queries(semantic_queries),
            dimension=dense_adapter.recipe.dimension,
            expected_count=len(semantic_queries),
        )
        dense_variants: list[list[RankedCandidate]] = []
        for vector in vectors:
            hits = dense_backend.search(vector, limit=dense_limit, payload_filter=payload_filter)
            dense_variants.append(active(_dense_hits_to_candidates(hits)))
        dense_branch = merge_ranked_variants(dense_variants, limit=dense_limit)

    branches: dict[str, Sequence[RankedCandidate]] = {"lexical": lexical_branch}
    if dense_requested:
        branches["dense"] = dense_branch
    fused = [
        candidate
        for candidate in reciprocal_rank_fusion(branches, k=rrf_k, limit=output_limit)
        if tombstone_ledger is None
        or not tombstone_blocks_identity(
            tombstone_ledger,
            str(candidate.payload.get("message_fullname", "")),
            candidate.payload.get("source_revision_id"),
            candidate.payload.get("context_message_refs", ()),
        )
    ]
    return CandidateGenerationResult(
        scenario_id=scenario_id,
        lexical=tuple(lexical_branch),
        dense=tuple(dense_branch),
        fused=tuple(fused),
    )

def run_feedback_pass(
    *,
    plan: FeedbackPlan,
    original: Sequence[RankedCandidate],
    search: FeedbackSearch,
    limit: int,
    payload_filter: Mapping[str, str] | None = None,
    tombstone_ledger: TombstoneLedger | None = None,
) -> FeedbackUnion:
    """Run exactly one full-index feedback query and audit the union."""
    if limit <= 0:
        raise ValueError("feedback limit must be positive")
    feedback = search(plan.vector, limit, payload_filter)
    if tombstone_ledger is not None:
        def active(candidate: RankedCandidate) -> bool:
            return not tombstone_blocks_identity(
                tombstone_ledger,
                str(candidate.payload.get("message_fullname", "")),
                candidate.payload.get("source_revision_id"),
                candidate.payload.get("context_message_refs", ()),
            )
        original = [candidate for candidate in original if active(candidate)]
        feedback = [candidate for candidate in feedback if active(candidate)]
    return union_feedback_candidates(original, feedback, limit=limit)


def _dense_hits_to_candidates(hits: Sequence[DenseHit]) -> list[RankedCandidate]:
    return [
        RankedCandidate(
            candidate_id=hit.candidate_id,
            rank=hit.rank,
            raw_score=hit.score,
            payload=hit.payload,
        )
        for hit in hits
    ]
