"""Deterministic lexical/dense rank fusion without raw-score arithmetic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One branch result; score semantics remain branch-local."""

    candidate_id: str
    rank: int
    raw_score: float
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if self.rank < 1:
            raise ValueError("rank must be one-based")
        if not isfinite(self.raw_score):
            raise ValueError("raw_score must be finite")


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """A fused result with inspectable branch ranks and raw scores."""

    candidate_id: str
    rank: int
    rrf_score: float
    branch_ranks: Mapping[str, int]
    branch_scores: Mapping[str, float]
    payload: Mapping[str, Any] = field(default_factory=dict)


def merge_ranked_variants(
    variant_rankings: Sequence[Sequence[RankedCandidate]], *, limit: int
) -> list[RankedCandidate]:
    """Collapse query variants into one branch ranking before fusion.

    A unit contributes once to a branch. Its best one-based rank wins, with
    variant order and candidate ID as deterministic tie-breakers.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    best: dict[str, tuple[tuple[int, int, str], RankedCandidate]] = {}
    for variant_index, ranking in enumerate(variant_rankings):
        for candidate in ranking:
            key = (candidate.rank, variant_index, candidate.candidate_id)
            existing = best.get(candidate.candidate_id)
            if existing is None or key < existing[0]:
                best[candidate.candidate_id] = (key, candidate)
    ordered = sorted(best.values(), key=lambda item: item[0])[:limit]
    return [
        RankedCandidate(
            candidate_id=candidate.candidate_id,
            rank=rank,
            raw_score=candidate.raw_score,
            payload=candidate.payload,
        )
        for rank, (_, candidate) in enumerate(ordered, start=1)
    ]


def reciprocal_rank_fusion(
    branch_rankings: Mapping[str, Sequence[RankedCandidate]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    limit: int = 100,
) -> list[FusedCandidate]:
    """Fuse branch ranks using ``weight / (k + one_based_rank)``.

    Raw BM25, cosine, and reranker scores are never combined. Duplicate IDs
    within a branch count only at their best rank, so query expansion cannot
    create extra votes for the same unit.
    """
    if k <= 0:
        raise ValueError("rrf k must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not branch_rankings:
        return []
    branch_weights = {name: 1.0 for name in branch_rankings}
    if weights is not None:
        unknown = set(weights) - set(branch_rankings)
        if unknown:
            raise ValueError(f"weights contain unknown branches: {sorted(unknown)}")
        for name, weight in weights.items():
            if weight < 0 or not isfinite(weight):
                raise ValueError(f"weight for {name} must be finite and non-negative")
            branch_weights[name] = weight
    if not any(weight > 0 for weight in branch_weights.values()):
        raise ValueError("at least one branch weight must be positive")

    candidates: dict[str, dict[str, Any]] = {}
    for branch_name, ranking in sorted(branch_rankings.items()):
        for candidate in ranking:
            current = candidates.setdefault(
                candidate.candidate_id,
                {
                    "rrf_score": 0.0,
                    "branch_ranks": {},
                    "branch_scores": {},
                    "payload": candidate.payload,
                },
            )
            previous_rank = current["branch_ranks"].get(branch_name)
            if previous_rank is not None and previous_rank <= candidate.rank:
                continue
            if previous_rank is not None:
                current["rrf_score"] -= branch_weights[branch_name] / (k + previous_rank)
            current["branch_ranks"][branch_name] = candidate.rank
            current["branch_scores"][branch_name] = candidate.raw_score
            current["rrf_score"] += branch_weights[branch_name] / (k + candidate.rank)

    ordered = sorted(
        candidates.items(),
        key=lambda item: (-item[1]["rrf_score"], item[0]),
    )[:limit]
    return [
        FusedCandidate(
            candidate_id=candidate_id,
            rank=rank,
            rrf_score=float(values["rrf_score"]),
            branch_ranks=dict(sorted(values["branch_ranks"].items())),
            branch_scores=dict(sorted(values["branch_scores"].items())),
            payload=values["payload"],
        )
        for rank, (candidate_id, values) in enumerate(ordered, start=1)
    ]
