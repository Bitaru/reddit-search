"""Bounded, scenario-local example-guided retrieval helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .dense import Vector, validate_vector
from .fusion import RankedCandidate

FeedbackLabel = Literal["yes", "no"]


@dataclass(frozen=True, slots=True)
class FeedbackExample:
    """A human-approved development example for one scenario."""

    candidate_id: str
    scenario_id: str
    label: FeedbackLabel
    vector: Vector

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("feedback candidate_id must not be empty")
        if not self.scenario_id.strip():
            raise ValueError("feedback scenario_id must not be empty")
        if self.label not in {"yes", "no"}:
            raise ValueError("feedback labels must be yes or no")


@dataclass(frozen=True, slots=True)
class FeedbackPlan:
    """One bounded average-vector pass and its audit identities."""

    scenario_id: str
    vector: Vector
    positive_ids: tuple[str, ...]
    negative_ids: tuple[str, ...]
    strategy: str = "average_vector"


@dataclass(frozen=True, slots=True)
class FeedbackUnion:
    """Original candidates plus one feedback retrieval pass."""

    candidates: tuple[RankedCandidate, ...]
    new_ids: tuple[str, ...]
    already_seen_ids: tuple[str, ...]
    seed_ids: tuple[str, ...]


def build_average_vector(
    examples: Sequence[FeedbackExample],
    *,
    dimension: int,
    max_positive: int = 5,
    max_negative: int = 5,
) -> FeedbackPlan:
    """Build a positive centroid, retaining negative IDs for audit/exclusion.

    Negative examples are not silently subtracted: their semantic meaning is
    scenario-specific and raw vector subtraction would be an unvalidated model
    change. They remain bounded and available to the caller for exclusion or
    later reranking.
    """
    if max_positive < 1 or max_negative < 0:
        raise ValueError("feedback example limits are invalid")
    if not examples:
        raise ValueError("feedback requires at least one positive example")
    scenario_ids = {example.scenario_id for example in examples}
    if len(scenario_ids) != 1:
        raise ValueError("feedback examples must belong to one scenario")
    positives = [example for example in examples if example.label == "yes"]
    negatives = [example for example in examples if example.label == "no"]
    if not positives:
        raise ValueError("feedback requires at least one positive example")
    if len(positives) > max_positive:
        raise ValueError(f"feedback has more than {max_positive} positive examples")
    if len(negatives) > max_negative:
        raise ValueError(f"feedback has more than {max_negative} negative examples")
    if len({example.candidate_id for example in examples}) != len(examples):
        raise ValueError("feedback examples must have unique candidate IDs")
    vectors = [validate_vector(example.vector, dimension) for example in positives]
    centroid = tuple(
        sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)
    )
    validated_centroid = validate_vector(centroid, dimension)
    return FeedbackPlan(
        scenario_id=positives[0].scenario_id,
        vector=validated_centroid,
        positive_ids=tuple(example.candidate_id for example in positives),
        negative_ids=tuple(example.candidate_id for example in negatives),
    )


def union_feedback_candidates(
    original: Sequence[RankedCandidate],
    feedback: Sequence[RankedCandidate],
    *,
    limit: int,
) -> FeedbackUnion:
    """Union one feedback result set, preserving original candidates first.

    This is the bounded pre-rerank union. The original ranking is never
    discarded; a later configured reranker may reorder this explicit union.
    """
    if limit <= 0:
        raise ValueError("feedback union limit must be positive")
    original_by_id: dict[str, RankedCandidate] = {}
    for candidate in original:
        original_by_id.setdefault(candidate.candidate_id, candidate)
    feedback_by_id: dict[str, RankedCandidate] = {}
    for candidate in feedback:
        feedback_by_id.setdefault(candidate.candidate_id, candidate)

    original_ids = list(original_by_id)
    new_ids = [
        candidate_id for candidate_id in feedback_by_id if candidate_id not in original_by_id
    ]
    ordered_ids = (original_ids + new_ids)[:limit]
    candidates = tuple(
        RankedCandidate(
            candidate_id=candidate_id,
            rank=rank,
            raw_score=(
                original_by_id[candidate_id].raw_score
                if candidate_id in original_by_id
                else feedback_by_id[candidate_id].raw_score
            ),
            payload=(
                original_by_id[candidate_id].payload
                if candidate_id in original_by_id
                else feedback_by_id[candidate_id].payload
            ),
        )
        for rank, candidate_id in enumerate(ordered_ids, start=1)
    )
    return FeedbackUnion(
        candidates=candidates,
        new_ids=tuple(candidate_id for candidate_id in new_ids if candidate_id in ordered_ids),
        already_seen_ids=tuple(
            candidate_id for candidate_id in feedback_by_id if candidate_id in original_by_id
        ),
        seed_ids=tuple(original_ids),
    )
