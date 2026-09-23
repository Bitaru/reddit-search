"""Inspectable, injectable focal-author reranking primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from reddit_search.ingest.invalidation import TombstoneLedger, tombstone_blocks_identity

from .fusion import FusedCandidate

FOCAL_AUTHOR_RERANK_INSTRUCTION = (
    "Rank messages where the focal author describes their own problem or asks for a tool "
    "matching the scenario. Mere topic mentions, answers to someone else, quotations, and "
    "promotion of the author's own product are not equivalent. Respect explicit negations "
    "and requirements. Judge the focus message with its labeled context."
)
RERANK_TEMPLATE_VERSION = "focal-author-v1"


class RerankerAdapter(Protocol):
    """Installed reranker contract; model loading remains outside this package."""

    model_id: str
    revision: str

    def score(
        self,
        *,
        scenario_query: str,
        candidate_texts: Sequence[str],
        instruction: str,
    ) -> Sequence[float]:
        """Return one relevance score per candidate text."""


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    candidate: FusedCandidate
    score: float
    rank: int


def rerank_cache_key(
    *,
    scenario_id: str,
    scenario_query: str,
    instruction: str,
    candidate: FusedCandidate,
    candidate_text: str,
    model_id: str,
    model_revision: str,
    template_version: str = RERANK_TEMPLATE_VERSION,
) -> str:
    """Hash every input that can change the reranker decision."""
    payload = {
        "scenario_id": scenario_id,
        "scenario_query": scenario_query,
        "instruction": instruction,
        "template_version": template_version,
        "candidate_id": candidate.candidate_id,
        "candidate_text": candidate_text,
        "context": candidate.payload.get("context_text"),
        "message_fullname": candidate.payload.get("message_fullname"),
        "source_revision_id": candidate.payload.get("source_revision_id"),
        "snapshot_id": candidate.payload.get("snapshot_id"),
        "model_id": model_id,
        "model_revision": model_revision,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def rerank_candidates(
    candidates: Sequence[FusedCandidate],
    *,
    scenario_id: str,
    scenario_query: str,
    candidate_texts: Mapping[str, str],
    adapter: RerankerAdapter,
    cache: MutableMapping[str, float] | None = None,
    instruction: str = FOCAL_AUTHOR_RERANK_INSTRUCTION,
    template_version: str = RERANK_TEMPLATE_VERSION,
    limit: int,
    tombstone_ledger: TombstoneLedger | None = None,
) -> list[RerankedCandidate]:
    """Score only the bounded union and preserve deterministic tie ordering."""
    if not scenario_id.strip() or not scenario_query.strip():
        raise ValueError("scenario_id and scenario_query must not be empty")
    if not instruction.strip() or not template_version.strip():
        raise ValueError("reranker instruction and template version must not be empty")
    if limit <= 0:
        raise ValueError("rerank limit must be positive")
    if tombstone_ledger is not None:
        candidates = [
            candidate
            for candidate in candidates
            if not tombstone_blocks_identity(
                tombstone_ledger,
                str(candidate.payload.get("message_fullname", "")),
                candidate.payload.get("source_revision_id"),
                candidate.payload.get("context_message_refs", ()),
            )
        ]
    bounded = list(candidates[:limit])
    missing_text = [
        candidate.candidate_id
        for candidate in bounded
        if candidate.candidate_id not in candidate_texts
    ]
    if missing_text:
        raise ValueError(f"missing reranker text for candidates: {missing_text}")
    score_by_id: dict[str, float] = {}
    uncached: list[FusedCandidate] = []
    for candidate in bounded:
        key = rerank_cache_key(
            scenario_id=scenario_id,
            scenario_query=scenario_query,
            instruction=instruction,
            candidate=candidate,
            candidate_text=candidate_texts[candidate.candidate_id],
            model_id=adapter.model_id,
            model_revision=adapter.revision,
            template_version=template_version,
        )
        if cache is not None and key in cache:
            score_by_id[candidate.candidate_id] = float(cache[key])
        else:
            uncached.append(candidate)
    if uncached:
        scores = adapter.score(
            scenario_query=scenario_query,
            candidate_texts=[candidate_texts[candidate.candidate_id] for candidate in uncached],
            instruction=instruction,
        )
        if len(scores) != len(uncached):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(uncached)} candidates"
            )
        for candidate, score in zip(uncached, scores, strict=True):
            numeric_score = float(score)
            if not isfinite(numeric_score):
                raise ValueError("reranker returned a non-finite score")
            score_by_id[candidate.candidate_id] = numeric_score
            if cache is not None:
                key = rerank_cache_key(
                    scenario_id=scenario_id,
                    scenario_query=scenario_query,
                    instruction=instruction,
                    candidate=candidate,
                    candidate_text=candidate_texts[candidate.candidate_id],
                    model_id=adapter.model_id,
                    model_revision=adapter.revision,
                    template_version=template_version,
                )
                cache[key] = numeric_score
    ordered = sorted(
        bounded,
        key=lambda candidate: (-score_by_id[candidate.candidate_id], candidate.candidate_id),
    )
    return [
        RerankedCandidate(
            candidate=candidate,
            score=score_by_id[candidate.candidate_id],
            rank=rank,
        )
        for rank, candidate in enumerate(ordered, start=1)
    ]
