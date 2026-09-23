from collections.abc import Mapping, Sequence

import pytest

from reddit_search.retrieval.dense import DenseBackendError, DenseHit, EmbeddingRecipe
from reddit_search.retrieval.feedback import FeedbackExample, build_average_vector
from reddit_search.retrieval.fusion import FusedCandidate, RankedCandidate
from reddit_search.retrieval.pipeline import generate_candidates, run_feedback_pass
from reddit_search.retrieval.rerank import rerank_candidates


class QueryAdapter:
    recipe = EmbeddingRecipe(
        model_id="qwen",
        revision="rev-1",
        dimension=2,
        query_instruction="Represent the query.",
    )

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [(1.0, 0.0) for _ in texts]

    def encode_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [(1.0, float(index)) for index, _ in enumerate(texts)]


class DenseBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, ...], int, Mapping[str, str] | None]] = []

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        payload_filter: Mapping[str, str] | None = None,
    ) -> Sequence[DenseHit]:
        self.calls.append((tuple(vector), limit, payload_filter))
        return [
            DenseHit("dense-a", 1, 0.2, {"context_text": "a"}),
            DenseHit("shared", 2, 0.1, {"context_text": "shared"}),
        ]


def lexical_search(query: str, limit: int) -> Sequence[RankedCandidate]:
    if query == "first":
        return [RankedCandidate("shared", 1, -4.0), RankedCandidate("lex-a", 2, -3.0)]
    return [RankedCandidate("shared", 1, -10.0), RankedCandidate("lex-b", 2, -2.0)]


def test_generate_candidates_exposes_branches_and_passes_identical_filter() -> None:
    backend = DenseBackend()
    result = generate_candidates(
        scenario_id="scenario",
        lexical_queries=("first", "second"),
        lexical_search=lexical_search,
        semantic_queries=("semantic one", "semantic two"),
        dense_adapter=QueryAdapter(),
        dense_backend=backend,
        payload_filter={"snapshot_id": "snap"},
        lexical_limit=5,
        dense_limit=5,
        output_limit=5,
    )

    assert {candidate.candidate_id for candidate in result.lexical} == {"shared", "lex-a", "lex-b"}
    assert {candidate.candidate_id for candidate in result.dense} == {"dense-a", "shared"}
    assert result.fused[0].branch_ranks["lexical"] == 1
    assert all(call[2] == {"snapshot_id": "snap"} for call in backend.calls)
    assert result.manifest()["scenario_id"] == "scenario"


def test_generate_candidates_refuses_unavailable_dense_branch() -> None:
    with pytest.raises(DenseBackendError, match="adapter and backend"):
        generate_candidates(
            scenario_id="scenario",
            lexical_queries=("first",),
            lexical_search=lexical_search,
            semantic_queries=("semantic",),
        )


def test_feedback_pipeline_runs_one_pass_and_records_new_ids() -> None:
    plan = build_average_vector(
        [FeedbackExample("seed", "scenario", "yes", (1.0, 0.0))],
        dimension=2,
    )
    calls: list[tuple[tuple[float, ...], int, Mapping[str, str] | None]] = []

    def search(
        vector: Sequence[float], limit: int, payload_filter: Mapping[str, str] | None
    ) -> Sequence[RankedCandidate]:
        calls.append((tuple(vector), limit, payload_filter))
        return [RankedCandidate("new", 1, 0.5)]

    union = run_feedback_pass(
        plan=plan,
        original=[RankedCandidate("seed", 1, 0.9)],
        search=search,
        limit=4,
        payload_filter={"snapshot_id": "snap"},
    )
    assert len(calls) == 1
    assert union.new_ids == ("new",)
    assert calls[0][2] == {"snapshot_id": "snap"}



def test_tombstones_filter_exact_fullname_and_context_candidates(tmp_path) -> None:
    from reddit_search.ingest.invalidation import load_tombstone_ledger

    ledger_path = tmp_path / "tombstones.jsonl"
    ledger_path.write_text(
        '{"message_fullname":"t1_exact","source_revision_id":"rev-1","reason":"removed"}\n'
        '{"message_fullname":"t1_all","source_revision_id":null,"reason":"removed"}\n',
        encoding="utf-8",
    )
    ledger = load_tombstone_ledger(ledger_path)

    def search(_query: str, _limit: int) -> Sequence[RankedCandidate]:
        return [
            RankedCandidate(
                "exact",
                1,
                1.0,
                {"message_fullname": "t1_exact", "source_revision_id": "rev-1"},
            ),
            RankedCandidate(
                "context",
                2,
                0.9,
                {
                    "message_fullname": "t1_other",
                    "source_revision_id": "rev-2",
                    "context_message_refs": ["t1_all"],
                },
            ),
            RankedCandidate(
                "live",
                3,
                0.8,
                {"message_fullname": "t1_live", "source_revision_id": "rev-3"},
            ),
        ]

    result = generate_candidates(
        scenario_id="scenario",
        lexical_queries=("query",),
        lexical_search=search,
        tombstone_ledger=ledger,
        output_limit=5,
    )
    assert [candidate.candidate_id for candidate in result.fused] == ["live"]

class Reranker:
    model_id = "qwen-reranker"
    revision = "rev-1"

    def __init__(self) -> None:
        self.calls = 0

    def score(
        self,
        *,
        scenario_query: str,
        candidate_texts: Sequence[str],
        instruction: str,
    ) -> Sequence[float]:
        self.calls += 1
        assert scenario_query == "need notes"
        assert "focal author" in instruction
        return [float(len(text)) for text in candidate_texts]


def test_reranker_cache_is_versioned_and_bounded() -> None:
    candidates = [
        FusedCandidate("short", 1, 0.5, {"lexical": 1}, {}, {}),
        FusedCandidate("long", 2, 0.4, {"dense": 1}, {}, {}),
    ]
    texts = {"short": "x", "long": "long text"}
    adapter = Reranker()
    cache: dict[str, float] = {}

    first = rerank_candidates(
        candidates,
        scenario_id="scenario",
        scenario_query="need notes",
        candidate_texts=texts,
        adapter=adapter,
        cache=cache,
        limit=2,
    )
    second = rerank_candidates(
        candidates,
        scenario_id="scenario",
        scenario_query="need notes",
        candidate_texts=texts,
        adapter=adapter,
        cache=cache,
        limit=2,
    )

    assert [item.candidate.candidate_id for item in first] == ["long", "short"]
    assert [item.candidate.candidate_id for item in second] == ["long", "short"]
    assert adapter.calls == 1
    assert len(cache) == 2


def test_rerank_rejects_stale_tombstoned_cache_hit(tmp_path) -> None:
    from reddit_search.ingest.invalidation import load_tombstone_ledger

    ledger_path = tmp_path / "tombstones.jsonl"
    ledger_path.write_text(
        '{"message_fullname":"t1_stale","source_revision_id":"rev-1","reason":"removed"}\n',
        encoding="utf-8",
    )
    ledger = load_tombstone_ledger(ledger_path)
    candidate = FusedCandidate(
        "stale", 1, 0.5, {"dense": 1}, {},
        {"message_fullname": "t1_stale", "source_revision_id": "rev-1"},
    )
    adapter = Reranker()
    result = rerank_candidates(
        [candidate],
        scenario_id="scenario",
        scenario_query="need notes",
        candidate_texts={"stale": "text"},
        adapter=adapter,
        cache={"already-cached": 1.0},
        limit=1,
        tombstone_ledger=ledger,
    )
    assert result == []
    assert adapter.calls == 0

def test_reranker_cache_key_changes_for_source_revision() -> None:
    from reddit_search.retrieval.rerank import rerank_cache_key

    base = FusedCandidate(
        "candidate",
        1,
        0.5,
        {"dense": 1},
        {},
        {"message_fullname": "t1_same", "source_revision_id": "rev-1", "snapshot_id": "snap"},
    )
    changed = FusedCandidate(
        "candidate",
        1,
        0.5,
        {"dense": 1},
        {},
        {"message_fullname": "t1_same", "source_revision_id": "rev-2", "snapshot_id": "snap"},
    )
    kwargs = dict(
        scenario_id="scenario",
        scenario_query="need notes",
        instruction="instruction",
        candidate_text="text",
        model_id="model",
        model_revision="model-rev",
    )
    assert rerank_cache_key(candidate=base, **kwargs) != rerank_cache_key(
        candidate=changed, **kwargs
    )
