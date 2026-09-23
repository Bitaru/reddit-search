"""Dense retrieval, rank fusion, and example-guided search primitives."""

from .comparison import ComparisonConfig, ComparisonVariant, load_comparison_config
from .comparison_local import run_local_development_comparison
from .comparison_run import (
    ComparisonPrerequisiteError,
    ComparisonRun,
    ComparisonScenario,
    ComparisonVariantResult,
    run_comparison,
)
from .dense import (
    DenseBackendError,
    DenseHit,
    DenseIndexJobStore,
    DensePoint,
    EmbeddingAdapter,
    EmbeddingRecipe,
    QdrantHttpIndex,
    validate_vector,
    write_index_manifest,
)
from .feedback import FeedbackExample, FeedbackPlan, build_average_vector, union_feedback_candidates
from .fusion import FusedCandidate, RankedCandidate, merge_ranked_variants, reciprocal_rank_fusion
from .indexing import DEFAULT_QUERY_INSTRUCTION, build_dense_index, write_index_result
from .pipeline import CandidateGenerationResult, generate_candidates, run_feedback_pass
from .qwen import QwenEmbeddingAdapter, QwenRerankerAdapter
from .rerank import (
    FOCAL_AUTHOR_RERANK_INSTRUCTION,
    RERANK_TEMPLATE_VERSION,
    RerankedCandidate,
    rerank_candidates,
)

__all__ = [
    "CandidateGenerationResult",
    "ComparisonConfig",
    "ComparisonPrerequisiteError",
    "ComparisonRun",
    "run_local_development_comparison",
    "ComparisonScenario",
    "ComparisonVariant",
    "ComparisonVariantResult",
    "DenseBackendError",
    "DenseHit",
    "DenseIndexJobStore",
    "DensePoint",
    "EmbeddingAdapter",
    "EmbeddingRecipe",
    "FeedbackExample",
    "FeedbackPlan",
    "FOCAL_AUTHOR_RERANK_INSTRUCTION",
    "FusedCandidate",
    "QdrantHttpIndex",
    "RERANK_TEMPLATE_VERSION",
    "RankedCandidate",
    "RerankedCandidate",
    "DEFAULT_QUERY_INSTRUCTION",
    "build_dense_index",
    "write_index_result",
    "build_average_vector",
    "QwenEmbeddingAdapter",
    "QwenRerankerAdapter",
    "generate_candidates",
    "load_comparison_config",
    "merge_ranked_variants",
    "reciprocal_rank_fusion",
    "rerank_candidates",
    "run_comparison",
    "run_feedback_pass",
    "union_feedback_candidates",
    "validate_vector",
    "write_index_manifest",
]
