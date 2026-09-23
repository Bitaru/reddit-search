"""Strict YAML configuration loading and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from .contracts import ProductProfile, Scenario, StrictModel, profile_missing_evidence
from .resources import (
    DEFAULT_MAX_PROCESS_RSS_BYTES,
    DEFAULT_MAX_STAGING_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
)


class NetworkSettings(StrictModel):
    allow_model_downloads: bool = False
    allow_remote_inference: bool = False
    allow_reddit_requests: bool = False
    allow_auto_publish: bool = False


class PathSettings(StrictModel):
    data_dir: str = "./data"
    reports_dir: str = "./reports"
    sqlite_path: str = "./data/sqlite/reddit_search.db"


class IngestionSettings(StrictModel):
    max_line_bytes: int = Field(default=8_388_608, gt=0)
    batch_records: int = Field(default=2_000, gt=0)
    checkpoint_interval_records: int = Field(default=5_000_000, gt=0)
    shard_target_records: int = Field(default=10_000, gt=0)
    reader_workers: Literal[1] = 1
    on_bad_json: Literal["quarantine_and_mark_incomplete"] = "quarantine_and_mark_incomplete"
    on_decompression_error: Literal["fail"] = "fail"
    max_staging_bytes: int = Field(default=DEFAULT_MAX_STAGING_BYTES, gt=0)
    minimum_free_disk_bytes: int = Field(default=DEFAULT_MINIMUM_FREE_DISK_BYTES, ge=0)
    max_process_rss_bytes: int = Field(default=DEFAULT_MAX_PROCESS_RSS_BYTES, gt=0)


class SamplingSettings(StrictModel):
    seed: int = 20_260_907
    pilot_target_units: int = Field(default=30_000, gt=0)
    rejected_control_target: int = Field(default=1_000, ge=0)


class ContextSettings(StrictModel):
    max_parent_messages: int = Field(default=4, ge=0)
    focus_chunk_tokens: int = Field(default=512, gt=0)
    focus_overlap_tokens: int = Field(default=64, ge=0)
    retrieval_context_tokens: int = Field(default=1_024, gt=0)
    evaluation_context_tokens: int = Field(default=4_096, gt=0)

    @model_validator(mode="after")
    def overlap_is_smaller_than_chunk(self) -> ContextSettings:
        if self.focus_overlap_tokens >= self.focus_chunk_tokens:
            raise ValueError("focus_overlap_tokens must be smaller than focus_chunk_tokens")
        return self


class RetrievalSettings(StrictModel):
    lexical_candidates_per_scenario: int = Field(default=100, gt=0)
    dense_candidates_per_scenario: int = Field(default=100, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    rerank_candidates: int = Field(default=100, gt=0)
    feedback_candidates: int = Field(default=100, gt=0)
    max_feedback_rounds: int = Field(default=1, ge=0, le=1)
    final_review_candidates: int = Field(default=40, gt=0)
    output_limit: int = Field(default=20, gt=0)
    output_max_questions_per_thread: int = Field(default=3, gt=0)


class ModelSettings(StrictModel):
    embedding_id: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_revision: str | None = None
    embedding_dimension: int = Field(default=1_024, gt=0)
    reranker_id: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_revision: str | None = None
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    batch_size: int = Field(default=8, gt=0)


class QdrantSettings(StrictModel):
    url: str = "http://127.0.0.1:6333"


class ReviewSettings(StrictModel):
    mode: Literal["manual", "model", "mock"] = "manual"
    evaluator_backend: str | None = None
    remote_request_budget: int = Field(default=0, ge=0)


class FeedbackSettings(StrictModel):
    enabled: bool = False
    strategy: Literal["average_vector"] = "average_vector"
    max_positive_examples: int = Field(default=5, ge=0, le=5)
    max_negative_examples: int = Field(default=5, ge=0, le=5)


class RuntimeConfig(StrictModel):
    schema_version: Literal[1] = 1
    mode: Literal["pilot"] = "pilot"
    language_policy: Literal["english_first"] = "english_first"
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)

    @model_validator(mode="after")
    def remote_inference_requires_explicit_budget(self) -> RuntimeConfig:
        if self.network.allow_remote_inference and self.review.remote_request_budget == 0:
            raise ValueError(
                "remote_request_budget must be positive when remote inference is enabled"
            )
        if self.review.mode == "model" and not self.review.evaluator_backend:
            raise ValueError("model review mode requires evaluator_backend")
        return self


def configuration_hash(value: Mapping[str, Any] | StrictModel) -> str:
    """Hash JSON-equivalent configuration deterministically, without filesystem state."""
    payload = value.model_dump(mode="json") if isinstance(value, StrictModel) else value
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def load_runtime_config(path: Path) -> RuntimeConfig:
    return RuntimeConfig.model_validate(_load_yaml_mapping(path))


def load_product_profiles(directory: Path) -> list[ProductProfile]:
    profiles = [
        ProductProfile.model_validate(_load_yaml_mapping(path))
        for path in sorted(directory.glob("*.yaml"))
    ]
    app_ids = [profile.app_id for profile in profiles]
    if len(app_ids) != len(set(app_ids)):
        raise ValueError("profile app IDs must be unique")
    return profiles


def load_scenarios(directory: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for path in sorted(directory.glob("*.yaml")):
        loaded = _load_yaml_mapping(path)
        entries = loaded.get("scenarios")
        if not isinstance(entries, list):
            raise ValueError(f"{path} must contain a scenarios list")
        scenarios.extend(Scenario.model_validate(entry) for entry in entries)
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("scenario IDs must be unique")
    return scenarios


def profile_validation_summary(profiles: list[ProductProfile]) -> dict[str, Any]:
    return {
        "valid": True,
        "profiles": {
            profile.app_id: {
                "profile_version": profile.profile_version,
                "missing_evidence": profile_missing_evidence(profile),
            }
            for profile in sorted(profiles, key=lambda item: item.app_id)
        },
    }
