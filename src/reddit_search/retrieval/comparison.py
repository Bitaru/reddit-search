"""Versioned A/B/C retrieval comparison configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

VariantID = Literal["A", "B", "C"]


@dataclass(frozen=True, slots=True)
class ComparisonVariant:
    variant_id: VariantID
    lexical: bool
    dense: bool
    reranker: bool
    feedback: bool

    def __post_init__(self) -> None:
        expected = {
            "A": (True, False, False, False),
            "B": (True, True, True, False),
            "C": (True, True, True, True),
        }[self.variant_id]
        actual = (self.lexical, self.dense, self.reranker, self.feedback)
        if actual != expected:
            raise ValueError(f"variant {self.variant_id} must have flags {expected}, got {actual}")

    def manifest(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "lexical": self.lexical,
            "dense": self.dense,
            "reranker": self.reranker,
            "feedback": self.feedback,
        }


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Shared-corpus and shared-human-budget contract for A/B/C."""

    snapshot_id: str
    split: Literal["dev", "test"]
    output_limit: int
    review_budget: int
    lexical_candidates: int
    dense_candidates: int
    rerank_candidates: int
    feedback_candidates: int
    rrf_k: int
    variants: tuple[ComparisonVariant, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("comparison schema_version must be 1")
        if not self.snapshot_id.strip():
            raise ValueError("comparison snapshot_id must not be empty")
        if self.split not in {"dev", "test"}:
            raise ValueError("comparison split must be dev or test")
        if self.output_limit <= 0 or self.review_budget <= 0:
            raise ValueError("comparison budgets must be positive")
        if self.output_limit > self.review_budget:
            raise ValueError("output_limit cannot exceed the shared review_budget")
        for name, value in (
            ("lexical_candidates", self.lexical_candidates),
            ("dense_candidates", self.dense_candidates),
            ("rerank_candidates", self.rerank_candidates),
            ("feedback_candidates", self.feedback_candidates),
            ("rrf_k", self.rrf_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if tuple(variant.variant_id for variant in self.variants) != ("A", "B", "C"):
            raise ValueError("comparison must contain variants A, B, and C in order")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "split": self.split,
            "output_limit": self.output_limit,
            "review_budget": self.review_budget,
            "lexical_candidates": self.lexical_candidates,
            "dense_candidates": self.dense_candidates,
            "rerank_candidates": self.rerank_candidates,
            "feedback_candidates": self.feedback_candidates,
            "rrf_k": self.rrf_k,
            "variants": [variant.manifest() for variant in self.variants],
        }

    def configuration_hash(self) -> str:
        encoded = json.dumps(
            self.manifest(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
def canonical_hash(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible comparison identity deterministically."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_comparison_manifest(path: Path) -> dict[str, Any]:
    """Load a comparison manifest and reject malformed identity contracts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "retrieval_comparison_manifest":
        raise ValueError("invalid comparison manifest")
    identity = payload.get("identity")
    expected = payload.get("comparison_identity")
    if not isinstance(identity, dict) or not isinstance(expected, str):
        raise ValueError("comparison manifest lacks identity contract")
    if canonical_hash(identity) != expected:
        raise ValueError("comparison manifest identity hash mismatch")
    return payload


def load_comparison_config(path: Path) -> ComparisonConfig:
    """Load and validate the checked-in A/B/C contract."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("comparison config must contain a mapping")
    variants_payload = payload.get("variants")
    if not isinstance(variants_payload, list):
        raise ValueError("comparison config must contain a variants list")
    variants: list[ComparisonVariant] = []
    for raw in variants_payload:
        if not isinstance(raw, Mapping):
            raise ValueError("comparison variant must be a mapping")
        variant_id = raw.get("variant_id")
        if variant_id not in {"A", "B", "C"}:
            raise ValueError(f"unknown comparison variant: {variant_id!r}")
        variants.append(
            ComparisonVariant(
                variant_id=variant_id,
                lexical=_require_bool(raw, "lexical"),
                dense=_require_bool(raw, "dense"),
                reranker=_require_bool(raw, "reranker"),
                feedback=_require_bool(raw, "feedback"),
            )
        )
    return ComparisonConfig(
        schema_version=int(payload.get("schema_version", 1)),
        snapshot_id=str(payload.get("snapshot_id", "")),
        split=payload.get("split", "test"),
        output_limit=int(payload.get("output_limit", 0)),
        review_budget=int(payload.get("review_budget", 0)),
        lexical_candidates=int(payload.get("lexical_candidates", 0)),
        dense_candidates=int(payload.get("dense_candidates", 0)),
        rerank_candidates=int(payload.get("rerank_candidates", 0)),
        feedback_candidates=int(payload.get("feedback_candidates", 0)),
        rrf_k=int(payload.get("rrf_k", 0)),
        variants=tuple(variants),
    )


def _require_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"comparison field {key} must be boolean")
    return value
