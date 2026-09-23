"""Test-only deterministic adapters that never represent real model output."""

from __future__ import annotations

import hashlib
from typing import Any


class MockEvaluator:
    """Return stable synthetic evaluation metadata from input text."""

    def __init__(self, version: str) -> None:
        self.version = version

    def evaluate(self, source_text: str) -> dict[str, Any]:
        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        return {
            "evaluator_kind": "mock",
            "evaluator_version": self.version,
            "synthetic": True,
            "source_digest": digest,
        }


class MockEmbedder:
    """Produce stable, non-semantic vectors for behavior tests only."""

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def encode_queries(self, texts: list[str], instruction: str) -> list[list[float]]:
        return [self._vector(f"{instruction}\n{text}") for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:8]]


class MockReranker:
    """Return deterministic lexical-overlap order for tests, not quality claims."""

    def rank(self, query: str, texts: list[str]) -> list[tuple[int, int]]:
        query_terms = set(query.casefold().split())
        return sorted(
            (
                (index, len(query_terms.intersection(text.casefold().split())))
                for index, text in enumerate(texts)
            ),
            key=lambda item: (-item[1], item[0]),
        )
