"""Offline preflight for the real dense-retrieval execution boundary."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import ValidationError

from reddit_search.config import configuration_hash, load_runtime_config

from .comparison import ComparisonConfig, load_comparison_config


@dataclass(frozen=True, slots=True)
class RetrievalPreflight:
    """Machine-readable readiness report; no model downloads or remote calls."""

    ready: bool
    blockers: tuple[str, ...]
    checks: dict[str, bool]
    runtime_configuration_hash: str | None
    comparison_configuration_hash: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
            "checks": dict(sorted(self.checks.items())),
            "runtime_configuration_hash": self.runtime_configuration_hash,
            "comparison_configuration_hash": self.comparison_configuration_hash,
        }


def collect_retrieval_preflight(
    *,
    runtime_path: Path,
    comparison_path: Path,
    probe_qdrant: bool = True,
) -> RetrievalPreflight:
    """Validate local-only config, installed runtime modules, and Qdrant health."""
    blockers: list[str] = []
    checks: dict[str, bool] = {}
    runtime_hash: str | None = None
    comparison_hash: str | None = None
    runtime = None
    comparison: ComparisonConfig | None = None

    try:
        runtime = load_runtime_config(runtime_path)
        runtime_hash = configuration_hash(runtime)
        checks["runtime_config_valid"] = True
    except (OSError, ValidationError, ValueError) as error:
        checks["runtime_config_valid"] = False
        blockers.append(f"runtime config invalid: {error}")

    try:
        comparison = load_comparison_config(comparison_path)
        comparison_hash = comparison.configuration_hash()
        checks["comparison_config_valid"] = True
    except (OSError, ValueError) as error:
        checks["comparison_config_valid"] = False
        blockers.append(f"comparison config invalid: {error}")

    if runtime is not None:
        local_only = not (
            runtime.network.allow_model_downloads
            or runtime.network.allow_remote_inference
            or runtime.network.allow_reddit_requests
        )
        checks["local_only_policy"] = local_only
        if not local_only:
            blockers.append(
                "runtime network policy permits downloads, remote inference, or Reddit requests"
            )

        embedding_pinned = bool(runtime.models.embedding_revision)
        reranker_pinned = bool(runtime.models.reranker_revision)
        checks["embedding_revision_pinned"] = embedding_pinned
        checks["reranker_revision_pinned"] = reranker_pinned
        if not embedding_pinned:
            blockers.append("models.embedding_revision is not pinned")
        if not reranker_pinned:
            blockers.append("models.reranker_revision is not pinned")

        torch_installed = importlib.util.find_spec("torch") is not None
        transformers_installed = importlib.util.find_spec("transformers") is not None
        checks["torch_installed"] = torch_installed
        checks["transformers_installed"] = transformers_installed
        if not torch_installed:
            blockers.append("torch is not installed")
        if not transformers_installed:
            blockers.append("transformers is not installed")

        qdrant_url = runtime.qdrant.url
        parsed = urlparse(qdrant_url)
        loopback = parsed.scheme in {"http", "https"} and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
        checks["qdrant_url_loopback"] = loopback
        if not loopback:
            blockers.append("qdrant.url must point to a loopback HTTP endpoint")
        elif probe_qdrant:
            healthy = _probe_qdrant(qdrant_url)
            checks["qdrant_healthy"] = healthy
            if not healthy:
                blockers.append("Qdrant is not healthy on the configured loopback endpoint")
        else:
            checks["qdrant_healthy"] = False

    if comparison is not None and runtime is not None:
        same_budget = (
            comparison.output_limit == runtime.retrieval.output_limit
            and comparison.rrf_k == runtime.retrieval.rrf_k
        )
        checks["comparison_matches_runtime_limits"] = same_budget
        if not same_budget:
            blockers.append("comparison config does not match runtime output_limit or rrf_k")

    return RetrievalPreflight(
        ready=not blockers and all(checks.values()),
        blockers=tuple(blockers),
        checks=checks,
        runtime_configuration_hash=runtime_hash,
        comparison_configuration_hash=comparison_hash,
    )


def resolve_dense_alias(alias: str, registry_path: Path) -> tuple[str, str] | None:
    """Resolve a managed alias to its canonical physical collection and snapshot."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for entry in registry.get("entries", []):
        if entry.get("alias") == alias:
            collection = entry.get("collection")
            snapshot = entry.get("snapshot_id")
            if isinstance(collection, str) and isinstance(snapshot, str):
                return collection, snapshot
    return None


def _probe_qdrant(base_url: str) -> bool:
    request = Request(base_url.rstrip("/") + "/healthz", method="GET")
    try:
        with urlopen(request, timeout=0.5) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("configs/retrieval_comparison.yaml"),
    )
    parser.add_argument("--no-qdrant-probe", action="store_true")
    args = parser.parse_args()
    report = collect_retrieval_preflight(
        runtime_path=args.runtime,
        comparison_path=args.comparison,
        probe_qdrant=not args.no_qdrant_probe,
    )
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
