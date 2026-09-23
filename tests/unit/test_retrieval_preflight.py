import importlib.util
from pathlib import Path

from reddit_search.retrieval.preflight import collect_retrieval_preflight

ROOT = Path(__file__).parents[2]


def test_current_runtime_preflight_reports_pinned_local_semantic_prerequisites() -> None:
    report = collect_retrieval_preflight(
        runtime_path=ROOT / "configs" / "runtime.yaml",
        comparison_path=ROOT / "configs" / "retrieval_comparison.yaml",
        probe_qdrant=False,
    )

    assert report.ready is False
    assert report.checks["runtime_config_valid"] is True
    assert report.checks["comparison_config_valid"] is True
    assert report.checks["qdrant_url_loopback"] is True
    assert report.checks["qdrant_healthy"] is False
    assert report.checks["embedding_revision_pinned"] is True
    assert report.checks["reranker_revision_pinned"] is True
    assert report.checks["torch_installed"] is (importlib.util.find_spec("torch") is not None)
    assert report.checks["transformers_installed"] is (
        importlib.util.find_spec("transformers") is not None
    )
    assert report.comparison_configuration_hash is not None

def test_preflight_rejects_remote_qdrant_url_without_probing(tmp_path: Path) -> None:
    runtime = (ROOT / "configs" / "runtime.yaml").read_text(encoding="utf-8")
    runtime = runtime.replace("http://127.0.0.1:6333", "https://qdrant.example.invalid")
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(runtime, encoding="utf-8")

    report = collect_retrieval_preflight(
        runtime_path=runtime_path,
        comparison_path=ROOT / "configs" / "retrieval_comparison.yaml",
        probe_qdrant=True,
    )

    assert report.checks["qdrant_url_loopback"] is False
    assert "qdrant.url must point to a loopback HTTP endpoint" in report.blockers
