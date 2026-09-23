import json
import subprocess
import sys
from pathlib import Path


def _write_sources_yaml(tmp_path: Path) -> Path:
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "sample_dump"
    source = fixtures / "RS_2026-01.jsonl"
    comments = fixtures / "RC_2026-01.jsonl"
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        f"""schema_version: 1
sources:
  - source_id: local-2026-01-submissions
    path: {source}
    source_kind: submission
    declared_month: 2026-01
    source_role: discovery
    usage_scope: authorized_local_research
  - source_id: local-2026-01-comments
    path: {comments}
    source_kind: comment
    declared_month: 2026-01
    source_role: discovery
    usage_scope: authorized_local_research
""",
        encoding="utf-8",
    )
    return sources


def _rules(tmp_path: Path) -> Path:
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """rules:
  - rule_id: expense.no_bank_link
    required_term_groups:
      - [expense, spending]
      - [bank, sync, link]
""",
        encoding="utf-8",
    )
    return rules


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reddit_search", *arguments],
        capture_output=True,
        check=False,
        text=True,
    )


def test_run_produces_manifest_and_review_artifacts(tmp_path: Path) -> None:
    sources = _write_sources_yaml(tmp_path)
    rules = _rules(tmp_path)
    output = tmp_path / "run"

    result = _run_cli(
        "run",
        "--sources",
        str(sources),
        "--rules",
        str(rules),
        "--output",
        str(output),
        "--target",
        "10",
        "--max-records",
        "10",
        "--queue-limit",
        "10",
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["kind"] == "run_manifest"
    assert manifest["snapshot_id"] == "discovery-selection-v1"
    assert set(manifest["stages"]) == {
        "sources_register",
        "ingest_discover",
        "review_cards",
        "review_queue",
        "review_worksheet",
    }
    assert manifest["stages"]["review_cards"]["card_count"] == 1

    run_manifest_path = output / "run_manifest.json"
    assert run_manifest_path.is_file()
    persisted = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert persisted["stages"] == manifest["stages"]

    assert (output / "review-cards" / "review_cards.jsonl").is_file()
    assert (output / "review-queue" / "review_cards.jsonl").is_file()
    worksheet = output / "review-worksheet" / "review_worksheet.jsonl"
    assert worksheet.is_file()
    worksheet_rows = worksheet.read_text(encoding="utf-8").splitlines()
    assert len(worksheet_rows) == 1


def test_run_rejects_second_run_into_same_output(tmp_path: Path) -> None:
    sources = _write_sources_yaml(tmp_path)
    rules = _rules(tmp_path)
    output = tmp_path / "run"

    first = _run_cli(
        "run",
        "--sources",
        str(sources),
        "--rules",
        str(rules),
        "--output",
        str(output),
        "--target",
        "10",
        "--max-records",
        "10",
    )
    assert first.returncode == 0, first.stderr

    second = _run_cli(
        "run",
        "--sources",
        str(sources),
        "--rules",
        str(rules),
        "--output",
        str(output),
        "--target",
        "10",
        "--max-records",
        "10",
    )
    assert second.returncode == 2, second.stdout
    assert "fresh or empty" in second.stderr
