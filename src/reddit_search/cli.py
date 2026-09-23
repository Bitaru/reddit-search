"""Command-line interface for local Reddit opportunity search."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer
import yaml
import zstandard
from pydantic import ValidationError

from .config import (
    configuration_hash,
    load_product_profiles,
    load_runtime_config,
    profile_validation_summary,
)
from .contracts import ClaimStatus
from .corpus.baseline import build_corpus_index, run_lexical_scenarios
from .corpus.units import CHUNKING_VERSION, CONTEXT_RECIPE_VERSION
from .demo import run_synthetic_demo
from .doctor import collect_doctor_report
from .evaluation.archive_audit import audit_registered_archives
from .evaluation.archive_coverage import derive_archive_coverage
from .evaluation.archive_partitioned import audit_registered_archives_partitioned
from .evaluation.findings import export_findings
from .evaluation.grouping_audit import audit_corpus_grouping
from .evaluation.labels import (
    collect_worksheet_labels,
    export_label_worksheet,
    import_label_rows,
    import_legacy_worksheet,
)
from .evaluation.metrics import compute_run_report, write_run_report
from .evaluation.pooling import build_blinded_pool
from .evaluation.reuse import reuse_labels
from .evaluation.splits import build_group_split
from .evaluation.unresolved_diagnostic import diagnose_unresolved
from .ingest.hydration import HydrationLimits, freeze_thread_ids, run_hydration
from .ingest.invalidation import audit_tombstone_outputs
from .ingest.pilot import run_discovery
from .ingest.review_export import export_discovery_review_cards
from .ingest.sources import (
    register_source_set,
    register_source_set_additive,
    validate_registered_sources,
)
from .operations import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    DenseCollectionProbe,
    PreflightConfig,
    build_purge_inventory,
    package_artifact,
    run_preflight,
    write_purge_inventory,
)
from .operations.artifact_package import ArtifactPackageError
from .reporting import operational_report, topic_report, write_report
from .retrieval.comparison_local import run_local_development_comparison
from .retrieval.dense_repair import DenseRepairError, repair_snapshot
from .retrieval.indexing import build_dense_index, write_index_result
from .review.evidence import validate_draft_rows
from .review.queue import build_balanced_review_queue
from .review.triage import build_luna_triage
from .review.worksheet import export_review_worksheet
from .review.workspace import create_review_server, review_server_report
from .run_pipeline import run_discovery_review

app = typer.Typer(
    name="reddit-search",
    help="Local, evidence-backed Reddit opportunity search.",
    no_args_is_help=True,
)

config_app = typer.Typer(help="Validate runtime configuration.")
profiles_app = typer.Typer(help="Validate product-profile evidence.")
sources_app = typer.Typer(help="Register explicitly declared local source files.")
ingest_app = typer.Typer(help="Build bounded local corpus stages.")
review_app = typer.Typer(help="Export bounded candidate sets for human review.")
dataset_app = typer.Typer(help="Freeze grouped evaluation datasets and pools.")
labels_app = typer.Typer(help="Export and validate structured evaluation labels.")
corpus_app = typer.Typer(help="Build and search the persistent lexical baseline corpus.")
retrieval_app = typer.Typer(help="Build local dense retrieval artifacts.")
report_app = typer.Typer(help="Produce deterministic operational and topic reports.")
findings_app = typer.Typer(help="Export judged-relevant findings from reviewed labels.")
tombstones_app = typer.Typer(help="Durable tombstone outbox inventory and local replay.")
app.add_typer(config_app, name="config")
app.add_typer(profiles_app, name="profiles")
app.add_typer(sources_app, name="sources")
app.add_typer(ingest_app, name="ingest")
app.add_typer(review_app, name="review")
app.add_typer(dataset_app, name="dataset")
app.add_typer(labels_app, name="labels")
app.add_typer(corpus_app, name="corpus")
app.add_typer(retrieval_app, name="retrieval")
app.add_typer(report_app, name="report")
app.add_typer(findings_app, name="findings")
app.add_typer(tombstones_app, name="tombstones")


def configure_logging() -> None:
    """Configure concise diagnostics without contaminating command output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.callback()
def cli() -> None:
    """Run local Reddit opportunity search commands."""


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable capability report."),
    ] = False,
) -> None:
    """Report local prerequisites without contacting external services."""
    report = collect_doctor_report()
    if json_output:
        typer.echo(json.dumps(report, sort_keys=True))
        return

    typer.echo("reddit-search local capability report")
    typer.echo(f"Python: {report['interpreter']['version']}")
    typer.echo(f"SQLite FTS5: {report['sqlite']['fts5_available']}")
    typer.echo(f"Docker on PATH: {report['optional_backends']['docker_on_path']}")
    typer.echo(
        f"Qdrant on 127.0.0.1:6333: {report['optional_backends']['qdrant_reachable_on_loopback']}"
    )


@config_app.command("validate")
def config_validate(
    config: Annotated[Path, typer.Option("--config", exists=True, readable=True)],
) -> None:
    """Validate runtime limits and report a stable configuration hash."""
    try:
        runtime_config = load_runtime_config(config)
    except (OSError, ValidationError, ValueError, yaml.YAMLError) as error:
        typer.echo(json.dumps({"valid": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error

    typer.echo(
        json.dumps(
            {"valid": True, "configuration_hash": configuration_hash(runtime_config)},
            sort_keys=True,
        )
    )


@profiles_app.command("validate")
def profiles_validate(
    profiles_dir: Annotated[Path, typer.Option("--profiles-dir", exists=True, file_okay=False)],
) -> None:
    """Summarize missing claim evidence without blocking topic-only search."""
    try:
        profiles = load_product_profiles(profiles_dir)
    except (OSError, ValidationError, ValueError, yaml.YAMLError) as error:
        typer.echo(json.dumps({"valid": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error

    typer.echo(json.dumps(profile_validation_summary(profiles), sort_keys=True))


@sources_app.command("register")
def sources_register(
    file: Annotated[Path, typer.Option("--file", exists=True, readable=True)],
    registry: Annotated[
        Path,
        typer.Option("--registry", help="Local registry path; archive content is not read."),
    ] = Path("data/manifests/source_registry.json"),
    additive: Annotated[
        bool,
        typer.Option(
            "--additive",
            help="Append new source identities while retaining existing registry rows.",
        ),
    ] = False,
) -> None:
    """Register source metadata; use --additive for monthly append-only updates."""
    try:
        register = register_source_set_additive if additive else register_source_set
        result = register(file, registry)
    except (OSError, ValidationError, ValueError, yaml.YAMLError) as error:
        typer.echo(json.dumps({"registered": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@sources_app.command("validate")
def sources_validate(
    registry: Annotated[
        Path,
        typer.Option("--registry", exists=True, readable=True),
    ] = Path("data/manifests/source_registry.json"),
) -> None:
    """Fully validate registered archives and persist their SHA-256 checksums."""
    try:
        result = validate_registered_sources(registry)
    except (OSError, ValidationError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        typer.echo(json.dumps({"validated": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))
    if result["failed_count"]:
        raise typer.Exit(code=3)


@ingest_app.command("discover")
def ingest_discover(
    registry: Annotated[
        Path,
        typer.Option("--registry", exists=True, readable=True),
    ] = Path("data/manifests/source_registry.json"),
    rules: Annotated[
        Path,
        typer.Option("--rules", exists=True, readable=True),
    ] = Path("configs/discovery.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Bounded normalized selection shard."),
    ] = Path("data/staging/discovery-selection.jsonl.zst"),
    target: Annotated[
        int,
        typer.Option("--target", min=1, help="Maximum selected records retained in memory."),
    ] = 30_000,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Stable selection priority seed."),
    ] = 20_260_907,
    max_records: Annotated[
        int,
        typer.Option(
            "--max-records",
            min=1,
            help="Maximum source records scanned; prevents a corpus-wide pilot run.",
        ),
    ] = 1_000_000,
    runtime: Annotated[
        Path,
        typer.Option("--runtime", exists=True, readable=True, help="Runtime YAML configuration."),
    ] = Path("configs/runtime.yaml"),
    tombstones: Annotated[
        Path | None,
        typer.Option(
            "--tombstones",
            exists=True,
            readable=True,
            help="Optional JSONL ledger of invalidated messages.",
        ),
    ] = None,
) -> None:
    """Discover a bounded candidate set across every registered discovery source."""
    try:
        runtime_config = load_runtime_config(runtime)
        result = run_discovery(
            registry,
            rules,
            output,
            target=target,
            seed=seed,
            max_records=max_records,
            max_staging_bytes=runtime_config.ingestion.max_staging_bytes,
            max_process_rss_bytes=runtime_config.ingestion.max_process_rss_bytes,
            tombstones_path=tombstones,
            minimum_free_disk_bytes=runtime_config.ingestion.minimum_free_disk_bytes,
        )
    except (OSError, ValidationError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        typer.echo(json.dumps({"discovered": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@ingest_app.command("freeze-threads")
def ingest_freeze_threads(
    selection: Annotated[
        Path,
        typer.Option(
            "--selection", exists=True, readable=True, help="Discovery selection JSONL.zst shard."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="JSON file receiving the frozen thread ID set."),
    ] = Path("data/staging/hydration-threads.json"),
) -> None:
    """Freeze the selected thread ID set for the hydration pass."""
    try:
        result = freeze_thread_ids(selection)
    except (OSError, ValueError, zstandard.ZstdError) as error:
        typer.echo(json.dumps({"frozen": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    typer.echo(json.dumps({**result, "output_file": str(output)}, sort_keys=True))


@ingest_app.command("hydrate")
def ingest_hydrate(
    selection: Annotated[
        Path,
        typer.Option(
            "--selection", exists=True, readable=True, help="Discovery selection JSONL.zst shard."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory receiving hydration shards and manifest."),
    ] = Path("data/staging/hydration"),
    context_messages: Annotated[
        int,
        typer.Option(
            "--context-messages",
            min=0,
            help="Maximum hydration-only context messages retained.",
        ),
    ] = 120_000,
    controls: Annotated[
        int,
        typer.Option("--controls", min=0, help="Deterministic rejected-control sample size."),
    ] = 1_000,
    seed: Annotated[int, typer.Option("--seed", help="Stable rank seed.")] = 20_260_912,
    max_records: Annotated[
        int | None,
        typer.Option(
            "--max-records",
            min=1,
            help="Scan bound for the pass; omit to read every source completely.",
        ),
    ] = None,
    runtime: Annotated[
        Path,
        typer.Option("--runtime", exists=True, readable=True, help="Runtime YAML configuration."),
    ] = Path("configs/runtime.yaml"),
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="Optional hydration checkpoint path; defaults inside the output directory.",
        ),
    ] = None,
    checkpoint_interval_records: Annotated[
        int | None,
        typer.Option(
            "--checkpoint-interval-records",
            min=1,
            help="Records between active-source checkpoint snapshots.",
        ),
    ] = None,
    tombstones: Annotated[
        Path | None,
        typer.Option(
            "--tombstones",
            exists=True,
            readable=True,
            help="Optional JSONL ledger of invalidated messages.",
        ),
    ] = None,
) -> None:
    """Reread registered sources to hydrate selected threads and sample controls."""
    try:
        runtime_config = load_runtime_config(runtime)
        result = run_hydration(
            Path("data/manifests/t05_source_registry.json"),
            selection,
            output,
            limits=HydrationLimits(
                max_context_messages=context_messages,
                seed=seed,
                control_target=controls,
                checkpoint_interval_records=(
                    checkpoint_interval_records
                    if checkpoint_interval_records is not None
                    else runtime_config.ingestion.checkpoint_interval_records
                ),
                max_staging_bytes=runtime_config.ingestion.max_staging_bytes,
                minimum_free_disk_bytes=runtime_config.ingestion.minimum_free_disk_bytes,
                max_process_rss_bytes=runtime_config.ingestion.max_process_rss_bytes,
            ),
            max_records=max_records,
            checkpoint_path=checkpoint,
            tombstones_path=tombstones,
        )
    except (OSError, ValueError, zstandard.ZstdError) as error:
        typer.echo(json.dumps({"hydrated": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))
    if not result["complete"]:
        raise typer.Exit(code=3)


@ingest_app.command("audit-tombstones")
def ingest_audit_tombstones(
    tombstones: Annotated[
        Path,
        typer.Option(
            "--tombstones",
            exists=True,
            readable=True,
            help="Canonical JSONL ledger of invalidated messages.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Audit report JSON output path."),
    ] = Path("reports/tombstone-audit/tombstone_audit_report.json"),
    selection: Annotated[
        Path | None,
        typer.Option(
            "--selection",
            exists=True,
            readable=True,
            help="Optional discovery selection shard to audit.",
        ),
    ] = None,
    hydration: Annotated[
        Path | None,
        typer.Option(
            "--hydration",
            exists=True,
            file_okay=False,
            readable=True,
            help="Optional hydration output directory to audit.",
        ),
    ] = None,
    review_cards: Annotated[
        Path | None,
        typer.Option(
            "--review-cards",
            exists=True,
            readable=True,
            help="Optional review_cards.jsonl file to audit.",
        ),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option(
            "--corpus",
            exists=True,
            readable=True,
            help="Optional SQLite corpus database to audit.",
        ),
    ] = None,
) -> None:
    """Audit persisted derivatives for messages covered by tombstones."""
    try:
        result = audit_tombstone_outputs(
            tombstones,
            output,
            selection_path=selection,
            hydration_directory=hydration,
            review_cards_path=review_cards,
            corpus_path=corpus,
        )
    except (OSError, ValueError, json.JSONDecodeError, zstandard.ZstdError) as error:
        typer.echo(json.dumps({"audited": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"audited": True} | result, sort_keys=True))


@review_app.command("cards")
def review_cards(
    selection: Annotated[
        Path,
        typer.Option(
            "--selection", exists=True, readable=True, help="Discovery selection JSONL.zst shard."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory for JSONL review cards."),
    ] = Path("reports/discovery-review"),
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", help="Stable identifier for this bounded selection."),
    ] = "discovery-selection-v1",
    tombstones: Annotated[
        Path | None,
        typer.Option(
            "--tombstones",
            exists=True,
            readable=True,
            help="Optional JSONL ledger of invalidated messages.",
        ),
    ] = None,
    focus_chunk_tokens: Annotated[int, typer.Option("--focus-chunk-tokens", min=1)] = 512,
    focus_overlap_tokens: Annotated[int, typer.Option("--focus-overlap-tokens", min=0)] = 64,
    chunking_version: Annotated[str, typer.Option("--chunking-version")] = CHUNKING_VERSION,
    context_recipe_version: Annotated[
        str,
        typer.Option(
            "--context-recipe-version",
            help="Version of the context construction recipe bound to this output.",
        ),
    ] = CONTEXT_RECIPE_VERSION,
) -> None:
    """Export selected messages with source evidence and selection provenance."""
    try:
        result = export_discovery_review_cards(
            selection,
            output,
            snapshot_id=snapshot_id,
            tombstones_path=tombstones,
            context_recipe_version=context_recipe_version,
            chunking_version=chunking_version,
            max_tokens=focus_chunk_tokens,
            overlap_tokens=focus_overlap_tokens,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"exported": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@review_app.command("queue")
def review_queue(
    cards: Annotated[
        Path,
        typer.Option("--cards", exists=True, readable=True, help="JSONL cards to stratify."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory for the bounded reviewer queue."),
    ] = Path("reports/review-queue"),
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum number of cards queued for review."),
    ] = 200,
) -> None:
    """Export a deterministic, rule-balanced subset of review cards."""
    try:
        result = build_balanced_review_queue(cards, output, limit=limit)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"queued": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@review_app.command("worksheet")
def review_worksheet(
    cards: Annotated[
        Path,
        typer.Option("--cards", exists=True, readable=True, help="Queued JSONL cards to annotate."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory for pending annotation rows."),
    ] = Path("reports/review-worksheet"),
) -> None:
    """Export a compact annotation worksheet without duplicating card evidence."""
    try:
        result = export_review_worksheet(cards, output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"exported": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@review_app.command("serve")
def review_serve(
    cards: Annotated[
        Path,
        typer.Option("--cards", exists=True, readable=True, help="Queued JSONL cards to review."),
    ],
    worksheet: Annotated[
        Path,
        typer.Option("--worksheet", exists=True, readable=True, help="Human annotation worksheet."),
    ],
    drafts: Annotated[
        Path | None,
        typer.Option("--drafts", help="Optional Luna machine-draft JSONL labels."),
    ] = None,
    candidate_ids: Annotated[
        Path | None,
        typer.Option(
            "--candidate-ids",
            exists=True,
            readable=True,
            help="JSON object with the candidate_ids to expose in this workspace.",
        ),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", min=0, max=65535, help="Loopback port; 0 selects a free port."),
    ] = 8765,
) -> None:
    """Serve a localhost-only workspace that saves completed worksheet rows."""
    try:
        server = create_review_server(
            cards_path=cards,
            worksheet_path=worksheet,
            drafts_path=drafts,
            selected_candidate_ids=(
                _read_candidate_ids(candidate_ids) if candidate_ids is not None else None
            ),
            host="127.0.0.1",
            port=port,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"serving": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(review_server_report(server), sort_keys=True))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@review_app.command("validate-drafts")
def review_validate_drafts(
    cards: Annotated[
        Path,
        typer.Option(
            "--cards", exists=True, readable=True, help="Queued JSONL cards to validate against."
        ),
    ],
    drafts: Annotated[
        Path,
        typer.Option(
            "--drafts", exists=True, readable=True, help="Machine-draft JSONL rows to validate."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="File for the source-free validation report JSON."),
    ],
    allow_incomplete: Annotated[
        bool,
        typer.Option("--allow-incomplete", help="Do not fail when some cards lack a draft row."),
    ] = False,
) -> None:
    """Validate machine-draft labels deterministically without transmitting source text."""
    try:
        report = validate_draft_rows(cards, drafts, require_complete=not allow_incomplete)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"validated": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(
        json.dumps({"validated": True, "output": str(output)} | report["summary"], sort_keys=True)
    )


@review_app.command("triage")
def review_triage(
    cards: Annotated[
        Path,
        typer.Option("--cards", exists=True, readable=True, help="Queued JSONL cards."),
    ],
    drafts: Annotated[
        Path,
        typer.Option("--drafts", exists=True, readable=True, help="Luna machine-draft JSONL."),
    ],
    validation: Annotated[
        Path,
        typer.Option(
            "--validation",
            exists=True,
            readable=True,
            help="Source-free output from review validate-drafts.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory for the deterministic triage queue."),
    ],
    negative_audit_rate: Annotated[
        float,
        typer.Option(
            "--negative-audit-rate",
            min=0.0,
            max=1.0,
            help="Seeded fraction of validated negatives sent to human audit.",
        ),
    ] = 0.1,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Stable seed for the negative audit sample."),
    ] = 20260912,
) -> None:
    """Route Luna drafts without treating model output as human judgment."""
    try:
        result = build_luna_triage(
            cards,
            drafts,
            validation,
            output,
            negative_audit_rate=negative_audit_rate,
            seed=seed,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"triaged": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"triaged": True} | result, sort_keys=True))


@dataset_app.command("pool")
def dataset_pool(
    runs: Annotated[
        Path,
        typer.Option(
            "--runs", exists=True, file_okay=False, readable=True, help="Ranked run directory."
        ),
    ],
    controls: Annotated[
        Path,
        typer.Option(
            "--controls", exists=True, readable=True, help="Zstandard rejected-control JSONL."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for the blinded pool."),
    ] = Path("reports/evaluation-pool"),
    target: Annotated[
        int,
        typer.Option("--target", min=1, help="Target number of pooled candidates."),
    ] = 240,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic pool seed."),
    ] = 20260912,
    corpus: Annotated[
        Path | None,
        typer.Option(
            "--corpus", exists=True, readable=True, help="Optional hydrated SQLite corpus."
        ),
    ] = None,
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            exists=True,
            readable=True,
            help="Optional retained source JSONL/Zstandard JSONL for canonical message fields.",
        ),
    ] = None,
    profiles_dir: Annotated[
        Path | None,
        typer.Option(
            "--profiles-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Optional product profiles bound to scenario app IDs.",
        ),
    ] = None,
) -> None:
    """Build a reviewer-safe pool without retrieval metadata."""
    try:
        result = build_blinded_pool(
            runs,
            controls,
            output,
            target_count=target,
            seed=seed,
            corpus_path=corpus,
            source_path=source,
            profiles_directory=profiles_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"built": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@dataset_app.command("split")
def dataset_split(
    pool: Annotated[
        Path,
        typer.Option("--pool", exists=True, readable=True, help="Blinded pool JSONL."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for grouped split artifacts."),
    ] = Path("reports/evaluation-splits"),
    duplicate_links: Annotated[
        Path | None,
        typer.Option(
            "--duplicate-links",
            exists=True,
            readable=True,
            help="Optional JSONL duplicate links from legacy annotations.",
        ),
    ] = None,
    development: Annotated[
        Path | None,
        typer.Option(
            "--development",
            exists=True,
            readable=True,
            help="Prior reviewed cards or candidate-ID manifest forced to development.",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic split seed."),
    ] = 20260912,
    test_fraction: Annotated[
        float,
        typer.Option(
            "--test-fraction", min=0.01, max=0.99, help="Fraction of groups assigned to test."
        ),
    ] = 0.5,
) -> None:
    """Freeze development/test membership at connected-group boundaries."""
    try:
        result = build_group_split(
            pool,
            output,
            duplicate_links_path=duplicate_links,
            seed=seed,
            test_fraction=test_fraction,
            development_path=development,
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"built": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@dataset_app.command("audit-grouping")
def dataset_audit_grouping(
    corpus: Annotated[Path, typer.Option("--corpus", exists=True, readable=True)],
    exposure: Annotated[list[Path], typer.Option("--exposure", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    duplicate_links: Annotated[
        Path | None, typer.Option("--duplicate-links", exists=True, readable=True)
    ] = None,
    candidate_splits: Annotated[
        Path | None, typer.Option("--candidate-splits", exists=True, readable=True)
    ] = None,
    corpus_manifest: Annotated[
        Path | None, typer.Option("--corpus-manifest", exists=True, readable=True)
    ] = None,
    snapshot_id: Annotated[str | None, typer.Option("--snapshot-id")] = None,
) -> None:
    """Audit retained-corpus grouping without changing split assignments."""
    try:
        result = audit_corpus_grouping(
            corpus,
            exposure,
            output,
            duplicate_links_path=duplicate_links,
            candidate_splits_path=candidate_splits,
            corpus_manifest_path=corpus_manifest,
            snapshot_id=snapshot_id,
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"audited": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"audited": True, **result}, sort_keys=True))


@dataset_app.command("audit-archives")
def dataset_audit_archives(
    registry: Annotated[Path, typer.Option("--registry", exists=True, readable=True)],
    corpus: Annotated[Path, typer.Option("--corpus", exists=True, readable=True)],
    exposure: Annotated[list[Path], typer.Option("--exposure", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    max_records: Annotated[int, typer.Option("--max-records", min=1)],
    max_bytes: Annotated[int | None, typer.Option("--max-bytes", min=1)] = None,
    snapshot_id: Annotated[str | None, typer.Option("--snapshot-id")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    index: Annotated[Path | None, typer.Option("--index")] = None,
    index_directory: Annotated[Path | None, typer.Option("--index-directory")] = None,
    max_index_bytes: Annotated[int, typer.Option("--max-index-bytes", min=1)] = 10 * 1024**3,
    progress: Annotated[Path | None, typer.Option("--progress")] = None,
    max_process_rss_bytes: Annotated[int, typer.Option("--max-process-rss-bytes", min=1)] = 4
    * 1024**3,
    min_free_disk_bytes: Annotated[int, typer.Option("--min-free-disk-bytes", min=0)] = 5 * 1024**3,
    duplicate_links: Annotated[
        Path | None, typer.Option("--duplicate-links", exists=True, readable=True)
    ] = None,
    candidate_splits: Annotated[
        Path | None, typer.Option("--candidate-splits", exists=True, readable=True)
    ] = None,
) -> None:
    """Audit registered archive exposure against the retained corpus index."""
    try:
        if index_directory is not None:
            if index is not None:
                raise ValueError("--index and --index-directory are mutually exclusive")
            result = audit_registered_archives_partitioned(
                registry,
                corpus,
                exposure,
                output,
                max_records=max_records,
                index_directory=index_directory,
                snapshot_id=snapshot_id,
                max_bytes=max_bytes,
                checkpoint_path=checkpoint,
                progress_path=progress,
                max_index_bytes=max_index_bytes,
                max_process_rss_bytes=max_process_rss_bytes,
                min_free_disk_bytes=min_free_disk_bytes,
            )
        else:
            result = audit_registered_archives(
                registry,
                corpus,
                exposure,
                output,
                max_records=max_records,
                max_bytes=max_bytes,
                snapshot_id=snapshot_id,
                checkpoint_path=checkpoint,
                index_path=index,
                progress_path=progress,
                max_process_rss_bytes=max_process_rss_bytes,
                min_free_disk_bytes=min_free_disk_bytes,
                duplicate_links_path=duplicate_links,
                candidate_splits_path=candidate_splits,
            )
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"audited": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"audited": True, **result}, sort_keys=True))
    if not result.get("complete", False):
        raise typer.Exit(code=3)


@dataset_app.command("coverage")
def dataset_coverage(
    audit: Annotated[
        Path,
        typer.Option(
            "--audit",
            exists=True,
            file_okay=False,
            readable=True,
            help="Completed archive audit directory.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for derived coverage artifacts."),
    ],
    rematch: Annotated[
        Path | None,
        typer.Option(
            "--rematch",
            exists=True,
            file_okay=False,
            readable=True,
            help=(
                "Verified rematch directory (archive_audit_rematch) over the same audit; "
                "derives coverage from the corrected per-(source_id, bucket) counts and "
                "records the supersession."
            ),
        ),
    ] = None,
) -> None:
    """Derive deterministic coverage from a completed exact-match archive audit."""
    try:
        result = derive_archive_coverage(audit, output, rematch_directory=rematch)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"derived": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"derived": True, **result}, sort_keys=True))


@dataset_app.command("diagnose-unresolved")
def dataset_diagnose_unresolved(
    audit: Annotated[Path, typer.Option("--audit", exists=True, file_okay=False, readable=True)],
    corpus: Annotated[Path, typer.Option("--corpus", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Diagnose unresolved archive exposure without rescanning or mutating inputs."""
    try:
        result = diagnose_unresolved(audit, corpus, output)
    except (OSError, ValueError, json.JSONDecodeError, TypeError, sqlite3.Error) as error:
        typer.echo(json.dumps({"diagnosed": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"diagnosed": True, **result}, sort_keys=True))


@labels_app.command("export")
def labels_export(
    pool: Annotated[
        Path,
        typer.Option("--pool", exists=True, readable=True, help="Blinded pool JSONL."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for label worksheet."),
    ] = Path("reports/evaluation-labels"),
) -> None:
    """Export structured §5.6 label stubs for the blinded pool."""
    try:
        result = export_label_worksheet(pool, output)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"exported": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@labels_app.command("import")
def labels_import(
    pool: Annotated[
        Path,
        typer.Option("--pool", exists=True, readable=True, help="Blinded pool JSONL."),
    ],
    labels: Annotated[
        Path,
        typer.Option("--labels", exists=True, readable=True, help="Completed §5.6 JSONL labels."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for validated labels and report."),
    ] = Path("reports/evaluation-labels-import"),
    allow_incomplete: Annotated[
        bool,
        typer.Option("--allow-incomplete", help="Permit unlabelled pool rows."),
    ] = False,
    profiles_dir: Annotated[
        Path | None,
        typer.Option(
            "--profiles-dir",
            exists=True,
            file_okay=False,
            help="Product profiles directory; verified claim IDs bind compatible verdicts.",
        ),
    ] = None,
) -> None:
    """Validate structured labels without accepting unsupported compatibility claims."""
    try:
        verified_claim_ids: list[str] = []
        if profiles_dir is not None:
            profiles = load_product_profiles(profiles_dir)
            verified_claim_ids = [
                claim.claim_id
                for profile in profiles
                for claim in profile.claims
                if claim.status is ClaimStatus.VERIFIED and claim.evidence_ref
            ]
        result = import_label_rows(
            pool,
            labels,
            output,
            require_complete=not allow_incomplete,
            verified_claim_ids=verified_claim_ids,
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"imported": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"imported": True} | result, sort_keys=True))


@labels_app.command("import-legacy")
def labels_import_legacy(
    worksheet: Annotated[
        Path,
        typer.Option("--worksheet", exists=True, readable=True, help="Legacy worksheet JSONL."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for legacy labels and sidecars."),
    ] = Path("reports/legacy-labels"),
    snapshot_id: Annotated[
        str | None,
        typer.Option("--snapshot-id", help="Snapshot identifier, if known."),
    ] = None,
) -> None:
    """Preserve legacy topic-fit judgments without inventing richer fields."""
    try:
        result = import_legacy_worksheet(worksheet, output, snapshot_id=snapshot_id)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"imported": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    """Temporarily change the working directory so relative volume probes bind."""
    previous = Path.cwd()
    os.chdir(path.expanduser().resolve())
    try:
        yield
    finally:
        os.chdir(previous)


def _read_candidate_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate ID file must be a JSON object")
    candidate_ids = payload.get("candidate_ids")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise ValueError("candidate ID file must contain a non-empty candidate_ids list")
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
        raise ValueError("candidate_ids must contain only non-empty strings")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids must not contain duplicates")
    return set(candidate_ids)


@labels_app.command("collect")
def labels_collect(
    pool: Annotated[
        Path,
        typer.Option("--pool", exists=True, readable=True, help="Blinded pool JSONL."),
    ],
    worksheet: Annotated[
        Path,
        typer.Option("--worksheet", exists=True, readable=True, help="Annotated worksheet JSONL."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for collected labels."),
    ] = Path("reports/step5-collected-labels"),
) -> None:
    """Convert completed worksheet annotations into §5.6 label rows."""
    try:
        result = collect_worksheet_labels(pool, worksheet, output)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"collected": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"collected": True} | result, sort_keys=True))


@labels_app.command("reuse")
def labels_reuse(
    source_pool: Annotated[
        Path,
        typer.Option("--source-pool", exists=True, readable=True, help="Prior blinded pool JSONL."),
    ],
    source_labels: Annotated[
        Path,
        typer.Option(
            "--source-labels",
            exists=True,
            readable=True,
            help="Prior validated labels JSONL.",
        ),
    ],
    target_pool: Annotated[
        Path,
        typer.Option(
            "--target-pool",
            exists=True,
            readable=True,
            help="Current blinded pool JSONL.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for reused labels and report."),
    ],
    rubric_version: Annotated[
        str,
        typer.Option("--rubric-version", help="Exact rubric contract version required for reuse."),
    ] = "topic-fit-v1",
) -> None:
    """Reuse only unchanged human judgments; all mismatches remain for review."""
    try:
        result = reuse_labels(
            source_pool,
            source_labels,
            target_pool,
            output,
            rubric_version=rubric_version,
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"reused": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"reused": True} | result, sort_keys=True))


@findings_app.command("export")
def findings_export(
    labels: Annotated[
        Path,
        typer.Option(
            "--labels",
            exists=True,
            readable=True,
            help="Validated §5.6 label rows (labels.jsonl from labels collect/import).",
        ),
    ],
    cards: Annotated[
        Path,
        typer.Option(
            "--cards",
            exists=True,
            readable=True,
            help="Review cards file or directory joined by (candidate_id, scenario_id).",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory for findings.jsonl and manifest."),
    ] = Path("reports/findings"),
) -> None:
    """Export reviewed-relevant findings, failing closed on incomplete rows."""
    try:
        result = export_findings(labels, cards, output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"exported": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@corpus_app.command("build")
def corpus_build(
    selection: Annotated[
        Path,
        typer.Option(
            "--selection", exists=True, readable=True, help="Discovery selection JSONL.zst shard."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Persistent SQLite corpus database path."),
    ] = Path("data/sqlite/june-2026-baseline.db"),
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", help="Stable identifier for the indexed selection."),
    ] = "june-2026-full-corpus",
    context_shard: Annotated[
        Path | None,
        typer.Option(
            "--context-shard",
            help="Optional hydration context shard extending ancestor lookups.",
        ),
    ] = None,
    runtime: Annotated[
        Path,
        typer.Option("--runtime", exists=True, readable=True, help="Runtime YAML configuration."),
    ] = Path("configs/runtime.yaml"),
    tombstones: Annotated[
        Path | None,
        typer.Option(
            "--tombstones",
            exists=True,
            readable=True,
            help="Optional JSONL ledger of invalidated messages.",
        ),
    ] = None,
    context_recipe_version: Annotated[
        str,
        typer.Option(
            "--context-recipe-version",
            help="Version of the context construction recipe bound to this corpus.",
        ),
    ] = CONTEXT_RECIPE_VERSION,
) -> None:
    """Index a frozen selection shard into a persistent lexical corpus."""
    try:
        runtime_config = load_runtime_config(runtime)
        result = build_corpus_index(
            selection,
            output,
            snapshot_id=snapshot_id,
            context_shard_path=context_shard,
            max_staging_bytes=runtime_config.ingestion.max_staging_bytes,
            minimum_free_disk_bytes=runtime_config.ingestion.minimum_free_disk_bytes,
            focus_chunk_tokens=runtime_config.context.focus_chunk_tokens,
            focus_overlap_tokens=runtime_config.context.focus_overlap_tokens,
            tombstones_path=tombstones,
            context_recipe_version=context_recipe_version,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"built": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@corpus_app.command("search")
def corpus_search(
    corpus: Annotated[
        Path,
        typer.Option("--corpus", exists=True, readable=True, help="Persistent corpus database."),
    ],
    scenarios: Annotated[
        Path,
        typer.Option(
            "--scenarios",
            exists=True,
            readable=True,
            help="Directory of scenario YAML definitions.",
        ),
    ] = Path("configs/scenarios"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory for per-scenario ranked run files."),
    ] = Path("reports/lexical-baseline"),
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", help="Snapshot to search within the corpus."),
    ] = "june-2026-full-corpus",
    candidates: Annotated[
        int,
        typer.Option(
            "--candidates",
            min=1,
            help="Per-query candidate limit before scenario merge and truncation.",
        ),
    ] = 100,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Final ranked hits persisted per scenario."),
    ] = 20,
) -> None:
    """Run deterministic lexical scenario searches against the corpus."""
    try:
        result = run_lexical_scenarios(
            corpus,
            scenarios,
            output,
            snapshot_id=snapshot_id,
            candidates_per_scenario=candidates,
            output_limit=limit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"searched": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@retrieval_app.command("index-dense")
def retrieval_index_dense(
    corpus: Annotated[
        Path,
        typer.Option("--corpus", exists=True, readable=True, help="Persistent corpus database."),
    ] = Path("data/sqlite/june-2026-baseline-hydrated.db"),
    runtime: Annotated[
        Path,
        typer.Option("--runtime", exists=True, readable=True, help="Runtime YAML configuration."),
    ] = Path("configs/runtime.yaml"),
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Dense index manifest output path."),
    ] = Path("reports/dense-index-manifest.json"),
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", help="Snapshot to encode into Qdrant."),
    ] = "june-2026-full-corpus",
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Optional Hugging Face cache directory."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size", min=1, help="Embedding batch size; defaults to runtime config."
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Torch device override: auto, cpu, mps, or cuda."),
    ] = None,
    job_store: Annotated[
        Path,
        typer.Option("--job-store", help="SQLite resume ledger for dense indexing."),
    ] = Path("reports/dense-index-jobs.sqlite"),
) -> None:
    """Build the pinned local Qwen index without enabling runtime downloads."""
    try:
        result = build_dense_index(
            corpus,
            runtime,
            manifest,
            snapshot_id=snapshot_id,
            cache_dir=cache_dir,
            batch_size=batch_size,
            device=device,
            job_store_path=job_store,
        )
        write_index_result(manifest.with_name("dense-index-result.json"), result)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        typer.echo(json.dumps({"indexed": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"indexed": True} | result, sort_keys=True))


@retrieval_app.command("snapshot-repair")
def retrieval_snapshot_repair(
    qdrant_url: Annotated[str, typer.Option("--qdrant-url")],
    registry: Annotated[Path, typer.Option("--registry", exists=True, readable=True)],
    corpus: Annotated[Path, typer.Option("--corpus", exists=True, readable=True)],
    alias: Annotated[str, typer.Option("--alias")],
    source_snapshot: Annotated[str, typer.Option("--source-snapshot")],
    target_snapshot: Annotated[str, typer.Option("--target-snapshot")],
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 256,
    apply: Annotated[bool, typer.Option("--apply")] = False,
) -> None:
    """Repair only snapshot_id payloads in one registry-managed collection."""
    try:
        result = repair_snapshot(
            qdrant_url=qdrant_url,
            registry_path=registry,
            corpus_path=corpus,
            alias=alias,
            source_snapshot=source_snapshot,
            target_snapshot=target_snapshot,
            batch_size=batch_size,
            dry_run=not apply,
        )
    except (DenseRepairError, OSError, ValueError, json.JSONDecodeError, sqlite3.Error) as error:
        typer.echo(json.dumps({"repaired": False, "error": str(error)}, sort_keys=True), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"repaired": True, **result}, sort_keys=True))


@retrieval_app.command("compare-dev")
def retrieval_compare_dev(
    corpus: Annotated[
        Path,
        typer.Option("--corpus", exists=True, readable=True, help="Persistent corpus database."),
    ] = Path("data/sqlite/june-2026-baseline-hydrated.db"),
    runtime: Annotated[
        Path,
        typer.Option("--runtime", exists=True, readable=True, help="Runtime YAML configuration."),
    ] = Path("configs/runtime.yaml"),
    comparison: Annotated[
        Path,
        typer.Option("--comparison", exists=True, readable=True, help="A/B/C comparison YAML."),
    ] = Path("configs/retrieval_comparison.yaml"),
    labels: Annotated[
        Path,
        typer.Option("--labels", exists=True, readable=True, help="Validated development labels."),
    ] = Path("reports/step6-validated-labels/labels.jsonl"),
    seed_cards: Annotated[
        Path | None,
        typer.Option(
            "--seed-cards",
            exists=True,
            readable=True,
            help="Optional source-bearing development cards for labels outside this corpus.",
        ),
    ] = None,
    dense_collection: Annotated[
        str | None,
        typer.Option(
            "--dense-collection",
            help="Verified registry collection override for this development comparison.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", help="Comparison report output path."),
    ] = Path("reports/step7-qwen-dev/comparison_run.json"),
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", help="Snapshot searched by A/B/C."),
    ] = "june-2026-full-corpus",
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Optional Hugging Face cache directory."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Torch device override: auto, cpu, mps, or cuda."),
    ] = None,
    reranker_batch_size: Annotated[
        int,
        typer.Option(
            "--reranker-batch-size",
            min=1,
            help="Reranker batch size; defaults to 1 for bounded MPS memory.",
        ),
    ] = 1,
) -> None:
    """Run local Qwen A/B/C on development scenarios with human positive seeds."""
    try:
        result = run_local_development_comparison(
            corpus,
            runtime,
            comparison,
            labels,
            output,
            dense_collection=dense_collection,
            snapshot_id=snapshot_id,
            cache_dir=cache_dir,
            device=device,
            reranker_batch_size=reranker_batch_size,
            seed_cards_path=seed_cards,
        )
    except (OSError, ValueError, RuntimeError, TypeError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"compared": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"compared": True, "output": str(output), **result}, sort_keys=True))


eval_app = typer.Typer(help="Measure saved runs against human judgment coverage.")
app.add_typer(eval_app, name="evaluation")


@eval_app.command("report")
def evaluation_report(
    runs: Annotated[
        Path,
        typer.Option(
            "--runs",
            exists=True,
            readable=True,
            help="Directory with per-scenario ranked JSONL and run_manifest.json.",
        ),
    ] = Path("reports/lexical-baseline-hydrated"),
    pool: Annotated[
        Path,
        typer.Option("--pool", exists=True, readable=True, help="Blinded pool JSONL."),
    ] = Path("reports/step4-blinded-pool/blinded_pool.jsonl"),
    labels: Annotated[
        Path,
        typer.Option("--labels", exists=True, readable=True, help="Completed §5.6 labels JSONL."),
    ] = Path("reports/step4-legacy-labels/labels.jsonl"),
    splits: Annotated[
        Path | None,
        typer.Option(
            "--splits",
            exists=True,
            readable=True,
            help="Optional scenario-aware split JSONL restricting measured tasks.",
        ),
    ] = None,
    split_name: Annotated[
        str | None,
        typer.Option(
            "--split-name",
            help="Split value to measure when --splits contains multiple values.",
        ),
    ] = None,
    pool_key: Annotated[
        Path | None,
        typer.Option(
            "--pool-key",
            exists=True,
            readable=True,
            help="Operator-only pool key JSON for context-stratum counts.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory for the metrics report JSON."),
    ] = Path("reports/step5-lexical-metrics"),
) -> None:
    """Measure a saved ranked run against human labels without fabricating judgments."""
    try:
        report = compute_run_report(
            runs,
            pool,
            labels,
            splits,
            pool_key_path=pool_key,
            split_name=split_name,
        )
        destination = write_run_report(report, output)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"reported": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"reported": True, "report": str(destination)} | report, sort_keys=True))


@report_app.command("topics")
def report_topics(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input", exists=True, readable=True, help="Review card JSONL file or directory."
        ),
    ],
    output: Annotated[Path, typer.Option("--output", help="Output report JSON path.")],
    snapshot: Annotated[str | None, typer.Option("--snapshot")] = None,
    app_name: Annotated[str | None, typer.Option("--app")] = None,
) -> None:
    """Summarize topic decisions, statuses, and observed source dates."""
    try:
        report = topic_report(input_path, snapshot_id=snapshot, app=app_name)
        destination = write_report(report, output)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"reported": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"reported": True, "report": str(destination), **report}, sort_keys=True))


@report_app.command("operations")
def report_operations(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input", exists=True, readable=True, help="Review card JSONL file or directory."
        ),
    ],
    output: Annotated[Path, typer.Option("--output", help="Output report JSON path.")],
    snapshot: Annotated[str | None, typer.Option("--snapshot")] = None,
    telemetry: Annotated[
        Path | None,
        typer.Option(
            "--telemetry",
            exists=True,
            readable=True,
            help="Optional atomically written stage telemetry manifest.",
        ),
    ] = None,
) -> None:
    """Summarize processing scope, context gaps, errors, and resource availability."""
    try:
        report = operational_report(input_path, snapshot_id=snapshot, telemetry_path=telemetry)
        destination = write_report(report, output)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        typer.echo(json.dumps({"reported": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"reported": True, "report": str(destination), **report}, sort_keys=True))


@app.command("run")
def run(
    sources: Annotated[
        Path,
        typer.Option(
            "--sources",
            exists=True,
            readable=True,
            help="Source declaration YAML (see configs/sources.example.yaml).",
        ),
    ],
    rules: Annotated[
        Path,
        typer.Option("--rules", exists=True, readable=True),
    ] = Path("configs/discovery.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Fresh run output directory."),
    ] = Path("runs/discovery-review"),
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", help="Stable identifier for this bounded selection."),
    ] = "discovery-selection-v1",
    target: Annotated[
        int,
        typer.Option("--target", min=1, help="Maximum selected records retained."),
    ] = 30_000,
    max_records: Annotated[
        int,
        typer.Option("--max-records", min=1, help="Maximum source records scanned."),
    ] = 1_000_000,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Stable selection priority seed."),
    ] = 20_260_907,
    queue_limit: Annotated[
        int,
        typer.Option("--queue-limit", min=1, help="Maximum cards queued for review."),
    ] = 200,
    runtime: Annotated[
        Path,
        typer.Option("--runtime", exists=True, readable=True, help="Runtime YAML configuration."),
    ] = Path("configs/runtime.yaml"),
    tombstones: Annotated[
        Path | None,
        typer.Option(
            "--tombstones",
            exists=True,
            readable=True,
            help="Optional JSONL ledger of invalidated messages.",
        ),
    ] = None,
) -> None:
    """Run sources register, discovery, cards, queue, and worksheet in one pass."""
    try:
        runtime_config = load_runtime_config(runtime)
        result = run_discovery_review(
            sources_config=sources,
            rules_path=rules,
            output_directory=output,
            snapshot_id=snapshot_id,
            target=target,
            max_records=max_records,
            seed=seed,
            queue_limit=queue_limit,
            runtime_limits={
                "max_staging_bytes": runtime_config.ingestion.max_staging_bytes,
                "minimum_free_disk_bytes": runtime_config.ingestion.minimum_free_disk_bytes,
                "max_process_rss_bytes": runtime_config.ingestion.max_process_rss_bytes,
            },
            tombstones_path=tombstones,
        )
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        typer.echo(json.dumps({"ran": False, "error": str(error)}), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True))


@app.command()
def demo(
    synthetic: Annotated[
        bool,
        typer.Option("--synthetic", help="Run only generated, explicitly synthetic source data."),
    ] = False,
    network: Annotated[
        str,
        typer.Option("--network", help="Must remain off for the first lexical milestone."),
    ] = "off",
    output: Annotated[
        Path, typer.Option("--output", help="Directory for deterministic review exports.")
    ] = Path("reports/demo"),
) -> None:
    """Run the first end-to-end lexical demonstration."""
    if not synthetic:
        raise typer.BadParameter("--synthetic is required; real sources are not accepted by demo")
    if network != "off":
        raise typer.BadParameter("--network must be off")
    typer.echo(json.dumps(run_synthetic_demo(output), sort_keys=True))


@tombstones_app.command("inventory")
def tombstones_inventory(
    outbox: Annotated[
        Path,
        typer.Option(
            "--outbox",
            help="Durable tombstone outbox SQLite database.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Purge inventory JSON output path."),
    ],
) -> None:
    """Project outbox rows into a read-only purge inventory without deleting anything."""
    try:
        result = build_purge_inventory(outbox=outbox, output=output)
        write_purge_inventory(output, result)
    except (OSError, ValueError, json.JSONDecodeError, sqlite3.Error) as error:
        typer.echo(json.dumps({"inventoried": False, "error": str(error)}), err=True)
        raise typer.Exit(code=3) from error
    typer.echo(json.dumps({"inventoried": True} | result["outbox"], sort_keys=True))


@tombstones_app.command("replay")
def tombstones_replay(
    outbox: Annotated[
        Path,
        typer.Option(
            "--outbox",
            help="Durable tombstone outbox SQLite database.",
        ),
    ],
    identity: Annotated[
        str,
        typer.Option("--identity", help="Canonical outbox row identity hash."),
    ],
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", min=1, help="Non-empty expected snapshot scope."),
    ],
    sqlite_input: Annotated[
        Path,
        typer.Option(
            "--sqlite-input",
            help="Existing SQLite corpus input; it is never modified.",
        ),
    ],
    sqlite_output: Annotated[
        Path,
        typer.Option(
            "--sqlite-output",
            help="Explicit SQLite derivative output path; must differ from input.",
        ),
    ],
    sqlite_manifest: Annotated[
        Path | None,
        typer.Option(
            "--sqlite-manifest",
            help="Optional explicit manifest JSON output path.",
        ),
    ] = None,
) -> None:
    """Replay one outbox row into a local SQLite derivative without live Qdrant."""
    from .operations import local_sqlite_operator, validate_replay_targets
    from .operations.outbox import TombstoneOutbox

    try:
        if not outbox.is_file():
            raise ValueError(f"outbox database does not exist: {outbox}")
        store = TombstoneOutbox(outbox)
        validate_replay_targets(
            store,
            identity,
            expected_snapshot_id=snapshot_id,
            sqlite_input=sqlite_input,
            sqlite_output=sqlite_output,
            sqlite_manifest=sqlite_manifest,
        )
        operator = local_sqlite_operator(
            store,
            sqlite_input=sqlite_input,
            sqlite_output=sqlite_output,
            sqlite_manifest=sqlite_manifest,
        )
        result = operator.replay(identity)
    except (OSError, ValueError, KeyError, RuntimeError, PermissionError, sqlite3.Error) as error:
        typer.echo(json.dumps({"replayed": False, "error": str(error)}), err=True)
        raise typer.Exit(code=3) from error
    typer.echo(
        json.dumps(
            {
                "replayed": True,
                "identity": identity,
                "row_status": result["status"],
                "snapshot_id": snapshot_id,
                "sqlite_result": result.get("backend_results"),
            },
            sort_keys=True,
        )
    )


@tombstones_app.command("operate")
def tombstones_operate(
    outbox: Annotated[
        Path,
        typer.Option(
            "--outbox",
            help="Durable tombstone outbox SQLite database; created when missing.",
        ),
    ],
    ledger: Annotated[
        Path,
        typer.Option(
            "--ledger",
            help="Operator-supplied JSONL tombstone ledger; must exist.",
        ),
    ],
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", min=1, help="Non-empty expected snapshot scope."),
    ],
    sqlite_input: Annotated[
        Path,
        typer.Option(
            "--sqlite-input",
            help="Existing SQLite corpus input; it is never modified.",
        ),
    ],
    sqlite_output: Annotated[
        Path,
        typer.Option(
            "--sqlite-output",
            help="Explicit SQLite derivative output path; must differ from input.",
        ),
    ],
    reconciliation: Annotated[
        Path,
        typer.Option(
            "--reconciliation",
            help="Deterministic reconciliation manifest JSON output path.",
        ),
    ],
    sqlite_manifest: Annotated[
        Path | None,
        typer.Option(
            "--sqlite-manifest",
            help="Optional explicit propagation manifest JSON output path.",
        ),
    ] = None,
    dense_mode: Annotated[
        str,
        typer.Option(
            "--dense-mode",
            help="Dense boundary mode: disabled (fail closed) or loopback.",
        ),
    ] = "disabled",
    dense_url: Annotated[
        str | None,
        typer.Option("--dense-url", help="Loopback Qdrant HTTP endpoint."),
    ] = None,
    dense_collection: Annotated[
        str | None,
        typer.Option("--dense-collection", help="Loopback Qdrant collection name."),
    ] = None,
    dense_model_id: Annotated[
        str | None,
        typer.Option("--dense-model-id", help="Embedding model identity."),
    ] = None,
    dense_revision: Annotated[
        str | None,
        typer.Option("--dense-revision", help="Resolved embedding model revision."),
    ] = None,
    dense_dimension: Annotated[
        int | None,
        typer.Option("--dense-dimension", min=1, help="Embedding dimension."),
    ] = None,
    dense_query_instruction: Annotated[
        str | None,
        typer.Option("--dense-query-instruction", help="Embedding query instruction."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the reconciliation manifest as JSON."),
    ] = False,
) -> None:
    """Run the full configured tombstone operator flow on one ledger."""
    from .operations import run_configured_operation
    from .operations.operator import OperatorBoundaryError, dense_scope_from_flags

    try:
        dense = dense_scope_from_flags(
            dense_mode=dense_mode,
            dense_url=dense_url,
            dense_collection=dense_collection,
            dense_model_id=dense_model_id,
            dense_revision=dense_revision,
            dense_dimension=dense_dimension,
            dense_query_instruction=dense_query_instruction,
        )
        from .operations.operator import OperatorScope

        scope = OperatorScope(
            outbox_path=outbox,
            ledger_path=ledger,
            snapshot_id=snapshot_id,
            sqlite_input=sqlite_input,
            sqlite_output=sqlite_output,
            sqlite_manifest=sqlite_manifest,
            reconciliation_path=reconciliation,
            dense=dense,
        )
        manifest = run_configured_operation(scope)
    except (OSError, ValueError, KeyError, RuntimeError, PermissionError, sqlite3.Error) as error:
        if isinstance(error, OperatorBoundaryError):
            typer.echo(
                json.dumps({"operated": False, "error": str(error)}),
                err=True,
            )
            raise typer.Exit(code=4) from error
        typer.echo(json.dumps({"operated": False, "error": str(error)}), err=True)
        raise typer.Exit(code=3) from error
    if json_output:
        typer.echo(json.dumps(manifest, sort_keys=True, indent=2))
        return
    typer.echo(
        json.dumps(
            {
                "operated": True,
                "status": manifest["status"],
                "identity": manifest["outbox"]["identity"],
                "reconciliation": manifest["scope"]["reconciliation"],
            },
            sort_keys=True,
        )
    )


@tombstones_app.command("purge")
def tombstones_purge(
    artifact_root: Annotated[
        Path,
        typer.Option(
            "--artifact-root",
            help="Directory holding the published SQLite derivative and its manifest.",
        ),
    ],
    outbox: Annotated[
        Path,
        typer.Option(
            "--outbox",
            help="Durable tombstone outbox SQLite database; created when missing.",
        ),
    ],
    ledger: Annotated[
        Path,
        typer.Option(
            "--ledger",
            help="Operator-supplied JSONL tombstone ledger; must exist.",
        ),
    ],
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", min=1, help="Non-empty expected snapshot scope."),
    ],
    backup_dir: Annotated[
        Path,
        typer.Option(
            "--backup-dir",
            help="Directory for the content-addressed backup; must be outside the artifact root.",
        ),
    ],
    reconciliation: Annotated[
        Path,
        typer.Option(
            "--reconciliation",
            help="Deterministic reconciliation manifest JSON output path.",
        ),
    ],
    artifact_db: Annotated[
        Path | None,
        typer.Option(
            "--artifact-db",
            help="Explicit published db inside the artifact root; default is its only .db file.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the reconciliation manifest as JSON."),
    ] = False,
) -> None:
    """Destructively purge tombstoned identities from a managed SQLite artifact.

    Backs up and verifies the current generation, stages a scrubbed new
    generation through the durable outbox flow, publishes it atomically, and
    records a hash-bound claim. Original archives are never touched.
    """
    from .operations import run_artifact_purge
    from .operations.artifact_purge import ArtifactPurgeScope
    from .operations.operator import OperatorBoundaryError

    scope = ArtifactPurgeScope(
        artifact_root=artifact_root,
        outbox_path=outbox,
        ledger_path=ledger,
        snapshot_id=snapshot_id,
        backup_dir=backup_dir,
        reconciliation_path=reconciliation,
        artifact_db=artifact_db,
    )
    try:
        manifest = run_artifact_purge(scope)
    except (OSError, ValueError, KeyError, RuntimeError, PermissionError, sqlite3.Error) as error:
        if isinstance(error, OperatorBoundaryError):
            typer.echo(
                json.dumps({"purged": False, "destructive": True, "error": str(error)}),
                err=True,
            )
            raise typer.Exit(code=4) from error
        typer.echo(
            json.dumps({"purged": False, "destructive": True, "error": str(error)}),
            err=True,
        )
        raise typer.Exit(code=3) from error
    if json_output:
        typer.echo(json.dumps(manifest, sort_keys=True, indent=2))
        return
    typer.echo(
        json.dumps(
            {
                "purged": True,
                "destructive": True,
                "status": manifest["status"],
                "idempotent": manifest["idempotent"],
                "identity": manifest["outbox"]["identity"],
                "reconciliation": manifest["scope"]["reconciliation"],
            },
            sort_keys=True,
        )
    )


@tombstones_app.command("preflight")
def tombstones_preflight(
    artifact_root: Annotated[
        Path | None,
        typer.Option("--artifact-root", help="Managed artifact root with its sidecar manifest."),
    ] = None,
    ledger: Annotated[
        Path | None,
        typer.Option("--ledger", help="Operator tombstone ledger SQLite database."),
    ] = None,
    outbox: Annotated[
        Path | None,
        typer.Option("--outbox", help="Durable tombstone outbox SQLite database."),
    ] = None,
    dense_manifest: Annotated[
        list[Path] | None,
        typer.Option(
            "--dense-manifest",
            help=(
                "Dense index manifest binding one collection probe; a manifest "
                "without a 'collection' key reports a mapping gap."
            ),
        ),
    ] = None,
    collection: Annotated[
        list[str] | None,
        typer.Option(
            "--collection",
            help="Explicit dense collection name to probe without a manifest.",
        ),
    ] = None,
    qdrant_base_url: Annotated[
        str | None,
        typer.Option(
            "--qdrant-base-url",
            help="Loopback Qdrant HTTP endpoint for dense collection probes.",
        ),
    ] = None,
    dense_registry: Annotated[
        Path | None, typer.Option("--dense-registry", exists=True, readable=True)
    ] = None,
    disk_path: Annotated[
        Path | None,
        typer.Option(
            "--disk-path",
            help="Volume measured for free disk when no --artifact-root is given.",
        ),
    ] = None,
    minimum_free_disk_bytes: Annotated[
        int,
        typer.Option(
            "--minimum-free-disk-bytes",
            help="Minimum free bytes required on the measured volume.",
        ),
    ] = DEFAULT_MINIMUM_FREE_DISK_BYTES,
    required: Annotated[
        list[str] | None,
        typer.Option(
            "--required",
            help="Check name (artifact|dense_collections|disk|ledger|outbox|paths) "
            "that must pass for readiness.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Atomically written preflight report JSON path."),
    ] = None,
) -> None:
    """Run a read-only preflight for a tombstone purge without deleting anything."""
    if disk_path is not None and not Path(disk_path).expanduser().is_dir():
        typer.echo(
            json.dumps({"ready": False, "error": f"disk path does not exist: {disk_path}"}),
            err=True,
        )
        raise typer.Exit(code=3)
    try:
        with _chdir(disk_path) if disk_path is not None else contextlib.nullcontext():
            report = run_preflight(
                PreflightConfig(
                    artifact_root=(artifact_root.expanduser().resolve() if artifact_root else None),
                    ledger_path=ledger.expanduser().resolve() if ledger else None,
                    outbox_path=outbox.expanduser().resolve() if outbox else None,
                    qdrant_base_url=qdrant_base_url,
                    dense_registry_path=dense_registry.expanduser().resolve()
                    if dense_registry
                    else None,
                    dense_collections=tuple(
                        DenseCollectionProbe(manifest_path=path.expanduser().resolve())
                        for path in dense_manifest or []
                    )
                    + tuple(
                        DenseCollectionProbe(collection_name=name) for name in collection or []
                    ),
                    output_path=output.expanduser().resolve() if output else None,
                    minimum_free_disk_bytes=minimum_free_disk_bytes,
                    required=tuple(required or ()),
                )
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(json.dumps({"ready": False, "error": str(error)}), err=True)
        raise typer.Exit(code=3) from error
    if output is None:
        typer.echo(json.dumps(report, sort_keys=True))
    if not report["ready"]:
        raise typer.Exit(code=3)


artifacts_app = typer.Typer(help="Package managed artifacts additively.")
app.add_typer(artifacts_app, name="artifacts")


@artifacts_app.command("package")
def artifacts_package(
    source: Annotated[
        Path,
        typer.Option("--source", help="Hydrated corpus SQLite database to package."),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="New managed artifact root; must be empty or absent."),
    ],
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", help="Snapshot identity recorded in the artifact manifest."),
    ],
    corpus_sha256: Annotated[
        str | None,
        typer.Option(
            "--corpus-sha256",
            help="Expected source database SHA-256; packaging fails on mismatch.",
        ),
    ] = None,
) -> None:
    """Publish a verified, additive managed-artifact copy without overwriting."""
    try:
        manifest = package_artifact(
            source,
            artifact_root,
            snapshot_id=snapshot_id,
            corpus_sha256=corpus_sha256,
        )
    except (ArtifactPackageError, OSError, ValueError) as error:
        typer.echo(json.dumps({"packaged": False, "error": str(error)}), err=True)
        raise typer.Exit(code=3) from error
    typer.echo(json.dumps({"packaged": True} | manifest, sort_keys=True))


def main() -> None:
    """Run the CLI with a stable program name for console and module entry points."""
    configure_logging()
    app(prog_name="reddit-search")
