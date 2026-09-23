# reddit-search

Turn authorized local Reddit dumps into a reviewable list of opportunities.
You describe the user needs to look for. `reddit-search` searches the archive,
prepares blinded review cards, and keeps a hash-bound record of how each
result was produced.

If you maintain a product, you can optionally bind product profiles to your
scenarios: a thread can then be marked compatible only when the verdict cites
a verified claim from the profile. Unsupported or incomplete judgments stay
visible instead of being counted as evidence.

The default runtime is local-only. It blocks Reddit requests, model downloads,
remote inference, and automatic publishing.

## Try it first

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run reddit-search doctor
uv run reddit-search config validate --config configs/runtime.yaml
uv run reddit-search profiles validate --profiles-dir configs/profiles
uv run reddit-search demo --synthetic
```

The demo creates a small synthetic archive, searches it, and writes a run
summary plus review cards under `reports/demo`. It does not read your dumps or
use the network.

The ingestion guard reserves 5 GiB of free disk by default. You can inspect the
other limits in `configs/runtime.yaml`.

## Where to get dumps

You can browse Reddit-related datasets on
[Academic Torrents](https://academictorrents.com/checkb.htm?search=reddit).
Dataset sizes, date ranges, and file layouts vary, so check the dataset page
before downloading. Use only data you are authorized to process, then point
`configs/sources.local.yaml` to the local `.zst` files.

## How the pipeline works

1. Ingest registers source files and reads them in bounded, checkpointed
   stages. Malformed rows go to quarantine.
2. Retrieve searches a deterministic SQLite FTS5 corpus. An optional dense
   backend can add local Qwen embeddings and reranking through Qdrant.
3. Review removes retrieval metadata before showing thread text in a localhost
   workspace.
4. Labels turn completed worksheets into structured judgments. When profiles
   are in use, compatible judgments must cite verified product claims.
5. Evaluate scores saved runs against reviewed labels and preserves the input
   hashes behind each report.

## Configure your data

### Declare source dumps

Edit `configs/sources.local.yaml`. Every source needs an explicit usage scope so
the registry records how you are allowed to use it.

```yaml
schema_version: 1
sources:
  - source_id: local-2026-01-submissions
    path: /path/to/reddit/submissions/RS_2026-01.zst
    source_kind: submission
    declared_month: 2026-01
    source_role: discovery
    usage_scope: authorized_local_research
```

Registering a source checks its metadata and path without reading the archive:

```bash
uv run reddit-search sources register --file configs/sources.local.yaml
```

Use `--additive` when adding a new month to an existing registry. Additive
registration keeps the existing rows and rejects changes to their identities.

### Describe the needs to search for

Scenario files live in `configs/scenarios/`. Each scenario covers one user need
and contains three to five lexical queries plus two to three semantic queries.

A scenario is a search target, not a product binding: `app_id` is optional.

```yaml
scenarios:
  - schema_version: 1
    scenario_id: needs.private_tracking
    scenario_version: 1
    description: Find first-person requests for a simple way to handle this need.
    lexical_queries:
      - "simple need tracker"
      - "need tracker without account"
      - "private need tracking"
    semantic_queries:
      - "I need a simple way to track this without creating an account."
      - "I want to keep this information private on my device."
```

If you maintain a product and want claim-bound fit judgments, add `app_id` and
`required_claim_ids_for_fit`; the IDs must exist as verified claims in the
matching profile (see the next section). Scenarios without an app binding stay
topic-only, and their `product_fit` verdicts remain `not_evaluated`.

Keep scenario IDs stable. Increase `scenario_version` when a query change could
alter which threads are retrieved.

If you use a coding agent, the
[scenario-authoring skill](.agents/skills/reddit-search-scenario-authoring/SKILL.md)
can turn verified product claims and user needs into schema-valid scenarios.

### Record product claims (optional extension)

Product profiles live in `configs/profiles/`. They are only needed for
claim-bound fit judgments; topic-only runs do not require them. A verified
claim needs an `evidence_ref`. Constraints tell the reviewer exactly where the
claim stops.

```yaml
app_id: yourproduct
profile_version: 1
verification_status: verified
verification_owner: "name or team responsible for verification"
capabilities:
  - claim_id: yourproduct.verified_claim
    status: verified
    evidence_ref: https://example.com/product-evidence
    version_scope: "1.0"
    constraints:
      - "Describe what this claim includes and excludes."
```

Validate profiles before a review round:

```bash
uv run reddit-search profiles validate --profiles-dir configs/profiles
```

## Choose a review path

The CLI supports two related workflows. Pick the one that matches the question
you are trying to answer.

### Review discovery matches

Use this path to inspect the bounded set selected by
`configs/discovery.yaml`.

```bash
uv run reddit-search ingest discover

uv run reddit-search review cards \
  --selection data/staging/discovery-selection.jsonl.zst \
  --snapshot-id local-discovery-v1 \
  --output reports/discovery-review

uv run reddit-search review queue \
  --cards reports/discovery-review/review_cards.jsonl \
  --output reports/review-queue

uv run reddit-search review worksheet \
  --cards reports/review-queue/review_cards.jsonl \
  --output reports/review-worksheet

uv run reddit-search review serve \
  --cards reports/review-queue/review_cards.jsonl \
  --worksheet reports/review-worksheet/review_worksheet.jsonl
```

`review serve` listens on `127.0.0.1:8765` by default. Stop it with
`Ctrl-C` when the review is complete.

### Build ranked runs and an evaluation pool

Use this path when you want to measure retrieval quality. The corpus and search
commands require explicit input identities:

```bash
uv run reddit-search corpus build \
  --selection data/staging/discovery-selection.jsonl.zst \
  --output data/sqlite/reddit-search.db \
  --snapshot-id local-snapshot-v1

uv run reddit-search corpus search \
  --corpus data/sqlite/reddit-search.db \
  --snapshot-id local-snapshot-v1 \
  --output reports/lexical-run
```

`dataset pool` combines ranked runs with rejected controls from hydration. Run
`uv run reddit-search ingest hydrate --help` and
`uv run reddit-search dataset pool --help` for the required paths. The resulting
`blinded_pool.jsonl` contains no retrieval scores or variant IDs.

The pool-to-review-card converter is currently a Python API:
`reddit_search.review.blinded_pool.convert_blinded_pool`. There is no CLI command
for this bridge yet. After review, use `labels collect`, `labels import`, and
`evaluation report`; each command's `--help` lists the pool, worksheet, and run
paths it requires.

## Optional dense retrieval

Install the semantic dependencies and start local Qdrant:

```bash
uv sync --group dev --group semantic
docker compose --profile semantic up -d qdrant
```

Then inspect the dense commands:

```bash
uv run reddit-search retrieval index-dense --help
uv run reddit-search retrieval compare-dev --help
```

The model revisions are pinned, and runtime loading uses
`local_files_only=True`. Put the model files in the local Hugging Face cache
before running dense retrieval. Device selection uses CUDA, Apple MPS, or CPU,
in that order when `device: auto`.

Qdrant listens on `127.0.0.1:6333` and stores data under `data/qdrant`.

## Outputs and failure behavior

Most state-changing commands return JSON on stdout. `doctor` prints a readable
report unless you add `--json`. Commands fail instead of guessing when an input
is missing, a hash changes, a budget is exceeded, or a compatible judgment cites
an unverified claim.

Generated data belongs under `data/`. Published run artifacts belong under
`reports/`. Start a new output directory for a new run so earlier evidence
remains reproducible.

## Repository layout

```text
configs/       runtime limits, sources, discovery rules, scenarios, profiles
data/          SQLite corpora, Qdrant storage, staging files (generated)
docs/          implementation status and decision log
reports/       hash-bound run artifacts (generated)
src/reddit_search/
  cli.py       Typer entry point and command groups
  ingest/      source registration, discovery, hydration, shard readers
  corpus/      lexical corpus building and search
  retrieval/   dense indexing, comparison, reranking
  review/      blinded cards, queues, worksheets, localhost workspace
  evaluation/  pools, labels, metrics, archive audits
  operations/  tombstone propagation and artifact purge
tests/         offline unit and acceptance tests
```

## Development

```bash
uv run pytest
uv run ruff check .
```

The test suite is deterministic and network-free. Ingestion tests still honor
the 5 GiB free-disk reserve, so they stop with `BudgetError` when the workspace
does not have enough room.

## License

MIT — see [LICENSE](LICENSE).
