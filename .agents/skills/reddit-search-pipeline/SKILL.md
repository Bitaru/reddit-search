---
name: reddit-search-pipeline
description: Run and debug the local reddit-search pipeline. Covers source registration, search, blinded review, claim-bound labels, evaluation, and fail-closed recovery.
---

# reddit-search pipeline

Use this runbook when changing the pipeline or helping someone run it. Keep the
workflow reproducible: explicit source identities, fresh output directories,
verified product claims, and hashes that still point to the files that produced
them.

## Start with the right workflow

There are three useful entry points.

### Check that the project works

```bash
uv sync --group dev
uv run reddit-search doctor
uv run reddit-search config validate --config configs/runtime.yaml
uv run reddit-search profiles validate --profiles-dir configs/profiles
uv run reddit-search demo --synthetic
```

The `--synthetic` flag is required. This path is local, quick, and safe to run
without access to an archive.

### Review discovery matches

Use this path to inspect the bounded selection created by discovery rules:

```bash
uv run reddit-search sources register --file configs/sources.local.yaml
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

`review serve` binds to loopback only. Stop it with `Ctrl-C`.

### Measure ranked retrieval

Use this path for lexical or dense evaluation:

```text
sources register
ingest discover
ingest hydrate
corpus build
corpus search
dataset pool
convert blinded pool to review cards
review worksheet
review serve
labels collect
labels import
evaluation report
```

Important boundary: `review cards` reads a discovery selection shard. It does
not convert `blinded_pool.jsonl`. For an evaluation pool, call
`reddit_search.review.blinded_pool.convert_blinded_pool` from Python. There is no
CLI command for this bridge yet.

Use each command's `--help` before assembling a run. Several stages require
explicit paths, and retaining those paths is part of the provenance contract.

## Write valid source declarations

`configs/sources.local.yaml` must name an existing, authorized file:

```yaml
schema_version: 1
sources:
  - source_id: local-2026-01-submissions
    path: /path/to/RS_2026-01.zst
    source_kind: submission
    declared_month: 2026-01
    source_role: discovery
    usage_scope: authorized_local_research
```

Use `sources register --additive` for a new month. Additive registration keeps
the existing rows and rejects changes to their identities. Never edit the
generated registry to get around that check.

## Write valid scenarios

Read `.agents/skills/reddit-search-scenario-authoring/SKILL.md` before creating or
changing scenario YAML. It covers opportunity boundaries, FTS5 query behavior,
semantic queries, verified claim mappings, versioning, and validation.

The pipeline contract still requires three to five `lexical_queries`, two to
three `semantic_queries`, a stable scenario ID, and a version of at least 1.
`app_id` and `required_claim_ids_for_fit` are optional; citing claims requires
an `app_id`, and claim-bound fit judgments additionally require verified
product profiles.

## Write useful product profiles

Place one profile per app in `configs/profiles/`:

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
      - "Describe what the claim includes and excludes."
```

A claim with `status: verified` requires `evidence_ref`. Constraints are not
schema-required, but a reviewer needs them to know the claim's boundary.
Increase `profile_version` when the set or meaning of claims changes.

Run this before pooling or importing labels:

```bash
uv run reddit-search profiles validate --profiles-dir configs/profiles
```

## Preserve review blinding

Reviewer-facing rows may contain the thread text, scenario identity, profile
identity, and review fields. They must not expose retrieval score, rank,
variant, or hidden stratum metadata.

`dataset pool` removes retrieval metadata. The blinded-pool converter then
creates review cards. Do not pass evaluation pools through `review cards`; that
command is for discovery selection shards.

The worksheet stores judgments by reference and does not duplicate card
evidence. Keep the cards, worksheet, and their manifests together.

## Apply product-fit rules

Allowed `product_fit` values:

- `compatible`
- `incompatible`
- `needs_clarification`
- `not_evaluated`

A compatible row must cite at least one verified claim ID in
`supported_claim_ids`. Every other verdict must leave that list empty.
`topic_fit` and `product_fit` are separate judgments; do not derive one from
the other.

For machine-assisted screening:

1. Compare the card's focus text with the verified profile claims.
2. Record `machine-assessed` in the round report.
3. After an operator reviews all compatible rows, record
   `machine-screened-operator-confirmed`.
4. Keep precision and recall null until a held-out set has real human labels.

## Preserve provenance

- Never fabricate an input hash or source identity.
- Never edit a published manifest after its artifact has been used.
- Write each run to a fresh output directory.
- Publish a superseding artifact when an earlier result is wrong. Name the
  superseded artifact and preserve its SHA-256.
- Record destructive or superseding actions in
  `docs/implementation_status.md`.
- Keep generated data under `data/` and published artifacts under `reports/`.

## Respect runtime limits

The default runtime blocks model downloads, remote inference, Reddit requests,
and automatic publishing. Ingestion also reserves 5 GiB of free disk and limits
process RSS, staging bytes, and record counts.

Do not lower a guard to make a run pass. Free disk space, reduce the declared
workload, or resume from the checkpoint as appropriate.

## Dense retrieval stays local

Dense retrieval is optional:

```bash
uv sync --group dev --group semantic
docker compose --profile semantic up -d qdrant
uv run reddit-search retrieval index-dense --help
uv run reddit-search retrieval compare-dev --help
```

Qwen model revisions are pinned and loaded with `local_files_only=True`.
Provision the model files before the run. With `device: auto`, the adapter
chooses CUDA, Apple MPS, or CPU.

## Diagnose common failures

| Message or symptom | Meaning | Action |
|---|---|---|
| `--synthetic is required` | The demo was started without its safety flag | Run `reddit-search demo --synthetic` |
| `BudgetError: insufficient free disk` | The workspace is below the 5 GiB reserve | Free space; keep the guard enabled |
| `supported_claim_ids requires compatible product_fit` | A non-compatible row cites claims | Clear the IDs or correct the verdict |
| `compatible product claim is not verified` | A cited ID is absent or unverified | Fix the ID or add evidence to the profile |
| `missing reranker text for candidates` | Dense candidates do not match the corpus text source | Check the collection, snapshot, and corpus identities |
| Hash mismatch | An input changed after its manifest was created | Re-run the producing stage into a new directory |
| Incomplete stage report | A configured cap stopped the stage | Inspect the report; resume when that stage supports checkpoints |

## Verify changes

For documentation changes, run every command or parser needed to prove the
examples are valid.

For pipeline code changes:

```bash
uv run pytest
uv run ruff check .
```

Also run the changed CLI path on synthetic or bounded local data. Tests alone do
not prove that a multi-stage command flow works.
