---
name: reddit-search-scenario-authoring
description: Create or improve reddit-search scenario YAML, including focused opportunity definitions, FTS5 lexical queries, semantic queries, and verified product-claim mappings. Use when adding searches for a product, tuning weak retrieval, or turning product capabilities into scenarios.
---

# reddit-search scenario authoring

Create scenarios that find how people describe a need before they know which
product might solve it. Keep each scenario narrow enough to review. A scenario
is a search target, not a product binding: `app_id` and
`required_claim_ids_for_fit` are optional. Bind claims only when you maintain
a product profile and want claim-bound fit judgments.

## Gather the inputs

Read these files before drafting:

1. `configs/scenarios/` for existing coverage and version history.
2. `configs/profiles/<app_id>.yaml` — only when the scenario will be
   claim-bound — for verified claims and their constraints.
3. Nearby scenario files for current wording and formatting conventions.

For a topic-only scenario, stop after defining the need and queries: leave
`app_id` unset and `required_claim_ids_for_fit` empty. The scenario is
complete and its `product_fit` verdicts stay `not_evaluated`.

Claim-bound scenarios require the optional product-profile extension. Use
product documentation only to fill a missing profile, not to invent a claim
inside a scenario. A claim can support product fit when it has
`status: verified` and an `evidence_ref`. If no matching verified claim
exists, create and verify the profile claim first.

## Turn capabilities into user needs

Start from the problem a person would describe on Reddit. Product features are
supporting evidence, not search language.

Good scenario boundaries:

- one job, pain, constraint, or workaround per scenario;
- for claim-bound scenarios, a clear reason the verified claim could satisfy
  that need;
- enough separation from existing scenarios that a reviewer can label it
  without guessing which intent applies.

Split a draft when it joins needs that could appear in different posts. Merge it
when the only difference is a synonym or minor wording change.

Use a stable ID in the form `<namespace>.<need_slug>`. The namespace is the
`app_id` for claim-bound scenarios; topic-only scenarios may use any stable
namespace that does not collide with an existing app ID. Increase
`scenario_version` when a query or description change can alter retrieval.

## Build lexical queries

Each scenario requires three to five lexical queries. The corpus passes each
string directly to SQLite FTS5. Space-separated terms use FTS5's implicit `AND`,
so every term usually needs to appear in the indexed focus or context text.

Write short groups of words a Reddit author is likely to use:

- Prefer two to four discriminating terms.
- Cover different vocabulary for the same need, not different needs.
- Include the task or pain in every query.
- Add a constraint, workaround, or current tool when it separates useful posts
  from generic discussion.
- Use common words from the user's point of view.
- Omit the product name, marketing language, and feature names that only the
  product team would know.
- Avoid long sentences. One rare extra term can reduce a useful query to zero
  hits because terms are combined with `AND`.
- Avoid raw FTS5 operators or punctuation unless the query intentionally uses
  valid FTS5 syntax.

YAML quotes only delimit the YAML string. For example, `"invoice from phone"`
still reaches FTS5 as three terms joined by `AND`; it is not an exact phrase
search.

A useful set often includes plain task language, the current workaround, and a
constraint. Keep all of them focused on the same underlying need.

## Build semantic queries

Each scenario requires two to three semantic queries. Write them as short,
first-person statements that could plausibly appear in a post:

- express the need or frustration directly;
- include an important constraint when it changes product fit;
- vary the wording while keeping the same intent;
- omit the product name and unsupported promises.

Semantic queries guide dense retrieval. They do not relax the verified-claim
requirement for product-fit labels.

## Map verified claims (claim-bound scenarios only)

`required_claim_ids_for_fit` is optional and only meaningful when the scenario
sets `app_id`. The loader fails closed: a scenario that cites claims without an
`app_id` is rejected, because claim IDs live in an app namespace.

For claim-bound scenarios, set `required_claim_ids_for_fit` to the smallest set
of claims needed to support the scenario. Every ID must exist in the matching
product profile and must be verified with evidence. The current scenario loader
does not cross-check these IDs against profiles, so perform this check before
finishing.

Use one claim when one capability is enough. Use several only when all are
needed for the opportunity. Read each claim's constraints before mapping it; a
similar feature with the wrong platform, pricing model, or limitation is not a
match.

## Add negative-example reasons when useful

`negative_example_reasons` is optional. Add concise reasons when the scenario
has predictable false positives, such as:

- the author recommends a tool but does not need one;
- the post discusses a business process outside the supported product scope;
- the required platform or workflow conflicts with the claim constraints.

These are review cues, not extra search queries.

## Produce schema-valid YAML

A topic-only target (no product binding):

```yaml
scenarios:
  - schema_version: 1
    scenario_id: needs.private_tracking
    scenario_version: 1
    description: Find first-person needs to track information without sharing account access.
    lexical_queries:
      - "private tracker account"
      - "track without login"
      - "offline personal tracker"
    semantic_queries:
      - "I need to track this without creating or sharing an account."
      - "I want a private tracker that keeps my information under my control."
```

A claim-bound target (optional product-profile extension):

```yaml
scenarios:
  - schema_version: 1
    scenario_id: yourproduct.private_tracking
    app_id: yourproduct
    scenario_version: 1
    description: Find first-person needs to track information without sharing account access.
    lexical_queries:
      - "private tracker account"
      - "track without login"
      - "offline personal tracker"
    semantic_queries:
      - "I need to track this without creating or sharing an account."
      - "I want a private tracker that keeps my information under my control."
    required_claim_ids_for_fit:
      - yourproduct.private_tracking
    negative_example_reasons:
      - "The author is asking for team access rather than private personal tracking."
```

Schema limits:

- `app_id` is optional; when unset the scenario is topic-only.
- `required_claim_ids_for_fit` requires `app_id`; the loader rejects claim
  citations without an app binding.
- `scenario_version` is an integer of at least 1.
- `lexical_queries` has three to five non-empty strings.
- `semantic_queries` has two to three non-empty strings.
- Scenario IDs are unique across every YAML file in `configs/scenarios/`.
- Extra fields are rejected.

## Validate before finishing

Load the whole scenario directory through the production parser:

```bash
uv run python -c "from pathlib import Path; from reddit_search.config import load_scenarios; rows = load_scenarios(Path('configs/scenarios')); print(f'{len(rows)} scenarios valid')"
```

Then verify every mapped claim ID against `configs/profiles/`. If a corpus is
available, run the new scenarios into a fresh output directory and inspect hit
counts plus sample text. Good queries retrieve distinct, reviewable needs. Zero
hits suggest over-constrained wording; broad generic hits suggest missing intent
terms.

Before finishing, confirm:

- each scenario covers one need;
- query counts meet the schema;
- lexical terms use likely Reddit vocabulary;
- semantic queries sound like real first-person requests;
- for claim-bound scenarios, mapped claims exist, are verified, and fit their
  constraints; topic-only scenarios cite no claims;
- new IDs do not duplicate existing scenarios;
- changed scenarios have an appropriate version increase.
