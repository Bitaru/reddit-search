# Implementation status

## 2026-09-30 — Offline preparation and pilot acceptance

Offline candidate-to-draft preparation is complete: the provisional candidate,
accepted-as-is draft, evidence and valid private handoff were prepared. The
previous **4/5 (80%)** checkpoint incorrectly treated live-observation evidence
as an original offline completion gate. It was not required by the original
acceptance definition under Implementation order and acceptance in
`docs/archive-engagement-plan.md`.
This supersedes that gate without reducing scope or changing the criteria.
Original offline acceptance is **5/5 milestones**. The private outcome audit
verified the named August pilot rows, no-original-draft evidence and existing
measurement/handoff proof. The prior 80% checkpoint remains historical; live
observation was not an original offline completion requirement.
The accepted model is pinned Qwen2.5-7B-Instruct-GGUF Q4_K_M, SHA-256
`1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49`,
with qualification-v2 and draft-v3. A separately provisioned 14B model has
not been exercised or accepted. Native tokenizer diagnostics checked that
escaped task JSON preserved source strings without introducing extra native
conversation roles.

The production CLI saved a genuine draft for `t3_1vpyqeq`, record SHA-256
`1cd1c6f8744893e3650e8093e25b147ffc874f33349ab1f5d94ce62bd016d7d1`. The
original `not_invoked` abstention remains unchanged in the two-record machine
history. The operator accepted this exact draft as-is: accepted=true,
edits_needed=false, text_modified=false. Since no text changed, no `human_edit`
record was created.

The approval v2 also records the user's earlier “Checked — suitable” report.
The latest report says `t3_1vpyqeq` is deleted and explicitly says to skip
posting. Respect both reports as historical user statements; the later
deletion does not rewrite the archive-fit assessment or approval of the
unchanged draft text. Neither report is a formal live-check record or a
posting authorization. The manual-check journal remains empty.

A valid private handoff packet was exported to
`reports/engagement/mieru-august-index-followup-20260927/handoff-20260930.json`,
SHA-256
`ab627cae79ad58168095ca1d0d4bf1192fa6954a7983c53a3e6b3d71bfac35ed`. It is
bound to the accepted draft and keeps external transmission and posting
disabled; its manual-check form begins `live_unverified`. No posting occurred.

The bounded local artifact check is recorded in
`reports/engagement/mieru-august-index-followup-20260927/handoff-validation-20260930-v2.json`,
SHA-256
`cfa87b8692bfaceaee66ef15131e68caf4ced8a74e78472c9248cf8b72b283b2`. It
verified that handoff, approval, review, context, draft and journal hashes
agree; the handoff reply exactly matches the latest draft and passes human-edit
validation with zero findings. The packet retains local-only/no-posting
permissions and a blank live-unverified form. The check used local artifacts
only; no Reddit or network request occurred.

The measurement checkpoint is
`reports/engagement/mieru-august-index-followup-20260927/human-handoff-checkpoint-20260930-v2.json`,
SHA-256
`b913cadba1a71bd6c019d75d988969cb027fd4a2539e51386435a4cf3812106f`. V2
supersedes only the invalid fabricated midnight placeholder `recorded_at`
metadata in the first validation/checkpoint receipts. Those original files are
preserved unchanged; the source operator-approval and handoff files are
unchanged too. Corrected `recorded_at` values identify receipt creation, not
the user's live-observation time.

The canonical machine summary `pilot-report-20260930-v5.json`, SHA-256
`1691e1fc72f4569769ec365d23cf7fefae4080bce6c861df6f646afdf3fb9467`, remains
current for its input journals, which have not changed; its schema has no
acceptance field. Its zero human-edit count does not mean the draft was not
accepted. Current-session acceptance is supplemented by approval v2 and
checkpoint v2: one reviewed/accepted draft, zero rejected, zero edits needed,
and zero actual edits. The `true_positive` adjudication still has denominator
1 and is not representative. Do not revise v5 or create a new report unless
its source journals change.

The old local-model acceptance receipt
`reports/engagement/mieru-august-index-followup-20260927/local-model-acceptance-20260930.json`,
SHA-256
`d8c6474311354e32c9f1399e00c8711d7518a1db450609510384553713282a96`, is a
pre-human-review snapshot, not the latest acceptance evidence. Preserve it
unchanged as history.

### Optional later live actions

No further user input is needed for offline preparation. A live check or
posting is not an unfinished assistant task; any later action for another
target is optional, manual, and the user's responsibility. No posting occurred.
The user's view that factual alternatives or feature information might be
allowed is cautious opinion, not authorization; factual product content does
not exempt self-promotion from current community rules.

The `ManualCheck` API and journal are unchanged. A formal
`human_checked_replyable` record still requires a UTC `observed_at`, permalink,
nonblank observation, rule evidence and final decision (`post`, `do_not_post`,
or `defer`). `/api/manual-check` omits `observed_at` and stamps save time in
UTC, which is not the live-check time; `append_manual_check` accepts the
human-supplied UTC observation time. Do not synthesize a formal row or mark
the deleted target `human_checked_replyable`.

The private outcome receipt
`reports/engagement/mieru-august-index-followup-20260927/operator-outcome-2026-09-30-v4.json`,
SHA-256
`14e58ed178e0445c9d4f7662027d785a326810d9af60e35019d0544673947f9c`,
supplements the unchanged v5 report with the original August pilot results
and the user-reported deletion/skip-posting outcome. The original pilot's
latest rows are `t3_1vi5fq7` (`reject`), `t3_1vn7ndz` (`reject`, Sumlyst
tester request), and `t3_1vsou2f` targeting `t1_p5kxkph` (`needs_context`);
its original draft history is empty. That original summary has no explicit
false-positive adjudication or draft acceptance. Current approval v2, draft
history, canonical v5 and the handoff supply the acceptance, edits, explicit
FP judgment and retained abstention evidence. The `true_positive` denominator
remains 1 and non-representative.

Finalized outcome v1, v2 and v3 are preserved. V3 corrects v2's embedded
acceptance-text digest and the original/current measurement attribution.
V4 corrects the tester rejection's reason-code attribution from the actual
latest review: `no_useful_reply` and `context_incomplete`. The separate
needs-context row carries `self_promotion_or_tester_request`; it is not the
Sumlyst rejection. Candidate metadata confirms the Sumlyst target identity.
V2 records the initial v1 drafting change and does not claim that the initial
draft body was separately preserved. The original acceptance-section text is
embedded and hashed, so later documentation edits do not break its evidence.

The actual `engagement pilot-report` CLI reproduced canonical v5 byte-for-byte
from the current four journals. Its temporary output was removed; the
canonical summary and source journals remain unchanged.

Previously exercised verification: **601 tests passed, 2 opt-in live-Qdrant
tests skipped; ruff passed**. The localhost UI rendered the current machine
draft, model, prompt versions and exact affiliation with no page errors. No
new code or runtime behavior was changed for this documentation update.
Runtime policy remains unchanged: no remote inference, Reddit request, posting
or repeated monthly archive scan. The sections below are historical
checkpoints, not the current state.

## Historical checkpoint — 2026-09-30 owner decisions before provisioning

The latest owner record,
`reports/engagement/mieru-august-index-followup-20260927/operator-approval-20260930.json`
(1605 bytes, mode 600, SHA-256
`2dd7c26f526e287c3c27d9b568c94497c676a842cf8eed092c545c243a8008f7`),
records Bitaru's approval of the exact affiliation “I’m the developer of
Mieru.”, approval of criterion `engagement-fp-v1`, and explicit
`true_positive` adjudication selection for review
`eddebe8e1a1686390397edf7ebda18242522f04af768d573f2402e0c3eefac07`.
The affiliation is approved for use on a draft; it is not a draft or generation.

The selected generator path is an installed llama.cpp executable plus GGUF
weights, but no executable path, model path or model version was supplied, and
downloads and remote inference remain unauthorized. A retry with the approved
disclosure exited 2 because those explicit local paths are required; the model
was not invoked and the draft-history journal stayed unchanged. The earlier
missing-affiliation abstention remains in history as a no-text abstention, not
a human decline. There is still no generated draft, human edit, manual/live
check or handoff.
The browser smoke also rendered an unfinished synthetic nested conversation at
depths 0, 1, 2 and 1 with no page errors. This exercises workspace rendering
only; it is not evidence of real-model generation or semantic verification.

The explicit human `true_positive` adjudication is recorded in
`false-positive-adjudications.jsonl` (549 bytes, mode 600, SHA-256
`c4ebe4123876ba74daeadf338a3fd3256279521b362e0d21b5a6507c15d6a0c8`),
bound to that exact review hash. The latest private summary,
`reports/engagement/mieru-august-index-followup-20260927/pilot-report-20260930-v4.json`
(1704 bytes, mode 600, SHA-256
`271777cc0f597409c8e2f468eed1eb600bae03b770e79c6d196abf9947aec644`),
records one adjudicated and zero unadjudicated identities, one
`true_positive`, zero `false_positive`, and false-positive rate 0.0 with
denominator 1. It still records one abstention, zero drafts, zero human edits
and zero manual checks. The owner-approved label is an explicit judgment, not
an inference from the review status or affiliation approval. This single
adjudication is not a representative quality rate, precision/recall result or
live verification. Original strict acceptance remains 3/5.

## Historical checkpoint — 2026-09-30 execution state before owner decisions

Checkpoint `reports/engagement/mieru-august-index-followup-20260927/execution-checkpoint-20260930.json`
was recorded at `2026-09-30T14:25:59.362244+00:00`. The newest real-candidate
record is a deterministic abstention: outcome `abstain`, reason
`a truthful, nonblank operator affiliation is required`, no generated text,
`model_version: not_invoked`, and no human decline recorded. Its SHA-256 is
`39f077e9e4aca1fe45af3efbf1007e783b048be1ae3fa4ff81cbedb5fdbab23a`.
This is not a real generation, semantic safety proof, live check or handoff.
Strict acceptance remains 3/5 original milestones.
This record belongs to app `mieru`, thread and reply target `t3_1vpyqeq`;
the review remains `provisional_candidate`, with seven retained messages and
workspace status `live_unverified`. These are archive/workspace facts only.

The v3 private summary is
`reports/engagement/mieru-august-index-followup-20260927/pilot-report-20260930-v3.json`
(1620 bytes, mode 600, SHA-256
`c8f25b66b68f354a9a8f1a88f40f7b56ca2ffe8e105f481bca5f5555d506cfee`).
Its review journal is 621 bytes, SHA-256
`4c792827a14dfddb66deabadb83719e0122c63281c9327fc4d6156bdf01882b6`;
draft history is 1392 bytes, SHA-256
`022f8beaaed3b6143d53e46ce181a461c117b65dddf4101e47fb9f426097ea5a`;
manual-check journal is 0 bytes, SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The summary's newest outcome counts are one abstention and zero drafts; its
review remains one `provisional_candidate`. The draft-history journal is not
empty: it includes the abstention. Historical v1, v2 and earlier checkpoint
state files remain byte-unchanged.

The running workspace was restarted and browser-verified at ephemeral
`http://127.0.0.1:53966`. Before the narrow repairs, rendering failed because
`renderConversation` expected a missing identity DOM node and review-history
rendering called undefined `reviewHistory`. Restoring the identity element and
rendering the review-history records resolved those failures. The executed
browser then showed seven messages, archive and live labels, prompt versions
`qualification-v1` and `draft-v1`, and the `not_invoked` abstention; no page
errors remained. This verifies rendering only, not live Reddit status.

No supported generation assets were found: the MLX Qwen cache entry is only
`.locks`, not weights; no model root or authorized asset identity is recorded.
The MLX generate CLI's presence does not supply a supported generator. A human
must authorize/provide an installed local generator, weights and exact version,
or explicitly decline, and approve truthful affiliation wording. The human
must still inspect the live target, newer replies and subreddit rules, then
review/edit an actual draft if one becomes available. The archived root lock
and moderator `t1_p41bvy9` Rule 11 notice are historical risk evidence, not
current live status. `engagement-fp-v1` remains a proposal to confirm or amend;
there are zero adjudications, so the false-positive rate and denominator are
null. One reviewed identity remains unadjudicated.

The checkpoint rechecked artifact hashes for the candidates, shortlist
manifest, registry, selection shard and manifest, privacy target, and rerun6
context bundle and manifest; those expected hashes matched. Source archives
were not rehashed. Registered stream digests matching the registry do not
establish fresh archive-file hashes or live status. Downloads, remote inference,
Reddit requests and auto-publishing remained disabled.

## Historical snapshot — 2026-09-30 archive-engagement assessment

Strict acceptance is 3/5 original milestones (60% fully accepted; no partial
credit): contracts/shortlist, batch context and the human workspace. Verified
product reference/local drafting and human handoff/pilot remain partial. This
counts fully accepted milestones, not effort, time or code volume.

The current Mieru review is one `provisional_candidate` for `t3_1vpyqeq`,
supporting `mieru.tracking_without_bank_link` on rerun6. The bundle has seven
linked rows and is complete for the registered sources scanned; one
non-target comment is archived as removed. This does not recover missing text
or establish live replyability. `decisions.jsonl` holds one record;
`reply_drafts.jsonl` and `manual-checks.jsonl` are 0 bytes. No real draft,
handoff, manual/live check or real-candidate measurement has happened. The
loopback workspace shows `live_unverified`; no local generator assets were found
on this machine, and no approved truthful affiliation wording is recorded.

The synthetic adjudication path has also been exercised with the `engagement adjudicate` command and the `engagement pilot-report --adjudication-journal` option against synthetic journals in a temporary directory. This is synthetic evidence, not a product measurement: the real private summary still reports zero adjudications.
With one `false_positive` adjudication for the single reviewed identity, the
report records `false_positive_rate` 1.0, `rate_denominator` 1,
`adjudicated_identity_count` 1, `unadjudicated_reviewed_identity_count` 0,
`label_counts` `{false_positive: 1, true_positive: 0, unclear: 0}`, and
criterion version `engagement-fp-v1`. With one `unclear` adjudication instead,
it records rate 0.0, denominator 1, one adjudicated identity, and label counts
`{false_positive: 0, true_positive: 0, unclear: 1}`. An unclear judgment counts
as adjudicated, not as a false positive; the denominator is adjudicated
identities. The open proposal status of `engagement-fp-v1` is unchanged.

The false-positive output is
`reports/engagement/synthetic-adjudication-smoke-20260930/report-with-adjudication.json`
(1638 bytes, SHA-256
`d77a23297ccbaa590ecc90c1723d634ce5a117357793173e707fcc7bd3e2ebc3`); the
unclear output is
`reports/engagement/synthetic-adjudication-smoke-20260930/report-unclear.json`
(1638 bytes, SHA-256
`577e462ff04b0314c953c216d284d6f3a16b59399d1854f7b18f4cf97bae37b9`).

A second adjudication for the same review is rejected with exit 2 and
`duplicate false-positive adjudication for this review`, whether it repeats or
conflicts with the existing label; the journal keeps one row per review. An
adjudication bound to a superseded review fails closed with exit 2 and
`stale adjudication references a superseded review`, without writing a report.
Writing an adjudication for no review is rejected with exit 2 and
`review_sha256 must identify exactly one review record`; a pre-existing unknown
review reference makes summary generation exit 2 with
`adjudication references an unknown review record`.

`adjudicate` requires `--review-journal`, `--adjudication-journal`, `--label`,
`--adjudicator` and `--adjudicated-at`, plus exactly one of `--review-sha256`
or `--app-id`/`--thread-fullname`/`--reply-target-fullname`; `--note` is
optional. Supplying neither or both identity forms fails with
`provide either --review-sha256 or all of --app-id, --thread-fullname, and --reply-target-fullname, but not both`.
`pilot-report` requires `--review-journal`, `--draft-history` and `--output`,
and accepts optional `--manual-check-journal` and `--adjudication-journal`.
Its output must be under gitignored `reports/`, or it fails with
`handoff output must be under the gitignored reports/ directory`.

This session's workspace verification executed the real page in a managed
headless Chromium against rerun6 and a synthetic nested thread. The conversation
rendered depth-first by `parent_fullname`, with indentation. Rerun6 showed
`t3_1vpyqeq` at depth 0 and its six `t1_*` replies at depth 1, with the
`version_scope`; draft/prompt history displayed `record_type`, `outcome`,
`validation_status`, `validation_message`, `qualification_prompt_version`,
`draft_prompt_version`, `model_version`, `operator_affiliation`,
`abstention_reasons` and current draft text. The synthetic tree rendered at
depths 0, 1, 2 and 1 with the expected ancestor chain; synthetic draft history
showed prompt and model versions. A direct-reply-to-root classification defect
was found and fixed.
`uv run pytest tests/acceptance/test_engagement_workspace.py tests/acceptance/test_engagement_serve_cli.py -q`
passed 7 tests; `uv run ruff check .` was clean; `uv run pytest -q` reported
599 passed and 2 skipped, which require `REDDIT_SEARCH_LIVE_QDRANT=1`.
There is no permanent automated client-side DOM test: jsdom and Playwright are
not available, and a Node/browser test would add a toolchain the repository
does not use. Verification was by executed browser, served HTML and source
review. `--dump-dom` timed out; the managed Chromium route worked.

Deterministic-check coverage is now closed with eight tests added while
implementations remained unchanged. The tests directly cover IDs, claim status,
version scope, preserved affiliation disclosure, input hashes and prohibited
live-status assertions. `DraftCompletion.model_validate` directly rejects
unknown/extra fields, missing required fields and wrong value types. No real
model response has been parsed; the end-to-end path where a real model produces
a response remains unproven without an authorized generator. Three tests were
proven load-bearing by temporary source-condition inversions with expected
failures, then restored byte-identically. Integrated verification on the
current tree: `uv run pytest -q` reported 599 passed and 2 skipped; both skips
require `REDDIT_SEARCH_LIVE_QDRANT=1`. `uv run ruff check .` was clean.

Seven added test cases cover cross-source provenance retention (T1), revision-conflict draft blocking (T2), stale-adjudication rejection with current-review counting (T3), reopening rejected reviews without overwriting history (T4), and strict-schema rejection of extra, missing, and wrong-type fields (T6); each invariant has a load-bearing inversion, and the restored source SHA-256 values are `context.py` `8c5dd86b03fd7daec3d3a83d633adc875acbeab14fac6fabafa88f0a0faec301`, `drafting.py` `e0565dafe57a0144fffe3b676837abbdb5f69e81ee54d989b1b0471cfcb965ec`, `handoff.py` `c9160d6948879b699628a3ee8f0d69fc67c3da2709919df28e9ae4f5d02edb94`, and `review.py` `63de2c99d6e78513308e8034b8ecb91b78a68d5b383b0441967ebe334624bfe7`.
These tests do not parse a real model response or establish a real-candidate draft; the semantic abstention scenarios still require an authorized local generator.

The executed-browser workspace now shows archive verdict and live status as
separately labelled exact values in candidate summary and reviewer detail,
plus the newest manual-check observation and compact earlier history. It says
only a human can change status when no check exists, and that recorded checks
are past human observations and the program never posts. The four live labels,
`needs_context` and a two-record history were exercised. Focused acceptance
tests passed 7 tests and verify every served `live_status` and
`manual_check_history` field, including `final_human_decision: defer`. There is
still no permanent automated client-side DOM test.

Milestone 5 measurement support now includes strict append-only adjudication
records, `engagement adjudicate`, and optional `--adjudication-journal` input
to `engagement pilot-report`. The summary includes the proposed criterion and
its denominator, counts by label, adjudicated and unadjudicated reviewed
identities, and a false-positive rate that distinguishes no adjudications
from zero false positives among adjudications. Criterion version
`engagement-fp-v1` is a proposal for the product owner to confirm or amend.
False positives are never inferred from `reject` or `do_not_post`.

The refreshed private summary
`reports/engagement/mieru-august-index-followup-20260927/pilot-report-20260930-v2.json`
is 1554 bytes, mode 600, SHA-256
`6a77bf4f427cd08de9eafb326416b4bba79d1cbc3626b8b8f1af1b2df8876d68`. It was
produced on the real private journals with
`uv run reddit-search engagement pilot-report --review-journal <decisions.jsonl> --draft-history <reply_drafts.jsonl> --manual-check-journal <manual-checks.jsonl> --output <v2 file>`.
It records review journal SHA-256
`4c792827a14dfddb66deabadb83719e0122c63281c9327fc4d6156bdf01882b6`,
empty draft and manual-check journals (each SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`), one
`provisional_candidate` reviewed identity, zero drafts and zero manual checks.
For `engagement-fp-v1`, the criterion is: label `false_positive` only when
the reviewed item clearly lacks a genuine, first-person, currently unanswered
need that the app can address using its verified claims; label `unclear` when
evidence is insufficient. False-positive adjudications have status
`not_recorded`; `false_positive_rate` and `rate_denominator` are null,
`adjudicated_identity_count` is 0, `unadjudicated_reviewed_identity_count` is
1, and `false_positive_adjudication_journal_sha256` is null. This summary
reports no measurement yet: zero adjudications means the rate is undefined,
not zero. A repeat run produced identical JSON, the journals were byte-unchanged,
and `uv run ruff check .` was clean. The optional `--adjudication-journal`
requires an existing file, so it was omitted because no adjudication file
exists; the output path must be fresh and under gitignored `reports/`.
The older
`reports/engagement/mieru-august-index-followup-20260927/pilot-report-20260930.json`
is byte-unchanged and remains a historical snapshot of the pre-change summary
shape (1112 bytes, SHA-256
`036dc7627251c334cd60c5e867fde962c1e94106e91a46ee4bef0787f21f59bd`).

The search used `command -v` for llama/ollama/mlx CLIs; checked
`/opt/homebrew/bin`, `/usr/local/bin`, `~/bin` and `~/.local/bin`; bounded
`find` for `*.gguf` under `/Users/bitaru`, `/opt/homebrew`, `/usr/local`,
`~/Library` and the mounted non-system volume `/Volumes/T7 Shield`
(excluding `reddit*` subtrees); and checked `~/.ollama`, `~/.lmstudio`,
`~/Library/Application Support`, `~/Library/Caches/huggingface` and
`~/.cache/mlx_lm`.
The local asset search found no llama.cpp or Ollama executable, no GGUF weights,
and no `torch`, `transformers`, `llama_cpp` or `mlx` importable in the project
venv. The only local generation runtime is MLX at
`/Users/bitaru/.local/bin/mlx_lm.generate`; it works, but no MLX-format weights
were found and `~/.cache/mlx_lm` is absent. The Hugging Face cache contains only
retrieval assets, `models--Qwen--Qwen3-Embedding-0.6B` and
`models--Qwen--Qwen3-Reranker-0.6B` (about 1.1 GB each). Their `config.json`
declares `architectures: ["Qwen3ForCausalLM"]`, `model_type: qwen3` and
`tie_word_embeddings: true`. This is a base causal-LM architecture reused for
pooling, so these retrieval checkpoints are not an authorized or proven text
generator. This matches the existing statement that the local Qwen
configuration covers retrieval, not a proven local text-generation backend.

`src/reddit_search/engagement/local_model.py` requires absolute executable and
model paths, accepts only executable basenames `llama-cli` or `llama`, passes
the prompt on stdin and invokes
`<executable> -m <model_path> -n 512 --no-display-prompt`. 
`src/reddit_search/engagement/drafting.py` validates the same paths. A
non-llama.cpp runtime therefore needs adapter and validation changes, not only
configuration.
Fresh read-only evidence remains in
`reports/engagement/mieru-august-index-followup-20260927/state-validation-20260930.json`.
It verifies the selection shard and manifest, exact followup allowlist
`data/staging/engagement/mieru-privacy-target-20260927.json`, candidate,
shortlist, registry and rerun6 bundle pins. Recorded stream digests match the
registry; source archives were not rehashed. The current pilot summary is the
refreshed v2 file described above. The running loopback workspace serves
`127.0.0.1:61362`; the port is ephemeral. `configs/runtime.yaml` keeps
downloads, remote inference, Reddit requests and auto-publish false.

The human/asset owner must choose and authorize one path: (a) a local llama.cpp
`llama-cli` or `llama` executable plus a GGUF model with its version identity,
which fits the existing adapter with no code change; (b) another local runtime
plus its weights, which also requires adapter and validation work; or (c)
explicitly decline. No downloads or remote inference. The synthetic real-model
safety checks and a real local generation smoke remain required before
milestone 4 can be accepted. The human approves truthful affiliation wording;
the assistant must not invent it. The value is supplied per run via
`engagement draft --affiliation <string>`, not from the profile. A blank or
whitespace-only value records an abstention with reason
`a truthful, nonblank operator affiliation is required` rather than failing;
when claims are used, validation also requires the supplied string in the
draft text. No value is recorded in configs, prompts, registry, docs or the
seven nested launch artifacts checked. The recorded identity fact
`verification_owner: "Mieru developer Vitaly Danilov; operator confirmed the published facts"`
is available to the human, who must supply or approve the wording.
A human may check the live target, newer replies and community rules in
parallel, and records an observation only after actually checking. The
assistant records a genuine draft or truthful abstention on the reviewed
target; if unusable, it preserves that result and reviews a narrowly selected
supported alternative rather than forcing a draft. The human reviews or edits
and the assistant revalidates. The assistant exports the private handoff, then
the human refreshes the live/rules check before any manual posting decision;
posting is optional, manual and outside software completion. After journal
changes, the assistant refreshes the pilot summary to a new file including
the adjudication journal. Close milestones 4/5 and overall completion only
when every original acceptance item is demonstrated. Do not repeat the
archive scan or broad discovery unless evidence invalidates the candidate.

## Historical implementation records

The dated entries below document prior states and experiments. Read them as
historical snapshots, not as the current state above.

## 2026-09-30: Contract-invariant and schema coverage

The seven added cases are:

1. `tests/unit/test_engagement_context.py:241`
   `test_identical_message_across_registered_sources_retains_each_provenance`
   preserves the `comment` and `comment-second` provenance entries for the
   identical message revision.
2. `tests/unit/test_engagement_drafting.py:296`
   `test_revision_conflict_in_context_blocks_drafting_with_coverage_reason`
   returns abstention with the coverage-gap reason and `model_version`
   `not_invoked`.
3. `tests/unit/test_engagement_handoff.py:498`
   `test_pilot_summary_rejects_stale_adjudication_and_counts_current_review`
   rejects the older review with `stale adjudication references a superseded
   review`, counts one adjudicated identity for the current review, and rejects
   a repeat append as duplicate.
4. `tests/unit/test_engagement_review.py:48`
   `test_rejected_review_can_be_reopened_without_overwriting_history` keeps the
   reject and newer provisional review in order with their separate reasons.
5. `tests/unit/test_engagement_drafting.py:310`
   `test_draft_completion_rejects_malformed_model_output` is parameterized for
   an unknown/extra field, a missing required field and a wrong value type;
   each is rejected by `DraftCompletion.model_validate`.

Load-bearing inversions, one per invariant: T1 disabling the identical-revision
merge branch failed with a SQLite unique-constraint error; T2 disabling the
coverage guard let drafting proceed to the missing-model error; T3 inverting
the stale-review check failed with `DID NOT RAISE ValueError`; T4 changing
append mode from `a` to `w` failed because only
`provisional_candidate` remained instead of `reject` then
`provisional_candidate`; T6 changing `extra="forbid"` to `extra="ignore"`
failed with `DID NOT RAISE ValueError`. The T5 synthetic-profile/unverified-
claim gate already has coverage from `test_synthetic_profile_cannot_authorize_a_draft`,
`test_missing_owner_approved_claim_statement_abstains_before_model` and
`test_unverified_supporting_claim_abstains_before_model`; no duplicate test
was added.

Integrated verification on the current tree is `uv run pytest -q`: 599 passed
and 2 skipped. Both skips are in
`tests/acceptance/test_tombstones_operate_live.py` (lines 147 and 281) and
require `REDDIT_SEARCH_LIVE_QDRANT=1`; `uv run ruff check .` passes. These tests
do not parse a real model response, establish a real-candidate draft, or prove
the semantic abstention scenarios, which still require an authorized local
generator.

## 2026-09-30: Coverage audits and workspace live status

Eight deterministic-check tests were added without changing implementations.
`tests/unit/test_engagement_drafting.py` adds
`test_unverified_supporting_claim_abstains_before_model` (about line 282),
`test_blank_version_scope_abstains_before_model` (290),
`test_input_hash_rejections_are_distinct` (298) rejects a malformed expected
SHA-256, manifest/expected mismatch, bundle/manifest mismatch, and
newest-review/bundle mismatch. `test_qualification_rejects_invalid_citations`
(356) and `test_completion_rejects_invalid_citations` (377) each reject
duplicate cited message IDs, duplicate claim IDs, and malformed message
fullnames. `test_draft_text_validation_flags_live_status_and_missing_affiliation`
(391) covers a live-status assertion and missing supplied affiliation.
`tests/unit/test_engagement_review.py` adds
`test_false_positive_adjudication_rejects_missing_required_field` (218) and
`test_manual_check_rejects_missing_required_reviewer` (238). The three targeted
source-condition inversions for version scope, verified status and duplicate
citations produced expected failures; `src/reddit_search/engagement/drafting.py`
was restored byte-identically (`cmp` clean). Current integrated verification
is 592 passed and 2 skipped from `uv run pytest -q`; both skips in
`tests/acceptance/test_tombstones_operate_live.py` require
`REDDIT_SEARCH_LIVE_QDRANT=1`. `uv run ruff check .` is clean.

In `src/reddit_search/engagement/workspace.html`, the archive verdict and live
status render as separately labelled exact values in the candidate summary and
reviewer detail. The page renders the newest manual-check observation
(observation time, reviewer, live status, observed permalink, observation text,
subreddit rule evidence and `final_human_decision`) and a compact earlier-history
line. With no check it says only a human can change the status; it identifies a
recorded check as a past human observation and says the program never posts.
Managed headless Chromium rendered `live_unverified`,
`human_checked_replyable`, `live_unavailable` and `live_not_replyable`
individually through the shared code path, with `needs_context` and a
two-record history. The four labels were smoke-exercised individually, not
through four independent implementations. There is no permanent automated
client-side DOM test.

The acceptance coverage in `tests/acceptance/test_engagement_workspace.py`
(about lines 60-64, 149-167) and
`tests/acceptance/test_engagement_serve_cli.py` (about lines 126-139, 188-208)
checks the served `live_status` and every `manual_check_history` field with
and without a recorded check; the recorded case has
`final_human_decision` `defer`. Those files passed 7 focused tests. The private
journals remained unchanged: `decisions.jsonl` is 621 bytes, SHA-256
`4c792827a14dfddb66deabadb83719e0122c63281c9327fc4d6156bdf01882b6`;
`reply_drafts.jsonl` and `manual-checks.jsonl` are each 0 bytes.

Each named deterministic check (IDs, claim status, version scope, preserved
affiliation disclosure, input hashes and prohibited live-status assertions)
has a direct test. Strict-schema parsing of model responses, the qualification
and completion JSON paths, remains indirect because no generator is available.
Deterministic abstention gates cover untrusted instruction-like text via the
regex set, blank affiliation, and structural/profile eligibility. Semantic
judgments remain unproven without an authorized local model: whether an
available claim answers the need, platform compatibility, unknown or
conflicting natural-language constraints, and whether the need is already
resolved. This is why milestone 4 stays partial. The plan's `archive_rejected`
wording corresponds to stored review verdict `reject`, which the UI renders
humanized; it is not a missing state.

## 2026-09-30 — Rerun6 completion and workspace cutover

Launchd job `com.omp.reddit-search.mieru-august-index-followup-context-rerun6` re-ran the August followup scan with the fixed classifier. `scan_complete` is true: 413035033 records scanned (45,491,721 submission lines and 367,543,312 comment lines). Both streamed source digests match the registry values: submissions `7c364b51f381da1987510ca6485ca922c81525e4600043e408651bc67b878d97` and comments `30da6cc45cb9b6cad5a910888bcebeb3213715365d3373e9bcd9f1d01850deaa`.

Output in `data/staging/engagement/mieru-august-index-followup-context-20260927-rerun6/`: `threads.jsonl.zst` is 2176 bytes, SHA-256 `fdbb68a431b903c4b1f34254df9f17d9775008e9dbb3a2d71bab326b34251436`, byte-identical to the rerun5 bundle; `context_manifest.json` SHA-256 is `a933cd7ac325ce966a4a0e35d262ceda501b345b9d200ec19e3262f2ebd31c75`.

Thread `t3_1vpyqeq` has `coverage_status` `complete_for_scanned_sources`, `thread_context_complete_for_sources` true, `root_missing` false, no missing parents or reply targets, and empty `coverage_reasons`. `archive_content_counts` is `{"historical_removal": 1}`; the single note is `t1_p419wc5: message text is archived as removed/deleted`. The thread has 7 retained messages.

The running workspace job `com.omp.reddit-search.engagement-workspace-followup` now serves `--context-bundle` and `--context-manifest` from the rerun6 directory with `--product-profile /Users/bitaru/projects/reddit-search/configs/profiles/mieru.yaml`, bound to loopback with `--port 0`. The current URL is the last line of `reports/engagement/mieru-august-index-followup-20260927/engagement-workspace.log`, currently `http://127.0.0.1:61362`. A served verified, non-synthetic profile is required for the `Potential reply` option to be selectable; `potentialAllowed` in `src/reddit_search/engagement/workspace.html` also requires verified claims. The observed live page renders the archived-removal note with no coverage warning and the option enabled. The previous instance's plist is archived as `reports/engagement/mieru-august-index-followup-20260927/workspace-followup-rerun5.plist`.

`decisions.jsonl`, `manual-checks.jsonl`, and `reply_drafts.jsonl` remain present and empty (0 bytes, SHA-256 of the empty string); no human verdict, draft, or manual check exists for `t3_1vpyqeq` (superseded by the review recorded in the entry below). Drafting remains blocked because no local text-generation model is provisioned and model downloads and remote inference remain disabled in `configs/runtime.yaml`. Live-page checking and posting remain manual.

## 2026-09-30 — First followup review recorded and pilot summary

The human review for candidate `t3_1vpyqeq` (`r/privacy`) was recorded at 2026-09-30T09:51:16Z with verdict `provisional_candidate`, `product_fit` `compatible`, supporting claim `mieru.tracking_without_bank_link`, evidence message `t3_1vpyqeq`, `context_complete` true, reason code `author_has_current_need`, and `bundle_sha256` `fdbb68a431b903c4b1f34254df9f17d9775008e9dbb3a2d71bab326b34251436` (the rerun6 bundle). The provisional decision was accepted by the workspace gate on the real bundle, proving the archived-removal fix end to end.

`reports/engagement/mieru-august-index-followup-20260927/decisions.jsonl` now holds one record (621 bytes, SHA-256 `4c792827a14dfddb66deabadb83719e0122c63281c9327fc4d6156bdf01882b6`).

The new private summary `reports/engagement/mieru-august-index-followup-20260927/pilot-report-20260930.json` (1112 bytes, SHA-256 `036dc7627251c334cd60c5e867fde962c1e94106e91a46ee4bef0787f21f59bd`) was written with `engagement pilot-report --review-journal ... --draft-history ... --manual-check-journal ... --output ...`. It reports one provisional decision, zero drafts, zero manual checks, zero human edits, and records that false-positive labels are not captured by the current schemas.

`reply_drafts.jsonl` and `manual-checks.jsonl` remain 0 bytes. No draft exists, so no handoff packet exists. Drafting stays blocked until an authorized local llama.cpp executable and GGUF model are provisioned; `configs/runtime.yaml` keeps model downloads and remote inference disabled. The human live-page check remains outstanding.

## 2026-09-30 — Archived removal notes and capture coverage

The context scanner previously classified every archived-removed row as a capture gap, incorrectly blocking provisional review when a non-target comment was retained but its text was already stored as `[removed]` or `[deleted]`. The scanner now keeps such rows in separate per-thread `archive_content_notes` and `archive_content_counts` fields; they do not alter completeness. Archived-removed roots and reply targets remain capture gaps, with descriptive `root_text_removed` and `target_text_removed` codes. The workspace presents notes separately from gap warnings, while only capture gaps disable a provisional decision. Context unit tests cover both classifications, and workspace acceptance tests cover separate note payloads and provisional acceptance with notes versus refusal with a gap.

## 2026-09-26 — Engagement context-scan cancellation provenance

Preserved the original `reports/engagement/august-deep-candidate-20260926/context-scan-cancellation.json` (SHA-256 `96223090b59770c1337034aee3670417928158739189fdde9cef6076e405b9ea`) unchanged. Its `cancelled_by` field incorrectly attributes cancellation directly to the user. Superseded it with `reports/engagement/august-deep-candidate-20260926/context-scan-cancellation-superseding.json` (SHA-256 `7ed8598f197a004c13acb068014f5c72e08746b40926d930225d54c776420607`), which corrects the attribution; the superseding file records the original path, hash, and reason. The context scan remains incomplete; partial staging is preserved.

## 2026-09-29 — Engagement context reader performance and rerun

The second real-data context attempt for candidate `t3_1vpyqeq` in `r/privacy`, submission `t3_1vpyqeq`, revision `3bdb7906a14a0effe01ed905cdd9841b9c0693888335070dab72cdcb587cc80a`, had run since `2026-09-27T20:48:58Z` as launchd job `com.omp.reddit-search.mieru-august-index-followup-context-rerun4`. The operator stopped it using `launchctl bootout gui/501/com.omp.reddit-search.mieru-august-index-followup-context-rerun4`; no TERM or KILL was needed. At the last live capture, the comments descriptor was at 11535384576 of 51525496195 bytes (22.39%); the submissions source had been fully read and its descriptor was closed. Python CPU time was 19:49 over elapsed 1-10:47:38. The staging `.context.sqlite3` was 24576 bytes with 0 message rows; no manifest or bundle existed. The stop record is `reports/engagement/mieru-august-index-followup-20260927/context-rerun4-stop-status.json`; partial staging at `data/staging/engagement/mieru-august-index-followup-context-20260927-rerun4/` is preserved. The earlier `rerun2` and `rerun3` attempts remain unexplained in `reports/engagement/mieru-august-index-followup-20260927/context-attempt-status-20260927.json` and `reports/engagement/mieru-august-index-followup-20260927/context-attempt-rerun3-status.json`.

Measured cause of the impractical runtime was repeated re-slicing of the remaining decoded buffer once per line in `ArchiveReader._iter_source_lines` using `split("\n", maxsplit=1)`, making line splitting quadratic in chunk size. On a real 250,000,000-byte comments slice, throughput was 0.77 MB/s compressed for 1,657,074 records in 324 seconds. The live job descriptor advanced about 16.5 MB/min (0.275 MB/s), leaving about 40.2 GB and 40 hours of awake running time. The staging SQLite file remains at schema size (24576 bytes) while matching rows remain in the SQLite page cache until final commit; a small/schema-only `.context.sqlite3` does not establish scan progress, and a surviving `.context.sqlite3` means the process ended without running the reader cleanup path.

`_iter_source_lines` now splits each decoded chunk once and carries the partial trailing line. Equivalence on the comments slice was 1,657,074 records and identical line-stream SHA-256 `bf39f6d14229f162d01e4311e5ac5d1a7721414535d465c46e62f8b172ff9529`, with throughput increasing from 0.77 to 11.35 MB/s (14.09x). An independent 250,000,000-byte submissions slice produced 500,350 records and identical line-stream SHA-256 `5c5707e3e521ffd98df820fa84ee3b8eb56314d42c8c426b40877d6cf8cf301b`; `lines_seen`, `valid_records`, `invalid_records`, `invalid_line_numbers`, `compressed_bytes_read`, `decompressed_bytes_read`, and the `SourceReadError` at the truncated tail were identical. Runtime decreased from 48.9 to 17.4 seconds (2.81x). A synthetic input whose first line crosses the 131072-byte decompressed chunk boundary yielded identical lines in both implementations. Four regression tests in `tests/unit/test_reader.py` cover the chunk boundary, over-limit long line, `\\r\\n` preservation, and invalid UTF-8 straddling a boundary. `uv run pytest -q` reported 571 passed and 2 skipped; both skips require `REDDIT_SEARCH_LIVE_QDRANT=1`. `uv run ruff check .` passed. The change is limited to `src/reddit_search/ingest/reader.py` (+10/-8) and `tests/unit/test_reader.py` (+91).

Launchd job `com.omp.reddit-search.mieru-august-index-followup-context-rerun5` started at `2026-09-29T07:42:49Z` from `reports/engagement/mieru-august-index-followup-20260927/context-rerun5.launch.sh`, with the same pinned registry, candidates, and shortlist hashes and the same limits. It runs under `/usr/bin/caffeinate -i -s -m`; `ProcessType` is omitted from the plist to avoid launchd I/O throttling. A 300-second progress sampler writes `reports/engagement/mieru-august-index-followup-20260927/context-rerun5.progress.jsonl`. Output is `data/staging/engagement/mieru-august-index-followup-context-20260927-rerun5`; sentinel is `reports/engagement/mieru-august-index-followup-20260927/context-rerun5.exit-status`. Observed compressed throughput is 12.4-17 MB/s, several times the previous attempt. `caffeinate` cannot survive a lid close or power loss. At this entry, no manifest or bundle exists for this attempt; completion and bundle verification are pending.

No human verdict exists for `t3_1vpyqeq`; no reply draft or manual check exists. No local text-generation model is provisioned, so `engagement draft` must abstain. `configs/runtime.yaml` keeps `allow_model_downloads`, `allow_remote_inference`, `allow_reddit_requests`, and `allow_auto_publish` false. Live-page checking and posting remain manual. The followup, pilot-three, and deep shortlist input chains were re-verified: every pinned hash resolves to an existing file, and selection run ID `898ef707c04bc8fb365fd96d7055905561b6419639757c5fe35264bbb4da9c90` is the `run_id` recorded inside the hashed selection manifest `data/staging/august-2026-topic-selection-full-scan-20260923.jsonl.zst.manifest.json`.

## 2026-09-29 — Followup context scan completed (rerun5) and wrapper offset drift

The rerun5 scan finished with `scan_complete` true: 413,035,033 records scanned (45,491,721 submission lines and 367,543,312 comment lines), matching the earlier full-discovery pass exactly. Both per-source streamed digests equal the registry digests: submissions `7c364b51f381da1987510ca6485ca922c81525e4600043e408651bc67b878d97` and comments `30da6cc45cb9b6cad5a910888bcebeb3213715365d3373e9bcd9f1d01850deaa`. In `data/staging/engagement/mieru-august-index-followup-context-20260927-rerun5/`, `threads.jsonl.zst` is 2176 bytes (SHA-256 `fdbb68a431b903c4b1f34254df9f17d9775008e9dbb3a2d71bab326b34251436`) and `context_manifest.json` is 2471 bytes. The bundle contains one thread, `t3_1vpyqeq`, with 7 matched and 7 retained messages; the root is present and all six captured comments parent to it. Coverage is `needs_context` with one `historical_removal` reason for `t1_p419wc5`. Independent verification found the input hashes match the manifest, the tombstone ledger is empty (`tombstone_canonical_sha256` null and `tombstone_sha256` equal to the empty-file digest), and `scanned_record_count` equals the sum of `lines_seen`. The staging `.context.sqlite3` was removed on this clean exit; a surviving `.context.sqlite3` indicates the process ended without the reader cleanup path, as in rerun4. `/usr/bin/time -l` reported 6940.79 seconds real, 6494.32 seconds user, 321.04 seconds system, maximum resident set size 690307072 bytes, and no swaps.

`context-rerun5.launch.sh` was rewritten at `2026-09-29T07:46:40Z`, 3 minutes 51 seconds after the job started at `07:42:49Z`. The running `/bin/sh` held byte offsets from the pre-rewrite file; when it read the tail at `09:38:30Z`, it resumed mid-word at `write_status finished 0` and logged `tatus: command not found` followed by `syntax error near unexpected token 'fi'`. Launchd records `last exit code = 2` for the wrapper process, while `context-rerun5.exit-status` records `status=finished`, `exit_code=0`, the CLI exit status captured by `rc=$?`; the records refer to different processes. The surviving script is 187 lines and 8220 bytes, passes `sh -n` and `bash -n`, and has `write_status finished 0` at line 182. Future attempts must not edit a launch script after bootstrap; bootout and re-bootstrap instead.

The local review workspace started at `2026-09-29T09:44:23Z` under launchd label `com.omp.reddit-search.engagement-workspace-followup`; its plist and log are in `reports/engagement/mieru-august-index-followup-20260927/`. It binds only to loopback at `http://127.0.0.1:64866` and serves the reviewed bundle. Its `--candidates`, `--shortlist-manifest`, `--context-bundle`, `--context-manifest`, `--registry`, `--decisions`, `--manual-checks`, and `--draft-history` paths point at the followup report directory. `GET /api/bootstrap` returns app ID `mieru`, thread `t3_1vpyqeq`, 7 messages, coverage `needs_context` with the `historical_removal` reason, and zero review, draft, and manual-check history. `decisions.jsonl`, `manual-checks.jsonl`, and `reply_drafts.jsonl` are empty and mode 0600; the operator created `reply_drafts.jsonl` empty because the workspace requires the draft journal to pre-exist.

No human verdict exists for `t3_1vpyqeq`; no draft or manual check exists. No local text-generation model is provisioned, so drafting remains blocked under `configs/runtime.yaml`. Live-page checking and posting remain manual.
