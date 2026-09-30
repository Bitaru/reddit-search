# Offline engagement operator guide

The commands and APIs in this guide use captured archive data to build a private candidate and optional provisional reply. They do not contact Reddit, call remote inference or publish anything. Archive status is historical; only a person can check the live page and decide whether to post.

The deliverable is a reviewed provisional draft with evidence and a private
handoff. A live-page check or published reply is not required to complete
preparation. Later live checking and any manual posting are separate human
actions.

## 1. Start with a frozen selection

Use an existing selected-message shard and its manifest. The manifest must attest to a complete, uncapped scan, and its output hash must match the shard. Do not substitute search snippets for thread context.

Build a fresh shortlist directory:

```sh
RUN_ID=review-001  # Choose a new ID for each run.
uv run reddit-search engagement shortlist \
  --selection PATH_TO_SELECTED_MESSAGES.jsonl.zst \
  --selection-manifest PATH_TO_SELECTION_MANIFEST.json \
  --registry PATH_TO_SOURCE_REGISTRY.json \
  --output "reports/engagement/$RUN_ID/shortlist" \
  --app-id YOUR_APP_ID
```

The command writes `candidates.jsonl` and `shortlist_manifest.json`. Optional `--ranked-directory` and `--run-manifest` must be supplied together. `--target-allowlist` accepts exact message fullnames. Shortlisting reads the frozen selection; it does not scan the registered archives.

## 2. Build bounded thread context

Use the shortlist outputs to scan registered sources once for the selected threads:

```sh
uv run reddit-search engagement context \
  --candidates "reports/engagement/$RUN_ID/shortlist/candidates.jsonl" \
  --shortlist-manifest "reports/engagement/$RUN_ID/shortlist/shortlist_manifest.json" \
  --registry PATH_TO_SOURCE_REGISTRY.json \
  --output "data/staging/engagement/$RUN_ID/context"
```

The output includes `threads.jsonl.zst` and `context_manifest.json`. If you use a tombstone ledger, pass the same `--tombstones PATH` to both `shortlist` and `context`; their manifests pin the ledger hash. Without a ledger, leave the option out of both commands. Keep the registry, selected shard, manifests, tombstone ledger and bundle together. Their hashes bind later stages to these exact inputs. Before drafting or exporting, compute the SHA-256 of the exact `context_manifest.json` bytes (on macOS, `shasum -a 256 "data/staging/engagement/$RUN_ID/context/context_manifest.json"`); use its first field for `PINNED_CONTEXT_MANIFEST_SHA256`, not the bundle hash.

Check `scan_complete`, each thread's `coverage_status`, missing root/parents/targets, omitted counts and coverage reasons before review. Resource caps or read/normalization errors can leave a useful partial bundle, but they do not make it complete. These capture gaps prevent a provisional-candidate decision or a draft. Archived-removed text in a captured non-target comment is reported separately as an archive content note and does not make capture incomplete; a removed root or reply target remains a gap. Historical removal, lock and reply-count fields do not establish current status.

Use a registered, checksum-verified source registry. The engagement pipeline accepts `authorized_local_research` and `user_provided_local_research_no_external_transmission` as `usage_scope` values. Respect the more restrictive permission for each source; never send archive text, excerpts or derived replies to an unapproved service. Keep source paths, declared months, verified SHA-256 values, registry hashes and tombstones explicit. `data/staging/` and `reports/` are gitignored in this repository, but still treat their contents as private.

New shortlist and context directories are mode `0700`; their artifacts are mode
`0600`. Draft histories are also private. Drafting refuses an existing history
unless it is a regular file with mode `0600`, not a symlink.

## 3. Review on loopback

Start the workspace with the exact shortlist, context, registry and journals:

```sh
uv run reddit-search engagement serve \
  --candidates "reports/engagement/$RUN_ID/shortlist/candidates.jsonl" \
  --shortlist-manifest "reports/engagement/$RUN_ID/shortlist/shortlist_manifest.json" \
  --context-bundle "data/staging/engagement/$RUN_ID/context/threads.jsonl.zst" \
  --context-manifest "data/staging/engagement/$RUN_ID/context/context_manifest.json" \
  --registry PATH_TO_SOURCE_REGISTRY.json \
  --decisions "reports/engagement/$RUN_ID/decisions.jsonl" \
  --manual-checks "reports/engagement/$RUN_ID/manual-checks.jsonl" \
  --product-profile PATH_TO_VERIFIED_PROFILE.yaml
```

`--draft-history` and `--product-profile` are optional for serving, but a matching verified profile is needed to record compatible product fit. The server binds to `127.0.0.1` by default and refuses non-loopback hosts. Keep it local; stop it with Ctrl-C when finished. Journals are append-only private files. The simplified queue reviews each shortlisted target in turn. The API can still review another unique captured message in the same thread; that target has its own history. Revisions are retained rather than overwritten.

On a first review, omit `--draft-history`. After drafting creates that file,
restart the workspace with `--draft-history` pointing to it to see the drafts.
The workspace rejects context bundles above 2 GiB compressed or decompressed,
JSONL rows above 2 MiB, or more than 1,000,000 messages. These are load limits,
not evidence that the archive capture is complete.

Read the highlighted target and captured replies. Enter your name once, then
choose **Skip**, **Not sure**, or **Potential reply**. Skip asks for one reason;
if the captured conversation already answered the question, select the message
that answered it. Not sure can include an optional note. Potential reply asks
you to choose one verified supporting claim and explain the fit. Select it
only if the author has a first-person need that the captured conversation has
not answered. A provisional reply requires complete capture coverage, a unique
target, a root that is not historically removed, and a verified claim that
matches the actual need. Archived-removed text in a non-target comment is a
factual note, not a capture gap; a removed target or root is a gap. Missing
archived replies remain missing; the workspace cannot infer a live page's
current state.

Click **Save & next** to append a review and advance to the next unfinished
conversation. A failed save stays on the current conversation and displays the
error. The progress bar counts saved archive reviews, not earlier quick
triage. Reloading resumes at the first unfinished conversation. When every
shortlisted target has a saved review, the page says **Thank you** and lists
the decisions; **Review again** appends a revision without deleting history.
Archive coverage warnings and source details remain available next to the
conversation. No archive choice checks Reddit, drafts a reply, or posts.

After completing the queue, a potential reply can have a separate, optional
human live-page observation recorded in the disclosure below the thank-you
screen. Inspect the current Reddit target, newer replies and community rules
yourself first. The workspace does not fetch Reddit or post a reply.

Journals must be regular mode-0600 files; the workspace rejects existing journals with broader permissions.


The workspace records archive verdicts and manual checks; it does not provide a reply-edit UI. Manual checks must be based on a fresh human inspection of the live permalink, target, new replies and applicable community rules. Record the observation, rule evidence, time and final decision. The available live statuses are `live_unverified`, `live_unavailable`, `live_not_replyable` and `human_checked_replyable`. The last requires a permalink, observation, rule evidence and a final human decision (`post`, `do_not_post` or `defer`). `live_unverified` is the starting state. A checked status is only a recorded observation, not a posting action or a promise the page will stay unchanged. Recheck immediately before any manual post.

## 4. Draft locally, or abstain

A product-specific provisional draft requires a current `provisional_candidate` review and a verified `ProductProfile` for the same `app_id`. Before drafting, the profile must have `profile_version` of at least 1, `verification_status: verified`, `synthetic: false`, a nonblank `verification_owner`, and a nonblank `verified_at` string. Every supporting claim must be in that profile, have `status: verified`, and include a nonblank `evidence_ref`, owner-approved `statement`, `version_scope`, and a `constraints` list with no blank entries. Verify those facts against the real product/version; do not copy claims from posts or model output. The profile and prompt hashes are recorded with the draft.

No text-generation model is installed by default, and the command will not download one. Provision weights separately, then explicitly provide the noninteractive llama.cpp `llama-completion` executable, a local GGUF model file, its pinned version, the profile, and a truthful affiliation. Do not use interactive `llama-cli`: its terminal output is not the bounded completion protocol.

Separate installation and weight downloads require explicit operator
authorization. Keep the pipeline's download and network safeguards unchanged;
authorizing local provisioning does not authorize remote inference or Reddit
access.

```sh
uv run reddit-search engagement draft \
  --app-id YOUR_APP_ID \
  --thread-fullname THREAD_FULLNAME \
  --reply-target-fullname REPLY_TARGET_FULLNAME \
  --context-bundle "data/staging/engagement/$RUN_ID/context/threads.jsonl.zst" \
  --context-manifest "data/staging/engagement/$RUN_ID/context/context_manifest.json" \
  --expected-context-manifest-sha256 PINNED_CONTEXT_MANIFEST_SHA256 \
  --review-journal "reports/engagement/$RUN_ID/decisions.jsonl" \
  --history "reports/engagement/$RUN_ID/reply_drafts.jsonl" \
  --product-profile PATH_TO_VERIFIED_PROFILE.yaml \
  --affiliation 'Your truthful relationship to the product' \
  --llama-cpp-executable PATH_TO_LOCAL_LLAMA_COMPLETION \
  --model-path PATH_TO_LOCAL_MODEL \
  --model-version LOCAL_MODEL_VERSION
```

The default prompt files are `qualification-v2.txt` and `draft-v3.txt` under
`configs/prompts/engagement/`. Qualification rechecks the source evidence and
claim fit rather than trusting a prior review verdict. Missing or uncertain
required support cannot become a provisional candidate. Old versioned prompts
remain available for interpreting their historical, hash-bound records.

Task JSON escapes angle brackets so native GGUF tokenization cannot treat
quoted control tokens as conversation boundaries. Decoding the JSON preserves
the exact source strings; this does not modify archived records or evidence.

The adapter passes the frozen prompt as the trusted system message and task
JSON as the user message. It uses a private temporary system file and unchanged
binary stdin, then removes the temporary file. The verified native transport
is llama.cpp 0.5.0 `llama-completion`, offline and single-turn, with bounded
context, tokens, input bytes, output bytes and execution time. It removes only
the observed native terminal frame; all remaining output must satisfy the
strict qualification or draft schema. It does not extract a JSON substring,
repair a response, insert a disclosure, retry, or fall back to a remote model.

Successful generation appends a draft with no abstention reasons. Earlier
abstentions remain in the append-only history; a later draft does not erase
them. Machine validation is not human acceptance or permission to post.

Provisional qualifications and drafts must cite at least one approved product
claim. Every accepted reply must preserve the supplied affiliation disclosure;
an empty claim list cannot bypass either requirement.

The command runs offline qualification and drafting. Existing behavior is
fail-closed: if affiliation is omitted or blank, it records a no-text
abstention for the missing truthful disclosure before invoking a model. That
omission is neither an approved disclosure nor an explicit human decline.
Missing generator assets alone cause an error only when earlier eligibility
and disclosure guards pass. There is no remote fallback. Preserve the exact
approved affiliation disclosure and verified claim statements. Never imply
current replyability, rule compliance, personal experience or product behavior
that the evidence does not support. An abstention is not a draft and cannot
proceed to handoff. For a genuine draft, supply an explicit supported local
generator, its weights and exact version, plus truthful approved affiliation;
without those assets, do not claim semantic model verification or real
generation.
For this Mieru run, Bitaru approved the exact disclosure “I’m the developer of
Mieru.” and separately authorized local installation and model downloads.
Private provisioning manifests record the executable, model revision and
hashes. Provisioning alone does not prove safe generation: a draft still must
preserve the exact disclosure and approved claim wording and pass every check.

## 5. Optional human edit and revalidation

Run `uv run reddit-search engagement edit --help` to see the required options.
Put the edited text in a local UTF-8 file and keep that file private (for
example, set its permissions to `0600`):

```sh
uv run reddit-search engagement edit \
  --history "reports/engagement/$RUN_ID/reply_drafts.jsonl" \
  --prior-record-sha256 CURRENT_RECORD_SHA256 \
  --text-file /path/to/private-edit.txt \
  --editor YOUR_EDITOR_ID
```

The command appends an immutable history record, reruns deterministic claim,
affiliation and live-status checks, and reports the result without printing
draft text. `--prior-record-sha256` must identify the latest record in the
history. An invalid edit remains in the journal for audit; handoff blocks on an
invalid latest edit.

A local-only humanizer or the `clean-user-facing-text` skill is optional. Do not use a hosted text service on archive-derived material. Keep the author’s meaning, verified claim wording, truthful affiliation and required disclosures. Revalidate after every edit; do not make authorship or watermark claims.

## 6. Private handoff and live decision

Run `uv run reddit-search engagement handoff --help` for the required options:

```sh
uv run reddit-search engagement handoff \
  --app-id YOUR_APP_ID \
  --thread-fullname THREAD_FULLNAME \
  --reply-target-fullname REPLY_TARGET_FULLNAME \
  --context-bundle "data/staging/engagement/$RUN_ID/context/threads.jsonl.zst" \
  --context-manifest "data/staging/engagement/$RUN_ID/context/context_manifest.json" \
  --expected-context-manifest-sha256 PINNED_CONTEXT_MANIFEST_SHA256 \
  --review-journal "reports/engagement/$RUN_ID/decisions.jsonl" \
  --draft-history "reports/engagement/$RUN_ID/reply_drafts.jsonl" \
  --registry PATH_TO_SOURCE_REGISTRY.json \
  --output "reports/engagement/$RUN_ID/handoff.json"
```

The command accepts only a hash-bound, complete context with a unique captured
target and parent chain, the latest matching provisional review and draft,
authorized sources, and a draft that passes deterministic validation. It
creates a private mode-0600 JSON packet with evidence, provenance and a blank
`live_unverified` manual-check form. It makes no network or posting call.

If you later consider posting, open the packet permalink yourself in a browser.
Inspect the current target, newer replies, removals or locks, and community
rules. Record what you saw and when. If you choose to post, do it manually and
recheck immediately beforehand. Never treat an archive verdict, a successful
export or `human_checked_replyable` as permission for software to publish.

Keep preparation acceptance separate from later target availability. If a
person reports that the target is deleted, retain the historical draft and
record the report and decision separately. Do not invent an observation time
to fill a formal manual-check row. Missing live-check details do not block
offline preparation.

A factual feature comparison or suggestion of an alternative can still count
as self-promotion. Check the current community rules before any manual use of
a draft; do not assume that neutral wording creates an exception. Preserve
the truthful developer disclosure when mentioning your own product.

For an aggregate pilot report, run `uv run reddit-search engagement pilot-report --help`
for the options. The manual-check and false-positive adjudication journals are
optional. The report summarizes recorded review decisions, manual checks, edits,
abstentions and explicit human adjudications; it does not report machine-triage
precision or recall as human judgments.

```sh
uv run reddit-search engagement pilot-report \
  --review-journal "reports/engagement/$RUN_ID/decisions.jsonl" \
  --draft-history "reports/engagement/$RUN_ID/reply_drafts.jsonl" \
  --manual-check-journal "reports/engagement/$RUN_ID/manual-checks.jsonl" \
  --adjudication-journal "reports/engagement/$RUN_ID/false-positive-adjudications.jsonl" \
  --output "reports/engagement/$RUN_ID/pilot-report.json"
```

Record a false-positive adjudication explicitly against a review record:

```sh
uv run reddit-search engagement adjudicate \
  --review-journal "reports/engagement/$RUN_ID/decisions.jsonl" \
  --adjudication-journal "reports/engagement/$RUN_ID/false-positive-adjudications.jsonl" \
  --app-id YOUR_APP_ID \
  --thread-fullname THREAD_FULLNAME \
  --reply-target-fullname REPLY_TARGET_FULLNAME \
  --label false_positive \
  --adjudicator YOUR_ADJUDICATOR_ID \
  --adjudicated-at 2026-09-30T12:00:00Z
```

The command selects the latest review for that exact identity. To select a
specific review record instead, replace the three identity options with
`--review-sha256 REVIEW_RECORD_SHA256`.

The record binds the label to the exact review hash, identity, review verdict,
and context-bundle hash. Criterion `engagement-fp-v1` is: "Label
false_positive only when the reviewed item clearly lacks a genuine,
first-person, currently unanswered need that the app can address using its
verified claims; label unclear when evidence is insufficient." This conservative
boundary is recorded in the report; confirm or amend it with the product owner
before relying on rate figures. The report counts each explicit adjudication in
the denominator and shows label counts, adjudicated and unadjudicated reviewed
identities, criterion versions, and the false-positive rate as false-positive
labels divided by adjudications. With no adjudications, the rate and denominator
remain `null` with status `not_recorded`; zero false positives among adjudicated
items is instead a numeric `0` with its denominator. The report never infers
false positives from `reject` or `do_not_post`.

For this Mieru pilot, Bitaru approved `engagement-fp-v1` and explicitly
recorded `true_positive` for the current review; the report contains that
adjudication. Criterion approval alone does not choose a label: record the
reviewer's explicit judgment against the exact review.