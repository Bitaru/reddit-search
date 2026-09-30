"""Build source-bound engagement candidates from frozen local selections."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reddit_search.engagement.contracts import ShortlistCandidate, ShortlistManifest
from reddit_search.ingest.invalidation import load_tombstone_ledger, tombstone_identity
from reddit_search.ingest.shard import read_selected_messages
from reddit_search.ingest.state import file_sha256

_ALLOWED_SCOPES = frozenset(
    {
        "authorized_local_research",
        "user_provided_local_research_no_external_transmission",
    }
)
_REDDIT_ID = re.compile(r"^t[13]_[A-Za-z0-9]+$")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _load_verified_selection(selection_path: Path, manifest_path: Path) -> tuple[str, str]:
    manifest = _read_json(manifest_path, "selection manifest")
    result = manifest.get("result")
    if (
        manifest.get("kind") != "discovery_manifest"
        or manifest.get("schema_version") != 2
        or not isinstance(result, dict)
        or result.get("scan_complete") is not True
        or manifest.get("budget_exhausted")
    ):
        raise ValueError("selection manifest must verify a complete, uncapped discovery scan")
    if manifest.get("output_sha256") != file_sha256(selection_path):
        raise ValueError("selection shard hash does not match its manifest")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("selection manifest lacks run_id")
    return run_id, file_sha256(manifest_path)


def _load_registry(registry_path: Path) -> dict[str, dict[str, Any]]:
    registry = _read_json(registry_path, "source registry")
    rows = registry.get("sources")
    if not isinstance(rows, list):
        raise ValueError("source registry must contain a sources list")
    sources: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source registry contains a malformed row")
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip() or source_id in sources:
            raise ValueError("source registry IDs must be non-empty and unique")
        if row.get("usage_scope") not in _ALLOWED_SCOPES:
            raise ValueError(f"source {source_id} usage_scope is not authorized locally")
        source_path = row.get("source_path")
        verified_sha = row.get("verified_sha256")
        expected_sha = row.get("expected_checksum")
        source = Path(source_path) if isinstance(source_path, str) else None
        input_size = row.get("input_size_bytes")
        if (
            row.get("status") != "complete"
            or row.get("invalid_records") != 0
            or not isinstance(verified_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", verified_sha)
            or (expected_sha is not None and expected_sha != verified_sha)
            or source is None
            or not source.is_file()
            or not isinstance(input_size, int)
            or source.stat().st_size != input_size
        ):
            raise ValueError(f"source {source_id} is not complete and checksum-verified")
        sources[source_id] = row
    return sources


def _load_ranked_scenarios(
    ranked_dir: Path, run_manifest_path: Path, app_id: str
) -> tuple[dict[str, tuple[str, set[str]]], str]:
    manifest = _read_json(run_manifest_path, "ranked run manifest")
    scenario_runs = manifest.get("scenario_runs")
    if not isinstance(scenario_runs, dict) or not scenario_runs:
        raise ValueError("ranked run manifest has no scenario_runs")
    matched: dict[str, tuple[str, set[str]]] = {}
    for scenario_id, info in scenario_runs.items():
        if not isinstance(scenario_id, str) or not isinstance(info, dict):
            raise ValueError("ranked run manifest contains malformed scenario entry")
        expected_hash = info.get("sha256")
        path = ranked_dir / f"{scenario_id}.jsonl"
        if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
            raise ValueError(f"ranked scenario hash mismatch: {scenario_id}")
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid ranked JSONL row {path}:{number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"ranked JSONL row {path}:{number} must be an object")
                fullname = row.get("message_fullname")
                thread = row.get("thread_fullname")
                row_app_id = row.get("app_id")
                retrieval = row.get("retrieval")
                if (
                    not isinstance(fullname, str)
                    or not _REDDIT_ID.fullmatch(fullname)
                    or not isinstance(thread, str)
                    or not thread.startswith("t3_")
                    or not _REDDIT_ID.fullmatch(thread)
                    or (row_app_id is not None and row_app_id != app_id)
                    or not isinstance(retrieval, dict)
                    or retrieval.get("scenario_id") != scenario_id
                ):
                    raise ValueError(
                        f"ranked JSONL row {path}:{number} has invalid identity/app/scenario"
                    )
                prior = matched.get(fullname)
                if prior is None:
                    matched[fullname] = (thread, {scenario_id})
                elif prior[0] != thread:
                    raise ValueError(f"contradictory ranked thread IDs for {fullname}")
                else:
                    prior[1].add(scenario_id)
    run_id = manifest.get("snapshot_id") or manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("ranked run manifest lacks snapshot_id or run_id")
    return matched, f"{run_id}:{file_sha256(run_manifest_path)}"


def _load_target_allowlist(path: Path) -> tuple[set[str], str]:
    raw = path.read_bytes()
    try:
        target_ids = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"target allowlist must be a UTF-8 JSON array: {path}") from error
    if not isinstance(target_ids, list) or not target_ids:
        raise ValueError("target allowlist must be a non-empty JSON array")
    if any(
        not isinstance(target_id, str) or not _REDDIT_ID.fullmatch(target_id)
        for target_id in target_ids
    ):
        raise ValueError("target allowlist entries must be exact t1_/t3_ fullnames")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("target allowlist IDs must be unique")
    return set(target_ids), hashlib.sha256(raw).hexdigest()


def build_shortlist(
    selection_path: Path,
    selection_manifest_path: Path,
    registry_path: Path,
    output_dir: Path,
    app_id: str,
    *,
    ranked_directory: Path | None = None,
    run_manifest_path: Path | None = None,
    tombstones_path: Path | None = None,
    target_allowlist_path: Path | None = None,
) -> dict[str, Any]:
    """Build fresh candidates from verified selection rows and explicit sources.

    If ranked inputs are supplied, both `ranked_directory` and
    `run_manifest_path` are required; selection messages survive only when their
    exact fullname occurs in a scenario run. Discovery rows use matched_rule_ids.
    The explicit app_id scopes this run; no app is inferred from card or text.
    An optional target allowlist is a UTF-8 JSON array of unique Reddit fullnames;
    every ID must occur in the selected shard and tombstoned targets are rejected.
    """
    selection_path = Path(selection_path)
    selection_manifest_path = Path(selection_manifest_path)
    registry_path = Path(registry_path)
    run_manifest_path = Path(run_manifest_path) if run_manifest_path is not None else None
    tombstones_path = Path(tombstones_path) if tombstones_path is not None else None
    target_allowlist_path = (
        Path(target_allowlist_path) if target_allowlist_path is not None else None
    )
    if not isinstance(app_id, str) or not app_id.strip():
        raise ValueError("app_id must not be blank")
    if (ranked_directory is None) != (run_manifest_path is None):
        raise ValueError("ranked_directory and run_manifest_path must be supplied together")
    for path in (selection_path, selection_manifest_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if ranked_directory is not None and not ranked_directory.is_dir():
        raise FileNotFoundError(ranked_directory)
    if run_manifest_path is not None and not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    if tombstones_path is not None and not tombstones_path.is_file():
        raise FileNotFoundError(tombstones_path)
    if target_allowlist_path is not None and not target_allowlist_path.is_file():
        raise FileNotFoundError(target_allowlist_path)
    if output_dir.exists():
        raise FileExistsError(f"output directory must be fresh: {output_dir}")

    selection_run_id, selection_manifest_sha = _load_verified_selection(
        selection_path, selection_manifest_path
    )
    sources = _load_registry(registry_path)
    tombstones = load_tombstone_ledger(tombstones_path)
    allowlist: set[str] | None = None
    allowlist_sha: str | None = None
    if target_allowlist_path is not None:
        allowlist, allowlist_sha = _load_target_allowlist(target_allowlist_path)
    ranked: dict[str, set[str]] | None = None
    ranked_identity: str | None = None
    if ranked_directory is not None and run_manifest_path is not None:
        ranked, ranked_identity = _load_ranked_scenarios(
            ranked_directory, run_manifest_path, app_id
        )

    gathered: dict[tuple[str, str, str], dict[str, Any]] = {}
    revisions: dict[str, tuple[str, str]] = {}
    seen_allowlist_ids: set[str] = set()
    for message in read_selected_messages(selection_path):
        if not _REDDIT_ID.fullmatch(message.fullname):
            raise ValueError(f"invalid selected message fullname: {message.fullname}")
        expected_prefix = "t3_" if message.kind == "submission" else "t1_"
        if not message.fullname.startswith(expected_prefix):
            raise ValueError(f"selected message {message.fullname} contradicts kind {message.kind}")
        if message.kind == "submission":
            if message.thread_fullname != message.fullname:
                raise ValueError(f"submission {message.fullname} contradicts its thread ID")
        elif not _REDDIT_ID.fullmatch(
            message.thread_fullname
        ) or not message.thread_fullname.startswith("t3_"):
            raise ValueError(f"comment {message.fullname} has invalid thread fullname")
        identity = (message.source_revision_id, message.thread_fullname)
        prior_identity = revisions.setdefault(message.fullname, identity)
        if prior_identity != identity:
            raise ValueError(
                f"contradictory revisions or thread ID for selected message {message.fullname}"
            )
        if allowlist is not None:
            if message.fullname not in allowlist:
                continue
            seen_allowlist_ids.add(message.fullname)
        if tombstones.matches(message.fullname, message.source_revision_id):
            if allowlist is not None:
                raise ValueError(f"target allowlist contains tombstoned message {message.fullname}")
            continue
        provenance_ids = {reference.source_id for reference in message.provenance}
        if not provenance_ids:
            raise ValueError(f"selected message {message.fullname} lacks provenance")
        for source_id in provenance_ids:
            if source_id not in sources:
                raise ValueError(f"selection references undeclared source ID: {source_id}")
        if ranked is None:
            scenario_ids = set(message.matched_rule_ids)
        else:
            ranked_record = ranked.get(message.fullname)
            if ranked_record is not None and ranked_record[0] != message.thread_fullname:
                raise ValueError(
                    f"ranked thread ID contradicts selected message {message.fullname}"
                )
            scenario_ids = ranked_record[1] if ranked_record is not None else set()
        if not scenario_ids:
            if allowlist is not None:
                raise ValueError(
                    "target allowlist message "
                    f"{message.fullname} has no discovery or ranked scenario"
                )
            continue
        key = (app_id, message.thread_fullname, message.fullname)
        item = gathered.get(key)
        if item is None:
            item = {
                "schema_version": 1,
                "app_id": app_id,
                "thread_fullname": message.thread_fullname,
                "reply_target_fullname": message.fullname,
                "source_message_fullname": message.fullname,
                "source_revision_id": message.source_revision_id,
                "permalink": message.permalink,
                "subreddit": message.subreddit,
                "scenario_references": set(),
                "source_references": set(),
                "reason_for_inclusion": (
                    "Matched explicit discovery rule or ranked scenario selection."
                ),
            }
            gathered[key] = item
        elif (
            item["source_revision_id"] != message.source_revision_id
            or item["permalink"] != message.permalink
            or item["subreddit"] != message.subreddit
            or item["thread_fullname"] != message.thread_fullname
        ):
            raise ValueError(f"contradictory identities for selected message {message.fullname}")
        item["scenario_references"].update(scenario_ids)
        for reference in message.provenance:
            item["source_references"].add(
                f"{reference.source_id}:{reference.line_number}:{message.fullname}:{message.source_revision_id}"
            )
    if allowlist is not None:
        missing_targets = allowlist - seen_allowlist_ids
        if missing_targets:
            raise ValueError(
                "target allowlist contains IDs absent from the selected shard: "
                + ", ".join(sorted(missing_targets))
            )
    rows: list[dict[str, Any]] = []
    for item in gathered.values():
        item["scenario_references"] = sorted(item["scenario_references"])
        item["source_references"] = sorted(item["source_references"])
        try:
            rows.append(ShortlistCandidate.model_validate(item).model_dump(mode="json"))
        except ValidationError as error:
            raise ValueError(f"invalid shortlist candidate: {error}") from error
    rows.sort(key=lambda row: (row["app_id"], row["thread_fullname"], row["reply_target_fullname"]))

    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    candidates_path = output_dir / "candidates.jsonl"
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        for row in rows
    )
    candidates_descriptor = os.open(
        candidates_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    os.fchmod(candidates_descriptor, 0o600)
    with os.fdopen(candidates_descriptor, "wb") as stream:
        stream.write(payload)
    inputs = {
        "selection_sha256": file_sha256(selection_path),
        "selection_manifest_sha256": selection_manifest_sha,
        "registry_sha256": file_sha256(registry_path),
        "tombstone_sha256": file_sha256(tombstones_path)
        if tombstones_path
        else hashlib.sha256(b"").hexdigest(),
        "selection_run_id": selection_run_id,
    }
    if allowlist_sha is not None:
        inputs["target_allowlist_sha256"] = allowlist_sha
    if run_manifest_path is not None and ranked_directory is not None:
        inputs["ranked_run_manifest_sha256"] = file_sha256(run_manifest_path)
        inputs["ranked_run_id_and_manifest_sha256"] = ranked_identity or ""
    manifest = ShortlistManifest(
        app_id=app_id,
        inputs=inputs,
        candidate_count=len(rows),
        candidates_sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest_path_out = output_dir / "shortlist_manifest.json"
    manifest_descriptor = os.open(
        manifest_path_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    os.fchmod(manifest_descriptor, 0o600)
    with os.fdopen(manifest_descriptor, "w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
    return {
        "candidate_count": len(rows),
        "candidates_path": str(candidates_path),
        "manifest_path": str(manifest_path_out),
        "tombstones": tombstone_identity(tombstones),
    }
