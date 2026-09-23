"""Deterministic, blinded evaluation pool construction.

Builds a reviewer-facing pool from ranked lexical runs plus deterministic
rejected controls. Pool rows hide retrieval rank, score, selection metadata,
and stratum; the operator keeps the unblinding key and manifest.

Pool rows are unique task identities: ``(candidate_id, scenario_id)`` for
ranked questions and ``(candidate_id, None)`` for controls. Repeated rows for
the same task across retrieval variants collapse to one row, but the same
message in different scenarios remains separately reviewable. Strata derive
from retrieval rank (ranked rows) or control origin:
- ``strong``: ranked row within the top ``_STRONG_MAX_RANK`` ranks
- ``adjacent``: ranked row beyond that
- ``unknown``: deterministic rejected-control row
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import zstandard

from reddit_search.config import load_product_profiles

_POOL_SCHEMA_VERSION = 2
_STRONG_MAX_RANK = 3
_STRATA = ("strong", "adjacent", "unknown")
_HIDDEN_REVIEWER_FIELDS = frozenset(
    {
        "rank",
        "score",
        "retrieval",
        "matched_rule_ids",
        "selection_channels",
        "selection_order",
        "selection_metadata",
        "archive_score",
        "stratum",
    }
)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".zst"):
        with path.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
                with io.TextIOWrapper(stream, encoding="utf-8") as text:
                    yield from _parse_lines(text, path)
    else:
        with path.open(encoding="utf-8") as text:
            yield from _parse_lines(text, path)


def _parse_lines(lines: Iterator[str], path: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} must be a JSON object")
        yield value


def _row_id(row: dict[str, Any]) -> str | None:
    for key in ("candidate_id", "unit_id", "fullname", "message_fullname"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _row_scenario(row: dict[str, Any]) -> str | None:
    value = row.get("scenario_id")
    if isinstance(value, str) and value:
        return value
    retrieval = row.get("retrieval")
    if isinstance(retrieval, dict):
        value = retrieval.get("scenario_id")
        if isinstance(value, str) and value:
            return value
    return None


def _row_thread(row: dict[str, Any]) -> str | None:
    value = row.get("thread_fullname")
    return value if isinstance(value, str) and value else None


def _row_rank(row: dict[str, Any]) -> int | None:
    value = row.get("rank")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _stratify_ranked(row: dict[str, Any]) -> str:
    rank = _row_rank(row)
    return "strong" if rank is not None and rank <= _STRONG_MAX_RANK else "adjacent"


def _run_files(runs_directory: Path) -> list[Path]:
    files = [
        path
        for path in sorted(runs_directory.iterdir())
        if path.is_file()
        and path.name != "run_manifest.json"
        and (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.zst"))
    ]
    if not files:
        raise ValueError(f"no ranked run JSONL files under {runs_directory}")
    return files


def _retained_sources(source_path: Path | None) -> dict[str, dict[str, Any]]:
    if source_path is None:
        return {}
    sources: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(source_path):
        identities = {
            row.get("candidate_id"),
            row.get("unit_id"),
            row.get("fullname"),
            row.get("message_fullname"),
        }
        for identity in identities:
            if isinstance(identity, str) and identity:
                sources[identity] = row
    return sources
def _enrich_retained_sources(
    retained: dict[str, dict[str, Any]], corpus: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Alias retained message sources by corpus unit and message identities."""
    enriched = dict(retained)
    for unit_id, unit in corpus.items():
        if unit_id in enriched:
            continue
        fullname = unit.get("message_fullname")
        if isinstance(fullname, str) and fullname in retained:
            enriched[unit_id] = retained[fullname]
    return enriched



def _corpus_units(corpus_path: Path | None) -> dict[str, dict[str, Any]]:
    if corpus_path is None:
        return {}
    connection = sqlite3.connect(f"file:{corpus_path}?mode=ro", uri=True)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(search_units)")}
        if not columns:
            raise ValueError(f"{corpus_path} has no search_units table")
        available = [
            name
            for name in (
                "unit_id",
                "snapshot_id",
                "message_fullname",
                "source_revision_id",
                "thread_fullname",
                "focus_field",
                "focus_start",
                "focus_end",
                "focus_text",
                "context_text",
                "context_only_text",
                "missing_context_ids",
                "ancestors_truncated",
                "context_message_refs",
                "permalink",
                "subreddit",
                "created_utc",
                "chunking_version",
                "context_recipe_version",
            )
            if name in columns
        ]
        units: dict[str, dict[str, Any]] = {}
        query = f"SELECT {', '.join(available)} FROM search_units"
        for values in connection.execute(query):
            unit = dict(zip(available, values, strict=True))
            unit_id = unit.get("unit_id")
            if isinstance(unit_id, str) and unit_id:
                units[unit_id] = unit
            message_fullname = unit.get("message_fullname")
            if isinstance(message_fullname, str) and message_fullname:
                units.setdefault(message_fullname, unit)
        return units
    finally:
        connection.close()


def _profile_identity(
    directory: Path | None,
) -> tuple[dict[str, int] | None, dict[str, str] | None, dict[str, Any]]:
    if directory is None:
        return None, None, {"path": None, "sha256": None, "versions": {}}
    profiles = load_product_profiles(directory)
    if not profiles:
        raise ValueError(f"profiles directory is empty: {directory}")
    entries = [
        {
            "app_id": profile.app_id,
            "profile_version": profile.profile_version,
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path, profile in zip(sorted(directory.glob("*.yaml")), profiles, strict=True)
    ]
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return (
        {entry["app_id"]: entry["profile_version"] for entry in entries},
        {entry["app_id"]: entry["sha256"] for entry in entries},
        {
            "path": str(directory.resolve()),
            "sha256": digest,
            "versions": {entry["app_id"]: entry["profile_version"] for entry in entries},
        },
    )


def _stable_order(row_id: str, tag: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{tag}\0{row_id}".encode()).hexdigest()


def _control_source(row: dict[str, Any]) -> dict[str, Any]:
    """Standardize a rejected-control message into a source bundle."""
    body = row.get("raw_body")
    title = row.get("raw_title")
    kind = row.get("kind")
    field = "selftext" if kind in {"submission", "post"} else "body"
    text = body if isinstance(body, str) and body else title
    if not isinstance(text, str):
        text = ""
    if not body and title:
        field = "title"
    source = {
        "message_fullname": row.get("fullname") or row.get("message_fullname"),
        "source_revision_id": row.get("source_revision_id"),
        "field": field,
        "start": 0,
        "end": len(text),
        "text": text,
        field: text,
        "kind": kind,
        "thread_fullname": _row_thread(row),
        "raw_title": title,
        "raw_body": body,
        "subreddit": row.get("subreddit"),
        "created_utc": row.get("created_utc"),
        "permalink": row.get("permalink"),
        "provenance": row.get("provenance"),
    }
    return source


def _ranked_source(row: dict[str, Any], unit: dict[str, Any] | None) -> dict[str, Any]:
    if unit is not None:
        focus_text = unit.get("focus_text")
        field = unit.get("focus_field")
        # Ranked rows may carry the canonical message field; offsets always
        # refer to that full field, never to the retrieval focus slice.
        canonical_text = None
        if field in {"body", "selftext"}:
            canonical_text = row.get("raw_body")
        elif field == "title":
            canonical_text = row.get("raw_title")
        if not isinstance(canonical_text, str):
            canonical_text = row.get(field) if isinstance(field, str) else None
        start = unit.get("focus_start")
        end = unit.get("focus_end")
        if not isinstance(canonical_text, str):
            if (
                isinstance(focus_text, str)
                and start == 0
                and end == len(focus_text)
            ):
                canonical_text = focus_text
            else:
                raise ValueError(
                    "ranked source lacks canonical full field text for absolute focus offsets"
                )
        source = {
            "message_fullname": unit.get("message_fullname"),
            "source_revision_id": unit.get("source_revision_id"),
            "field": field,
            "start": unit.get("focus_start"),
            "end": unit.get("focus_end"),
            "text": canonical_text,
            "focus_text": focus_text,
            "focus_start": unit.get("focus_start"),
            "focus_end": unit.get("focus_end"),
            "title": row.get("title", row.get("raw_title")),
            "thread_fullname": unit.get("thread_fullname"),
            "permalink": unit.get("permalink"),
            "subreddit": unit.get("subreddit"),
            "created_utc": unit.get("created_utc"),
            "context_text": unit.get("context_text"),
            "context_only_text": unit.get("context_only_text"),
            "missing_context_ids": unit.get("missing_context_ids"),
            "ancestors_truncated": unit.get("ancestors_truncated"),
            "context_message_refs": unit.get("context_message_refs"),
            "chunking_version": unit.get("chunking_version"),
            "context_recipe_version": unit.get("context_recipe_version"),
        }
        if isinstance(field, str) and isinstance(canonical_text, str):
            source[field] = canonical_text
        return {key: value for key, value in source.items() if value is not None}

    source = {
        key: value
        for key, value in row.items()
        if key not in _HIDDEN_REVIEWER_FIELDS
        and key not in {"candidate_id", "unit_id", "scenario_id", "provenance",
                        "source_revision_id", "selection"}
    }
    focus_text = source.get("focus_text")
    field = source.get("field")
    if not isinstance(field, str) or not field:
        field = "selftext" if "selftext" in source else "body"
    canonical_text = source.get(field)
    if not isinstance(canonical_text, str):
        canonical_text = source.get("text")
    if not isinstance(canonical_text, str):
        focus_text = source.get("focus_text")
        start = source.get("focus_start", 0)
        end = source.get("focus_end", len(focus_text) if isinstance(focus_text, str) else 0)
        if not (isinstance(focus_text, str) and start == 0 and end == len(focus_text)):
            raise ValueError(
                "ranked source lacks canonical full field text for absolute focus offsets"
            )
        canonical_text = focus_text
    source.setdefault("message_fullname", row.get("message_fullname"))
    source.setdefault("source_revision_id", row.get("source_revision_id"))
    source.setdefault("field", field)
    source.setdefault("start", row.get("focus_start", 0))
    source.setdefault("end", row.get("focus_end", len(canonical_text)))
    source["text"] = canonical_text
    source[field] = canonical_text
    if "provenance" in row:
        source["provenance"] = row["provenance"]
    return source


def _context_complete(unit: dict[str, Any] | None, row: dict[str, Any]) -> bool | None:
    if unit is None:
        value = row.get("context_complete")
        return value if isinstance(value, bool) else None
    truncated = unit.get("ancestors_truncated")
    missing = unit.get("missing_context_ids")
    if truncated not in (0, False, None):
        return False
    if missing is None:
        return False
    if isinstance(missing, str):
        try:
            missing = json.loads(missing)
        except json.JSONDecodeError:
            return False
    return isinstance(missing, list) and not missing


def _scenario_app_ids(runs_directory: Path) -> dict[str, str | None]:
    """Map scenario IDs to their optional app binding, loaded from run rows."""
    app_ids: dict[str, str | None] = {}
    for path in _run_files(runs_directory):
        for row in _iter_jsonl(path):
            scenario_id = _row_scenario(row) or path.stem
            if scenario_id not in app_ids:
                raw = row.get("app_id")
                app_ids[scenario_id] = raw if isinstance(raw, str) else None
    return app_ids


def _reviewer_row(
    row: dict[str, Any],
    *,
    pool_id: str,
    corpus: dict[str, dict[str, Any]],
    candidate_id: str,
    is_control: bool,
    scenario_app_ids: dict[str, str | None],
    profile_versions: dict[str, int] | None,
    profile_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    unit = None if is_control else corpus.get(candidate_id)
    source = _control_source(row) if is_control else _ranked_source(row, unit)
    scenario_id = None if is_control else _row_scenario(row)
    app_id = scenario_app_ids.get(scenario_id) if scenario_id else None
    if profile_versions is None:
        app_profile_version = None
        app_profile_sha256 = None
    else:
        # Profile mode binds profiles only through explicit app_id in run rows;
        # topic-only scenarios (no app_id) stay unbound rather than guessed.
        if app_id is not None and app_id not in profile_versions:
            raise ValueError(f"no product profile found for app_id: {app_id}")
        app_profile_version = profile_versions.get(app_id) if app_id else None
        app_profile_sha256 = profile_hashes.get(app_id) if app_id and profile_hashes else None
    thread = _row_thread(row) or (
        unit.get("thread_fullname")
        if unit and isinstance(unit.get("thread_fullname"), str)
        else source.get("thread_fullname")
    )
    snapshot_id = unit.get("snapshot_id") if unit else row.get("snapshot_id")
    return {
        "pool_id": pool_id,
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "app_id": app_id,
        "app_profile_version": app_profile_version,
        "app_profile_sha256": app_profile_sha256,
        "thread_fullname": thread,
        "snapshot_id": snapshot_id,
        "context_complete": _context_complete(unit, row),
        "source": source,
    }
def build_blinded_pool(
    runs_directory: Path,
    controls_path: Path,
    output_directory: Path,
    target_count: int = 240,
    seed: int = 20260912,
    corpus_path: Path | None = None,
    profiles_directory: Path | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic blinded pool of unique message-task identities.

    Repeated ``candidate_id`` rows collapse only within the same
    ``(candidate_id, scenario_id)`` task.
    """
    if target_count < 1:
        raise ValueError("target_count must be positive")
    runs_directory = Path(runs_directory)
    controls_path = Path(controls_path)
    output_directory = Path(output_directory)
    if corpus_path is not None:
        corpus_path = Path(corpus_path)
    if profiles_directory is not None:
        profiles_directory = Path(profiles_directory)
    if source_path is not None:
        source_path = Path(source_path)
    profile_versions, profile_hashes, profile_identity = _profile_identity(profiles_directory)
    scenario_app_ids = _scenario_app_ids(runs_directory)

    # Collapse repeated ranked rows to one task per candidate and scenario.
    ranked_rows = 0
    by_task: dict[tuple[str, str], tuple[int, str, dict[str, Any]]] = {}
    for path in _run_files(runs_directory):
        for row in _iter_jsonl(path):
            candidate_id = _row_id(row)
            if not candidate_id:
                raise ValueError(f"ranked run row in {path} lacks candidate identity")
            ranked_rows += 1
            scenario_id = _row_scenario(row) or path.stem
            rank = _row_rank(row) or 2**31
            task = (candidate_id, scenario_id)
            key = (rank, path.name)
            existing = by_task.get(task)
            if existing is None or key < (existing[0], existing[1]):
                by_task[task] = (rank, scenario_id, row)
    duplicate_row_count = ranked_rows - len(by_task)
    duplicate_cross_scenario_count = sum(
        1
        for candidate_id in {candidate for candidate, _ in by_task}
        if sum(1 for candidate, _ in by_task if candidate == candidate_id) > 1
    )

    ranked = sorted(
        by_task.items(),
        key=lambda item: (
            _stable_order(f"{item[0][0]}\0{item[0][1]}", "ranked", seed),
            item[0],
        ),
    )
    ranked_candidate_ids = {candidate_id for candidate_id, _ in by_task}

    control_rows: list[dict[str, Any]] = []
    seen_control_ids: set[str] = set()
    for row in _iter_jsonl(controls_path):
        candidate_id = _row_id(row)
        if not candidate_id:
            raise ValueError("control row lacks candidate identity")
        if candidate_id in ranked_candidate_ids or candidate_id in seen_control_ids:
            continue
        seen_control_ids.add(candidate_id)
        control_rows.append(row)
    control_rows.sort(
        key=lambda row: (
            _stable_order(str(_row_id(row)), "control", seed),
            str(_row_id(row)),
        )
    )

    chosen: list[tuple[dict[str, Any], str, bool]] = []
    # Reserve a proportional control slice so ranked rows cannot starve controls.
    ranked_limit = target_count - min(len(control_rows), target_count // 3)
    for _, (_, _, row) in ranked:
        if len(chosen) >= ranked_limit:
            break
        chosen.append((row, _stratify_ranked(row), False))
    for row in control_rows:
        if len(chosen) >= target_count:
            break
        chosen.append((row, "unknown", True))

    corpus = _corpus_units(corpus_path)
    retained = _enrich_retained_sources(_retained_sources(source_path), corpus)
    pool_rows: list[dict[str, Any]] = []
    stratum_counts = dict.fromkeys(_STRATA, 0)
    for index, (row, stratum, is_control) in enumerate(chosen, start=1):
        stratum_counts[stratum] += 1
        candidate_id = str(_row_id(row))
        if not is_control and candidate_id in retained:
            row = {**retained[candidate_id], **row}
        pool_rows.append(
            _reviewer_row(
                row,
                pool_id=f"pool-{index:04d}",
                corpus=corpus,
                candidate_id=candidate_id,
                is_control=is_control,
                scenario_app_ids=scenario_app_ids,
                profile_versions=profile_versions,
                profile_hashes=profile_hashes,
            )
        )
    pool_keys = [(row["candidate_id"], row["scenario_id"]) for row in pool_rows]
    if len(pool_keys) != len(set(pool_keys)):
        raise AssertionError("pool contains duplicate candidate/scenario task identities")

    output_directory.mkdir(parents=True, exist_ok=True)
    pool_path = output_directory / "blinded_pool.jsonl"
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pool_rows
    )
    temporary_path = pool_path.with_suffix(".jsonl.tmp")
    temporary_path.write_text(payload, encoding="utf-8")
    temporary_path.replace(pool_path)

    manifest = {
        "kind": "blinded_pool_manifest",
        "schema_version": _POOL_SCHEMA_VERSION,
        "seed": seed,
        "target_count": target_count,
        "pool_count": len(pool_rows),
        "ranked_task_count": len(by_task),
        "duplicate_row_count": duplicate_row_count,
        "duplicate_cross_scenario_count": duplicate_cross_scenario_count,
        "strong_max_rank": _STRONG_MAX_RANK,
        "stratum_counts": stratum_counts,
        "runs_directory": str(runs_directory),
        "controls_path": str(controls_path),
        "controls_sha256": hashlib.sha256(controls_path.read_bytes()).hexdigest(),
        "corpus_path": str(corpus_path) if corpus_path is not None else None,
        "profiles": profile_identity,
        "pool_file": pool_path.name,
        "pool_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    manifest_path = output_directory / "pool_manifest.json"
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)

    key = {
        f"pool-{index:04d}": {
            "candidate_id": str(_row_id(row)),
            "scenario_id": _row_scenario(row),
            "stratum": stratum,
            "source_kind": "control" if is_control else "ranked",
            "rank": _row_rank(row),
            "score": row.get("score"),
        }
        for index, (row, stratum, is_control) in enumerate(chosen, start=1)
    }
    key_path = output_directory / "pool_key.json"
    key_temporary = key_path.with_suffix(".json.tmp")
    key_temporary.write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    key_temporary.replace(key_path)
    return manifest
