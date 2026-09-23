"""Deterministic, leakage-safe development/test group splits."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import zstandard

_SPLIT_SCHEMA_VERSION = 3
_IDENTITY_FIELDS = (
    "candidate_id",
    "scenario_id",
    "snapshot_id",
    "app_id",
    "app_profile_version",
    "app_profile_sha256",
    "source_revision_id",
    "source_fingerprint",
    "context_recipe_version",
    "context_dependency_identity",
)
_SHINGLE_SIZE = 5
_NEAR_DUPLICATE_JACCARD = 0.9


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".zst"):
        with path.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
                with io.TextIOWrapper(stream, encoding="utf-8") as text:
                    yield from _parse_lines(text, path)
        return
    with path.open(encoding="utf-8") as stream:
        yield from _parse_lines(stream, path)


def _parse_lines(lines: Iterator[str], path: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} line {line_number} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} must be a JSON object")
        yield value


def _pool_rows(pool_path: Path) -> list[dict[str, Any]]:
    rows = list(_iter_jsonl(pool_path))
    if not rows:
        raise ValueError(f"pool {pool_path} is empty")
    identities = [(row.get("candidate_id"), row.get("scenario_id")) for row in rows]
    if any(
        not isinstance(candidate_id, str) or not candidate_id
        or (scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id))
        for candidate_id, scenario_id in identities
    ):
        raise ValueError(f"pool {pool_path} contains an invalid task identity")
    if len(identities) != len(set(identities)):
        raise ValueError(f"pool {pool_path} contains duplicate candidate/scenario tasks")
    return rows


def _duplicate_links(path: Path) -> list[tuple[str, str, str | None]]:
    links: list[tuple[str, str, str | None]] = []
    for row in _iter_jsonl(path):
        left = row.get("candidate_id")
        right = row.get("duplicate_of")
        if not isinstance(left, str) or not left or not isinstance(right, str) or not right:
            raise ValueError(f"{path} duplicate link rows need candidate_id and duplicate_of")
        relation = row.get("relation") or row.get("link_type") or row.get("kind")
        links.append((left, right, relation if isinstance(relation, str) else None))
    return links


def _development_keys(path: Path) -> tuple[set[str], set[str], set[str], list[str]]:
    """Read prior IDs, source keys, and texts for development forcing."""
    candidate_ids: set[str] = set()
    message_fullnames: set[str] = set()
    thread_fullnames: set[str] = set()
    source_texts: list[str] = []
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("candidate_ids", []) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError(f"{path} must contain a candidate_ids list or JSON rows")
        for value in values:
            if isinstance(value, str) and value:
                candidate_ids.add(value)
            elif isinstance(value, dict):
                _collect_development_keys(
                    value,
                    candidate_ids,
                    message_fullnames,
                    thread_fullnames,
                    source_texts,
                )
        return candidate_ids, message_fullnames, thread_fullnames, source_texts

    for row in _iter_jsonl(path):
        _collect_development_keys(
            row,
            candidate_ids,
            message_fullnames,
            thread_fullnames,
            source_texts,
        )
    return candidate_ids, message_fullnames, thread_fullnames, source_texts


def _collect_development_keys(
    row: dict[str, Any],
    candidate_ids: set[str],
    message_fullnames: set[str],
    thread_fullnames: set[str],
    source_texts: list[str],
) -> None:
    candidate_id = row.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id:
        candidate_ids.add(candidate_id)
    duplicate_of = row.get("duplicate_of")
    if isinstance(duplicate_of, str) and duplicate_of:
        candidate_ids.add(duplicate_of)
    message_fullname = row.get("message_fullname")
    if isinstance(message_fullname, str) and message_fullname:
        message_fullnames.add(message_fullname)
    thread_fullname = row.get("thread_fullname")
    if isinstance(thread_fullname, str) and thread_fullname:
        thread_fullnames.add(thread_fullname)
    source = row.get("source")
    if isinstance(source, dict):
        source_message = source.get("message_fullname")
        if isinstance(source_message, str) and source_message:
            message_fullnames.add(source_message)
        source_thread = source.get("thread_fullname")
        if isinstance(source_thread, str) and source_thread:
            thread_fullnames.add(source_thread)
    text = _source_text(row)
    if text:
        source_texts.append(text)

def _source_text(row: dict[str, Any]) -> str:
    source = row.get("source")
    if isinstance(source, dict):
        for key in ("text", "focus_text", "body", "selftext"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
    for key in ("focus_text", "text", "body", "selftext"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _thread_fullname(row: dict[str, Any]) -> str | None:
    value = row.get("thread_fullname")
    if isinstance(value, str) and value:
        return value
    source = row.get("source")
    if isinstance(source, dict):
        value = source.get("thread_fullname")
        if isinstance(value, str) and value:
            return value
    return None


def _tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _near_duplicate_edges(
    pool: list[dict[str, Any]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    normalized: dict[str, set[str]] = {}
    token_sets: dict[str, set[tuple[str, ...]]] = {}
    for row in pool:
        candidate_id = row["candidate_id"]
        text = _source_text(row)
        tokens = _tokens(text)
        if len(tokens) < _SHINGLE_SIZE:
            continue
        normalized.setdefault(" ".join(tokens), set()).add(candidate_id)
        token_sets[candidate_id] = {
            tokens[index : index + _SHINGLE_SIZE]
            for index in range(len(tokens) - _SHINGLE_SIZE + 1)
        }

    exact: set[tuple[str, str]] = set()
    for candidate_ids in normalized.values():
        for left, right in itertools.combinations(sorted(candidate_ids), 2):
            exact.add((left, right))

    buckets: dict[tuple[str, ...], set[str]] = {}
    for candidate_id, shingles in token_sets.items():
        for shingle in shingles:
            buckets.setdefault(shingle, set()).add(candidate_id)
    possible: set[tuple[str, str]] = set()
    for candidate_ids in buckets.values():
        for left, right in itertools.combinations(sorted(candidate_ids), 2):
            possible.add((left, right))
    near: set[tuple[str, str]] = set()
    for left, right in sorted(possible - exact):
        left_shingles = token_sets[left]
        right_shingles = token_sets[right]
        union = left_shingles | right_shingles
        if union and len(left_shingles & right_shingles) / len(union) >= _NEAR_DUPLICATE_JACCARD:
            near.add((left, right))
    return exact, near


def _development_near_duplicate_matches(
    pool: list[dict[str, Any]], development_texts: list[str]
) -> set[str]:
    """Find pool candidates matching a previously exposed source text."""
    development_token_sets: list[set[tuple[str, ...]]] = []
    development_normalized: set[str] = set()
    buckets: dict[tuple[str, ...], set[int]] = {}
    for text in development_texts:
        tokens = _tokens(text)
        if len(tokens) < _SHINGLE_SIZE:
            continue
        development_normalized.add(" ".join(tokens))
        shingles = {
            tokens[index : index + _SHINGLE_SIZE]
            for index in range(len(tokens) - _SHINGLE_SIZE + 1)
        }
        development_token_sets.append(shingles)
        for shingle in shingles:
            buckets.setdefault(shingle, set()).add(len(development_token_sets) - 1)

    matches: set[str] = set()
    for row in pool:
        candidate_id = row["candidate_id"]
        tokens = _tokens(_source_text(row))
        if len(tokens) < _SHINGLE_SIZE:
            continue
        if " ".join(tokens) in development_normalized:
            matches.add(candidate_id)
            continue
        shingles = {
            tokens[index : index + _SHINGLE_SIZE]
            for index in range(len(tokens) - _SHINGLE_SIZE + 1)
        }
        possible = {
            index for shingle in shingles for index in buckets.get(shingle, set())
        }
        if any(
            (union := shingles | development_token_sets[index])
            and len(shingles & development_token_sets[index]) / len(union)
            >= _NEAR_DUPLICATE_JACCARD
            for index in possible
        ):
            matches.add(candidate_id)
    return matches


def _group_sort_key(group_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0group\0{group_id}".encode()).hexdigest()
    return digest, group_id


def _group_id(members: tuple[str, ...]) -> str:
    digest = hashlib.sha256("\0".join(members).encode()).hexdigest()
    return f"group-{digest[:16]}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _identity_projection(row: dict[str, Any]) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for key in _IDENTITY_FIELDS:
        if key == "candidate_id":
            value = row.get(key)
        else:
            value = row.get(key)
            if key in ("source_revision_id", "context_recipe_version") and key not in row:
                value = next(
                    (
                        source[key]
                        for source in (row.get("source"), row.get("source_bundle"))
                        if isinstance(source, dict) and key in source
                    ),
                    None,
                )
        if value is not None:
            projection[key] = value
    return projection


def _identity_digest(rows: list[dict[str, Any]]) -> str:
    projections = [_identity_projection(row) for row in rows]
    payload = json.dumps(projections, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity_coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {key: sum(key in _identity_projection(row) for row in rows) for key in _IDENTITY_FIELDS}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_group_split(
    pool_path: Path,
    output_directory: Path,
    duplicate_links_path: Path | None = None,
    seed: int = 20260912,
    test_fraction: float = 0.5,
    development_path: Path | None = None,
) -> dict[str, Any]:
    """Assign whole connected groups to development or test deterministically."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1 exclusive")
    pool_path = Path(pool_path)
    pool = _pool_rows(pool_path)
    pool_ids = {row["candidate_id"] for row in pool}
    union_find = _UnionFind()
    for row in pool:
        candidate_id = row["candidate_id"]
        union_find.add(candidate_id)
        thread = _thread_fullname(row)
        if thread is not None:
            union_find.union(candidate_id, f"thread:{thread}")

    duplicate_links: list[tuple[str, str, str | None]] = []
    resolved_duplicate_links: list[tuple[str, str, str | None]] = []
    unresolved_duplicate_links: list[tuple[str, str]] = []
    if duplicate_links_path is not None:
        duplicate_links = _duplicate_links(duplicate_links_path)
        for left, right, relation in duplicate_links:
            if left in pool_ids and right in pool_ids:
                union_find.union(left, right)
                resolved_duplicate_links.append((left, right, relation))
            else:
                unresolved_duplicate_links.append((left, right))

    exact_edges, near_edges = _near_duplicate_edges(pool)
    for left, right in exact_edges | near_edges:
        union_find.union(left, right)

    members_by_root: dict[str, set[str]] = {}
    for row in pool:
        candidate_id = row["candidate_id"]
        members_by_root.setdefault(union_find.find(candidate_id), set()).add(candidate_id)
    members_by_group: dict[str, tuple[str, ...]] = {}
    candidate_group: dict[str, str] = {}
    for members in members_by_root.values():
        ordered_members = tuple(sorted(members))
        group_id = _group_id(ordered_members)
        members_by_group[group_id] = ordered_members
        for candidate_id in ordered_members:
            candidate_group[candidate_id] = group_id
    task_count_by_group = {
        group_id: sum(1 for row in pool if candidate_group[row["candidate_id"]] == group_id)
        for group_id in members_by_group
    }

    development_matches: set[str] = set()
    development_candidate_matches: set[str] = set()
    development_message_matches: set[str] = set()
    development_thread_matches: set[str] = set()
    development_near_duplicate_matches: set[str] = set()
    development_candidate_ids: set[str] = set()
    development_message_fullnames: set[str] = set()
    development_thread_fullnames: set[str] = set()
    development_texts: list[str] = []
    if development_path is not None:
        (
            development_candidate_ids,
            development_message_fullnames,
            development_thread_fullnames,
            development_texts,
        ) = _development_keys(development_path)
        for row in pool:
            candidate_id = row["candidate_id"]
            if candidate_id in development_candidate_ids:
                development_candidate_matches.add(candidate_id)
            source = row.get("source")
            message_fullname = source.get("message_fullname") if isinstance(source, dict) else None
            if message_fullname in development_message_fullnames:
                development_message_matches.add(candidate_id)
            thread = _thread_fullname(row)
            if thread in development_thread_fullnames:
                development_thread_matches.add(candidate_id)
        development_near_duplicate_matches = _development_near_duplicate_matches(
            pool, development_texts
        )
        development_matches = (
            development_candidate_matches
            | development_message_matches
            | development_thread_matches
            | development_near_duplicate_matches
        )
    forced_development_groups = {
        candidate_group[candidate_id] for candidate_id in development_matches
    }

    ordered_groups = sorted(
        members_by_group,
        key=lambda group_id: _group_sort_key(group_id, seed),
    )
    target_test = round(len(pool) * test_fraction)
    test_groups: set[str] = set()
    assigned_test = 0
    for group_id in ordered_groups:
        if group_id in forced_development_groups:
            continue
        group_task_count = task_count_by_group[group_id]
        if assigned_test + group_task_count <= target_test or not test_groups:
            test_groups.add(group_id)
            assigned_test += group_task_count
        if assigned_test >= target_test:
            break

    mapping: dict[str, dict[str, str]] = {}
    for candidate_id in sorted(candidate_group):
        group_id = candidate_group[candidate_id]
        mapping[candidate_id] = {
            "group_id": group_id,
            "split": "test" if group_id in test_groups else "dev",
        }
    dev_candidates = {
        candidate_id for candidate_id, entry in mapping.items() if entry["split"] == "dev"
    }
    test_candidates = set(mapping) - dev_candidates
    if dev_candidates & test_candidates or dev_candidates | test_candidates != pool_ids:
        raise AssertionError("split assignment does not cover the pool disjointly")
    for group_id in members_by_group:
        group_splits = {
            mapping[candidate_id]["split"] for candidate_id in members_by_group[group_id]
        }
        if len(group_splits) != 1:
            raise AssertionError(f"group leakage: {group_id} crosses splits")

    output_directory.mkdir(parents=True, exist_ok=True)
    mapping_path = output_directory / "group_split_mapping.json"
    _write_json(mapping_path, mapping)

    split_rows: list[dict[str, Any]] = []
    for row in pool:
        candidate_id = row["candidate_id"]
        entry: dict[str, Any] = {
            "candidate_id": candidate_id,
            "group_id": mapping[candidate_id]["group_id"],
            "split": mapping[candidate_id]["split"],
        }
        if row.get("scenario_id") is not None:
            entry["scenario_id"] = row["scenario_id"]
        split_rows.append(entry)
    split_rows_path = output_directory / "candidate_splits.jsonl"
    temporary_rows_path = split_rows_path.with_suffix(".jsonl.tmp")
    temporary_rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows),
        encoding="utf-8",
    )
    temporary_rows_path.replace(split_rows_path)

    manifest = {
        "kind": "group_split_manifest",
        "schema_version": _SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "test_fraction": test_fraction,
        "pool_path": str(pool_path),
        "pool_sha256": _sha(pool_path),
        "pool_count": len(pool),
        "pool_candidate_count": len(pool_ids),
        "group_count": len(members_by_group),
        "group_sizes": sorted(len(members) for members in members_by_group.values()),
        "group_task_sizes": sorted(task_count_by_group.values()),
        "dev_count": len(dev_candidates),
        "test_count": len(test_candidates),
        "dev_task_count": sum(
            task_count_by_group[group_id]
            for group_id in members_by_group
            if mapping[members_by_group[group_id][0]]["split"] == "dev"
        ),
        "test_task_count": sum(
            task_count_by_group[group_id]
            for group_id in members_by_group
            if mapping[members_by_group[group_id][0]]["split"] == "test"
        ),
        "duplicate_links_input_count": len(duplicate_links),
        "duplicate_links_used": len(resolved_duplicate_links),
        "duplicate_links_unresolved_count": len(unresolved_duplicate_links),
        "duplicate_links_unresolved": [
            {"candidate_id": left, "duplicate_of": right}
            for left, right in unresolved_duplicate_links
        ],
        "duplicate_links_sha256": _sha(duplicate_links_path) if duplicate_links_path else None,
        "known_crosspost_links_used": sum(
            1
            for _, _, relation in resolved_duplicate_links
            if relation and "cross" in relation.casefold()
        ),
        "exact_duplicate_link_count": len(exact_edges),
        "near_duplicate_link_count": len(near_edges),
        "development_source": str(development_path) if development_path else None,
        "development_source_sha256": _sha(development_path) if development_path else None,
        "development_match_count": len(development_matches),
        "development_candidate_match_count": len(development_candidate_matches),
        "development_message_match_count": len(development_message_matches),
        "development_thread_match_count": len(development_thread_matches),
        "development_near_duplicate_match_count": len(development_near_duplicate_matches),
        "development_group_count": len(forced_development_groups),
        "mapping_file": mapping_path.name,
        "candidate_splits_file": split_rows_path.name,
    }
    manifest["dependency_identity_digest"] = _identity_digest(pool)
    manifest["dependency_identity_coverage"] = _identity_coverage(pool)
    manifest["mapping_sha256"] = _sha(mapping_path)
    manifest["candidate_splits_sha256"] = _sha(split_rows_path)
    manifest_path = output_directory / "group_split_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest
