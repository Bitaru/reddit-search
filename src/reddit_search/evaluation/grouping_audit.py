"""Read-only corpus-wide exposure and grouping audit."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import zstandard

from reddit_search.ingest.state import atomic_write_json, file_sha256

_SCHEMA_VERSION = 1
_SHINGLE_SIZE = 5
_NEAR_DUPLICATE_JACCARD = 0.9


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".zst"):
        with path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8") as text:
                yield from _parse_lines(text, path)
    else:
        with path.open(encoding="utf-8") as text:
            yield from _parse_lines(text, path)


def _parse_lines(lines: Iterable[str], path: Path) -> Iterator[dict[str, Any]]:
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


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold(), re.UNICODE))


def _source(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("source", "source_bundle"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                yield item


def _identity(row: dict[str, Any]) -> tuple[str | None, str | None, str | None, tuple[str, ...]]:
    source = _source(row)
    candidate = row.get("candidate_id") or row.get("unit_id")
    message = row.get("message_fullname") or source.get("message_fullname")
    thread = row.get("thread_fullname") or source.get("thread_fullname")
    refs = tuple(_strings(source.get("context_message_refs")))
    if not refs:
        refs = tuple(_strings(row.get("context_message_refs")))
    return (
        candidate if isinstance(candidate, str) and candidate else None,
        message if isinstance(message, str) and message else None,
        thread if isinstance(thread, str) and thread else None,
        refs,
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha_bytes(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for row in rows:
            encoded = _canonical(row) + b"\n"
            output.write(encoded)
            digest.update(encoded)
    temporary.replace(path)
    return digest.hexdigest()


def _union(connection: sqlite3.Connection, left: str, right: str) -> str:
    def root(node: str) -> str:
        current = node
        while True:
            parent = connection.execute(
                "SELECT parent FROM parent WHERE node=?", (current,)
            ).fetchone()
            if parent is None:
                connection.execute(
                    "INSERT INTO parent(node,parent) VALUES (?,?)", (current, current)
                )
                return current
            if parent[0] == current:
                return current
            current = str(parent[0])

    left_root, right_root = root(left), root(right)
    if left_root != right_root:
        first, second = sorted((left_root, right_root))
        connection.execute("UPDATE parent SET parent=? WHERE node=?", (first, second))
        return first
    return left_root


def audit_corpus_grouping(
    corpus_path: Path,
    exposure_paths: Iterable[Path],
    output_directory: Path,
    *,
    duplicate_links_path: Path | None = None,
    candidate_splits_path: Path | None = None,
    corpus_manifest_path: Path | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Audit retained-corpus exposure/grouping relationships without assigning splits."""
    corpus_path, output_directory = Path(corpus_path), Path(output_directory)
    exposure_paths = tuple(Path(path) for path in exposure_paths)
    if not exposure_paths:
        raise ValueError("at least one exposure path is required")
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(corpus_path)
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript("""
        CREATE TEMP TABLE nodes (node TEXT PRIMARY KEY, kind TEXT NOT NULL, message TEXT, thread TEXT, text TEXT);
        CREATE TEMP TABLE parent (node TEXT PRIMARY KEY, parent TEXT NOT NULL);
        CREATE TEMP TABLE edges (left_node TEXT NOT NULL, right_node TEXT NOT NULL, relation TEXT NOT NULL, PRIMARY KEY(left_node,right_node,relation));
        CREATE TEMP TABLE events (ordinal INTEGER PRIMARY KEY, artifact TEXT, row_number INTEGER, payload TEXT, status TEXT, reason TEXT, nodes TEXT);
        CREATE TEMP TABLE candidate_map (candidate TEXT PRIMARY KEY, node TEXT NOT NULL);
        CREATE TEMP TABLE splits (node TEXT NOT NULL, split TEXT NOT NULL);
    """)
    relation_counts: Counter[str] = Counter()
    corpus_count = 0
    missing_message = 0
    missing_thread = 0
    unresolved_context_refs = 0
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(search_units)")}
        required = {
            "unit_id",
            "message_fullname",
            "thread_fullname",
            "focus_text",
            "context_message_refs",
        }
        if not required <= columns:
            raise ValueError(
                f"corpus lacks required search_units columns: {sorted(required - columns)}"
            )
        query = "SELECT unit_id,message_fullname,thread_fullname,focus_text,context_message_refs FROM search_units"
        if snapshot_id is not None:
            query += " WHERE snapshot_id = ?"
            rows = connection.execute(query + " ORDER BY message_fullname,unit_id", (snapshot_id,))
        else:
            rows = connection.execute(query + " ORDER BY message_fullname,unit_id")
        pending_context_refs: list[tuple[str, str]] = []
        for unit_id, message, thread, text, refs_raw in rows:
            corpus_count += 1
            if not isinstance(message, str) or not message:
                missing_message += 1
                continue
            node = f"message:{message}"
            connection.execute(
                "INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?)",
                (node, "message", message, thread, text),
            )
            connection.execute(
                "INSERT OR IGNORE INTO candidate_map VALUES (?,?)", (str(unit_id), node)
            )
            connection.execute(
                "INSERT OR IGNORE INTO candidate_map VALUES (?,?)", (str(message), node)
            )
            if isinstance(thread, str) and thread:
                thread_node = f"thread:{thread}"
                connection.execute(
                    "INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?)",
                    (thread_node, "connector", None, thread, None),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?,?,?)", (node, thread_node, "same_thread")
                )
                relation_counts["same_thread"] += 1
            else:
                missing_thread += 1
            try:
                refs = json.loads(refs_raw) if isinstance(refs_raw, str) else refs_raw
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid context_message_refs for search unit {unit_id}"
                ) from error
            if refs is None:
                refs = []
            if not isinstance(refs, list) or any(
                not isinstance(reference, str) or not reference for reference in refs
            ):
                raise ValueError(
                    f"context_message_refs for search unit {unit_id} must be a list of non-empty strings"
                )
            for reference in refs:
                pending_context_refs.append((node, reference))
        for node, reference in pending_context_refs:
            ref_node = f"message:{reference}"
            if connection.execute("SELECT 1 FROM nodes WHERE node=?", (ref_node,)).fetchone():
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?,?,?)",
                    (node, ref_node, "context_ref"),
                )
                relation_counts["context_ref"] += 1
            else:
                unresolved_context_refs += 1
        connection.commit()
        exact: dict[str, str] = {}
        for node, text in connection.execute(
            "SELECT node,text FROM nodes WHERE kind='message' AND text IS NOT NULL ORDER BY node"
        ):
            normalized = " ".join(_tokens(str(text)))
            if len(normalized) >= 1 and normalized in exact:
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?,?,?)",
                    (node, exact[normalized], "exact_text"),
                )
                relation_counts["exact_text"] += 1
            else:
                exact[normalized] = node
            tokens = _tokens(str(text))
            if len(tokens) >= _SHINGLE_SIZE:
                shingles = {
                    " ".join(tokens[i : i + _SHINGLE_SIZE])
                    for i in range(len(tokens) - _SHINGLE_SIZE + 1)
                }
                for shingle in shingles:
                    connection.execute(
                        "INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?)",
                        (f"shingle:{_sha_bytes(shingle)[:32]}", "connector", None, None, None),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO edges VALUES (?,?,?)",
                        (node, f"shingle:{_sha_bytes(shingle)[:32]}", "near_text"),
                    )
        connection.commit()
        # Compare only nodes sharing a shingle; all state is held by SQLite temp tables.
        by_shingle: dict[str, list[str]] = {}
        for node, connector in connection.execute(
            "SELECT left_node,right_node FROM edges WHERE relation='near_text'"
        ):
            by_shingle.setdefault(connector, []).append(node)
        texts = {
            node: text
            for node, text in connection.execute(
                "SELECT node,text FROM nodes WHERE kind='message' AND text IS NOT NULL"
            )
        }
        seen: set[tuple[str, str]] = set()
        for members in by_shingle.values():
            for index, left in enumerate(sorted(set(members))):
                for right in sorted(set(members))[index + 1 :]:
                    pair = (left, right)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    left_tokens = _tokens(str(texts[left]))
                    right_tokens = _tokens(str(texts[right]))
                    a = {
                        " ".join(left_tokens[i : i + _SHINGLE_SIZE])
                        for i in range(len(left_tokens) - _SHINGLE_SIZE + 1)
                    }
                    b = {
                        " ".join(right_tokens[i : i + _SHINGLE_SIZE])
                        for i in range(len(right_tokens) - _SHINGLE_SIZE + 1)
                    }
                    if a and b and len(a & b) / len(a | b) >= _NEAR_DUPLICATE_JACCARD:
                        connection.execute(
                            "INSERT OR IGNORE INTO edges VALUES (?,?,?)", (left, right, "near_text")
                        )
                        relation_counts["near_text"] += 1
        connection.commit()
        events: list[dict[str, Any]] = []
        for artifact in exposure_paths:
            for row_number, row in enumerate(_iter_jsonl(artifact), start=1):
                candidate, message, thread, refs = _identity(row)
                matches: set[str] = set()
                for value in (candidate, message):
                    if value:
                        found = connection.execute(
                            "SELECT node FROM candidate_map WHERE candidate=?", (value,)
                        ).fetchone()
                        if found:
                            matches.add(str(found[0]))
                if thread:
                    found = connection.execute(
                        "SELECT node FROM nodes WHERE node=?", (f"thread:{thread}",)
                    ).fetchone()
                    if found:
                        matches.update(
                            str(node[0])
                            for node in connection.execute(
                                "SELECT left_node FROM edges WHERE right_node=? AND relation='same_thread'",
                                (found[0],),
                            )
                        )
                for reference in refs:
                    if connection.execute(
                        "SELECT 1 FROM nodes WHERE node=?", (f"message:{reference}",)
                    ).fetchone():
                        matches.add(f"message:{reference}")
                status = "matched" if matches else "unresolved"
                reason = (
                    None
                    if matches
                    else "no exact candidate, unit, message, thread, or context reference match"
                )
                events.append(
                    {
                        "artifact": str(artifact.resolve()),
                        "row_number": row_number,
                        "payload_identity": {
                            "candidate_id": candidate,
                            "message_fullname": message,
                            "thread_fullname": thread,
                            "scenario_id": row.get("scenario_id"),
                        },
                        "status": status,
                        "reason": reason,
                        "nodes": sorted(matches),
                    }
                )
                connection.execute(
                    "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                    (
                        len(events),
                        str(artifact.resolve()),
                        row_number,
                        _canonical(row).decode(),
                        status,
                        reason,
                        json.dumps(sorted(matches)),
                    ),
                )
        connection.commit()
        duplicate_unresolved = 0
        if duplicate_links_path is not None:
            for row in _iter_jsonl(Path(duplicate_links_path)):
                left, right = row.get("candidate_id"), row.get("duplicate_of")
                if not isinstance(left, str) or not isinstance(right, str):
                    raise ValueError("duplicate link rows require candidate_id and duplicate_of")
                a = connection.execute(
                    "SELECT node FROM candidate_map WHERE candidate=?", (left,)
                ).fetchone()
                b = connection.execute(
                    "SELECT node FROM candidate_map WHERE candidate=?", (right,)
                ).fetchone()
                if a and b:
                    connection.execute(
                        "INSERT OR IGNORE INTO edges VALUES (?,?,?)", (a[0], b[0], "duplicate_link")
                    )
                    relation_counts["duplicate_link"] += 1
                else:
                    duplicate_unresolved += 1
        connection.commit()
        for left, right, _relation in connection.execute(
            "SELECT left_node,right_node,relation FROM edges ORDER BY relation,left_node,right_node"
        ):
            _union(connection, str(left), str(right))
        if candidate_splits_path is not None:
            for row in _iter_jsonl(Path(candidate_splits_path)):
                candidate, split = row.get("candidate_id"), row.get("split")
                if isinstance(candidate, str) and isinstance(split, str):
                    found = connection.execute(
                        "SELECT node FROM candidate_map WHERE candidate=?", (candidate,)
                    ).fetchone()
                    if found:
                        connection.execute("INSERT INTO splits VALUES (?,?)", (found[0], split))
        connection.commit()
        membership_rows = []
        for node, kind, message in connection.execute(
            "SELECT node,kind,message FROM nodes WHERE kind='message' ORDER BY node"
        ):
            group = _union(connection, str(node), str(node))
            relations = [
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT relation FROM edges WHERE left_node=? OR right_node=? ORDER BY relation",
                    (node, node),
                )
            ]
            membership_rows.append(
                {
                    "node_id": node,
                    "node_kind": kind,
                    "message_fullname": message,
                    "group_id": f"group-{hashlib.sha256(group.encode()).hexdigest()[:16]}",
                    "relation_types": relations,
                }
            )
        membership_hash = _atomic_jsonl(
            output_directory / "group_membership.jsonl", membership_rows
        )
        event_rows = [
            {
                "artifact": row[1],
                "row_number": row[2],
                "payload_identity": json.loads(row[3]),
                "status": row[4],
                "reason": row[5],
                "nodes": json.loads(row[6]),
            }
            for row in connection.execute(
                "SELECT ordinal,artifact,row_number,payload,status,reason,nodes FROM events ORDER BY ordinal"
            )
        ]
        events_hash = _atomic_jsonl(output_directory / "exposure_matches.jsonl", event_rows)
        unresolved = sum(row["status"] == "unresolved" for row in event_rows)
        manifest: dict[str, Any] = {
            "kind": "corpus_grouping_audit",
            "schema_version": _SCHEMA_VERSION,
            "complete": True,
            "retained_corpus_boundary": "audit covers only rows retained in supplied SQLite search_units snapshot; source archives are not rescanned",
            "snapshot_id": snapshot_id,
            "algorithm": {
                "token_normalization": "NFKC casefold Unicode word tokens",
                "shingle_size": _SHINGLE_SIZE,
                "near_text_jaccard_threshold": _NEAR_DUPLICATE_JACCARD,
            },
            "inputs": {
                "corpus": {"path": str(corpus_path.resolve()), "sha256": file_sha256(corpus_path)},
                "exposures": [
                    {"path": str(path.resolve()), "sha256": file_sha256(path)}
                    for path in exposure_paths
                ],
                "duplicate_links": {
                    "path": str(Path(duplicate_links_path).resolve()),
                    "sha256": file_sha256(Path(duplicate_links_path)),
                }
                if duplicate_links_path
                else None,
                "candidate_splits": {
                    "path": str(Path(candidate_splits_path).resolve()),
                    "sha256": file_sha256(Path(candidate_splits_path)),
                }
                if candidate_splits_path
                else None,
                "corpus_manifest": {
                    "path": str(Path(corpus_manifest_path).resolve()),
                    "sha256": file_sha256(Path(corpus_manifest_path)),
                }
                if corpus_manifest_path
                else None,
            },
            "counts": {
                "corpus_rows": corpus_count,
                "missing_message_rows": missing_message,
                "missing_thread_rows": missing_thread,
                "unresolved_context_refs": unresolved_context_refs,
                "exposure_events": len(event_rows),
                "matched_events": len(event_rows) - unresolved,
                "unresolved_events": unresolved,
                "membership_rows": len(membership_rows),
                "relation_rows": dict(sorted(relation_counts.items())),
                "split_crossing_groups": 0,
            },
            "unresolved": {
                "exposure_events": unresolved,
                "duplicate_link_references": duplicate_unresolved,
                "context_references": unresolved_context_refs,
            },
            "outputs": {
                "group_membership": {"sha256": membership_hash, "row_count": len(membership_rows)},
                "exposure_matches": {"sha256": events_hash, "row_count": len(event_rows)},
            },
        }
        if candidate_splits_path is not None:
            roots: dict[str, set[str]] = {}
            for node, split in connection.execute("SELECT node,split FROM splits"):
                roots.setdefault(_union(connection, str(node), str(node)), set()).add(str(split))
            crossings = sum(len(splits) > 1 for splits in roots.values())
            manifest["counts"]["split_crossing_groups"] = crossings
        manifest_path = output_directory / "grouping_audit_manifest.json"
        atomic_write_json(manifest_path, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest
    finally:
        connection.close()
