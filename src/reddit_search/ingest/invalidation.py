"""Deterministic tombstone ledgers for source-preserving invalidation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reddit_search.ingest.state import atomic_write_json, canonical_json_bytes, file_sha256


@dataclass(frozen=True, slots=True)
class Tombstone:
    """Invalidate one message, optionally only one source revision."""

    message_fullname: str
    source_revision_id: str | None
    reason: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "message_fullname": self.message_fullname,
            "source_revision_id": self.source_revision_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TombstoneProjection:
    """Content-free, identity-only contract for explicitly supplied tombstones.

    This is deliberately not a propagation claim: downstream artifacts are only
    named by their supplied byte hashes and are not inferred from archive rows.
    """

    records: tuple[dict[str, str | None], ...]
    source_artifact_hashes: dict[str, str]
    ledger_digest: str
    counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "tombstone_identity_projection",
            "records": [dict(record) for record in self.records],
            "source_artifact_hashes": dict(self.source_artifact_hashes),
            "ledger_digest": self.ledger_digest,
            "counts": dict(self.counts),
            "propagation_status": "not_claimed",
        }


def project_tombstone_identities(
    records: Iterable[dict[str, Any]],
    *,
    source_artifacts: dict[str, tuple[Path, str]] | None = None,
) -> TombstoneProjection:
    """Build a deterministic contract from explicit, fully identified records.

    ``source_revision_id=None`` denotes a fullname-wide tombstone.
    """
    projected: list[dict[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"tombstone record {number} must be an object")
        for field in ("status", "coverage_status"):
            if record.get(field) in {"matched", "unresolved"}:
                raise ValueError(
                    f"tombstone record {number} is archive coverage, not a tombstone"
                )
        fullname = record.get("message_fullname")
        revision = record.get("source_revision_id")
        if (
            not isinstance(fullname, str)
            or not fullname.startswith(("t1_", "t3_"))
            or (revision is not None and (not isinstance(revision, str) or not revision.strip()))
        ):
            raise ValueError(f"tombstone record {number} has invalid identity")
        key = (fullname, revision)
        if key in seen:
            raise ValueError(f"duplicate tombstone identity: {fullname}/{revision}")
        seen.add(key)
        item: dict[str, str | None] = {
            "message_fullname": fullname,
            "source_revision_id": revision,
        }
        for field in ("candidate_id", "unit_id", "snapshot_id"):
            value = record.get(field)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"invalid {field} for tombstone record {number}")
                item[field] = value
        projected.append(item)
    projected.sort(
        key=lambda item: tuple(
            item.get(k) or ""
            for k in (
                "message_fullname",
                "source_revision_id",
                "candidate_id",
                "unit_id",
                "snapshot_id",
            )
        )
    )
    hashes: dict[str, str] = {}
    for name, (path, expected) in sorted((source_artifacts or {}).items()):
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"source artifact hash mismatch: {name}")
        hashes[name] = actual
    digest = hashlib.sha256(canonical_json_bytes(projected)).hexdigest()
    return TombstoneProjection(
        records=tuple(projected),
        source_artifact_hashes=hashes,
        ledger_digest=digest,
        counts={"records": len(projected), "artifacts": len(hashes)},
    )

@dataclass(frozen=True, slots=True)
class TombstoneLedger:
    """Canonical tombstones and the identity of the ledger that supplied them."""

    path: Path | None
    digest: str | None
    records: tuple[Tombstone, ...]
    _all_revisions: frozenset[str]
    _by_revision: dict[str, frozenset[str]]

    def matches(self, message_fullname: str, source_revision_id: str) -> bool:
        """Return whether a message or its exact source revision is invalidated."""
        return message_fullname in self._all_revisions or message_fullname in self._by_revision.get(
            source_revision_id, frozenset()
        )
    @property
    def count(self) -> int:
        return len(self.records)
def tombstone_blocks_identity(
    ledger: TombstoneLedger,
    message_fullname: str,
    source_revision_id: str | None,
    context_message_refs: Iterable[str] = (),
) -> bool:
    """Return whether a source or context identity is invalidated.

    Context rows do not carry contributor revisions, so any known tombstone
    for a referenced fullname conservatively blocks the row.
    """
    if source_revision_id is not None and ledger.matches(message_fullname, source_revision_id):
        return True
    if source_revision_id is None and message_fullname in ledger._all_revisions:
        return True
    context_fullnames = {record.message_fullname for record in ledger.records}
    return any(fullname in context_fullnames for fullname in context_message_refs)


def row_has_structured_context_identity(row: Mapping[str, object]) -> bool:
    """Return whether a row carries any structured context identity field.

    Mirrors the fields ``tombstone_blocks_row`` reads: ``context_message_refs``
    at row or nested-source level, ``context.message_fullnames`` at either
    level. Rows that carry their context only as free text cannot be verified
    against an identity-only tombstone ledger, so callers can fail closed.
    """
    source = row.get("source")
    source_map = source if isinstance(source, Mapping) else row
    for container in (row, source_map):
        if isinstance(container.get("context_message_refs"), list):
            return True
        context = container.get("context")
        if isinstance(context, Mapping) and isinstance(
            context.get("message_fullnames"), list
        ):
            return True
    return False


def tombstone_blocks_row(ledger: TombstoneLedger, row: Mapping[str, object]) -> bool:
    """Check direct identity plus all supported nested/sibling context identities."""
    source = row.get("source")
    source_map = source if isinstance(source, Mapping) else row
    refs: set[str] = set()
    for container in (row, source_map):
        direct_refs = container.get("context_message_refs")
        if isinstance(direct_refs, Iterable) and not isinstance(direct_refs, (str, bytes)):
            refs.update(str(ref) for ref in direct_refs)
        context = container.get("context")
        if isinstance(context, Mapping):
            nested_refs = context.get("message_fullnames", ())
            if isinstance(nested_refs, Iterable) and not isinstance(nested_refs, (str, bytes)):
                refs.update(str(ref) for ref in nested_refs)
    return tombstone_blocks_identity(
        ledger,
        str(source_map.get("message_fullname", "")),
        source_map.get("source_revision_id"),
        refs,
    )



EMPTY_TOMBSTONE_LEDGER = TombstoneLedger(
    path=None,
    digest=None,
    records=(),
    _all_revisions=frozenset(),
    _by_revision={},
)


def load_tombstone_ledger(path: Path | None) -> TombstoneLedger:
    """Load and canonicalize an operator-supplied JSONL tombstone ledger."""
    if path is None:
        return EMPTY_TOMBSTONE_LEDGER
    records: list[Tombstone] = []
    seen: set[tuple[str, str | None]] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid tombstone JSON at {path}:{line_number}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"tombstone row at {path}:{line_number} must be an object")
            unexpected = set(payload) - {"message_fullname", "source_revision_id", "reason"}
            if unexpected:
                raise ValueError(
                    f"tombstone row at {path}:{line_number} has unknown fields: "
                    + ", ".join(sorted(str(field) for field in unexpected))
                )
            message_fullname = payload.get("message_fullname")
            source_revision_id = payload.get("source_revision_id")
            reason = payload.get("reason")
            if (
                not isinstance(message_fullname, str)
                or not message_fullname.startswith(("t1_", "t3_"))
                or len(message_fullname) <= 3
            ):
                raise ValueError(
                    f"tombstone row at {path}:{line_number} has invalid message_fullname"
                )
            if source_revision_id is not None and not isinstance(source_revision_id, str):
                raise ValueError(
                    f"tombstone row at {path}:{line_number} has invalid source_revision_id"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"tombstone row at {path}:{line_number} has an empty reason")
            key = (message_fullname, source_revision_id)
            if key in seen:
                raise ValueError(f"duplicate tombstone at {path}:{line_number}")
            seen.add(key)
            records.append(Tombstone(message_fullname, source_revision_id, reason.strip()))

    records.sort(
        key=lambda item: (item.message_fullname, item.source_revision_id or "", item.reason)
    )
    all_revisions = frozenset(
        record.message_fullname for record in records if record.source_revision_id is None
    )
    by_revision: dict[str, set[str]] = {}
    for record in records:
        if record.source_revision_id is not None:
            by_revision.setdefault(record.source_revision_id, set()).add(record.message_fullname)
    digest = hashlib.sha256(
        canonical_json_bytes([record.as_dict() for record in records])
    ).hexdigest()
    return TombstoneLedger(
        path=path.resolve(),
        digest=digest,
        records=tuple(records),
        _all_revisions=all_revisions,
        _by_revision={key: frozenset(value) for key, value in by_revision.items()},
    )


def tombstone_identity(ledger: TombstoneLedger) -> dict[str, Any]:
    """Return stable manifest metadata without embedding ledger reason text."""
    return {
        "path": str(ledger.path) if ledger.path is not None else None,
        "sha256": ledger.digest,
        "count": ledger.count,
    }


def audit_tombstone_outputs(
    ledger_path: Path,
    output_path: Path,
    *,
    selection_path: Path | None = None,
    hydration_directory: Path | None = None,
    review_cards_path: Path | None = None,
    corpus_path: Path | None = None,
) -> dict[str, Any]:
    """Audit persisted derivatives for records still covered by tombstones."""
    ledger = load_tombstone_ledger(ledger_path)
    artifact_specs: list[tuple[str, Path]] = []
    if selection_path is not None:
        artifact_specs.append(("selection_shard", selection_path))
    if hydration_directory is not None:
        artifact_specs.extend(
            (
                ("hydrated_context_shard", hydration_directory / "hydrated-context.jsonl.zst"),
                ("rejected_controls_shard", hydration_directory / "rejected-controls.jsonl.zst"),
            )
        )
    if review_cards_path is not None:
        artifact_specs.append(("review_cards", review_cards_path))
    if corpus_path is not None:
        artifact_specs.append(("corpus_sqlite", corpus_path))
    if any(output_path.resolve() == path.resolve() for _, path in artifact_specs):
        raise ValueError("audit report output must differ from every audited artifact")
    if not artifact_specs:
        raise ValueError("at least one persisted output must be supplied for audit")

    artifacts = [
        _audit_artifact(kind, path, ledger)
        for kind, path in sorted(artifact_specs, key=lambda item: (item[0], str(item[1])))
    ]
    summary = {
        "artifact_count": len(artifacts),
        "clean_artifact_count": sum(item["status"] == "clean" for item in artifacts),
        "stale_artifact_count": sum(item["status"] == "stale" for item in artifacts),
        "missing_artifact_count": sum(item["status"] == "missing" for item in artifacts),
        "invalidated_record_count": sum(
            int(item["invalidated_record_count"]) for item in artifacts
        ),
        "matched_tombstone_count": sum(int(item["matched_tombstone_count"]) for item in artifacts),
    }
    report = {
        "kind": "tombstone_audit_report",
        "schema_version": 1,
        "ledger": tombstone_identity(ledger),
        "scope": (
            "Audits only the supplied persisted derivatives. It does not reread source "
            "archives or infer invalidation for omitted artifacts."
        ),
        "artifacts": artifacts,
        "summary": summary,
        "all_supplied_artifacts_clean": (
            summary["stale_artifact_count"] == 0 and summary["missing_artifact_count"] == 0
        ),
        "report_file": str(output_path),
    }
    atomic_write_json(output_path, report)
    return report


def _audit_artifact(
    kind: str,
    path: Path,
    ledger: TombstoneLedger,
) -> dict[str, Any]:
    resolved_path = path.resolve()
    if not resolved_path.is_file():
        return {
            "kind": kind,
            "path": str(resolved_path),
            "sha256": None,
            "status": "missing",
            "record_count": 0,
            "invalidated_record_count": 0,
            "matched_tombstone_count": 0,
            "unmatched_tombstone_count": ledger.count,
            "matched_tombstones": [],
        }

    if kind in {
        "selection_shard",
        "hydrated_context_shard",
        "rejected_controls_shard",
    }:
        from .shard import read_selected_messages

        rows = (
            (message.fullname, message.source_revision_id)
            for message in read_selected_messages(resolved_path)
        )
    elif kind == "review_cards":
        rows = _iter_review_card_messages(resolved_path)
    elif kind == "corpus_sqlite":
        rows = _iter_corpus_messages(resolved_path)
    else:
        raise ValueError(f"unsupported tombstone audit artifact kind: {kind}")
    return _audit_rows(kind, resolved_path, rows, ledger)


def _audit_rows(
    kind: str,
    path: Path,
    rows: Iterable[tuple[str, str]],
    ledger: TombstoneLedger,
) -> dict[str, Any]:
    record_count = 0
    invalidated_record_count = 0
    matched_indices: set[int] = set()
    for message_fullname, source_revision_id in rows:
        record_count += 1
        matching = {
            index
            for index, record in enumerate(ledger.records)
            if record.message_fullname == message_fullname
            and (
                record.source_revision_id is None or record.source_revision_id == source_revision_id
            )
        }
        if matching:
            invalidated_record_count += 1
            matched_indices.update(matching)
    matched_tombstones = [ledger.records[index].as_dict() for index in sorted(matched_indices)]
    return {
        "kind": kind,
        "path": str(path),
        "sha256": file_sha256(path),
        "status": "stale" if invalidated_record_count else "clean",
        "record_count": record_count,
        "invalidated_record_count": invalidated_record_count,
        "matched_tombstone_count": len(matched_tombstones),
        "unmatched_tombstone_count": ledger.count - len(matched_tombstones),
        "matched_tombstones": matched_tombstones,
    }


def _iter_review_card_messages(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid review card JSON at {path}:{line_number}") from error
            if not isinstance(payload, dict) or not isinstance(payload.get("source"), dict):
                raise ValueError(f"review card at {path}:{line_number} lacks source metadata")
            source = payload["source"]
            fullname = source.get("message_fullname")
            revision = source.get("source_revision_id")
            if not isinstance(fullname, str) or not isinstance(revision, str):
                raise ValueError(f"review card at {path}:{line_number} lacks message identity")
            yield fullname, revision


def _iter_corpus_messages(path: Path) -> Iterator[tuple[str, str]]:
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ValueError(f"could not open corpus SQLite database: {path}") from error
    try:
        try:
            cursor = connection.execute(
                "SELECT message_fullname, source_revision_id FROM search_units"
            )
        except sqlite3.Error as error:
            raise ValueError(f"corpus lacks the search_units table: {path}") from error
        for fullname, revision in cursor:
            if not isinstance(fullname, str) or not isinstance(revision, str):
                raise ValueError(f"corpus contains malformed message identity: {path}")
            yield fullname, revision
    finally:
        connection.close()
