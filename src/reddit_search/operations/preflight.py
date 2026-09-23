"""Read-only production preflight for a tombstone-purge run.

Every check observes without mutating: files are opened read-only, SQLite is
opened ``mode=ro``, Qdrant is probed with GET/POST reads only, and nothing is
written unless an explicit ``output_path`` is requested (via
``atomic_write_json``). A missing optional input reports ``missing`` — never
``fail`` — while a present-but-broken input (unparseable manifest, hash
mismatch, count drift, aliasing) reports ``fail`` with a measured reason.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from reddit_search.ingest.invalidation import load_tombstone_ledger
from reddit_search.ingest.state import atomic_write_json, file_sha256
from reddit_search.retrieval.dense import (
    DenseBackendError,
    EmbeddingRecipe,
    QdrantHttpIndex,
)

from .operator import _reject_alias

PREFLIGHT_REPORT_KIND = "tombstone_preflight_report"
PREFLIGHT_SCHEMA_VERSION = 1
CHECK_NAMES = ("artifact", "dense_collections", "disk", "ledger", "outbox", "paths")
DEFAULT_MINIMUM_FREE_DISK_BYTES = 5_368_709_120  # configs/runtime.yaml minimum_free_disk_bytes
_SAMPLE_LIMIT = 8
_QDRANT_TIMEOUT_SECONDS = 10.0
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True, slots=True)
class DenseCollectionProbe:
    """One explicitly bound dense collection to probe.

    ``collection_name`` and the manifest ``collection`` field are the only
    naming sources; nothing is inferred from collection-name digests. A probe
    with neither an explicit name nor a manifest binding is a mapping gap.
    """

    manifest_path: Path | None = None
    collection_name: str | None = None
    expected_snapshot_id: str | None = None


@dataclass(frozen=True, slots=True)
class PreflightConfig:
    """Fully explicit preflight inputs; absent fields mean "not configured"."""

    artifact_root: Path | None = None
    artifact_db: Path | None = None
    ledger_path: Path | None = None
    outbox_path: Path | None = None
    qdrant_base_url: str | None = None
    dense_collections: tuple[DenseCollectionProbe, ...] = ()
    dense_registry_path: Path | None = None
    backup_dir: Path | None = None
    output_path: Path | None = None
    minimum_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES
    required: tuple[str, ...] = ()
    missing_allowed: tuple[str, ...] = ()


def _rest_json(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None,
) -> dict[str, Any]:
    """Perform one loopback-only Qdrant REST read and return the JSON object."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("dense index URL must point to a loopback HTTP endpoint")
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=_QDRANT_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw = response.read()
    except HTTPError as error:
        raise DenseBackendError(
            f"Qdrant {method} {path} failed with HTTP {error.code}", status=error.code
        ) from error
    except URLError as error:
        raise DenseBackendError(f"Qdrant {method} {path} unavailable: {error.reason}") from error
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DenseBackendError(f"Qdrant returned invalid JSON for {method} {path}") from error
    if not isinstance(payload, dict):
        raise DenseBackendError(f"Qdrant returned a non-object response for {method} {path}")
    return payload


def _collection_path(collection: str) -> str:
    return f"/collections/{quote(collection, safe='')}"


def _check_artifact(config: PreflightConfig) -> tuple[str, dict[str, Any], list[str]]:
    """Verify one published artifact generation against its sidecar manifest."""
    if config.artifact_root is None:
        return "missing", {"reason": "artifact_root not configured"}, []
    detail: dict[str, Any] = {"artifact_root": str(config.artifact_root)}
    root = Path(config.artifact_root).expanduser().resolve()
    if not root.is_dir():
        return "missing", {**detail, "reason": f"artifact root does not exist: {root}"}, []
    if config.artifact_db is not None:
        db = Path(config.artifact_db).expanduser().resolve()
        if not db.is_file() or db.parent != root:
            return (
                "fail",
                detail,
                [f"artifact db must be an existing direct child of the artifact root: {db}"],
            )
    else:
        candidates = sorted(
            path for path in root.iterdir() if path.is_file() and path.suffix == ".db"
        )
        if len(candidates) != 1:
            return (
                "fail",
                {**detail, "db_files": [str(path) for path in candidates]},
                [f"artifact root must contain exactly one .db file; found {len(candidates)}"],
            )
        db = candidates[0]
    manifest = db.with_name(db.name + ".manifest.json")
    detail["db"] = str(db)
    detail["manifest"] = str(manifest)
    if not manifest.is_file():
        return "fail", detail, [f"artifact root is missing the sidecar manifest: {manifest}"]
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return "fail", detail, [f"artifact manifest is not readable JSON: {error}"]
    if not isinstance(value, dict):
        return "fail", detail, ["artifact manifest must contain a JSON object"]
    kind = value.get("kind")
    if kind is not None and not isinstance(kind, str):
        return "fail", detail, ["artifact manifest kind must be a string when present"]
    detail["manifest_kind"] = kind
    detail["manifest_schema_version"] = value.get("schema_version")
    recorded = value.get("output_sha256") or value.get("packaged_sha256")
    actual = file_sha256(db)
    detail["recorded_sha256"] = recorded if isinstance(recorded, str) else None
    detail["actual_sha256"] = actual
    detail["hash_match"] = isinstance(recorded, str) and recorded == actual
    if not detail["hash_match"]:
        return (
            "fail",
            detail,
            [
                "artifact manifest hash mismatch: recorded "
                f"{recorded!r}, actual {actual}; refusing a tampered or stale artifact"
            ],
        )
    detail["reason"] = "artifact generation matches its manifest hash binding"
    return "ok", detail, []


def _check_ledger(config: PreflightConfig) -> tuple[str, dict[str, Any], list[str]]:
    """Parse the operator tombstone ledger via the canonical loader."""
    if config.ledger_path is None:
        return "missing", {"reason": "ledger_path not configured"}, []
    detail: dict[str, Any] = {"ledger_path": str(config.ledger_path)}
    path = Path(config.ledger_path).expanduser().resolve()
    if not path.is_file():
        return "missing", {**detail, "reason": f"tombstone ledger does not exist: {path}"}, []
    try:
        ledger = load_tombstone_ledger(path)
    except (ValueError, OSError) as error:
        return "fail", detail, [f"tombstone ledger failed to load: {error}"]
    detail["record_count"] = len(ledger.records)
    detail["digest"] = ledger.digest
    detail["reason"] = f"ledger parsed with {len(ledger.records)} tombstone records"
    return "ok", detail, []


def _check_outbox(config: PreflightConfig) -> tuple[str, dict[str, Any], list[str]]:
    """Open the outbox read-only and count rows by status."""
    if config.outbox_path is None:
        return "missing", {"reason": "outbox_path not configured"}, []
    detail: dict[str, Any] = {"outbox_path": str(config.outbox_path)}
    path = Path(config.outbox_path).expanduser().resolve()
    if not path.is_file():
        # The outbox is created lazily by the purge run; absence is expected.
        return "missing", {**detail, "reason": f"outbox database does not exist yet: {path}"}, []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as error:
        return "fail", detail, [f"outbox could not be opened read-only: {error}"]
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'tombstone_outbox'"
        ).fetchone()
        if table is None:
            return "fail", detail, ["outbox database has no tombstone_outbox table"]
        counts = {
            str(status): count
            for status, count in connection.execute(
                "SELECT status, COUNT(*) FROM tombstone_outbox GROUP BY status ORDER BY status"
            )
        }
    except sqlite3.Error as error:
        return "fail", detail, [f"outbox read failed: {error}"]
    finally:
        connection.close()
    detail["rows_by_status"] = counts
    detail["total_rows"] = sum(counts.values())
    detail["reason"] = f"outbox readable with {sum(counts.values())} rows"
    return "ok", detail, []


def _probe_collection(
    base_url: str, probe: DenseCollectionProbe
) -> tuple[str, dict[str, Any], list[str]]:
    """Probe one dense collection read-only and compare it to its manifest."""
    detail: dict[str, Any] = {}
    reasons: list[str] = []
    manifest_doc: dict[str, Any] | None = None
    if probe.manifest_path is not None:
        detail["manifest_path"] = str(probe.manifest_path)
        try:
            loaded = json.loads(Path(probe.manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return "fail", detail, [f"dense manifest is unreadable {probe.manifest_path}: {error}"]
        if not isinstance(loaded, dict):
            return "fail", detail, [f"dense manifest must be a JSON object: {probe.manifest_path}"]
        manifest_doc = loaded
        detail["manifest_kind"] = loaded.get("kind")
        detail["manifest_status"] = loaded.get("status")
    collection = probe.collection_name
    if collection is None and manifest_doc is not None:
        raw_collection = manifest_doc.get("collection")
        collection = raw_collection if isinstance(raw_collection, str) else None
    if collection is None:
        detail["collection"] = None
        detail["reason"] = (
            "mapping gap: probe has neither an explicit collection_name nor a "
            "manifest collection binding"
        )
        return "warn", detail, []
    detail["collection"] = collection
    expected_count = manifest_doc.get("expected_count") if manifest_doc else None
    detail["expected_count"] = expected_count if isinstance(expected_count, int) else None
    expected_snapshot = probe.expected_snapshot_id
    if expected_snapshot is None and manifest_doc is not None:
        raw_snapshot = manifest_doc.get("snapshot_id")
        expected_snapshot = raw_snapshot if isinstance(raw_snapshot, str) else None
    detail["expected_snapshot_id"] = expected_snapshot

    index: QdrantHttpIndex | None = None
    recipe_doc = manifest_doc.get("recipe") if manifest_doc else None
    if isinstance(recipe_doc, dict):
        try:
            recipe = EmbeddingRecipe(**recipe_doc)
        except (TypeError, ValueError) as error:
            return "fail", detail, [f"{collection}: manifest recipe is invalid: {error}"]
        index = QdrantHttpIndex(
            base_url,
            collection=collection,
            recipe=recipe,
            timeout_seconds=_QDRANT_TIMEOUT_SECONDS,
        )

    try:
        info = (
            index._request("GET", index.collection_path)  # noqa: SLF001
            if index is not None
            else _rest_json(base_url, "GET", _collection_path(collection), None)
        )
    except (DenseBackendError, ValueError) as error:
        detail["reachable"] = False
        return "fail", detail, [f"{collection}: collection probe failed: {error}"]
    result = info.get("result")
    params = result.get("config", {}).get("params", {}) if isinstance(result, dict) else {}
    vectors = params.get("vectors", {}) if isinstance(params, dict) else {}
    dimension = vectors.get("size") if isinstance(vectors, dict) else None
    distance = vectors.get("distance") if isinstance(vectors, dict) else None
    detail["dimension"] = dimension
    detail["distance"] = distance

    try:
        if index is not None:
            observed_count = index.count()
        else:
            payload = _rest_json(
                base_url,
                "POST",
                f"{_collection_path(collection)}/points/count",
                {"exact": True},
            )
            if payload.get("status") != "ok":
                raise DenseBackendError("Qdrant count response status is not ok")
            count_result = payload.get("result")
            observed_count = count_result.get("count") if isinstance(count_result, dict) else None
            if not isinstance(observed_count, int) or observed_count < 0:
                raise DenseBackendError("Qdrant count response lacks a non-negative integer count")
    except (DenseBackendError, ValueError) as error:
        detail["reachable"] = False
        return "fail", detail, [f"{collection}: count probe failed: {error}"]
    detail["reachable"] = True
    detail["observed_count"] = observed_count

    if isinstance(detail["expected_count"], int):
        detail["count_match"] = observed_count == detail["expected_count"]
        if not detail["count_match"]:
            reasons.append(
                f"{collection}: observed point count {observed_count} does not match "
                f"manifest expected_count {detail['expected_count']}"
            )

    if index is not None:
        distance_match = (
            str(distance).lower() == index.distance.lower() if distance is not None else None
        )
        detail["recipe"] = {
            "dimension": index.dimension,
            "dimension_match": dimension == index.dimension,
            "distance": index.distance,
            "distance_match": distance_match,
        }
        if dimension != index.dimension:
            reasons.append(
                f"{collection}: collection dimension {dimension} does not match "
                f"manifest recipe dimension {index.dimension}"
            )
        if distance_match is False:
            reasons.append(
                f"{collection}: collection distance {distance} does not match "
                f"manifest recipe distance {index.distance}"
            )

    try:
        scroll_body = {"limit": _SAMPLE_LIMIT, "with_payload": True, "with_vector": False}
        response = (
            index._request("POST", f"{index.collection_path}/points/scroll", scroll_body)  # noqa: SLF001
            if index is not None
            else _rest_json(
                base_url,
                "POST",
                f"{_collection_path(collection)}/points/scroll",
                scroll_body,
            )
        )
        if response.get("status") != "ok":
            raise DenseBackendError("Qdrant scroll response status is not ok")
        scroll_result = response.get("result")
        points = scroll_result.get("points") if isinstance(scroll_result, dict) else None
        if not isinstance(points, list):
            raise DenseBackendError("Qdrant scroll response lacks a point list")
    except (DenseBackendError, ValueError) as error:
        return "fail", detail, [f"{collection}: scroll probe failed: {error}"]
    payloads = [
        point["payload"]
        for point in points
        if isinstance(point, dict) and isinstance(point.get("payload"), dict)
    ]
    detail["sampled_points"] = len(payloads)
    if observed_count == 0:
        coverage = "empty"
    elif payloads and all("context_message_refs" in payload for payload in payloads):
        coverage = "covered"
    elif payloads:
        coverage = "legacy"
    else:
        coverage = "unknown"
    detail["payload_coverage"] = coverage
    observed_snapshots = sorted(
        {
            str(payload["snapshot_id"])
            for payload in payloads
            if isinstance(payload.get("snapshot_id"), str)
        }
    )
    detail["observed_snapshot_ids"] = observed_snapshots
    if expected_snapshot is not None and observed_snapshots:
        detail["snapshot_match"] = expected_snapshot in observed_snapshots
    else:
        detail["snapshot_match"] = None

    if reasons:
        return "fail", detail, reasons
    if coverage == "legacy":
        detail["reason"] = (
            "sampled points predate context_message_refs payloads; tombstone "
            "context deletion is not coverable without a reindex"
        )
        return "warn", detail, []
    if coverage == "unknown":
        detail["reason"] = "collection reports points but none could be sampled"
        return "warn", detail, []
    detail["reason"] = f"collection reachable with {observed_count} points; coverage {coverage}"
    return "ok", detail, []


def _check_dense(config: PreflightConfig) -> tuple[str, dict[str, Any], list[str]]:
    """Probe every explicitly bound dense collection plus the live listing."""
    if config.qdrant_base_url is None or not config.dense_collections:
        return "missing", {"reason": "qdrant_base_url/dense_collections not configured"}, []
    parsed = urlparse(config.qdrant_base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOOPBACK_HOSTS:
        return (
            "fail",
            {"qdrant_base_url": config.qdrant_base_url},
            ["qdrant_base_url must point at a loopback HTTP endpoint"],
        )
    detail: dict[str, Any] = {"qdrant_base_url": config.qdrant_base_url}
    collections: list[dict[str, Any]] = []
    reasons: list[str] = []
    registry_entries: dict[str, dict[str, Any]] = {}
    if config.dense_registry_path is not None and config.dense_registry_path.exists():
        raw_registry = json.loads(config.dense_registry_path.read_text(encoding="utf-8"))
        registry_entries = {
            entry["alias"]: entry
            for entry in raw_registry.get("entries", [])
            if isinstance(entry, dict) and isinstance(entry.get("alias"), str)
        }
    for probe in config.dense_collections:
        if probe.collection_name in registry_entries:
            entry = registry_entries[probe.collection_name]
            probe = DenseCollectionProbe(
                manifest_path=probe.manifest_path,
                collection_name=entry.get("collection"),
                expected_snapshot_id=entry.get("snapshot_id", probe.expected_snapshot_id),
            )
        try:
            status, probe_detail, probe_reasons = _probe_collection(config.qdrant_base_url, probe)
        except Exception as error:  # defensive: one bad probe never aborts the check
            status, probe_detail, probe_reasons = (
                "fail",
                {"manifest_path": str(probe.manifest_path) if probe.manifest_path else None},
                [f"collection probe crashed: {error}"],
            )
        probe_detail["status"] = status
        if probe_reasons:
            probe_detail["reasons"] = probe_reasons
        collections.append(probe_detail)
        reasons.extend(probe_reasons)
    detail["collections"] = collections

    unmapped: list[str] = []
    listing_failed = False
    try:
        listing = _rest_json(config.qdrant_base_url, "GET", "/collections", None)
        listing_result = listing.get("result")
        entries = listing_result.get("collections", []) if isinstance(listing_result, dict) else []
        live = sorted(
            entry["name"]
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        )
        named = {
            report["collection"] for report in collections if report.get("collection") is not None
        }
        unmapped = sorted(set(live) - named)
    except (DenseBackendError, ValueError) as error:
        listing_failed = True
        reasons.append(f"collection listing failed: {error}")
    detail["unmapped_collections"] = unmapped
    if unmapped:
        reasons.append(
            "live collections not named by any configured manifest: " + ", ".join(unmapped)
        )

    statuses = {report["status"] for report in collections}
    if listing_failed or "fail" in statuses:
        overall = "fail"
    elif unmapped or "warn" in statuses:
        overall = "warn"
    else:
        overall = "ok"
    detail["reason"] = (
        f"probed {len(collections)} configured collection(s) with "
        f"{len(unmapped)} unmapped live collection(s)"
    )
    return overall, detail, reasons


def _check_disk(config: PreflightConfig) -> tuple[str, dict[str, Any], list[str]]:
    """Compare free bytes on the artifact-root volume against the minimum."""
    target = (
        Path(config.artifact_root).expanduser() if config.artifact_root is not None else Path.cwd()
    )
    # Measure the nearest existing ancestor: disk usage is a property of the
    # volume, so a not-yet-created artifact root still has a real volume.
    probe = target
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free = usage.free
    detail = {
        "path": str(probe),
        "configured_path": str(target),
        "free_bytes": free,
        "minimum_free_disk_bytes": config.minimum_free_disk_bytes,
    }
    if free < config.minimum_free_disk_bytes:
        return (
            "fail",
            detail,
            [
                f"only {free} free bytes on {detail['path']}; minimum is "
                f"{config.minimum_free_disk_bytes}"
            ],
        )
    detail["reason"] = f"{free} free bytes available on {detail['path']}"
    return "ok", detail, []


def _check_paths(config: PreflightConfig) -> tuple[str, dict[str, Any], list[str]]:
    """Reject aliasing and backup containment across every configured path.

    Reuses the ``_reject_alias`` rail from ``operations.operator`` (imported,
    not re-implemented: it is the same semantics ``artifact_purge`` enforces).
    """
    named: list[tuple[str, Path]] = []

    def add(label: str, value: Path | None) -> None:
        if value is not None:
            named.append((label, Path(value).expanduser()))

    add("artifact_root", config.artifact_root)
    add("artifact_db", config.artifact_db)
    add("ledger", config.ledger_path)
    add("outbox", config.outbox_path)
    add("backup_dir", config.backup_dir)
    add("output_path", config.output_path)
    for index, probe in enumerate(config.dense_collections):
        add(f"dense_manifest_{index}", probe.manifest_path)
    if not named:
        return "missing", {"reason": "no paths configured"}, []
    detail: dict[str, Any] = {
        "checked_paths": [{"label": label, "path": str(path)} for label, path in named]
    }
    reasons: list[str] = []
    try:
        for index, (_label, _path) in enumerate(named):
            _reject_alias(named[index][0], named[index:])
    except ValueError as error:
        reasons.append(str(error))
    if config.backup_dir is not None and config.artifact_root is not None:
        backup_resolved = Path(config.backup_dir).expanduser().resolve()
        root_resolved = Path(config.artifact_root).expanduser().resolve()
        if backup_resolved == root_resolved or backup_resolved.is_relative_to(root_resolved):
            reasons.append(
                f"backup directory must be outside the artifact root: {config.backup_dir}"
            )
    if reasons:
        return "fail", detail, reasons
    detail["reason"] = f"no aliasing among {len(named)} configured paths"
    return "ok", detail, []


_CHECKS = (
    ("artifact", _check_artifact),
    ("dense_collections", _check_dense),
    ("disk", _check_disk),
    ("ledger", _check_ledger),
    ("outbox", _check_outbox),
    ("paths", _check_paths),
)


def run_preflight(config: PreflightConfig) -> dict[str, Any]:
    """Run every configured preflight check read-only and report readiness.

    Returns a deterministic JSON-serializable report. ``ready`` is true iff no
    check failed and every check named in ``required`` is ``ok`` (or ``missing``
    while also named in ``missing_allowed``). Warnings never block readiness on
    their own; every ``fail`` and every unsatisfied required check lands in
    ``blockers`` with a measured reason.
    """
    unknown_required = sorted(set(config.required) - set(CHECK_NAMES))
    if unknown_required:
        raise ValueError(f"unknown required check names: {', '.join(unknown_required)}")
    unknown_allowed = sorted(set(config.missing_allowed) - set(config.required))
    if unknown_allowed:
        raise ValueError(
            "missing_allowed names must also be required; unknown/unrequired: "
            + ", ".join(unknown_allowed)
        )

    checks: dict[str, dict[str, Any]] = {}
    for name, check in _CHECKS:
        try:
            status, detail, reasons = check(config)
        except Exception as error:  # defensive: preflight itself must never raise
            status = "fail"
            detail = {"reason": f"preflight check crashed: {error}"}
            reasons = [f"preflight check {name} crashed: {error}"]
        entry = {"status": status, **detail}
        if reasons:
            entry["reasons"] = reasons
        checks[name] = entry

    blockers: list[dict[str, str]] = []
    for name, _check in _CHECKS:
        entry = checks[name]
        if entry["status"] == "fail":
            reason = "; ".join(entry.get("reasons", [entry.get("reason", "failed")]))
            blockers.append({"check": name, "reason": reason})
        elif (
            name in config.required
            and entry["status"] != "ok"
            and not (entry["status"] == "missing" and name in config.missing_allowed)
        ):
            blockers.append(
                {"check": name, "reason": f"required check '{name}' is {entry['status']}"}
            )
    report = {
        "kind": PREFLIGHT_REPORT_KIND,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "ready": not blockers,
        "blockers": blockers,
        "required": sorted(config.required),
        "missing_allowed": sorted(config.missing_allowed),
        "checks": checks,
    }
    if config.output_path is not None:
        atomic_write_json(Path(config.output_path), report)
    return report


__all__ = [
    "CHECK_NAMES",
    "DEFAULT_MINIMUM_FREE_DISK_BYTES",
    "DenseCollectionProbe",
    "PREFLIGHT_REPORT_KIND",
    "PREFLIGHT_SCHEMA_VERSION",
    "PreflightConfig",
    "run_preflight",
]
