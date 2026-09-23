"""Fail-closed loopback repair for a mislabelled dense snapshot payload."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


class DenseRepairError(RuntimeError):
    pass


def _loopback(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise DenseRepairError("repair requires a loopback Qdrant URL")


def _request(base: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = Request(
        base.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=30) as response:  # noqa: S310
        result = json.loads(response.read())
    if not isinstance(result, dict) or result.get("status") not in {None, "ok"}:
        raise DenseRepairError(f"unexpected Qdrant response for {method} {path}")
    return result


def repair_snapshot(
    *,
    qdrant_url: str,
    registry_path: Path,
    corpus_path: Path,
    alias: str,
    source_snapshot: str,
    target_snapshot: str,
    batch_size: int = 256,
    dry_run: bool = True,
) -> dict:
    _loopback(qdrant_url)
    if batch_size <= 0:
        raise DenseRepairError("batch_size must be positive")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = [e for e in registry.get("entries", []) if e.get("alias") == alias]
    if len(entries) != 1:
        raise DenseRepairError("alias is not uniquely managed by registry")
    entry = entries[0]
    collection = entry.get("collection")
    if not isinstance(collection, str) or not collection.endswith("-ctx2"):
        raise DenseRepairError("registry target is not a canonical ctx2 collection")
    if entry.get("snapshot_id") != target_snapshot or not isinstance(entry.get("point_count"), int):
        raise DenseRepairError("registry identity does not match requested repair")
    with sqlite3.connect(corpus_path) as db:
        row = db.execute(
            "select count(*) from search_units where snapshot_id = ?", (target_snapshot,)
        ).fetchone()
    corpus_count = int(row[0])
    if corpus_count != entry["point_count"]:
        raise DenseRepairError("corpus count does not match registry point_count")
    path = f"/collections/{quote(collection, safe='')}/points"
    before_total = int(
        _request(
            qdrant_url,
            "POST",
            f"/collections/{quote(collection, safe='')}/points/count",
            {"exact": True},
        )["result"]["count"]
    )
    before_source = int(
        _request(
            qdrant_url,
            "POST",
            f"/collections/{quote(collection, safe='')}/points/count",
            {
                "exact": True,
                "filter": {"must": [{"key": "snapshot_id", "match": {"value": source_snapshot}}]},
            },
        )["result"]["count"]
    )
    before_target = int(
        _request(
            qdrant_url,
            "POST",
            f"/collections/{quote(collection, safe='')}/points/count",
            {
                "exact": True,
                "filter": {"must": [{"key": "snapshot_id", "match": {"value": target_snapshot}}]},
            },
        )["result"]["count"]
    )
    if before_source != corpus_count or before_target != 0:
        raise DenseRepairError(
            f"preflight counts mismatch: source={before_source}, "
            f"target={before_target}, corpus={corpus_count}"
        )
    changed = 0
    offset = None
    while True:
        body = {"limit": batch_size, "with_payload": ["unit_id", "snapshot_id"]}
        if offset is not None:
            body["offset"] = offset
        result = _request(qdrant_url, "POST", path + "/scroll", body).get("result", {})
        points = result.get("points", [])
        if not isinstance(points, list):
            raise DenseRepairError("invalid scroll response")
        ids = [
            p.get("id")
            for p in points
            if isinstance(p, dict) and p.get("payload", {}).get("snapshot_id") == source_snapshot
        ]
        if ids and not dry_run:
            _request(
                qdrant_url,
                "PUT",
                path + "/payload?wait=true",
                {"payload": {"snapshot_id": target_snapshot}, "points": ids},
            )
        changed += len(ids)
        offset = result.get("next_page_offset")
        if not points or offset is None:
            break
    after_total = (
        before_total
        if dry_run
        else int(
            _request(
                qdrant_url,
                "POST",
                f"/collections/{quote(collection, safe='')}/points/count",
                {"exact": True},
            )["result"]["count"]
        )
    )
    after_target = (
        changed
        if dry_run
        else int(
            _request(
                qdrant_url,
                "POST",
                f"/collections/{quote(collection, safe='')}/points/count",
                {
                    "exact": True,
                    "filter": {
                        "must": [{"key": "snapshot_id", "match": {"value": target_snapshot}}]
                    },
                },
            )["result"]["count"]
        )
    )
    if after_total != before_total or after_target != changed:
        raise DenseRepairError("post-repair count invariant failed")
    return {
        "alias": alias,
        "collection": collection,
        "dry_run": dry_run,
        "changed": changed,
        "before_total": before_total,
        "after_total": after_total,
        "before_source": before_source,
        "after_target": after_target,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--source-snapshot", required=True)
    parser.add_argument("--target-snapshot", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            repair_snapshot(
                qdrant_url=args.qdrant_url,
                registry_path=args.registry,
                corpus_path=args.corpus,
                alias=args.alias,
                source_snapshot=args.source_snapshot,
                target_snapshot=args.target_snapshot,
                batch_size=args.batch_size,
                dry_run=not args.apply,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
