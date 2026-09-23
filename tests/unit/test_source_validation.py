import hashlib
import json
from pathlib import Path


def test_validation_updates_registered_source_with_checksum_and_counts(tmp_path: Path) -> None:
    from reddit_search.ingest.sources import validate_registered_sources

    source = tmp_path / "RS_2026-06.jsonl"
    source.write_text('{"id":"one"}\n')
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "source_id": "june-submissions",
                        "source_path": str(source),
                        "source_kind": "submission",
                        "declared_month": "2026-06",
                        "source_role": "discovery",
                        "usage_scope": "test",
                        "input_size_bytes": source.stat().st_size,
                        "expected_checksum": None,
                        "verified_sha256": None,
                        "status": "registered",
                    }
                ],
            }
        )
    )

    summary = validate_registered_sources(registry)

    persisted = json.loads(registry.read_text())
    record = persisted["sources"][0]
    assert summary == {"validated_count": 1, "failed_count": 0}
    assert record["status"] == "complete"
    assert record["valid_records"] == 1
    assert record["verified_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
