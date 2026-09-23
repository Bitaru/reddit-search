import json
from pathlib import Path

import pytest
from pydantic import ValidationError


def test_source_registration_requires_explicit_usage_scope(tmp_path: Path) -> None:
    from reddit_search.ingest.sources import SourceSet

    with pytest.raises(ValidationError):
        SourceSet.model_validate(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "source_id": "june-submissions",
                        "path": str(tmp_path / "RS_2026-06.zst"),
                        "source_kind": "submission",
                        "declared_month": "2026-06",
                        "source_role": "discovery",
                        "usage_scope": "",
                    }
                ],
            }
        )


def test_registration_records_explicit_source_metadata_without_reading_contents(
    tmp_path: Path,
) -> None:
    from reddit_search.ingest.sources import register_source_set

    source = tmp_path / "RS_2026-06.zst"
    source.write_bytes(b"compressed bytes are not read during registration")
    source_config = tmp_path / "sources.yaml"
    source_config.write_text(
        """schema_version: 1
sources:
  - source_id: june-submissions
    path: ./RS_2026-06.zst
    source_kind: submission
    declared_month: 2026-06
    source_role: discovery
    usage_scope: authorized_test_fixture
"""
    )
    registry = tmp_path / "registered_sources.json"

    result = register_source_set(source_config, registry)

    assert result["registered_count"] == 1
    persisted = json.loads(registry.read_text())
    assert persisted["sources"][0]["status"] == "registered"
    assert persisted["sources"][0]["input_size_bytes"] == source.stat().st_size
    assert persisted["sources"][0]["verified_sha256"] is None
 
def _write_source_config(
    path: Path,
    source: Path,
    *,
    month: str = "2026-06",
    checksum: str | None = None,
    usage_scope: str = "authorized_test_fixture",
) -> None:
    checksum_line = f"    expected_checksum: {checksum}\n" if checksum is not None else ""
    path.write_text(
        "schema_version: 1\n"
        "sources:\n"
        "  - source_id: june-submissions\n"
        f"    path: {source.name}\n"
        "    source_kind: submission\n"
        f"    declared_month: {month}\n"
        "    source_role: discovery\n"
        f"    usage_scope: {usage_scope}\n"
        f"{checksum_line}",
        encoding="utf-8",
    )


def test_additive_registration_appends_new_sources_without_replacing_existing(
    tmp_path: Path,
) -> None:
    from reddit_search.ingest.sources import register_source_set, register_source_set_additive

    first = tmp_path / "first.zst"
    second = tmp_path / "second.zst"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    config = tmp_path / "sources.yaml"
    _write_source_config(config, first)
    registry = tmp_path / "registered_sources.json"
    register_source_set(config, registry)
    config.write_text(
        config.read_text()
        .replace("june-submissions", "july-submissions")
        .replace("first.zst", "second.zst")
        .replace("2026-06", "2026-07")
    )

    result = register_source_set_additive(config, registry)

    assert result["added_count"] == 1
    assert [row["source_id"] for row in json.loads(registry.read_text())["sources"]] == [
        "june-submissions", "july-submissions"
    ]


def test_additive_registration_is_idempotent_and_preserves_existing_metadata(
    tmp_path: Path,
) -> None:
    from reddit_search.ingest.sources import register_source_set, register_source_set_additive

    source = tmp_path / "source.zst"
    source.write_bytes(b"source")
    config = tmp_path / "sources.yaml"
    _write_source_config(config, source)
    registry = tmp_path / "registered_sources.json"
    register_source_set(config, registry)
    persisted = json.loads(registry.read_text())
    persisted["sources"][0]["status"] = "complete"
    registry.write_text(json.dumps(persisted) + "\n")

    result = register_source_set_additive(config, registry)

    assert result["added_count"] == 0
    assert result["idempotent_count"] == 1
    assert json.loads(registry.read_text())["sources"][0]["status"] == "complete"


@pytest.mark.parametrize("field,replacement", [
    ("path", "changed.zst"),
    ("declared_month", "2026-07"),
    ("usage_scope", "different_scope"),
])
def test_additive_registration_rejects_identity_conflicts_without_mutation(
    tmp_path: Path, field: str, replacement: str,
) -> None:
    from reddit_search.ingest.sources import register_source_set, register_source_set_additive

    source = tmp_path / "source.zst"
    source.write_bytes(b"source")
    changed = tmp_path / "changed.zst"
    changed.write_bytes(b"changed")
    config = tmp_path / "sources.yaml"
    _write_source_config(config, source)
    registry = tmp_path / "registered_sources.json"
    register_source_set(config, registry)
    before = registry.read_bytes()
    old_value = {
        "path": source.name,
        "declared_month": "2026-06",
        "usage_scope": "authorized_test_fixture",
    }[field]
    config_text = config.read_text().replace(
        f"    {field}: {old_value}",
        f"    {field}: {replacement}",
    )
    config.write_text(config_text)

    with pytest.raises(ValueError, match="conflict"):
        register_source_set_additive(config, registry)

    assert registry.read_bytes() == before
 
def test_additive_registration_rejects_expected_checksum_conflict(tmp_path: Path) -> None:
    from reddit_search.ingest.sources import register_source_set, register_source_set_additive

    source = tmp_path / "source.zst"
    source.write_bytes(b"source")
    config = tmp_path / "sources.yaml"
    _write_source_config(config, source)
    registry = tmp_path / "registered_sources.json"
    register_source_set(config, registry)
    before = registry.read_bytes()
    _write_source_config(config, source, checksum="a" * 64)

    with pytest.raises(ValueError, match="expected_checksum"):
        register_source_set_additive(config, registry)

    assert registry.read_bytes() == before
