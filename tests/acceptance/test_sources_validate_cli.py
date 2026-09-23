import json
import subprocess
import sys
from pathlib import Path


def test_sources_validate_updates_registered_registry(tmp_path: Path) -> None:
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

    result = subprocess.run(
        [sys.executable, "-m", "reddit_search", "sources", "validate", "--registry", str(registry)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"failed_count": 0, "validated_count": 1}
