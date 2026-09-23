import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_operations_report_reads_telemetry(tmp_path: Path) -> None:
    cards = tmp_path / "cards.jsonl"
    cards.write_text(
        json.dumps({"review_status": "complete", "context": {"context_complete": True}})
        + "\n"
    )
    digest = hashlib.sha256(cards.read_bytes()).hexdigest()
    telemetry = tmp_path / "telemetry.json"
    telemetry.write_text(
        json.dumps(
            {
                "kind": "stage_telemetry",
                "schema_version": 1,
                "status": "complete",
                "stage": "review",
                "run_identity": None,
                "input_hashes": {"files": [{"path": str(cards), "sha256": digest}]},
                "limits": {},
                "observations": {
                    "elapsed_seconds": 1.25,
                    "rss_bytes": 123,
                    "free_disk_bytes": 456,
                    "sampled_path": str(tmp_path),
                    "cache": "not_attempted",
                    "cleanup": "partial",
                },
                "errors": [],
                "limitations": [],
            }
        )
    )
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reddit_search",
            "report",
            "operations",
            "--input",
            str(cards),
            "--output",
            str(output),
            "--telemetry",
            str(telemetry),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["resource_usage"] == {
        "elapsed_seconds": 1.25,
        "rss_bytes": 123,
        "disk_bytes": 456,
        "cache": "not_attempted",
        "cleanup_state": "partial",
    }
