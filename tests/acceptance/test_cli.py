import json
import subprocess
import sys


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reddit_search", *arguments],
        capture_output=True,
        check=False,
        text=True,
    )


def test_help_identifies_the_cli() -> None:
    result = run_cli("--help")

    assert result.returncode == 0, result.stderr
    assert "reddit-search" in result.stdout
    assert "doctor" in result.stdout


def test_doctor_reports_local_lexical_prerequisites() -> None:
    result = run_cli("doctor", "--json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["sqlite"]["fts5_available"] is True
    assert report["network"]["external_backends_configured"] is False
