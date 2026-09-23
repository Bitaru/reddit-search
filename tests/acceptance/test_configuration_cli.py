import json
import subprocess
import sys
from pathlib import Path


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reddit_search", *arguments],
        capture_output=True,
        check=False,
        text=True,
    )


def test_config_validate_accepts_default_runtime_configuration() -> None:
    result = run_cli("config", "validate", "--config", "configs/runtime.yaml")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["valid"] is True


def test_profiles_validate_reports_missing_evidence_for_template_profile() -> None:
    result = run_cli("profiles", "validate", "--profiles-dir", "configs/profiles")

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["valid"] is True
    assert summary["profiles"]["example_app"]["missing_evidence"] == [
        "example_app.free_with_iap",
        "example_app.iphone_app",
        "example_app.mobile_invoicing",
        "example_app.statement_import",
    ]


def test_verified_claims_reject_missing_evidence(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "broken.yaml").write_text(
        "app_id: broken\n"
        "profile_version: 1\n"
        "verification_status: verified\n"
        "capabilities:\n"
        "  - claim_id: broken.unbacked_claim\n"
        "    status: verified\n",
        encoding="utf-8",
    )

    result = run_cli("profiles", "validate", "--profiles-dir", str(profile_dir))

    assert result.returncode != 0
    assert "verified claims require an evidence_ref" in (result.stderr + result.stdout)


def test_scenario_files_are_data_driven_and_loadable() -> None:
    from reddit_search.config import load_scenarios

    scenarios = load_scenarios(Path("configs/scenarios"))

    assert {scenario.app_id for scenario in scenarios} == {None, "example_app"}
    assert {scenario.scenario_id for scenario in scenarios} == {
        "example_app.expense_tracking",
        "example_app.statement_import",
        "example_app.invoicing_on_the_go",
    }
    # A topic-only scenario must not cite claims; claim citations require app_id.
    unbound = [scenario for scenario in scenarios if scenario.app_id is None]
    assert unbound and all(not scenario.required_claim_ids_for_fit for scenario in unbound)
