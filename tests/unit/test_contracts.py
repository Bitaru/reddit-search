import pytest
from pydantic import ValidationError


def test_scenario_requires_configured_retrieval_variants() -> None:
    from reddit_search.contracts import Scenario

    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {
                "schema_version": 1,
                "scenario_id": "mieru.no_bank_link",
                "app_id": "mieru",
                "scenario_version": 1,
                "description": "Track expenses without connecting a bank.",
                "lexical_queries": ["expense tracker"],
                "semantic_queries": ["Need an offline expense tracker."],
            }
        )


def test_real_messages_cannot_use_synthetic_verified_claims() -> None:
    from reddit_search.contracts import (
        ProductClaim,
        ProductProfile,
        assert_claims_usable_for_record,
    )

    profile = ProductProfile.model_validate(
        {
            "app_id": "fixture-budget",
            "profile_version": 1,
            "verification_status": "verified",
            "synthetic": True,
            "capabilities": [
                ProductClaim(
                    claim_id="fixture-budget.offline",
                    status="verified",
                    evidence_ref="fixture:budget-profile",
                )
            ],
        }
    )

    with pytest.raises(ValueError, match="synthetic"):
        assert_claims_usable_for_record(profile, record_is_synthetic=False)


def test_unknown_claim_is_the_default_and_remains_reported() -> None:
    from reddit_search.contracts import ProductClaim, ProductProfile, profile_missing_evidence

    profile = ProductProfile(
        app_id="mieru",
        profile_version=1,
        capabilities=[ProductClaim(claim_id="mieru.no_bank_link")],
    )

    assert profile.capabilities[0].status == "unknown"
    assert profile_missing_evidence(profile) == ["mieru.no_bank_link"]
