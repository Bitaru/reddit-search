"""Typed, source-preserving contracts for the lexical milestone."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject undeclared input so archive/configuration mistakes remain visible."""

    model_config = ConfigDict(extra="forbid")


class ClaimStatus(StrEnum):
    VERIFIED = "verified"
    UNKNOWN = "unknown"
    PLANNED = "planned"
    UNSUPPORTED = "unsupported"


class ProductClaim(StrictModel):
    claim_id: Annotated[str, Field(min_length=1)]
    status: ClaimStatus = ClaimStatus.UNKNOWN
    evidence_ref: str | None = None
    version_scope: str | None = None
    constraints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def verified_claim_has_evidence(self) -> ProductClaim:
        if self.status is ClaimStatus.VERIFIED and not self.evidence_ref:
            raise ValueError("verified claims require an evidence_ref")
        return self


class ProductProfile(StrictModel):
    app_id: Annotated[str, Field(min_length=1)]
    profile_version: Annotated[int, Field(ge=1)]
    verified_at: str | None = None
    verification_owner: str | None = None
    platforms: list[ProductClaim] = Field(default_factory=list)
    capabilities: list[ProductClaim] = Field(default_factory=list)
    pricing_claims: list[ProductClaim] = Field(default_factory=list)
    device_constraints: list[ProductClaim] = Field(default_factory=list)
    unsupported_requirements: list[ProductClaim] = Field(default_factory=list)
    limitations: list[ProductClaim] = Field(default_factory=list)
    verification_status: ClaimStatus = ClaimStatus.UNKNOWN
    synthetic: bool = False

    @property
    def claims(self) -> list[ProductClaim]:
        return [
            *self.platforms,
            *self.capabilities,
            *self.pricing_claims,
            *self.device_constraints,
            *self.unsupported_requirements,
            *self.limitations,
        ]

    @model_validator(mode="after")
    def claim_ids_are_unique(self) -> ProductProfile:
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("product profile claim IDs must be unique")
        return self


class Scenario(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    scenario_id: Annotated[str, Field(min_length=1)]
    app_id: str | None = None
    scenario_version: Annotated[int, Field(ge=1)]
    description: Annotated[str, Field(min_length=1)]
    lexical_queries: list[Annotated[str, Field(min_length=1)]] = Field(min_length=3, max_length=5)
    semantic_queries: list[Annotated[str, Field(min_length=1)]] = Field(min_length=2, max_length=3)
    required_claim_ids_for_fit: list[str] = Field(default_factory=list)
    negative_example_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def claims_require_app_binding(self) -> Scenario:
        if self.required_claim_ids_for_fit and self.app_id is None:
            raise ValueError(
                "required_claim_ids_for_fit requires app_id; claim IDs live in an app namespace"
            )
        return self


def profile_missing_evidence(profile: ProductProfile) -> list[str]:
    """Return claim IDs that cannot support a confirmed product-fit assertion."""
    return sorted(
        claim.claim_id
        for claim in profile.claims
        if claim.status is not ClaimStatus.VERIFIED or not claim.evidence_ref
    )


def assert_claims_usable_for_record(profile: ProductProfile, *, record_is_synthetic: bool) -> None:
    """Keep fixture verification from establishing claims about real Reddit records."""
    if profile.synthetic and not record_is_synthetic:
        raise ValueError("synthetic product claims cannot support a real record")
