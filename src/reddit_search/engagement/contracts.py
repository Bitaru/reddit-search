from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from reddit_search.contracts import StrictModel


class ShortlistCandidate(StrictModel):
    """A source-bound candidate for human engagement review."""

    schema_version: Literal[1] = 1
    app_id: str = Field(min_length=1)
    thread_fullname: str = Field(pattern=r"^t3_[A-Za-z0-9]+$")
    reply_target_fullname: str = Field(pattern=r"^t[13]_[A-Za-z0-9]+$")
    source_message_fullname: str = Field(pattern=r"^t[13]_[A-Za-z0-9]+$")
    source_revision_id: str = Field(min_length=1)
    permalink: str = Field(min_length=1)
    subreddit: str = Field(min_length=1)
    scenario_references: list[str]
    source_references: list[str]
    reason_for_inclusion: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ShortlistManifest(StrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["engagement_shortlist_manifest"] = "engagement_shortlist_manifest"
    app_id: str = Field(min_length=1)
    inputs: dict[str, str]
    candidate_count: int = Field(ge=0)
    candidates_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    model_config = ConfigDict(extra="forbid")
