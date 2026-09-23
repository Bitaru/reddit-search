"""Deterministic evidence validation for machine-draft labels (spec T09).

Offset checks establish grounding, not correct interpretation: semantic
entailment and author attribution still require human review. Validation
never promotes a draft beyond ``machine_draft``; invalid rows become
``needs_review`` for the campaign harness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, model_validator

from reddit_search.contracts import StrictModel

TOPIC_FIT_VALUES = ("relevant", "not_relevant", "uncertain")
NEED_CLARITY_VALUES = ("clear", "unclear", "uncertain")
SPEAKER_INTENT_VALUES = (
    "seeking_solution",
    "describing_pain",
    "sharing_experience",
    "answering_someone",
    "self_promotion",
    "quoting",
    "unknown",
)
PRODUCT_FIT_VALUES = ("compatible", "incompatible", "needs_clarification", "not_evaluated")
RESOLUTION_VALUES = ("resolved", "no_resolution_observed", "unknown")
SUPPORTS_VALUES = ("topic_fit", "need_clarity", "author_requirement")
ATTRIBUTION_VALUES = ("focal_author", "quoted_other", "uncertain")
TITLE_FIELD = "title"
_MAX_DIAGNOSTIC_CHARS = 200
_MAX_SCHEMA_ERROR_MESSAGES = 5


def _bounded(value: object) -> str:
    """Render a diagnostic without echoing untrusted text: bounded, one line."""

    flat = " ".join(repr(value).split())
    if len(flat) > _MAX_DIAGNOSTIC_CHARS:
        flat = flat[:_MAX_DIAGNOSTIC_CHARS] + "..."
    return flat


def _schema_failure_messages(line_number: int, error: ValidationError) -> list[str]:
    """Describe pydantic errors by type and location, never by input content."""

    messages = [
        "line "
        f"{line_number}: draft row failed schema validation at "
        f"{'.'.join(str(part) for part in item.get('loc', ()))}: "
        f"{_bounded(item.get('type'))} {_bounded(item.get('msg'))}".strip()
        for item in error.errors()[:_MAX_SCHEMA_ERROR_MESSAGES]
    ]
    omitted = len(error.errors()) - len(messages)
    if omitted > 0:
        messages.append(f"line {line_number}: {omitted} additional schema errors omitted")
    return messages


class DraftAnnotation(StrictModel):
    """Strict label payload; unknown fields are rejected, not ignored."""

    review_status: str
    topic_fit: str
    need_clarity: str
    speaker_intent: str = "unknown"
    product_fit: str = "not_evaluated"
    resolution_in_available_context: str = "unknown"
    duplicate_of: str | None = None
    reviewer_note: str | None = None

    @model_validator(mode="after")
    def labels_use_declared_values(self) -> DraftAnnotation:
        if self.review_status != "machine_draft":
            raise ValueError("review_status must be machine_draft for validator input")
        if self.topic_fit not in TOPIC_FIT_VALUES:
            raise ValueError(f"topic_fit must be one of {TOPIC_FIT_VALUES}")
        if self.need_clarity not in NEED_CLARITY_VALUES:
            raise ValueError(f"need_clarity must be one of {NEED_CLARITY_VALUES}")
        if self.speaker_intent not in SPEAKER_INTENT_VALUES:
            raise ValueError(f"speaker_intent must be one of {SPEAKER_INTENT_VALUES}")
        if self.product_fit not in PRODUCT_FIT_VALUES:
            raise ValueError(f"product_fit must be one of {PRODUCT_FIT_VALUES}")
        if self.resolution_in_available_context not in RESOLUTION_VALUES:
            raise ValueError(f"resolution_in_available_context must be one of {RESOLUTION_VALUES}")
        return self


class DraftEvidence(StrictModel):
    """One claimed quote: exact substring of a named field at code-point offsets."""

    message_fullname: str
    source_revision_id: str
    field: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str = Field(min_length=1)
    supports: str
    attribution: str

    @model_validator(mode="after")
    def declared_values_are_known(self) -> DraftEvidence:
        if self.end <= self.start:
            raise ValueError("evidence end offset must be greater than start offset")
        if self.supports not in SUPPORTS_VALUES:
            raise ValueError(f"supports must be one of {SUPPORTS_VALUES}")
        if self.attribution not in ATTRIBUTION_VALUES:
            raise ValueError(f"attribution must be one of {ATTRIBUTION_VALUES}")
        if self.supports == "author_requirement" and self.attribution != "focal_author":
            raise ValueError("author_requirement evidence must be attributed to the focal author")
        return self


class DraftRow(StrictModel):
    """One machine-draft result for one candidate/scenario task."""

    schema_version: int = Field(default=1, ge=1)
    candidate_id: str = Field(min_length=1)
    scenario_id: str | None = Field(default=None, min_length=1)
    model_provenance: dict[str, Any] = Field(min_length=1)
    annotation: DraftAnnotation
    evidence: list[DraftEvidence] = Field(default_factory=list)


class ValidationReportRow(StrictModel):
    candidate_id: str
    scenario_id: str | None = None
    validation_status: str
    failures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def invalid_rows_list_failures(self) -> ValidationReportRow:
        if self.validation_status == "invalid" and not self.failures:
            raise ValueError("invalid rows must record at least one failure")
        return self


def validate_draft_rows(
    cards_path: Path, drafts_path: Path, *, require_complete: bool = True
) -> dict[str, Any]:
    """Deterministically validate machine-draft rows against queued tasks."""
    sources = _card_sources(cards_path)
    drafts, draft_file_failures = _read_drafts(drafts_path)
    rows: list[ValidationReportRow] = []
    seen: set[TaskKey] = set()
    if draft_file_failures:
        rows.append(
            ValidationReportRow(
                candidate_id="<file>",
                validation_status="invalid",
                failures=list(draft_file_failures),
            )
        )
    for task, row in drafts:
        failures = _row_failures(task, row, sources, seen)
        seen.add(task)
        rows.append(
            ValidationReportRow(
                candidate_id=task[0],
                scenario_id=task[1],
                validation_status="invalid" if failures else "valid",
                failures=failures,
            )
        )
    if require_complete:
        for candidate_id, scenario_id in sorted(set(sources) - seen):
            rows.append(
                ValidationReportRow(
                    candidate_id=candidate_id,
                    scenario_id=scenario_id,
                    validation_status="invalid",
                    failures=["missing draft row for candidate"],
                )
            )
    rows.sort(key=lambda row: (row.candidate_id, row.scenario_id or ""))
    valid = sum(row.validation_status == "valid" for row in rows)
    return {
        "kind": "machine_draft_evidence_validation",
        "schema_version": 1,
        "cards_sha256": _sha256(cards_path),
        "drafts_sha256": _sha256(drafts_path),
        "require_complete": require_complete,
        "summary": {
            "candidate_count": len({candidate_id for candidate_id, _ in sources}),
            "task_count": len(sources),
            "draft_row_count": len(drafts),
            "valid": valid,
            "invalid": len(rows) - valid,
        },
        "rows": [row.model_dump() for row in rows],
    }


def _row_failures(
    task: TaskKey,
    row: DraftRow,
    sources: dict[TaskKey, dict[str, Any]],
    seen: set[TaskKey],
) -> list[str]:
    failures: list[str] = []
    if task in seen:
        failures.append("duplicate candidate_id/scenario task")
    if task not in sources:
        failures.append("unknown candidate_id/scenario task")
        return failures
    source = sources[task]
    if row.annotation.topic_fit == "relevant" and not row.evidence:
        failures.append("relevant topic_fit requires at least one evidence entry")
    for index, item in enumerate(row.evidence):
        failures.extend(_evidence_failures(index, item, source))
    return failures


def _evidence_failures(index: int, item: DraftEvidence, source: dict[str, Any]) -> list[str]:
    prefix = f"evidence[{index}]"
    if item.message_fullname != source["message_fullname"]:
        return [f"{prefix} message_fullname does not match the candidate source"]
    if item.source_revision_id != source["source_revision_id"]:
        return [f"{prefix} source_revision_id does not match the candidate source"]
    if item.field == TITLE_FIELD:
        text = source["title"]
    elif item.field == source["field"]:
        text = source["text"]
    else:
        return [f"{prefix} unsupported field of length {len(item.field)}"]
    if not isinstance(text, str):
        return [f"{prefix} cited field is unavailable on this card"]
    if item.end > len(text):
        return [f"{prefix} end offset {item.end} exceeds field length {len(text)}"]
    if text[item.start : item.end] != item.quote:
        return [f"{prefix} quote does not match source[field][start:end]"]
    return []


def _card_sources(cards_path: Path) -> dict[TaskKey, dict[str, Any]]:
    sources: dict[TaskKey, dict[str, Any]] = {}
    with cards_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            card = json.loads(line)
            candidate_id = card.get("candidate_id")
            scenario_id = card.get("scenario_id")
            source = card.get("source")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or (
                    scenario_id is not None
                    and (not isinstance(scenario_id, str) or not scenario_id)
                )
                or not isinstance(source, dict)
            ):
                raise ValueError(f"card line {line_number} lacks candidate_id/scenario_id/source")
            task = (candidate_id, scenario_id)
            if task in sources:
                raise ValueError(f"card line {line_number} repeats candidate/scenario task")
            for key in (
                "message_fullname",
                "source_revision_id",
                "field",
                "start",
                "end",
                "text",
                "title",
            ):
                if key not in source:
                    raise ValueError(f"card line {line_number} source lacks {key!r}")
            sources[task] = source
    if not sources:
        raise ValueError("review cards are empty")
    return sources


TaskKey = tuple[str, str | None]


def _read_drafts(
    drafts_path: Path,
) -> tuple[list[tuple[TaskKey, DraftRow]], list[str]]:
    drafts: list[tuple[TaskKey, DraftRow]] = []
    failures: list[str] = []
    with drafts_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                row = DraftRow.model_validate(payload)
            except json.JSONDecodeError as error:
                failures.append(f"line {line_number}: invalid JSON ({error.msg})")
                continue
            except ValidationError as error:
                failures.extend(_schema_failure_messages(line_number, error))
                continue
            drafts.append(((row.candidate_id, row.scenario_id), row))
    if not drafts and not failures:
        raise ValueError("machine-draft labels are empty")
    return drafts, failures


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
