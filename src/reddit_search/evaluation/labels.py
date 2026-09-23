"""Strict human evaluation labels and source-preserving worksheet conversion."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from reddit_search.contracts import StrictModel
from reddit_search.ingest.invalidation import TombstoneLedger, tombstone_blocks_row


class TopicFit(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class SpeakerIntent(StrEnum):
    SEEKING_SOLUTION = "seeking_solution"
    DESCRIBING_PAIN = "describing_pain"
    SHARING_EXPERIENCE = "sharing_experience"
    ANSWERING_SOMEONE = "answering_someone"
    SELF_PROMOTION = "self_promotion"
    QUOTING = "quoting"
    UNKNOWN = "unknown"


class ProductFit(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    NEEDS_CLARIFICATION = "needs_clarification"
    NOT_EVALUATED = "not_evaluated"


class Resolution(StrEnum):
    RESOLVED = "resolved"
    NO_RESOLUTION_OBSERVED = "no_resolution_observed"
    UNKNOWN = "unknown"


class EvaluatorKind(StrEnum):
    HUMAN = "human"
    MODEL = "model"
    MOCK = "mock"


class ValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class Evidence(StrictModel):
    message_fullname: str = Field(min_length=1)
    source_revision_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    quote: str = Field(min_length=1)
    supports: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    quoted: bool = False

    @model_validator(mode="after")
    def validate_structure(self) -> Evidence:
        if self.end <= self.start:
            raise ValueError("evidence end offset must be greater than start offset")
        if self.attribution not in {"focal_author", "quoted_other", "uncertain"}:
            raise ValueError("evidence attribution is invalid")
        if self.supports not in {"topic_fit", "need_clarity", "author_requirement", "product_fit"}:
            raise ValueError("evidence supports is invalid")
        if self.supports == "author_requirement" and self.attribution != "focal_author":
            raise ValueError("author_requirement evidence must be attributed to the focal author")
        return self


class LabelRecord(StrictModel):
    candidate_id: str = Field(min_length=1)
    snapshot_id: str | None = None
    scenario_id: str | None = None
    app_id: str | None = None
    app_profile_version: int | None = Field(default=None, ge=1)
    topic_fit: TopicFit = TopicFit.UNKNOWN
    speaker_intent: SpeakerIntent = SpeakerIntent.UNKNOWN
    product_fit: ProductFit = ProductFit.NOT_EVALUATED
    platform_requirement: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    resolution_in_available_context: Resolution = Resolution.UNKNOWN
    evidence: list[Evidence] = Field(default_factory=list)
    supported_claim_ids: list[str] = Field(default_factory=list)
    proposed_review_bucket: str | None = None
    live_status: str = "unverified"
    policy_status: str = "unverified"
    rubric_version: str = Field(default="topic-fit-v1", min_length=1)
    evaluator_kind: EvaluatorKind = EvaluatorKind.HUMAN
    evaluator_version: str | None = None
    validation_status: ValidationStatus = ValidationStatus.UNKNOWN
    created_at: str | None = None

    @model_validator(mode="after")
    def validate_statuses(self) -> LabelRecord:
        if self.live_status != "unverified" or self.policy_status != "unverified":
            raise ValueError("live_status and policy_status must remain unverified")
        if self.evaluator_kind is EvaluatorKind.MOCK:
            raise ValueError("mock labels are not accepted")
        if self.product_fit is ProductFit.COMPATIBLE and not self.supported_claim_ids:
            raise ValueError("compatible product_fit requires supported_claim_ids")
        return self


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_IDENTITY_FIELDS = (
    "candidate_id",
    "scenario_id",
    "snapshot_id",
    "app_id",
    "app_profile_version",
    "app_profile_sha256",
    "source_revision_id",
    "source_fingerprint",
    "context_recipe_version",
    "context_dependency_identity",
)


def _identity_projection(row: dict[str, Any], candidate_id: str | None = None) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    source_candidates = (row.get("source"), row.get("source_bundle"))
    for key in _IDENTITY_FIELDS:
        if key == "candidate_id":
            value = candidate_id if candidate_id is not None else row.get(key) or row.get("unit_id")
        else:
            # Pooling keeps these two provenance values in the source bundle.
            # Top-level values are authoritative when present; only approved
            # identity fields may be recovered from nested source mappings.
            value = row.get(key)
            if value is None and key in ("source_revision_id", "context_recipe_version"):
                value = next(
                    (
                        source[key]
                        for source in source_candidates
                        if isinstance(source, dict) and source.get(key) is not None
                    ),
                    None,
                )
        if value is not None:
            projection[key] = value
    return projection


def _identity_digest(rows: list[dict[str, Any]]) -> str:
    projections = [_identity_projection(row) for row in rows]
    payload = json.dumps(projections, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity_coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {key: sum(key in _identity_projection(row) for row in rows) for key in _IDENTITY_FIELDS}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True))
            stream.write("\n")
    temporary.replace(path)
    return path


def _identity(candidate_id: str, scenario_id: object) -> tuple[str, str | None]:
    if not candidate_id:
        raise ValueError("candidate_id must not be empty")
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
        raise ValueError("scenario_id must be a non-empty string or null")
    return candidate_id, scenario_id


def export_label_worksheet(pool_path: Path, output_directory: Path) -> dict[str, Any]:
    """Export source-bearing §5.6 label stubs without retrieval metadata."""
    pool = _read_jsonl(pool_path)
    if not pool:
        raise ValueError("pool is empty")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str | None]] = set()
    for item in pool:
        candidate_id = item.get("candidate_id") or item.get("unit_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("pool row lacks candidate_id")
        identity = _identity(candidate_id, item.get("scenario_id"))
        if identity in identities:
            raise ValueError("pool contains duplicate candidate/scenario identities")
        identities.add(identity)
        metadata = _identity_projection(item, candidate_id)
        label_metadata = {
            key: metadata[key]
            for key in ("snapshot_id", "scenario_id", "app_id", "app_profile_version")
            if key in metadata
        }
        label = LabelRecord(candidate_id=candidate_id, **label_metadata).model_dump(mode="json")
        row: dict[str, Any] = {"schema_version": 1, **metadata, "label": label}
        source = item.get("source") or item.get("source_bundle")
        if isinstance(source, dict):
            row["source"] = source
        rows.append(row)
    output = _write_jsonl(output_directory / "label_worksheet.jsonl", rows)
    manifest = output_directory / "label_worksheet_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pool_file": str(pool_path),
                "pool_sha256": _sha(pool_path),
                "dependency_identity_sha256": _identity_digest(pool),
                "identity_coverage": _identity_coverage(pool),
                "row_count": len(rows),
                "blind": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"worksheet_file": str(output), "manifest_file": str(manifest), "row_count": len(rows)}


def import_label_rows(
    pool_path: Path,
    labels_path: Path,
    output_directory: Path,
    verified_claim_ids=(),
    require_complete: bool = True,
    *,
    tombstone_ledger: TombstoneLedger | None = None,
) -> dict[str, Any]:
    pool = _read_jsonl(pool_path)
    labels = _read_jsonl(labels_path)
    sources: dict[tuple[str, str | None], Any] = {}
    pool_rows_by_task: dict[tuple[str, str | None], dict[str, Any]] = {}
    pool_identity_rows: dict[tuple[str, str | None], dict[str, Any]] = {}
    for row in pool:
        candidate_id = row.get("candidate_id") or row.get("unit_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("pool row lacks candidate_id")
        identity = _identity(candidate_id, row.get("scenario_id"))
        if identity in sources:
            raise ValueError("pool contains duplicate candidate/scenario identities")
        sources[identity] = row.get("source") or row.get("source_bundle")
        pool_rows_by_task[identity] = row
        pool_identity_rows[identity] = _identity_projection(row, candidate_id)

    failures: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    verified = {str(claim_id) for claim_id in verified_claim_ids}
    for line_number, raw in enumerate(labels, start=1):
        nested = raw.get("label")
        if isinstance(nested, dict):
            payload = dict(nested)
            for key in (
                "candidate_id",
                "snapshot_id",
                "scenario_id",
                "app_id",
                "app_profile_version",
            ):
                if key not in payload and key in raw:
                    payload[key] = raw[key]
        else:
            payload = raw
        candidate_value = raw.get("candidate_id") or payload.get("candidate_id")
        failures_for_row: list[str] = []
        try:
            record = LabelRecord.model_validate(payload)
            candidate_id = record.candidate_id
            identity = _identity(candidate_id, record.scenario_id)
            if candidate_value != candidate_id:
                failures_for_row.append("candidate_id does not match label")
            if identity in seen:
                failures_for_row.append("duplicate candidate/scenario identity")
            if (
                record.product_fit is ProductFit.COMPATIBLE
                and not set(record.supported_claim_ids) <= verified
            ):
                failures_for_row.append("compatible product claim is not verified")
            expected = pool_identity_rows.get(identity)
            source = sources.get(identity)
            pool_row = pool_rows_by_task.get(identity, {})
            if tombstone_ledger is not None and tombstone_blocks_row(tombstone_ledger, pool_row):
                failures_for_row.append("source or context is invalidated")
            if expected is None:
                failures_for_row.append("candidate/scenario identity does not match pool")
            else:
                for key, value in _identity_projection(raw, candidate_id).items():
                    if key in expected and value != expected[key]:
                        failures_for_row.append(f"{key} does not match pool")
            for index, evidence in enumerate(record.evidence):
                failures_for_row.extend(_evidence_failures(index, evidence, source))
            seen.add(identity)
            if failures_for_row:
                failures.append(
                    {
                        "candidate_id": candidate_id,
                        "scenario_id": record.scenario_id,
                        "line": line_number,
                        "failures": failures_for_row,
                    }
                )
            else:
                valid.append(record.model_dump(mode="json"))
        except Exception as error:
            failures.append(
                {
                    "candidate_id": str(candidate_value or "<unknown>"),
                    "line": line_number,
                    "failures": [f"schema validation failed ({type(error).__name__})"],
                }
            )

    if require_complete:
        for candidate_id, scenario_id in sorted(set(sources) - seen):
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "scenario_id": scenario_id,
                    "failures": ["missing label row"],
                }
            )

    labels_output = _write_jsonl(output_directory / "labels.jsonl", valid)
    report = {
        "kind": "label_validation",
        "schema_version": 1,
        "pool_sha256": _sha(pool_path),
        "labels_sha256": _sha(labels_path),
        "dependency_identity_sha256": _identity_digest(pool),
        "identity_coverage": _identity_coverage(pool),
        "valid_count": len(valid),
        "invalid_count": len(failures),
        "rows": sorted(
            failures,
            key=lambda row: (row["candidate_id"], row.get("scenario_id") or ""),
        ),
    }
    report_path = output_directory / "label_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "labels_file": str(labels_output), "report_file": str(report_path)}


def _evidence_failures(index: int, ev: Evidence, source: Any) -> list[str]:
    p = f"evidence[{index}]"
    if not isinstance(source, dict):
        return [f"{p} source is unavailable"]
    if ev.message_fullname != source.get("message_fullname"):
        return [f"{p} message_fullname does not match source"]
    if ev.source_revision_id != source.get("source_revision_id"):
        return [f"{p} source_revision_id does not match source"]
    text = source.get(ev.field)
    if text is None and ev.field == source.get("field"):
        text = source.get("text")
    if not isinstance(text, str):
        return [f"{p} cited field is unavailable"]
    if ev.end > len(text):
        return [f"{p} end offset exceeds field length"]
    if text[ev.start : ev.end] != ev.quote:
        return [f"{p} quote does not match source"]
    if ev.supports == "author_requirement" and ev.attribution != "focal_author":
        return [f"{p} author attribution is not focal_author"]
    return []


def collect_worksheet_labels(
    pool_path: Path, worksheet_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Convert completed workspace annotations into §5.6 label rows.

    Topic fit maps relevant/not_relevant/uncertain to yes/no/unknown; clarity
    is preserved as the proposed review bucket. Pool rows provide the
    snapshot/scenario/app provenance so reviewer-facing anonymity survives.
    Pending rows are skipped, never counted as negatives.
    """
    pool = _read_jsonl(pool_path)
    pool_by_task: dict[tuple[str, str | None], dict[str, Any]] = {}
    pool_tasks_by_candidate: dict[str, set[tuple[str, str | None]]] = {}
    for row in pool:
        candidate_id = row.get("candidate_id") or row.get("unit_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("pool row lacks candidate_id")
        scenario_id = row.get("scenario_id")
        if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
            raise ValueError("pool row scenario_id must be a non-empty string or null")
        task = (candidate_id, scenario_id)
        if task in pool_by_task:
            raise ValueError("pool contains duplicate candidate/scenario task rows")
        pool_by_task[task] = row
        pool_tasks_by_candidate.setdefault(candidate_id, set()).add(task)

    fit_by_status = {
        "relevant": TopicFit.YES,
        "not_relevant": TopicFit.NO,
        "uncertain": TopicFit.UNKNOWN,
    }
    clarity_by_bucket = {
        "clear": "clear",
        "unclear": "needs_clarification",
    }
    unknown_sentinels = {"unreviewed": "unknown"}
    rows = _read_jsonl(worksheet_path)
    labels: list[dict[str, Any]] = []
    seen_tasks: set[tuple[str, str | None]] = set()
    pending = 0
    for line_number, row in enumerate(rows, start=1):
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"worksheet row {line_number} lacks candidate_id")
        scenario_id = row.get("scenario_id")
        if scenario_id is None:
            candidates = pool_tasks_by_candidate.get(candidate_id, set())
            if len(candidates) == 1:
                task = next(iter(candidates))
            else:
                task = (candidate_id, None)
        else:
            if not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError(f"worksheet row {line_number} scenario_id is invalid")
            task = (candidate_id, scenario_id)
        if task not in pool_by_task:
            raise ValueError(f"worksheet row {line_number} task is not in the blinded pool")
        if task in seen_tasks:
            raise ValueError("worksheet contains duplicate candidate/scenario task rows")
        seen_tasks.add(task)
        annotation = row.get("annotation")
        if not isinstance(annotation, dict):
            raise ValueError(f"worksheet row {line_number} lacks an annotation object")
        if annotation.get("review_status") != "complete":
            pending += 1
            continue
        fit_value = annotation.get("topic_fit")
        if fit_value not in fit_by_status:
            raise ValueError(
                f"worksheet row {line_number} topic_fit {fit_value!r} is not a decision"
            )
        clarity_value = annotation.get("need_clarity")
        if clarity_value not in clarity_by_bucket:
            raise ValueError(
                f"worksheet row {line_number} need_clarity {clarity_value!r} is invalid"
            )
        claim_ids = annotation.get("supported_claim_ids", [])
        if not isinstance(claim_ids, list) or not all(
            isinstance(claim_id, str) and claim_id for claim_id in claim_ids
        ):
            raise ValueError(
                f"worksheet row {line_number} supported_claim_ids must be a string list"
            )
        if annotation.get("product_fit") == "compatible" and not claim_ids:
            raise ValueError(
                f"worksheet row {line_number} compatible product_fit requires supported_claim_ids"
            )
        if annotation.get("product_fit") != "compatible" and claim_ids:
            raise ValueError(
                f"worksheet row {line_number} supported_claim_ids requires compatible product_fit"
            )
        pool_row = pool_by_task[task]
        record = LabelRecord(
            candidate_id=candidate_id,
            snapshot_id=pool_row.get("snapshot_id"),
            scenario_id=pool_row.get("scenario_id"),
            app_id=pool_row.get("app_id"),
            app_profile_version=pool_row.get("app_profile_version"),
            topic_fit=fit_by_status[fit_value],
            speaker_intent=unknown_sentinels.get(
                annotation.get("speaker_intent", "unknown"),
                annotation.get("speaker_intent", "unknown"),
            ),
            product_fit=annotation.get("product_fit", "not_evaluated"),
            supported_claim_ids=claim_ids,
            resolution_in_available_context=unknown_sentinels.get(
                annotation.get("resolution_in_available_context", "unknown"),
                annotation.get("resolution_in_available_context", "unknown"),
            ),
            proposed_review_bucket=clarity_by_bucket[clarity_value],
            evaluator_kind=EvaluatorKind.HUMAN,
        )
        labels.append(record.model_dump(mode="json"))
    output = _write_jsonl(output_directory / "labels.jsonl", labels)
    report = {
        "kind": "worksheet_label_collection",
        "schema_version": 1,
        "pool_sha256": _sha(pool_path),
        "worksheet_sha256": _sha(worksheet_path),
        "dependency_identity_sha256": _identity_digest(pool),
        "identity_coverage": _identity_coverage(pool),
        "collected_count": len(labels),
        "pending_count": pending,
        "input_row_count": len(rows),
    }
    report_path = output_directory / "collection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "collected_count": len(labels),
        "pending_count": pending,
        "labels_file": str(output),
        "report_file": str(report_path),
    }


def import_legacy_worksheet(
    worksheet_path: Path, output_directory: Path, snapshot_id: str | None = None
) -> dict[str, Any]:
    rows = _read_jsonl(worksheet_path)
    labels = []
    sidecar = []
    duplicates = []
    skipped_unreviewed = 0
    for line_number, raw in enumerate(rows, start=1):
        cid = raw.get("candidate_id") or raw.get("id")
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"legacy row {line_number} lacks candidate_id")
        ann = raw.get("annotation") if isinstance(raw.get("annotation"), dict) else raw
        sidecar.append({"candidate_id": cid, "legacy": raw})
        old = ann.get("topic_fit", ann.get("relevance", ann.get("label", "uncertain")))
        review_status = ann.get("review_status")
        if review_status in {"pending", "unreviewed"} or old == "unreviewed":
            skipped_unreviewed += 1
            continue
        fit = {
            "relevant": "yes",
            "not_relevant": "no",
            "uncertain": "unknown",
            "yes": "yes",
            "no": "no",
        }.get(old, "unknown")
        rec = LabelRecord(
            candidate_id=cid,
            snapshot_id=snapshot_id,
            topic_fit=fit,
            evaluator_kind=EvaluatorKind.HUMAN,
        )
        labels.append(rec.model_dump(mode="json"))
        duplicate_of = ann.get("duplicate_of")
        if isinstance(duplicate_of, str) and duplicate_of:
            duplicates.append({"candidate_id": cid, "duplicate_of": duplicate_of})
    output = _write_jsonl(output_directory / "labels.jsonl", labels)
    sidecar_output = _write_jsonl(output_directory / "legacy_sidecar.jsonl", sidecar)
    duplicate_output = _write_jsonl(output_directory / "duplicate_links.jsonl", duplicates)
    return {
        "labels_file": str(output),
        "legacy_sidecar_file": str(sidecar_output),
        "duplicate_links_file": str(duplicate_output),
        "input_row_count": len(rows),
        "row_count": len(labels),
        "skipped_unreviewed_count": skipped_unreviewed,
    }
