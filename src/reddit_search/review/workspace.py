"""Serve a localhost-only workspace for reviewing bounded candidate cards."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any

from reddit_search.ingest.invalidation import TombstoneLedger, tombstone_blocks_row
from reddit_search.ingest.state import atomic_write_json

from .queue import _write_jsonl

_MAX_REQUEST_BYTES = 65_536


def _tombstone_status_sidecar_path(worksheet_path: Path) -> Path:
    return worksheet_path.with_name(f"{worksheet_path.name}.tombstone_status.json")


_TOPIC_FIT_VALUES = {"relevant", "not_relevant", "uncertain", "unreviewed"}
_CLARITY_VALUES = {"clear", "unclear", "unreviewed"}
_SPEAKER_INTENT_VALUES = {
    "seeking_solution",
    "describing_pain",
    "sharing_experience",
    "answering_someone",
    "self_promotion",
    "quoting",
    "unknown",
    "unreviewed",
}
_PRODUCT_FIT_VALUES = {"compatible", "incompatible", "needs_clarification", "not_evaluated"}
_RESOLUTION_VALUES = {"resolved", "no_resolution_observed", "unknown", "unreviewed"}
_REVIEW_STATUS_VALUES = {"complete", "pending"}


def create_review_server(
    *,
    cards_path: Path,
    worksheet_path: Path,
    drafts_path: Path | None,
    selected_candidate_ids: set[str] | None = None,
    tombstone_ledger: TombstoneLedger | None = None,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    workspace = _ReviewWorkspace(
        cards_path,
        worksheet_path,
        drafts_path,
        selected_candidate_ids=selected_candidate_ids,
        tombstone_ledger=tombstone_ledger,
    )

    class ReviewWorkspaceHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send_bytes(_PAGE_HTML.encode(), content_type="text/html; charset=utf-8")
                return
            if self.path == "/api/bootstrap":
                self._send_json(workspace.snapshot())
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/worksheet":
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > _MAX_REQUEST_BYTES:
                    raise ValueError("request body must be between 1 and 65536 bytes")
                payload = json.loads(self.rfile.read(content_length))
                candidate_id = workspace.save_annotation(payload)
            except (json.JSONDecodeError, ValueError) as error:
                self._send_json(
                    {"saved": False, "error": str(error)}, status=HTTPStatus.BAD_REQUEST
                )
                return
            self._send_json({"saved": True, "candidate_id": candidate_id})

        def log_message(self, _format: str, *_args: object) -> None:
            """Keep local review interaction from polluting the terminal."""

        def _send_json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send_bytes(
                json.dumps(payload, sort_keys=True).encode(),
                content_type="application/json; charset=utf-8",
                status=status,
            )

        def _send_bytes(
            self,
            payload: bytes,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer((host, port), ReviewWorkspaceHandler)
    server.review_card_count = workspace.card_count
    return server


def review_server_report(server: ThreadingHTTPServer) -> dict[str, object]:
    """Return the stable launch data printed before the blocking server loop."""
    host, port = server.server_address[:2]
    return {
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}",
        "card_count": server.review_card_count,
    }


class _ReviewWorkspace:
    def __init__(
        self,
        cards_path: Path,
        worksheet_path: Path,
        drafts_path: Path | None,
        *,
        selected_candidate_ids: set[str] | None,
        tombstone_ledger: TombstoneLedger | None,
    ) -> None:
        self._worksheet_path = worksheet_path
        all_cards = _read_jsonl(cards_path, label="review cards")
        self._all_worksheet = _read_jsonl(worksheet_path, label="worksheet")
        all_drafts = [] if drafts_path is None else _read_jsonl(drafts_path, label="draft labels")
        if tombstone_ledger is not None:
            for row in (*all_cards, *all_drafts):
                if tombstone_blocks_row(tombstone_ledger, row):
                    raise ValueError("review cards or drafts contain invalidated source/context")
            self._excluded_worksheet = [
                row for row in self._all_worksheet if tombstone_blocks_row(tombstone_ledger, row)
            ]
            excluded_worksheet_count = len(self._excluded_worksheet)
        else:
            self._excluded_worksheet = []
            excluded_worksheet_count = 0
        self._tombstone_status: dict[str, object] = (
            {
                "status": "checked",
                "worksheet_rows_excluded": excluded_worksheet_count,
            }
            if tombstone_ledger is not None
            else {"status": "not_checked", "worksheet_rows_excluded": 0}
        )
        if tombstone_ledger is not None:
            atomic_write_json(
                _tombstone_status_sidecar_path(worksheet_path),
                {
                    "kind": "review_worksheet_tombstone_status",
                    "status": "checked",
                    "worksheet_rows_excluded": excluded_worksheet_count,
                    "ledger_sha256": tombstone_ledger.digest,
                },
            )
        all_cards_by_task = _rows_by_task_key(all_cards, label="review cards")
        all_worksheet_by_task = _rows_by_task_key(
            self._all_worksheet,
            label="worksheet",
            known_tasks=set(all_cards_by_task),
        )
        missing_tasks = set(all_cards_by_task) - set(all_worksheet_by_task)
        if missing_tasks:
            missing = ", ".join(
                sorted(
                    candidate_id if scenario_id is None else f"{candidate_id}/{scenario_id}"
                    for candidate_id, scenario_id in missing_tasks
                )
            )
            raise ValueError(
                "review cards contain task identities absent from the worksheet "
                f"(worksheet regeneration gap): {missing}"
            )
        blocked_ids = {id(row) for row in self._excluded_worksheet}
        excluded_tasks = {
            task for task, row in all_worksheet_by_task.items() if id(row) in blocked_ids
        }
        all_drafts_by_task = _rows_by_task_key(
            all_drafts,
            label="draft labels",
            known_tasks=set(all_cards_by_task),
        )
        if set(all_cards_by_task) != set(all_worksheet_by_task):
            raise ValueError("review cards and worksheet must contain the same task identities")
        if not set(all_drafts_by_task).issubset(all_cards_by_task):
            raise ValueError("draft labels contain task identities absent from review cards")
        card_tasks = {id(card): task for task, card in all_cards_by_task.items()}
        draft_tasks = {id(row): task for task, row in all_drafts_by_task.items()}
        blocked_ids = {id(row) for row in self._excluded_worksheet}
        reviewable_worksheet = [row for row in self._all_worksheet if id(row) not in blocked_ids]
        reviewable_cards = [
            card for card in all_cards if card_tasks[id(card)] not in excluded_tasks
        ]
        reviewable_drafts = [
            row for row in all_drafts if draft_tasks[id(row)] not in excluded_tasks
        ]
        all_candidate_ids = {card["candidate_id"] for card in reviewable_cards}
        if selected_candidate_ids is not None:
            missing_ids = selected_candidate_ids - all_candidate_ids
            if missing_ids:
                raise ValueError("candidate selection contains IDs absent from review cards")
            if not selected_candidate_ids:
                raise ValueError("candidate selection must contain at least one candidate ID")
            self._cards = [
                card for card in reviewable_cards if card["candidate_id"] in selected_candidate_ids
            ]
            self._worksheet = [
                row for row in reviewable_worksheet if row["candidate_id"] in selected_candidate_ids
            ]
            self._drafts = [
                row for row in reviewable_drafts if row["candidate_id"] in selected_candidate_ids
            ]
        else:
            self._cards = reviewable_cards
            self._worksheet = reviewable_worksheet
            self._drafts = reviewable_drafts
        self._cards_by_task = _rows_by_task_key(self._cards, label="review cards")
        self._worksheet_by_task = _rows_by_task_key(self._worksheet, label="worksheet")
        self._drafts_by_task = _rows_by_task_key(self._drafts, label="draft labels")
        self._all_candidate_ids = all_candidate_ids
        self._lock = RLock()

    @property
    def card_count(self) -> int:
        return len(self._cards)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "cards": self._cards,
                "worksheet": self._worksheet,
                "drafts": self._drafts,
                "tombstone_status": self._tombstone_status,
            }

    def save_annotation(self, payload: object) -> str:
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        candidate_id = payload.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise ValueError("candidate_id must be a non-empty string")
        scenario_id = payload.get("scenario_id")
        if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
            raise ValueError("scenario_id must be a non-empty string or null")
        matching_tasks = [task for task in self._worksheet_by_task if task[0] == candidate_id]
        if scenario_id is None:
            if len(matching_tasks) != 1:
                raise ValueError("scenario_id is required for a candidate used by multiple tasks")
            task = matching_tasks[0]
        else:
            task = (candidate_id, scenario_id)
        if task not in self._worksheet_by_task:
            raise ValueError("candidate_id/scenario_id must identify a queued worksheet row")
        annotation = _validated_annotation(
            payload.get("annotation"), candidate_ids=self._all_candidate_ids
        )
        with self._lock:
            self._worksheet_by_task[task]["annotation"] = annotation
            self._worksheet = sorted(self._worksheet, key=_queue_order)
            # Write back ALL original rows: tombstone-excluded rows live in
            # self._all_worksheet too, so the on-disk worksheet stays a
            # superset-preserving artifact and never loses human records.
            self._all_worksheet = sorted(self._all_worksheet, key=_queue_order)
            _write_jsonl(self._worksheet_path, self._all_worksheet)
        return candidate_id


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid {label} row {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} are empty")
    return sorted(rows, key=_queue_order)


def _rows_by_task_key(
    rows: list[dict[str, Any]],
    *,
    label: str,
    known_tasks: set[tuple[str, str | None]] | None = None,
) -> dict[tuple[str, str | None], dict[str, Any]]:
    rows_by_task: dict[tuple[str, str | None], dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{label} candidate_id must be a non-empty string")
        scenario_id = row.get("scenario_id")
        if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id):
            raise ValueError(f"{label} scenario_id must be a non-empty string or null")
        if scenario_id is None and known_tasks is not None:
            candidates = {task for task in known_tasks if task[0] == candidate_id}
            if len(candidates) > 1:
                raise ValueError(
                    f"{label} scenario_id is required for a candidate used by multiple tasks"
                )
            if len(candidates) == 1:
                scenario_id = next(iter(candidates))[1]
                row["scenario_id"] = scenario_id
        task = (candidate_id, scenario_id)
        if task in rows_by_task:
            raise ValueError(f"{label} contain duplicate candidate/scenario tasks")
        rows_by_task[task] = row
    return rows_by_task


def _queue_order(row: dict[str, Any]) -> int:
    review_queue = row.get("review_queue")
    if not isinstance(review_queue, dict):
        raise ValueError("review row must contain a review_queue object")
    queue_order = review_queue.get("queue_order")
    if isinstance(queue_order, bool) or not isinstance(queue_order, int) or queue_order < 1:
        raise ValueError("review row queue_order must be a positive integer")
    return queue_order


def _validated_annotation(value: object, *, candidate_ids: set[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("annotation must be an object")
    review_status = value.get("review_status")
    topic_fit = value.get("topic_fit")
    need_clarity = value.get("need_clarity")
    speaker_intent = value.get("speaker_intent", "unknown")
    product_fit = value.get("product_fit", "not_evaluated")
    resolution = value.get("resolution_in_available_context", "unknown")
    duplicate_of = value.get("duplicate_of")
    reviewer_note = value.get("reviewer_note")
    claim_ids = value.get("supported_claim_ids", [])
    if review_status not in _REVIEW_STATUS_VALUES:
        raise ValueError("annotation review_status is invalid")
    if topic_fit not in _TOPIC_FIT_VALUES:
        raise ValueError("annotation topic_fit is invalid")
    if need_clarity not in _CLARITY_VALUES:
        raise ValueError("annotation need_clarity is invalid")
    if (
        speaker_intent not in _SPEAKER_INTENT_VALUES
        or product_fit not in _PRODUCT_FIT_VALUES
        or resolution not in _RESOLUTION_VALUES
    ):
        raise ValueError("annotation structured judgment is invalid")
    if not isinstance(claim_ids, list) or not all(
        isinstance(claim_id, str) and claim_id for claim_id in claim_ids
    ):
        raise ValueError("annotation supported_claim_ids must be a string list")
    if product_fit == "compatible" and not claim_ids:
        raise ValueError("compatible product_fit requires supported_claim_ids")
    if product_fit != "compatible" and claim_ids:
        raise ValueError("supported_claim_ids requires compatible product_fit")
    if duplicate_of is not None and (
        not isinstance(duplicate_of, str) or not duplicate_of or duplicate_of not in candidate_ids
    ):
        raise ValueError("annotation duplicate_of must be a queued candidate ID or null")
    if not isinstance(reviewer_note, str | type(None)) or (
        isinstance(reviewer_note, str) and len(reviewer_note) > 1_000
    ):
        raise ValueError("annotation reviewer_note must be a string up to 1000 characters or null")
    if review_status == "complete" and (topic_fit == "unreviewed" or need_clarity == "unreviewed"):
        raise ValueError("completed annotations require topic_fit and need_clarity")
    return {
        "review_status": review_status,
        "topic_fit": topic_fit,
        "need_clarity": need_clarity,
        "speaker_intent": speaker_intent,
        "product_fit": product_fit,
        "supported_claim_ids": claim_ids,
        "resolution_in_available_context": resolution,
        "duplicate_of": duplicate_of,
        "reviewer_note": reviewer_note or None,
    }


_PAGE_HTML = Path(__file__).with_name("workspace.html").read_text(encoding="utf-8")
