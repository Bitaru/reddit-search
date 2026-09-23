import json
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen


def test_review_workspace_serves_evidence_and_persists_annotation(tmp_path: Path) -> None:
    from reddit_search.review.workspace import create_review_server

    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    drafts_path = tmp_path / "luna_draft_labels.jsonl"
    cards_path.write_text(json.dumps(_card()) + "\n", encoding="utf-8")
    worksheet_path.write_text(json.dumps(_worksheet()) + "\n", encoding="utf-8")
    drafts_path.write_text(json.dumps(_draft()) + "\n", encoding="utf-8")

    server = create_review_server(
        cards_path=cards_path,
        worksheet_path=worksheet_path,
        drafts_path=drafts_path,
        host="127.0.0.1",
        port=0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urlopen(f"{base_url}/", timeout=5).read().decode()
        bootstrap = json.loads(urlopen(f"{base_url}/api/bootstrap", timeout=5).read())
        request = Request(
            f"{base_url}/api/worksheet",
            data=json.dumps(
                {
                    "candidate_id": "candidate-1",
                    "annotation": {
                        "review_status": "complete",
                        "topic_fit": "relevant",
                        "need_clarity": "clear",
                        "duplicate_of": None,
                        "reviewer_note": "Explicit first-person need.",
                    },
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        saved = json.loads(urlopen(request, timeout=5).read())
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert "Local review workspace" in page
    assert bootstrap["cards"][0]["source"]["title"] == "Need a tracker without bank sync"
    assert bootstrap["drafts"][0]["annotation"]["review_status"] == "machine_draft"
    assert saved == {"saved": True, "candidate_id": "candidate-1"}
    persisted = json.loads(worksheet_path.read_text(encoding="utf-8"))
    assert persisted["candidate_id"] == "candidate-1"
    assert persisted["review_queue"] == {"queue_order": 1, "stratum_rule_id": "mieru.no_bank_link"}
    assert persisted["annotation"]["review_status"] == "complete"


def test_review_workspace_limits_snapshot_and_preserves_unselected_rows(tmp_path: Path) -> None:
    from reddit_search.review.workspace import create_review_server

    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    first_card, second_card = _card(), _card()
    first_row, second_row = _worksheet(), _worksheet()
    second_card["candidate_id"] = "candidate-2"
    second_card["review_queue"] = {"queue_order": 2, "stratum_rule_id": "mieru.no_bank_link"}
    second_row["candidate_id"] = "candidate-2"
    second_row["review_queue"] = {"queue_order": 2, "stratum_rule_id": "mieru.no_bank_link"}
    cards_path.write_text(
        "\n".join(json.dumps(row) for row in (first_card, second_card)) + "\n", encoding="utf-8"
    )
    worksheet_path.write_text(
        "\n".join(json.dumps(row) for row in (first_row, second_row)) + "\n", encoding="utf-8"
    )

    server = create_review_server(
        cards_path=cards_path,
        worksheet_path=worksheet_path,
        drafts_path=None,
        selected_candidate_ids={"candidate-1"},
        host="127.0.0.1",
        port=0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        bootstrap = json.loads(urlopen(f"{base_url}/api/bootstrap", timeout=5).read())
        request = Request(
            f"{base_url}/api/worksheet",
            data=json.dumps(
                {
                    "candidate_id": "candidate-1",
                    "annotation": {
                        "review_status": "complete",
                        "topic_fit": "relevant",
                        "need_clarity": "clear",
                        "duplicate_of": "candidate-2",
                        "reviewer_note": None,
                    },
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        saved = json.loads(urlopen(request, timeout=5).read())
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert [card["candidate_id"] for card in bootstrap["cards"]] == ["candidate-1"]
    assert [row["candidate_id"] for row in bootstrap["worksheet"]] == ["candidate-1"]
    assert bootstrap["drafts"] == []
    assert saved == {"saved": True, "candidate_id": "candidate-1"}
    persisted = [
        json.loads(line) for line in worksheet_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["candidate_id"] for row in persisted] == ["candidate-1", "candidate-2"]
    assert persisted[0]["annotation"]["review_status"] == "complete"
    assert persisted[0]["annotation"]["duplicate_of"] == "candidate-2"
    assert persisted[1]["annotation"]["review_status"] == "pending"



def test_review_workspace_persists_duplicate_candidate_across_scenarios(tmp_path: Path) -> None:
    from reddit_search.review.workspace import create_review_server

    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    cards = [
        _card("candidate-1", scenario_id="scenario-a"),
        _card("candidate-1", scenario_id="scenario-b"),
    ]
    worksheets = [
        _worksheet("candidate-1", scenario_id="scenario-a"),
        _worksheet("candidate-1", scenario_id="scenario-b"),
    ]
    cards_path.write_text("".join(json.dumps(row) + "\n" for row in cards), encoding="utf-8")
    worksheet_path.write_text(
        "".join(json.dumps(row) + "\n" for row in worksheets),
        encoding="utf-8",
    )

    server = create_review_server(
        cards_path=cards_path,
        worksheet_path=worksheet_path,
        drafts_path=None,
        host="127.0.0.1",
        port=0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        bootstrap = json.loads(urlopen(f"{base_url}/api/bootstrap", timeout=5).read())
        request = Request(
            f"{base_url}/api/worksheet",
            data=json.dumps(
                {
                    "candidate_id": "candidate-1",
                    "scenario_id": "scenario-b",
                    "annotation": {
                        "review_status": "complete",
                        "topic_fit": "relevant",
                        "need_clarity": "clear",
                        "duplicate_of": None,
                        "reviewer_note": "Scenario-specific need.",
                    },
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        saved = json.loads(urlopen(request, timeout=5).read())
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert len(bootstrap["cards"]) == 2
    assert saved == {"saved": True, "candidate_id": "candidate-1"}
    persisted = [
        json.loads(line) for line in worksheet_path.read_text(encoding="utf-8").splitlines()
    ]
    by_scenario = {row["scenario_id"]: row for row in persisted}
    assert by_scenario["scenario-a"]["annotation"]["review_status"] == "pending"
    assert by_scenario["scenario-b"]["annotation"]["review_status"] == "complete"



def test_review_workspace_infers_unambiguous_legacy_scenario(tmp_path: Path) -> None:
    from reddit_search.review.workspace import create_review_server

    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    cards_path.write_text(
        json.dumps(_card("candidate-1", scenario_id="scenario-a")) + "\n",
        encoding="utf-8",
    )
    worksheet_path.write_text(json.dumps(_worksheet()) + "\n", encoding="utf-8")
    server = create_review_server(
        cards_path=cards_path,
        worksheet_path=worksheet_path,
        drafts_path=None,
        host="127.0.0.1",
        port=0,
    )

    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        bootstrap = json.loads(urlopen(f"{base_url}/api/bootstrap", timeout=5).read())
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert bootstrap["worksheet"][0]["scenario_id"] == "scenario-a"


def _card(
    candidate_id: str = "candidate-1", *, scenario_id: str | None = None
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "review_queue": {"queue_order": 1, "stratum_rule_id": "mieru.no_bank_link"},
        "selection": {"matched_rule_ids": ["mieru.no_bank_link"]},
        "source": {
            "title": "Need a tracker without bank sync",
            "text": "I want to track spending privately.",
        },
    }
    if scenario_id is not None:
        row["scenario_id"] = scenario_id
    return row


def _worksheet(
    candidate_id: str = "candidate-1", *, scenario_id: str | None = None
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "review_queue": {"queue_order": 1, "stratum_rule_id": "mieru.no_bank_link"},
        "selection": {"matched_rule_ids": ["mieru.no_bank_link"]},
        "annotation": {
            "review_status": "pending",
            "topic_fit": "unreviewed",
            "need_clarity": "unreviewed",
            "duplicate_of": None,
            "reviewer_note": None,
        },
    }
    if scenario_id is not None:
        row["scenario_id"] = scenario_id
    return row


def _draft() -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_id": "candidate-1",
        "review_queue": {"queue_order": 1, "stratum_rule_id": "mieru.no_bank_link"},
        "selection": {"matched_rule_ids": ["mieru.no_bank_link"]},
        "annotation": {
            "review_status": "machine_draft",
            "topic_fit": "relevant",
            "need_clarity": "clear",
            "duplicate_of": None,
            "reviewer_note": "Potential fit.",
        },
        "model_validation": {"label_kind": "machine_draft"},
    }


def test_review_workspace_rejects_invalidated_sibling_context(tmp_path: Path) -> None:
    from reddit_search.ingest.invalidation import load_tombstone_ledger
    from reddit_search.review.workspace import create_review_server

    card = _card()
    card["context"] = {"message_fullnames": ["t1_context"]}
    cards = tmp_path / "cards.jsonl"
    worksheet = tmp_path / "worksheet.jsonl"
    cards.write_text(json.dumps(card) + "\n", encoding="utf-8")
    worksheet.write_text(json.dumps(_worksheet()) + "\n", encoding="utf-8")
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        '{"message_fullname":"t1_context","source_revision_id":null,"reason":"gone"}\n',
        encoding="utf-8",
    )
    import pytest
    with pytest.raises(ValueError, match="invalidated"):
        create_review_server(
            cards_path=cards,
            worksheet_path=worksheet,
            drafts_path=None,
            host="127.0.0.1",
            port=0,
            tombstone_ledger=load_tombstone_ledger(tombstones),
        )


def test_review_workspace_records_tombstone_status_honestly(tmp_path: Path) -> None:
    from reddit_search.ingest.invalidation import load_tombstone_ledger
    from reddit_search.review.workspace import _ReviewWorkspace

    cards = [_card(), _card("candidate-2")]
    worksheet = [_worksheet(), _worksheet("candidate-2")]
    tombstoned = _worksheet("candidate-2")
    tombstoned["source"] = {
        "message_fullname": "t1_two",
        "source_revision_id": "rev-1",
        "text": "body",
    }
    tombstoned_card = _card("candidate-2")
    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    cards_path.write_text(
        "".join(json.dumps(row) + "\n" for row in [cards[0], tombstoned_card]),
        encoding="utf-8",
    )
    worksheet_path.write_text(
        "".join(json.dumps(row) + "\n" for row in [worksheet[0], tombstoned]),
        encoding="utf-8",
    )
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps(
            {
                "message_fullname": "t1_two",
                "source_revision_id": "rev-1",
                "reason": "gone",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = load_tombstone_ledger(tombstones)

    checked = _ReviewWorkspace(
        cards_path,
        worksheet_path,
        None,
        selected_candidate_ids=None,
        tombstone_ledger=ledger,
    )
    assert checked.snapshot()["tombstone_status"] == {
        "status": "checked",
        "worksheet_rows_excluded": 1,
    }
    assert [row["candidate_id"] for row in checked.snapshot()["worksheet"]] == ["candidate-1"]

    unchecked = _ReviewWorkspace(
        cards_path,
        worksheet_path,
        None,
        selected_candidate_ids=None,
        tombstone_ledger=None,
    )
    assert unchecked.snapshot()["tombstone_status"] == {
        "status": "not_checked",
        "worksheet_rows_excluded": 0,
    }


def test_review_workspace_save_preserves_tombstone_excluded_rows(tmp_path: Path) -> None:
    from reddit_search.ingest.invalidation import load_tombstone_ledger
    from reddit_search.review.workspace import _ReviewWorkspace

    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    done_row = _worksheet("candidate-done")
    done_row["context"] = {"message_fullnames": ["t1_gone"]}
    done_row["annotation"] = {
        "review_status": "complete",
        "topic_fit": "relevant",
        "need_clarity": "clear",
        "duplicate_of": None,
        "reviewer_note": "human verdict",
    }
    cards_path.write_text(
        "".join(json.dumps(row) + "\n" for row in [_card(), _card("candidate-done")]),
        encoding="utf-8",
    )
    worksheet_path.write_text(
        "".join(json.dumps(row) + "\n" for row in [_worksheet(), done_row]),
        encoding="utf-8",
    )
    tombstones = tmp_path / "tombstones.jsonl"
    tombstones.write_text(
        json.dumps({"message_fullname": "t1_gone", "reason": "user deletion"}) + "\n",
        encoding="utf-8",
    )
    ledger = load_tombstone_ledger(tombstones)

    workspace = _ReviewWorkspace(
        cards_path,
        worksheet_path,
        None,
        selected_candidate_ids=None,
        tombstone_ledger=ledger,
    )
    assert [row["candidate_id"] for row in workspace.snapshot()["worksheet"]] == [
        "candidate-1"
    ]

    workspace.save_annotation(
        {
            "candidate_id": "candidate-1",
            "annotation": {
                "review_status": "complete",
                "topic_fit": "not_relevant",
                "need_clarity": "clear",
                "duplicate_of": None,
                "reviewer_note": None,
            },
        }
    )

    persisted = [
        json.loads(line)
        for line in worksheet_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sorted(row["candidate_id"] for row in persisted) == [
        "candidate-1",
        "candidate-done",
    ]
    done_persisted = next(row for row in persisted if row["candidate_id"] == "candidate-done")
    assert done_persisted["annotation"]["reviewer_note"] == "human verdict"

    sidecar_path = worksheet_path.with_name(
        worksheet_path.name + ".tombstone_status.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "checked"
    assert sidecar["worksheet_rows_excluded"] == 1
    assert sidecar["ledger_sha256"] == ledger.digest


def test_review_workspace_missing_worksheet_tasks_fail_closed(tmp_path: Path) -> None:
    import pytest

    from reddit_search.review.workspace import _ReviewWorkspace

    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    cards_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [_card(), _card("candidate-2"), _card("candidate-3")]
        ),
        encoding="utf-8",
    )
    worksheet_path.write_text(json.dumps(_worksheet()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        _ReviewWorkspace(
            cards_path,
            worksheet_path,
            None,
            selected_candidate_ids=None,
            tombstone_ledger=None,
        )
    message = str(excinfo.value)
    assert "worksheet regeneration gap" in message
    assert "candidate-2" in message
    assert "candidate-3" in message


def test_review_workspace_persists_supported_claim_ids(tmp_path: Path) -> None:
    from reddit_search.review.workspace import create_review_server

    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    cards_path.write_text(json.dumps(_card()) + "\n", encoding="utf-8")
    worksheet_path.write_text(json.dumps(_worksheet()) + "\n", encoding="utf-8")

    server = create_review_server(
        cards_path=cards_path,
        worksheet_path=worksheet_path,
        drafts_path=None,
        host="127.0.0.1",
        port=0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(
            f"{base_url}/api/worksheet",
            data=json.dumps(
                {
                    "candidate_id": "candidate-1",
                    "annotation": {
                        "review_status": "complete",
                        "topic_fit": "relevant",
                        "need_clarity": "clear",
                        "product_fit": "compatible",
                        "supported_claim_ids": ["mieru.no_bank_link"],
                    },
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        saved = json.loads(urlopen(request, timeout=5).read())
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert saved == {"saved": True, "candidate_id": "candidate-1"}
    persisted = json.loads(worksheet_path.read_text(encoding="utf-8"))
    assert persisted["annotation"]["product_fit"] == "compatible"
    assert persisted["annotation"]["supported_claim_ids"] == ["mieru.no_bank_link"]


def test_review_workspace_rejects_claim_ids_without_compatible_fit(tmp_path: Path) -> None:
    from reddit_search.review.workspace import create_review_server

    cards_path = tmp_path / "review_cards.jsonl"
    worksheet_path = tmp_path / "review_worksheet.jsonl"
    cards_path.write_text(json.dumps(_card()) + "\n", encoding="utf-8")
    worksheet_path.write_text(json.dumps(_worksheet()) + "\n", encoding="utf-8")

    server = create_review_server(
        cards_path=cards_path,
        worksheet_path=worksheet_path,
        drafts_path=None,
        host="127.0.0.1",
        port=0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    request = Request(
        f"{base_url}/api/worksheet",
        data=json.dumps(
            {
                "candidate_id": "candidate-1",
                "annotation": {
                    "review_status": "complete",
                    "topic_fit": "relevant",
                    "need_clarity": "clear",
                    "product_fit": "incompatible",
                    "supported_claim_ids": ["mieru.no_bank_link"],
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    import urllib.error

    try:
        try:
            urlopen(request, timeout=5)
            raised = False
        except urllib.error.HTTPError as error:
            raised = error.code == 400
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert raised
