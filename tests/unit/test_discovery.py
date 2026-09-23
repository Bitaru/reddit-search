from reddit_search.ingest.reader import RawRecordEnvelope


def record(kind: str, line: int, payload: dict[str, object]) -> RawRecordEnvelope:
    return RawRecordEnvelope(
        source_id="synthetic",
        source_kind=kind,  # type: ignore[arg-type]
        line_number=line,
        payload=payload,
        raw_line="fixture\n",
    )


def test_relevant_deep_zero_score_comment_survives_unrelated_title_and_missing_parent() -> None:
    from reddit_search.ingest.discovery import DiscoveryRule, DiscoverySelector
    from reddit_search.ingest.hydrate import ContextBuilder
    from reddit_search.ingest.normalize import normalize_record

    root = normalize_record(
        record(
            "submission",
            1,
            {
                "name": "t3_weekly",
                "title": "Weekly chat",
                "selftext": "General discussion.",
                "subreddit": "personalfinance",
                "created_utc": 1_725_148_700,
                "permalink": "/r/personalfinance/comments/weekly/",
            },
        )
    )
    candidate = normalize_record(
        record(
            "comment",
            2,
            {
                "name": "t1_need",
                "link_id": "t3_weekly",
                "parent_id": "t1_older_month",
                "body": "I need to track expenses without linking my bank account.",
                "subreddit": "personalfinance",
                "created_utc": 1_725_148_800,
                "permalink": "/r/personalfinance/comments/weekly/comment/need/",
                "score": 0,
                "depth": 12,
            },
        )
    )
    selector = DiscoverySelector(
        [DiscoveryRule("mieru.no_bank_link", (("expense", "spending"), ("bank", "sync")))]
    )

    decision = selector.select(candidate)
    context = ContextBuilder({root.fullname: root, candidate.fullname: candidate}).build(
        candidate.fullname
    )

    assert decision.selected is True
    assert decision.matched_rule_ids == ["mieru.no_bank_link"]
    assert candidate.raw_body == "I need to track expenses without linking my bank account."
    assert [message.fullname for message in context.ancestors] == ["t3_weekly"]
    assert context.missing_parent_ids == ["t1_older_month"]


def test_deep_fully_present_chain_reports_truncation_as_incomplete() -> None:
    from reddit_search.ingest.hydrate import ContextBuilder
    from reddit_search.ingest.normalize import normalize_record

    messages = {}
    payload = {
        "name": "t3_deep",
        "title": "Weekly chat",
        "selftext": "General discussion.",
        "subreddit": "personalfinance",
        "created_utc": 1_725_148_700,
        "permalink": "/r/personalfinance/comments/deep/",
    }
    messages["t3_deep"] = normalize_record(record("submission", 1, payload))
    parent_fullname = "t3_deep"
    for number in range(1, 7):
        fullname = f"t1_chain{number}"
        payload = {
            "name": fullname,
            "link_id": "t3_deep",
            "parent_id": parent_fullname,
            "body": f"Chain message {number}.",
            "subreddit": "personalfinance",
            "created_utc": 1_725_148_700 + number,
            "permalink": f"/r/personalfinance/comments/deep/comment/{number}/",
            "score": 0,
            "depth": number,
        }
        messages[fullname] = normalize_record(record("comment", number + 1, payload))
        parent_fullname = fullname

    context = ContextBuilder(messages, max_parent_messages=4).build("t1_chain6")

    assert [message.fullname for message in context.ancestors] == [
        "t3_deep",
        "t1_chain2",
        "t1_chain3",
        "t1_chain4",
        "t1_chain5",
    ]
    assert context.missing_parent_ids == []
    assert context.ancestors_truncated is True
    assert context.context_complete is False


def test_control_sampling_is_reproducible_and_keeps_exact_id_namespaces() -> None:
    from reddit_search.ingest.normalize import normalize_record
    from reddit_search.ingest.sampling import deterministic_control_sample

    submission = normalize_record(
        record(
            "submission",
            1,
            {
                "name": "t3_same",
                "title": "Shared suffix",
                "selftext": "",
                "subreddit": "test",
                "created_utc": 1,
                "permalink": "/r/test/comments/same/",
            },
        )
    )
    comment = normalize_record(
        record(
            "comment",
            2,
            {
                "name": "t1_same",
                "link_id": "t3_same",
                "parent_id": "t3_same",
                "body": "Not selected.",
                "subreddit": "test",
                "created_utc": 2,
                "permalink": "/r/test/comments/same/comment/same/",
            },
        )
    )

    first = deterministic_control_sample([submission, comment], target=2, seed=7)
    second = deterministic_control_sample([comment, submission], target=2, seed=7)

    assert [message.fullname for message in first] == [message.fullname for message in second]
    assert {message.fullname for message in first} == {"t3_same", "t1_same"}
