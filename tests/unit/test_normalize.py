import pytest


def envelope(kind: str, payload: dict[str, object]):
    from reddit_search.ingest.reader import RawRecordEnvelope

    return RawRecordEnvelope(
        source_id="fixture-source",
        source_kind=kind,  # type: ignore[arg-type]
        line_number=1,
        payload=payload,
        raw_line="fixture\n",
    )


def test_normalizer_keeps_exact_fullnames_and_source_text() -> None:
    from reddit_search.ingest.normalize import normalize_record

    message = normalize_record(
        envelope(
            "comment",
            {
                "name": "t1_same",
                "link_id": "t3_same",
                "parent_id": "t1_parent",
                "body": "I need an expense tracker without bank sync.",
                "subreddit": "personalfinance",
                "created_utc": 1_725_148_800,
                "permalink": "/r/personalfinance/comments/same/comment/same/",
            },
        )
    )

    assert message.fullname == "t1_same"
    assert message.thread_fullname == "t3_same"
    assert message.raw_body == "I need an expense tracker without bank sync."


def test_normalizer_rejects_non_string_comment_body() -> None:
    from reddit_search.ingest.normalize import NormalizationError, normalize_record

    with pytest.raises(NormalizationError, match="body"):
        normalize_record(
            envelope(
                "comment",
                {
                    "name": "t1_invalid",
                    "link_id": "t3_thread",
                    "body": None,
                    "subreddit": "personalfinance",
                    "created_utc": 1_725_148_800,
                    "permalink": "/r/personalfinance/comments/thread/comment/invalid/",
                },
            )
        )
