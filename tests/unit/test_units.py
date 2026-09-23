from reddit_search.ingest.hydrate import ContextBuilder
from reddit_search.ingest.normalize import normalize_record
from reddit_search.ingest.reader import RawRecordEnvelope


def message(payload: dict[str, object], kind: str = "comment"):
    return normalize_record(
        RawRecordEnvelope(
            source_id="synthetic",
            source_kind=kind,  # type: ignore[arg-type]
            line_number=1,
            payload=payload,
            raw_line="fixture\n",
        )
    )


def test_message_unit_keeps_focus_separate_from_ancestor_context() -> None:
    from reddit_search.corpus.units import build_message_unit

    root = message(
        {
            "name": "t3_root",
            "title": "Weekly chat",
            "selftext": "Root context.",
            "subreddit": "test",
            "created_utc": 1,
            "permalink": "/r/test/comments/root/",
        },
        "submission",
    )
    comment = message(
        {
            "name": "t1_need",
            "link_id": "t3_root",
            "parent_id": "t3_root",
            "body": "I need an expense tracker without bank sync.",
            "subreddit": "test",
            "created_utc": 2,
            "permalink": "/r/test/comments/root/comment/need/",
        }
    )

    unit = build_message_unit(
        "synthetic-v1",
        ContextBuilder({root.fullname: root, comment.fullname: comment}).build(comment.fullname),
    )

    assert unit.focus_text == comment.raw_body
    assert unit.context_only_text.startswith("[SUBMISSION t3_root]")
    assert unit.focus_text not in unit.context_only_text
    assert unit.context_text.endswith(comment.raw_body)
    assert (
        unit.unit_id
        == build_message_unit(
            "synthetic-v1",
            ContextBuilder({root.fullname: root, comment.fullname: comment}).build(
                comment.fullname
            ),
        ).unit_id
    )


def test_context_builder_does_not_add_focus_submission_as_ancestor() -> None:
    root = message(
        {
            "name": "t3_root",
            "title": "Weekly chat",
            "selftext": "Root context.",
            "subreddit": "test",
            "created_utc": 1,
            "permalink": "/r/test/comments/root/",
        },
        "submission",
    )

    context = ContextBuilder({root.fullname: root}).build(root.fullname)

    assert context.ancestors == []
    assert context.missing_parent_ids == []


def test_long_message_tail_chunk_preserves_offsets_and_identity() -> None:
    from reddit_search.corpus.units import build_message_units

    body = "\n\n".join(
        ["early words " * 20, "middle words " * 20, "REQUIRED_TAIL_PHRASE exact requirement"]
    )
    msg = message(
        {
            "name": "t1_long",
            "link_id": "t3_root",
            "parent_id": "t3_root",
            "body": body,
            "subreddit": "test",
            "created_utc": 2,
            "permalink": "/long/",
        }
    )
    units = build_message_units(
        "snap",
        ContextBuilder({msg.fullname: msg}).build(msg.fullname),
        max_tokens=16,
        overlap_tokens=2,
    )
    tail = next(unit for unit in units if "REQUIRED_TAIL_PHRASE" in unit.focus_text)
    assert body[tail.focus_start : tail.focus_end] == tail.focus_text
    assert tail.message_fullname == msg.fullname and tail.thread_fullname == msg.thread_fullname
    assert tail.source_revision_id == msg.source_revision_id
    assert (
        tail.unit_id
        == next(
            unit
            for unit in build_message_units(
                "snap",
                ContextBuilder({msg.fullname: msg}).build(msg.fullname),
                max_tokens=16,
                overlap_tokens=2,
            )
            if unit.focus_text == tail.focus_text
        ).unit_id
    )


def test_oversized_paragraph_chunks_do_not_double_overlap() -> None:
    from reddit_search.corpus.units import build_message_units

    body = " ".join(f"word{i}" for i in range(40)) + "\n\nfinal requirement tail"
    msg = message(
        {
            "name": "t1_oversize",
            "link_id": "t3_root",
            "parent_id": "t3_root",
            "body": body,
            "subreddit": "test",
            "created_utc": 2,
            "permalink": "/x/",
        }
    )
    units = build_message_units(
        "snap",
        ContextBuilder({msg.fullname: msg}).build(msg.fullname),
        max_tokens=8,
        overlap_tokens=2,
    )
    import re

    assert all(len(re.findall(r"\S+", unit.focus_text)) <= 8 for unit in units)
    assert any("final requirement tail" in unit.focus_text for unit in units)
