def test_mock_evaluator_is_deterministic_and_marked_synthetic() -> None:
    from reddit_search.models.mock_backends import MockEvaluator

    evaluator = MockEvaluator(version="fixture-v1")
    first = evaluator.evaluate("same source text")
    second = evaluator.evaluate("same source text")

    assert first == second
    assert first["evaluator_kind"] == "mock"
    assert first["synthetic"] is True
