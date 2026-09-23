import pytest
from pydantic import ValidationError


def test_runtime_config_defaults_to_offline_mode() -> None:
    from reddit_search.config import RuntimeConfig

    config = RuntimeConfig.model_validate({})

    assert config.network.allow_model_downloads is False
    assert config.network.allow_remote_inference is False
    assert config.network.allow_reddit_requests is False
    assert config.review.remote_request_budget == 0


def test_runtime_config_rejects_remote_inference_without_budget() -> None:
    from reddit_search.config import RuntimeConfig

    with pytest.raises(ValidationError, match="remote_request_budget"):
        RuntimeConfig.model_validate(
            {"network": {"allow_remote_inference": True}, "review": {"remote_request_budget": 0}}
        )


def test_runtime_config_rejects_non_positive_ingestion_limit() -> None:
    from reddit_search.config import RuntimeConfig

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"ingestion": {"max_line_bytes": 0}})


def test_config_hash_is_canonical_across_mapping_order() -> None:
    from reddit_search.config import configuration_hash

    assert configuration_hash({"b": [2, 1], "a": {"x": True}}) == configuration_hash(
        {"a": {"x": True}, "b": [2, 1]}
    )


def test_production_scenarios_load_and_v2_repairs_are_wellformed() -> None:
    from pathlib import Path

    from reddit_search.config import load_scenarios

    scenarios = load_scenarios(Path("configs/scenarios"))
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}

    # The shipped example set is schema-valid and well-formed.
    assert set(by_id) == {
        "example_app.expense_tracking",
        "example_app.statement_import",
        "example_app.invoicing_on_the_go",
    }

    # Topic-only targets are first-class: no app binding, no claim citations.
    topic_only = by_id["example_app.expense_tracking"]
    assert topic_only.app_id is None
    assert topic_only.required_claim_ids_for_fit == []
    assert topic_only.scenario_version == 1
    assert 3 <= len(topic_only.lexical_queries) <= 5
    assert all(query.strip() for query in topic_only.lexical_queries)

    # Claim-bound targets cite verified claims in their app namespace.
    for scenario_id, claim_id in (
        ("example_app.statement_import", "example_app.statement_import"),
        ("example_app.invoicing_on_the_go", "example_app.mobile_invoicing"),
    ):
        scenario = by_id[scenario_id]
        assert scenario.app_id == "example_app"
        assert scenario.required_claim_ids_for_fit == [claim_id]
        assert 3 <= len(scenario.lexical_queries) <= 5
        assert 2 <= len(scenario.semantic_queries) <= 3
        assert all(query.strip() for query in scenario.lexical_queries)
