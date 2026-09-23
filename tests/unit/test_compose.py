from pathlib import Path

import yaml


def test_qdrant_compose_service_is_opt_in_and_loopback_only() -> None:
    compose = yaml.safe_load(Path("compose.yaml").read_text())
    qdrant = compose["services"]["qdrant"]

    assert qdrant["image"] == "qdrant/qdrant:v1.19.1"
    assert qdrant["ports"] == ["127.0.0.1:6333:6333"]
    assert qdrant["profiles"] == ["semantic"]
