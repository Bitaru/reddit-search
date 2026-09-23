"""Cheap, declarative lexical selection for independent message discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .normalize import NormalizedMessage


@dataclass(frozen=True, slots=True)
class DiscoveryRule:
    rule_id: str
    required_term_groups: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if (
            not self.rule_id
            or not self.required_term_groups
            or any(not group for group in self.required_term_groups)
        ):
            raise ValueError("discovery rules require an ID and non-empty term groups")

    def matches(self, text: str) -> bool:
        folded = text.casefold()
        return all(
            any(term.casefold() in folded for term in group) for group in self.required_term_groups
        )


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    selected: bool
    selection_channels: list[str]
    matched_rule_ids: list[str]


class DiscoverySelector:
    """Select a union of declarative topic rules; never uses score or depth."""

    def __init__(self, rules: list[DiscoveryRule]) -> None:
        self.rules = tuple(rules)

    def select(self, message: NormalizedMessage) -> SelectionDecision:
        text = "\n".join((message.raw_title, message.raw_body))
        matched_rule_ids = [rule.rule_id for rule in self.rules if rule.matches(text)]
        return SelectionDecision(
            selected=bool(matched_rule_ids),
            selection_channels=["topic_rule"] if matched_rule_ids else [],
            matched_rule_ids=matched_rule_ids,
        )


def load_discovery_rules(path: Path) -> list[DiscoveryRule]:
    """Load declarative discovery rules without deriving behavior from app names."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("rules"), list):
        raise ValueError(f"{path} must contain a rules list")

    rules: list[DiscoveryRule] = []
    for raw_rule in loaded["rules"]:
        if not isinstance(raw_rule, dict):
            raise ValueError("each discovery rule must be a mapping")
        rules.append(
            DiscoveryRule(
                rule_id=_required_string(raw_rule, "rule_id"),
                required_term_groups=_term_groups(raw_rule),
            )
        )
    return rules


def _required_string(value: dict[str, Any], key: str) -> str:
    raw_value = value.get(key)
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"discovery rule {key} must be a non-empty string")
    return raw_value


def _term_groups(value: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    raw_groups = value.get("required_term_groups")
    if not isinstance(raw_groups, list):
        raise ValueError("discovery rule required_term_groups must be a list")
    groups: list[tuple[str, ...]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, list) or not all(
            isinstance(term, str) and term for term in raw_group
        ):
            raise ValueError("discovery rule term groups must contain non-empty strings")
        groups.append(tuple(raw_group))
    return tuple(groups)
