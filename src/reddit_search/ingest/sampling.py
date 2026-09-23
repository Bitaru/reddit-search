"""Deterministic control sampling without dependence on input order."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .normalize import NormalizedMessage


def deterministic_control_sample(
    messages: Iterable[NormalizedMessage], *, target: int, seed: int
) -> list[NormalizedMessage]:
    """Select the lowest stable hash ranks, preserving full ID namespaces."""
    if target < 0:
        raise ValueError("target must not be negative")
    unique_messages = {message.fullname: message for message in messages}
    return sorted(
        unique_messages.values(),
        key=lambda message: (_stable_rank(message.fullname, seed), message.fullname),
    )[:target]


def _stable_rank(fullname: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{fullname}".encode()).digest()
