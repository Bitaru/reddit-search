"""Build bounded available parent context without fabricating missing messages."""

from __future__ import annotations

from dataclasses import dataclass

from .normalize import NormalizedMessage


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    focus: NormalizedMessage
    ancestors: list[NormalizedMessage]
    missing_parent_ids: list[str]
    context_complete: bool
    parent_cycle_detected: bool
    ancestors_truncated: bool = False


class ContextBuilder:
    """Resolve exact fullnames after source passes have collected available records."""

    def __init__(
        self, messages_by_fullname: dict[str, NormalizedMessage], *, max_parent_messages: int = 4
    ) -> None:
        if max_parent_messages < 0:
            raise ValueError("max_parent_messages must not be negative")
        self.messages_by_fullname = messages_by_fullname
        self.max_parent_messages = max_parent_messages

    def build(self, message_fullname: str) -> EvidenceBundle:
        focus = self.messages_by_fullname[message_fullname]
        direct_to_root: list[NormalizedMessage] = []
        missing_parent_ids: list[str] = []
        seen = {focus.fullname}
        parent_cycle_detected = False
        ancestors_truncated = False
        parent_fullname = focus.parent_fullname

        while parent_fullname and len(direct_to_root) < self.max_parent_messages:
            if parent_fullname in seen:
                parent_cycle_detected = True
                missing_parent_ids.append(parent_fullname)
                break
            parent = self.messages_by_fullname.get(parent_fullname)
            if parent is None:
                missing_parent_ids.append(parent_fullname)
                break
            direct_to_root.append(parent)
            seen.add(parent.fullname)
            parent_fullname = parent.parent_fullname
            if parent_fullname and len(direct_to_root) >= self.max_parent_messages:
                ancestors_truncated = True
                break

        ancestors = list(reversed(direct_to_root))
        root = self.messages_by_fullname.get(focus.thread_fullname)
        if root is None:
            if focus.kind == "comment" and focus.thread_fullname not in missing_parent_ids:
                missing_parent_ids.insert(0, focus.thread_fullname)
        elif root.fullname != focus.fullname and root.fullname not in {
            message.fullname for message in ancestors
        }:
            ancestors.insert(0, root)

        return EvidenceBundle(
            focus=focus,
            ancestors=ancestors,
            missing_parent_ids=missing_parent_ids,
            context_complete=(
                not missing_parent_ids and not parent_cycle_detected and not ancestors_truncated
            ),
            parent_cycle_detected=parent_cycle_detected,
            ancestors_truncated=ancestors_truncated,
        )
