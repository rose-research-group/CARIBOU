"""The item a web session's user-input queue carries.

Every `user_input_queue.put` passes a `UserTurn`, and every consumer expects
one (anything else is a programming error and raises TypeError). `block_id`
is set when the message came from the workbench about a specific block; the
runner then focuses that block (see `BlockTracker.set_focus`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class UserTurn:
    content: str
    block_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError(
                f"UserTurn.content must be str, got {type(self.content).__name__}"
            )
        if self.block_id is not None and (
            not isinstance(self.block_id, str) or not self.block_id
        ):
            raise TypeError("UserTurn.block_id must be a non-empty str or None")


def require_user_turn(item: Any) -> UserTurn:
    """Return `item` if it is a UserTurn; raise TypeError otherwise."""
    if not isinstance(item, UserTurn):
        raise TypeError(
            f"user input queue item must be a UserTurn, got {type(item).__name__}"
        )
    return item
