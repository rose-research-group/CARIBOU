"""Identifiers shared by every engine's event stream.

`action_id` pairs a `code_submitted` event with its `code_result` (and, on
the web, with the `artifact` events the block produced). The CLI/control
loop and the web loop must build it identically so downstream consumers
(checkpoint action ledgers, the frontend) see one format. Consumers treat
the string as opaque — compare it for equality, never parse it — so
sessions recorded under an older format still pair correctly.
"""

from __future__ import annotations


def make_action_id(owner_id: str, turn: int, block_index: int) -> str:
    """The id of code block `block_index` (1-based) in `turn` of `owner_id`.

    `owner_id` is the durable run id (CLI/control) or the session id (web).
    """
    if not owner_id:
        raise ValueError("owner_id must be non-empty")
    return f"{owner_id}:turn:{turn}:block:{block_index}"
