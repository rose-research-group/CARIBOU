from __future__ import annotations

import dataclasses

import pytest

from caribou.execution.user_input import UserTurn, require_user_turn


def test_user_turn_defaults_to_no_block_and_is_frozen() -> None:
    turn = UserTurn("hello")
    assert (turn.content, turn.block_id) == ("hello", None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        turn.content = "changed"  # type: ignore[misc]


def test_user_turn_carries_a_block_id() -> None:
    assert UserTurn("fix this", block_id="blk-0002").block_id == "blk-0002"


@pytest.mark.parametrize(
    "kwargs",
    [{"content": None}, {"content": 3}, {"content": "x", "block_id": ""},
     {"content": "x", "block_id": 2}],
)
def test_user_turn_rejects_bad_fields(kwargs) -> None:
    with pytest.raises(TypeError):
        UserTurn(**kwargs)


def test_require_user_turn_rejects_other_queue_items() -> None:
    turn = UserTurn("hi")
    assert require_user_turn(turn) is turn
    with pytest.raises(TypeError):
        require_user_turn("hi")
