from __future__ import annotations

import pytest

from caribou.execution.event_ids import make_action_id


def test_action_id_uses_the_control_plane_format():
    assert make_action_id("run_20260101-000000", 3, 2) == (
        "run_20260101-000000:turn:3:block:2"
    )
    assert make_action_id("0cf97f85-40b0", 1, 1) == "0cf97f85-40b0:turn:1:block:1"


def test_action_id_requires_an_owner():
    with pytest.raises(ValueError, match="owner_id"):
        make_action_id("", 1, 1)
