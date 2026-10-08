"""D4: a workbench message carries its block id to the runner's queue."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect

from caribou.execution.blocks import BLOCKS_FILENAME
from caribou.execution.user_input import UserTurn
from caribou.server.routes import websocket as websocket_route

from .test_blocks_route import _block, _write_blocks
from .test_session_delete import FakeRunningTask, _make_manager, _make_session


def _waiting_session(tmp_path: Path, manager, *, blocks=None):
    output_dir = tmp_path / "session-focus" / "outputs"
    session = _make_session("session-focus", output_dir)
    session.config = session.config.model_copy(update={"mode": "interactive"})
    session.runner_task = FakeRunningTask()
    manager._sessions[session.id] = session
    if blocks is not None:
        _write_blocks(output_dir.parent / BLOCKS_FILENAME, blocks, session.id)
    return session


def _send(manager, session, content, **kwargs):
    return asyncio.run(manager.send_user_message(session.id, content, **kwargs))


def test_plain_message_queues_user_turn_without_block(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _waiting_session(tmp_path, manager)

    assert _send(manager, session, "Hello") is True
    assert session.user_input_queue.get_nowait() == UserTurn("Hello", None)


def test_known_block_id_is_queued_with_the_turn(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _waiting_session(
        tmp_path, manager, blocks=[_block(1, "session-focus"), _block(2, "session-focus")]
    )

    assert _send(manager, session, "Fix the plot", block_id="blk-0002") is True
    assert session.user_input_queue.get_nowait() == UserTurn("Fix the plot", "blk-0002")
    assert session.events[-1]["data"]["reason"] == "user_message_queued"


@pytest.mark.parametrize("with_index", [True, False])
def test_unknown_block_id_is_rejected_visibly(tmp_path, monkeypatch, with_index):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _waiting_session(
        tmp_path, manager, blocks=[_block(1, "session-focus")] if with_index else None
    )

    assert _send(manager, session, "About this", block_id="blk-0009") is False
    assert session.user_input_queue.empty()
    rejection = session.events[-1]
    assert rejection["type"] == "error"
    assert rejection["data"]["code"] == "UNKNOWN_BLOCK"
    assert rejection["data"]["fatal"] is False
    assert "blk-0009" in rejection["data"]["message"]
    # Nothing was queued, so the session stays idle and can take a retry.
    assert session.status == "idle"


@pytest.mark.parametrize("bad", [2, ["blk-0001"], {"id": "blk-0001"}, ""])
def test_non_string_block_id_is_rejected_visibly(tmp_path, monkeypatch, bad):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _waiting_session(tmp_path, manager, blocks=[_block(1, "session-focus")])

    assert _send(manager, session, "About this", block_id=bad) is False
    assert session.user_input_queue.empty()
    rejection = session.events[-1]
    assert rejection["type"] == "error"
    assert rejection["data"]["code"] == "INVALID_BLOCK_ID"
    assert rejection["data"]["fatal"] is False


def test_busy_session_rejects_unknown_block_instead_of_queueing(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _waiting_session(tmp_path, manager, blocks=[_block(1, "session-focus")])
    session.status = "running"

    assert _send(manager, session, "About this", block_id="blk-0009") is False
    assert session.events[-1]["data"]["code"] == "UNKNOWN_BLOCK"
    assert session.message_queue == []
    assert session.user_input_queue.empty()


def test_auto_session_reports_not_accepted_before_checking_block(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _waiting_session(tmp_path, manager, blocks=[_block(1, "session-focus")])
    session.config = session.config.model_copy(update={"mode": "auto"})

    assert _send(manager, session, "About this", block_id="blk-0009") is False
    assert session.events[-1]["data"]["code"] == "MESSAGE_NOT_ACCEPTED"
    assert session.user_input_queue.empty()


class _FakeWebSocket:
    def __init__(self, messages):
        self._messages = [json.dumps(m) for m in messages]

    async def receive_text(self):
        if not self._messages:
            raise WebSocketDisconnect()
        return self._messages.pop(0)


class _RecordingManager:
    def __init__(self):
        self.calls = []

    async def send_user_message(self, session_id, content, block_id=None):
        self.calls.append((session_id, content, block_id))
        return True


@pytest.mark.parametrize(
    "message, expected_block",
    [
        ({"type": "user_message", "content": "hi"}, None),
        ({"type": "user_message", "content": "hi", "block_id": "blk-0001"}, "blk-0001"),
        # Non-string ids are passed on so the manager rejects them visibly.
        ({"type": "user_message", "content": "hi", "block_id": 7}, 7),
    ],
)
def test_websocket_passes_block_id_to_manager(monkeypatch, message, expected_block):
    recorder = _RecordingManager()
    monkeypatch.setattr(websocket_route, "session_manager", recorder)

    class _Session:
        id = "session-ws"

    asyncio.run(websocket_route._receive_messages(_FakeWebSocket([message]), _Session()))

    assert recorder.calls == [("session-ws", "hi", expected_block)]
