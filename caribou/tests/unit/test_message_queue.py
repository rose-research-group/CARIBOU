"""Chat message queue: messages sent while the agent is busy wait on the
server (persisted, one delivered per idle, removable until delivered)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from caribou.execution.blocks import BLOCKS_FILENAME
from caribou.execution.user_input import UserTurn
from caribou.server.models import (
    QueuedMessage,
    SessionForkRequest,
    SessionResponse,
    SessionStatus,
)
from caribou.server.routes import websocket as websocket_route
from caribou.server.session_persistence import load_persisted_sessions, save_session

from .test_block_focus_messages import _FakeWebSocket
from .test_blocks_route import _block, _write_blocks
from .test_branch_api import _post, setup  # noqa: F401 — `setup` is a fixture
from .test_session_delete import FakeRunningTask, _make_manager, _make_session
from .test_session_resume_fork_lifecycle import _manager, _stopped_session

SESSION_ID = "session-queue"


class FinishedTask:
    def done(self):
        return True


def _busy_session(tmp_path: Path, manager, *, blocks=None, status="running"):
    output_dir = tmp_path / SESSION_ID / "outputs"
    session = _make_session(SESSION_ID, output_dir)
    session.config = session.config.model_copy(update={"mode": "interactive"})
    session.status = SessionStatus(status)
    session.runner_task = FakeRunningTask()
    manager._sessions[session.id] = session
    if blocks is not None:
        _write_blocks(output_dir.parent / BLOCKS_FILENAME, blocks, session.id)
    return session


def _send(manager, session, content, **kwargs):
    return asyncio.run(manager.send_user_message(session.id, content, **kwargs))


def _unqueue(manager, session, message_id):
    return asyncio.run(manager.unqueue_message(session.id, message_id))


def _event(session, event_type, data):
    return {
        "type": event_type,
        "session_id": session.id,
        "turn": session.current_turn,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }


def _go_idle(manager, session, reason=None):
    """The runner's `_wait_for_user` emitting idle, as delivered on the loop."""

    async def run():
        manager._on_event(
            session, _event(session, "status_change", {"status": "idle", "reason": reason})
        )

    asyncio.run(run())


def _events_after(session, seq):
    return [e for e in session.events if e["seq"] > seq]


def _last_seq(session):
    return session.events[-1]["seq"] if session.events else 0


def _queue_contents(session):
    return [item.content for item in session.message_queue]


# --- Queueing --------------------------------------------------------------


@pytest.mark.parametrize("status", ["running", "initializing", "recovering"])
def test_message_is_queued_while_busy(tmp_path, monkeypatch, status):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager, status=status)
    if status != "running":
        # Initializing / recovering: the runner is still starting.
        session.runner_task = None

    assert _send(manager, session, "Next, cluster") is True

    assert session.user_input_queue.empty()
    assert session.status == status
    [item] = session.message_queue
    assert item.content == "Next, cluster"
    assert item.block_id is None
    assert len(item.id) == 32 and int(item.id, 16) >= 0  # uuid4 hex
    event = session.events[-1]
    assert event["type"] == "message_queue_changed"
    assert isinstance(event["seq"], int)
    [wire] = event["data"]["queue"]
    assert set(wire) == {"id", "content", "block_id", "created_at"}
    assert wire["id"] == item.id
    assert wire["content"] == "Next, cluster"
    assert wire["block_id"] is None
    datetime.fromisoformat(wire["created_at"])


def test_queued_message_keeps_a_valid_block_id(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(
        tmp_path, manager, blocks=[_block(1, SESSION_ID), _block(2, SESSION_ID)]
    )

    assert _send(manager, session, "Fix the plot", block_id="blk-0002") is True

    assert session.message_queue[0].block_id == "blk-0002"
    assert session.events[-1]["data"]["queue"][0]["block_id"] == "blk-0002"


@pytest.mark.parametrize(
    "block_id, blocks, code",
    [
        ("blk-0009", [_block(1, SESSION_ID)], "UNKNOWN_BLOCK"),
        (
            "blk-0001",
            [
                _block(
                    1,
                    SESSION_ID,
                    schema_version="caribou.block.v2",
                    entry=None,
                    inherited_from={"session_id": "parent-id", "block_id": "blk-0001"},
                )
            ],
            "INHERITED_BLOCK",
        ),
        ("", [_block(1, SESSION_ID)], "INVALID_BLOCK_ID"),
    ],
)
def test_bad_block_id_is_rejected_at_queue_time(
    tmp_path, monkeypatch, block_id, blocks, code
):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager, blocks=blocks)

    assert _send(manager, session, "About this", block_id=block_id) is False

    assert session.message_queue == []
    rejection = session.events[-1]
    assert rejection["type"] == "error"
    assert rejection["data"]["code"] == code
    assert rejection["data"]["fatal"] is False


def test_auto_mode_still_rejects(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager)
    session.config = session.config.model_copy(update={"mode": "auto"})

    assert _send(manager, session, "Hello") is False

    assert session.message_queue == []
    assert session.user_input_queue.empty()
    rejection = session.events[-1]
    assert rejection["type"] == "error"
    assert rejection["data"]["code"] == "MESSAGE_NOT_ACCEPTED"
    assert rejection["data"]["fatal"] is False


@pytest.mark.parametrize(
    "status, runner",
    [
        ("stopped", None),
        ("stopped", FinishedTask()),
        ("error", None),
        # Running/idle with a finished runner: nothing would ever deliver it.
        ("running", FinishedTask()),
        ("idle", None),
    ],
)
def test_sessions_without_a_live_runner_still_reject(tmp_path, monkeypatch, status, runner):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager, status=status)
    session.runner_task = runner

    assert _send(manager, session, "Hello") is False

    assert session.message_queue == []
    assert session.user_input_queue.empty()
    assert session.events[-1]["data"]["code"] == "MESSAGE_NOT_ACCEPTED"


def test_waiting_runner_still_gets_the_message_immediately(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager, status="idle")

    assert _send(manager, session, "Hello") is True

    assert session.message_queue == []
    assert session.user_input_queue.get_nowait() == UserTurn("Hello")
    assert session.events[-1]["data"] == {
        "status": "running",
        "reason": "user_message_queued",
    }
    assert not any(e["type"] == "message_queue_changed" for e in session.events)


# --- Delivery --------------------------------------------------------------


def test_idle_delivers_the_head_keeping_its_block_id(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(
        tmp_path, manager, blocks=[_block(1, SESSION_ID), _block(2, SESSION_ID)]
    )
    _send(manager, session, "Fix the plot", block_id="blk-0002")
    before = _last_seq(session)

    _go_idle(manager, session)

    assert session.user_input_queue.get_nowait() == UserTurn("Fix the plot", "blk-0002")
    assert session.user_input_queue.empty()
    assert session.message_queue == []
    events = _events_after(session, before)
    # idle first, then the queue without the head, then the same status
    # change an immediate send produces.
    assert [e["type"] for e in events] == [
        "status_change",
        "message_queue_changed",
        "status_change",
    ]
    assert events[0]["data"]["status"] == "idle"
    assert events[1]["data"] == {"queue": []}
    assert events[2]["data"] == {"status": "running", "reason": "user_message_queued"}
    assert session.status == SessionStatus.running


def test_fifo_one_message_per_idle(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager)
    for content in ("first", "second", "third"):
        assert _send(manager, session, content) is True
    assert _queue_contents(session) == ["first", "second", "third"]

    delivered = []
    for remaining in (["second", "third"], ["third"], []):
        _go_idle(manager, session)
        delivered.append(session.user_input_queue.get_nowait().content)
        # Exactly one per idle: nothing else reached the runner.
        assert session.user_input_queue.empty()
        assert _queue_contents(session) == remaining
        assert session.events[-2]["data"]["queue"] == [
            item.model_dump(mode="json") for item in session.message_queue
        ]
    assert delivered == ["first", "second", "third"]

    # An idle with an empty queue just stays idle.
    before = _last_seq(session)
    _go_idle(manager, session)
    assert session.status == SessionStatus.idle
    assert [e["type"] for e in _events_after(session, before)] == ["status_change"]


def test_new_message_while_waiting_goes_behind_older_queued_ones(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager)
    _send(manager, session, "older")
    # Idle with a stop requested: nothing is delivered.
    session.stop_flag.set()
    _go_idle(manager, session)
    assert session.user_input_queue.empty()
    session.stop_flag.clear()

    assert _send(manager, session, "newer") is True

    assert session.user_input_queue.get_nowait() == UserTurn("older")
    assert _queue_contents(session) == ["newer"]


def test_no_delivery_while_a_turn_is_already_on_the_runner_queue(tmp_path, monkeypatch):
    """A branch child's first idle already has its branch turn waiting."""
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager, status="recovering")
    session.runner_task = None
    _send(manager, session, "queued during recovery")
    session.runner_task = FakeRunningTask()
    session.user_input_queue.put(UserTurn("branch turn"))

    _go_idle(manager, session, "recovered_ready")

    assert session.user_input_queue.get_nowait() == UserTurn("branch turn")
    assert session.user_input_queue.empty()
    assert _queue_contents(session) == ["queued during recovery"]

    _go_idle(manager, session)
    assert session.user_input_queue.get_nowait() == UserTurn("queued during recovery")


def test_block_invalid_at_delivery_is_dropped_visibly(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    blocks_path = tmp_path / SESSION_ID / BLOCKS_FILENAME
    session = _busy_session(
        tmp_path, manager, blocks=[_block(1, SESSION_ID), _block(2, SESSION_ID)]
    )
    _send(manager, session, "About blk-0002", block_id="blk-0002")
    _send(manager, session, "Plain follow-up")
    dropped_id = session.message_queue[0].id
    # blk-0002 disappears before the agent next waits (e.g. a re-run).
    _write_blocks(blocks_path, [_block(1, SESSION_ID)], SESSION_ID)
    before = _last_seq(session)

    _go_idle(manager, session)

    events = _events_after(session, before)
    assert [e["type"] for e in events] == [
        "status_change",
        "message_queue_changed",
        "error",
        "message_queue_changed",
        "status_change",
    ]
    error = events[2]["data"]
    assert error["code"] == "UNKNOWN_BLOCK"
    assert error["fatal"] is False
    assert dropped_id in error["message"] and "blk-0002" in error["message"]
    assert [item["content"] for item in events[1]["data"]["queue"]] == ["Plain follow-up"]
    # Never sent with the wrong focus; the next message still goes out.
    assert session.user_input_queue.get_nowait() == UserTurn("Plain follow-up")
    assert session.user_input_queue.empty()
    assert session.message_queue == []


def test_inherited_block_at_delivery_is_dropped(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    blocks_path = tmp_path / SESSION_ID / BLOCKS_FILENAME
    session = _busy_session(tmp_path, manager, blocks=[_block(1, SESSION_ID)])
    _send(manager, session, "About blk-0001", block_id="blk-0001")
    _write_blocks(
        blocks_path,
        [
            _block(
                1,
                SESSION_ID,
                schema_version="caribou.block.v2",
                entry=None,
                inherited_from={"session_id": "parent-id", "block_id": "blk-0001"},
            )
        ],
        SESSION_ID,
    )

    _go_idle(manager, session)

    assert session.events[-1]["data"]["code"] == "INHERITED_BLOCK"
    assert session.user_input_queue.empty()
    assert session.message_queue == []
    assert session.status == SessionStatus.idle


# --- Removing --------------------------------------------------------------


def test_unqueued_message_is_never_sent(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager)
    _send(manager, session, "never mind")
    _send(manager, session, "keep this")
    removed = session.message_queue[0].id

    assert _unqueue(manager, session, removed) is True

    assert _queue_contents(session) == ["keep this"]
    assert session.events[-1]["type"] == "message_queue_changed"
    assert [i["content"] for i in session.events[-1]["data"]["queue"]] == ["keep this"]

    _go_idle(manager, session)
    _go_idle(manager, session)
    assert session.user_input_queue.get_nowait() == UserTurn("keep this")
    assert session.user_input_queue.empty()
    assert not any(
        e["type"] == "message_complete" or "never mind" in json.dumps(e)
        for e in session.events
        if e["type"] != "message_queue_changed"
    )


@pytest.mark.parametrize("how", ["delivered", "removed", "unknown", "missing"])
def test_unqueue_of_an_id_no_longer_queued_is_gone(tmp_path, monkeypatch, how):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager)
    _send(manager, session, "hello")
    message_id = session.message_queue[0].id
    if how == "delivered":
        _go_idle(manager, session)
    elif how == "removed":
        _unqueue(manager, session, message_id)
    elif how == "unknown":
        message_id = "0" * 32
    else:
        message_id = None
    before = _last_seq(session)

    assert _unqueue(manager, session, message_id) is False

    [error] = _events_after(session, before)
    assert error["type"] == "error"
    assert error["data"]["code"] == "QUEUED_MESSAGE_GONE"
    assert error["data"]["fatal"] is False


# --- Stop and resume -------------------------------------------------------


def test_stopping_keeps_the_queue_and_it_stays_removable(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager)
    _send(manager, session, "one")
    _send(manager, session, "two")

    asyncio.run(manager.stop_session(session.id))

    async def runner_stops():
        manager._on_event(
            session,
            _event(session, "status_change", {"status": "stopped", "reason": "user_requested"}),
        )

    asyncio.run(runner_stops())
    session.runner_task = FinishedTask()

    assert session.status == SessionStatus.stopped
    assert _queue_contents(session) == ["one", "two"]
    assert session.user_input_queue.empty()
    # New messages are refused, but queued ones can still be removed.
    assert _send(manager, session, "three") is False
    assert session.events[-1]["data"]["code"] == "MESSAGE_NOT_ACCEPTED"
    assert _unqueue(manager, session, session.message_queue[0].id) is True
    assert _queue_contents(session) == ["two"]


def test_resumed_session_delivers_its_persisted_queue_head(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager)
    _send(manager, session, "after restart")
    _send(manager, session, "and then this")
    session.status = SessionStatus.running
    manager._save_session(session)

    # Server restart: running -> stopped, queue reloaded from session.json.
    reloaded = load_persisted_sessions(tmp_path)[SESSION_ID]
    assert reloaded.status == SessionStatus.stopped
    assert _queue_contents(reloaded) == ["after restart", "and then this"]
    manager._sessions[SESSION_ID] = reloaded

    # Resume: recovering, then the recovered runner's first wait.
    reloaded.status = SessionStatus.recovering
    reloaded.runner_task = FakeRunningTask()
    _go_idle(manager, reloaded, "recovered_ready")

    assert reloaded.user_input_queue.get_nowait() == UserTurn("after restart")
    assert _queue_contents(reloaded) == ["and then this"]


# --- Persistence and REST --------------------------------------------------


def test_queue_round_trips_through_session_json(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(
        tmp_path, manager, blocks=[_block(1, SESSION_ID)], status="idle"
    )
    session.status = SessionStatus.running
    _send(manager, session, "plain")
    _send(manager, session, "focused", block_id="blk-0001")
    session.status = SessionStatus.stopped
    save_session(session, lambda _id: False, tmp_path)

    record = json.loads((tmp_path / SESSION_ID / "session.json").read_text())
    assert record["schema_version"] == "caribou.web_session.v5"
    assert record["message_queue"] == [
        item.model_dump(mode="json") for item in session.message_queue
    ]

    reloaded = load_persisted_sessions(tmp_path)[SESSION_ID]
    assert reloaded.message_queue == session.message_queue
    assert reloaded.message_queue[1].block_id == "blk-0001"


def test_files_without_a_queue_load_with_an_empty_one(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager, status="stopped")
    save_session(session, lambda _id: False, tmp_path)
    path = tmp_path / SESSION_ID / "session.json"
    record = json.loads(path.read_text())
    del record["message_queue"]
    path.write_text(json.dumps(record))

    assert load_persisted_sessions(tmp_path)[SESSION_ID].message_queue == []


def test_a_malformed_persisted_queue_is_not_loaded(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager, status="stopped")
    save_session(session, lambda _id: False, tmp_path)
    path = tmp_path / SESSION_ID / "session.json"
    record = json.loads(path.read_text())
    record["message_queue"] = [{"id": "x", "content": "missing created_at"}]
    path.write_text(json.dumps(record))

    assert SESSION_ID not in load_persisted_sessions(tmp_path)


def test_session_response_carries_the_queue(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _busy_session(tmp_path, manager)
    assert session.to_response().message_queue == []
    _send(manager, session, "queued")

    response = session.to_response()

    assert isinstance(response, SessionResponse)
    assert response.message_queue == session.message_queue
    body = json.loads(response.model_dump_json())
    assert body["message_queue"] == session.events[-1]["data"]["queue"]
    # A snapshot: later queue changes don't alter an earlier response.
    session.message_queue.clear()
    assert len(response.message_queue) == 1


# --- Fork and branch -------------------------------------------------------


def _queued(content="left behind") -> QueuedMessage:
    return QueuedMessage(
        id="a" * 32, content=content, block_id=None, created_at=datetime.utcnow()
    )


def test_fork_does_not_copy_the_queue(tmp_path, monkeypatch):
    async def run():
        source = _stopped_session(tmp_path)
        source.message_queue = [_queued()]
        manager = _manager(source)

        async def fake_complete(_source, _child, _request):
            return None

        monkeypatch.setattr(manager, "_complete_fork", fake_complete)
        monkeypatch.setattr("caribou.server.session_manager.SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(
            "caribou.server.session_manager._create_session_logger", lambda *_: None
        )
        response = await manager.fork_session(
            source.id, SessionForkRequest(name="fork", recovery_mode="smart")
        )
        child = manager.get_session(response.id)
        assert child.message_queue == []
        assert response.message_queue == []
        assert len(source.message_queue) == 1

    asyncio.run(run())


def test_branch_does_not_copy_the_queue(setup):  # noqa: F811
    source, manager, _started, client = setup
    source.message_queue = [_queued()]

    response = _post(client)

    assert response.status_code == 201, response.text
    assert response.json()["message_queue"] == []
    assert manager.get_session(response.json()["id"]).message_queue == []
    assert len(source.message_queue) == 1


# --- WebSocket -------------------------------------------------------------


class _RecordingManager:
    def __init__(self):
        self.calls = []

    async def unqueue_message(self, session_id, message_id):
        self.calls.append((session_id, message_id))
        return True


@pytest.mark.parametrize(
    "message, expected",
    [
        ({"type": "unqueue_message", "id": "abc"}, "abc"),
        # Passed on so the manager reports QUEUED_MESSAGE_GONE visibly.
        ({"type": "unqueue_message"}, None),
    ],
)
def test_websocket_passes_unqueue_to_manager(monkeypatch, message, expected):
    recorder = _RecordingManager()
    monkeypatch.setattr(websocket_route, "session_manager", recorder)

    class _Session:
        id = "session-ws"

    asyncio.run(websocket_route._receive_messages(_FakeWebSocket([message]), _Session()))

    assert recorder.calls == [("session-ws", expected)]
