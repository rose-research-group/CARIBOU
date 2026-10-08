import asyncio
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event

from caribou.execution.user_input import UserTurn
from caribou.server.models import SessionCreateRequest, SessionStatus
from caribou.server.session_manager import SessionManager, _Session


class FakeSandbox:
    def __init__(self):
        self.stop_calls = 0

    def stop_container(self):
        self.stop_calls += 1
        return True


class FakeRunningTask:
    def done(self):
        return False


def _make_session(session_id: str, output_dir: Path, sandbox=None) -> _Session:
    return _Session(
        id=session_id,
        config=SessionCreateRequest(
            mode="auto",
            run_mode="full_system",
            agent_system="caribou",
            llm_backend="chatgpt",
            sandbox_type="singularity",
            dataset_path="/tmp/example.h5ad",
            max_turns=1,
            initial_prompt="run",
        ),
        status=SessionStatus.idle,
        current_agent="agent",
        current_turn=0,
        messages=[],
        artifacts=[],
        code_events=[],
        output_dir=output_dir,
        events=[],
        event_condition=asyncio.Condition(),
        stop_flag=Event(),
        cancel_response_flag=Event(),
        user_input_queue=Queue(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        sandbox_manager=sandbox,
    )


def _make_manager(tmp_path: Path, monkeypatch) -> SessionManager:
    monkeypatch.setattr("caribou.server.session_manager._SESSIONS_DIR", tmp_path)
    manager = object.__new__(SessionManager)
    manager._sessions = {}
    manager._deleted_session_ids = set()
    manager._lock = asyncio.Lock()
    return manager


def test_delete_session_removes_persisted_session_directory(tmp_path, monkeypatch):
    async def run_test():
        manager = _make_manager(tmp_path, monkeypatch)
        sandbox = FakeSandbox()
        session = _make_session("session-a", tmp_path / "session-a" / "outputs", sandbox)
        manager._sessions[session.id] = session
        manager._save_session(session)

        assert (tmp_path / "session-a" / "session.json").exists()

        await manager.delete_session(session.id)

        assert session.id not in manager._sessions
        assert session.stop_flag.is_set()
        assert sandbox.stop_calls == 1
        assert not (tmp_path / "session-a").exists()

    asyncio.run(run_test())


def test_late_events_after_delete_do_not_recreate_session_file(tmp_path, monkeypatch):
    async def run_test():
        manager = _make_manager(tmp_path, monkeypatch)
        session = _make_session("session-b", tmp_path / "session-b" / "outputs")
        manager._sessions[session.id] = session
        manager._save_session(session)

        await manager.delete_session(session.id)
        manager._on_event(session, {
            "type": "status_change",
            "session_id": session.id,
            "turn": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"status": "idle", "reason": "late_event"},
        })

        assert not (tmp_path / "session-b" / "session.json").exists()
        assert session.events == []

    asyncio.run(run_test())


def test_stopped_session_rejects_user_messages(tmp_path, monkeypatch):
    async def run_test():
        manager = _make_manager(tmp_path, monkeypatch)
        session = _make_session("session-stopped", tmp_path / "session-stopped" / "outputs")
        session.status = SessionStatus.stopped
        manager._sessions[session.id] = session

        accepted = await manager.send_user_message(session.id, "Are you still there?")

        assert accepted is False
        assert session.user_input_queue.empty()
        # The rejection is logged as a non-fatal error, not dropped silently.
        rejection = session.events[-1]
        assert rejection["type"] == "error"
        assert rejection["data"]["code"] == "MESSAGE_NOT_ACCEPTED"
        assert rejection["data"]["fatal"] is False
        assert isinstance(rejection["seq"], int)

    asyncio.run(run_test())


def test_second_message_while_busy_is_queued_not_delivered(tmp_path, monkeypatch):
    async def run_test():
        manager = _make_manager(tmp_path, monkeypatch)
        session = _make_session("session-interactive", tmp_path / "session-interactive" / "outputs")
        session.config = session.config.model_copy(update={"mode": "interactive"})
        session.runner_task = FakeRunningTask()
        manager._sessions[session.id] = session

        first = await manager.send_user_message(session.id, "First")
        second = await manager.send_user_message(session.id, "Second")

        assert first is True
        assert second is True
        assert session.events[-1]["type"] == "message_queue_changed"
        assert [item.content for item in session.message_queue] == ["Second"]
        assert session.status == SessionStatus.running
        # Only the first reached the runner; the second waits for its idle.
        assert session.user_input_queue.get_nowait() == UserTurn("First")
        assert session.user_input_queue.empty()

    asyncio.run(run_test())


def test_cancel_response_keeps_interactive_session_alive(tmp_path, monkeypatch):
    async def run_test():
        manager = _make_manager(tmp_path, monkeypatch)
        session = _make_session("session-running", tmp_path / "session-running" / "outputs")
        session.config = session.config.model_copy(update={"mode": "interactive"})
        session.status = SessionStatus.running
        session.runner_task = FakeRunningTask()
        manager._sessions[session.id] = session

        accepted = await manager.cancel_response(session.id)

        assert accepted is True
        assert session.cancel_response_flag.is_set()
        assert not session.stop_flag.is_set()
        assert session.status == SessionStatus.running

    asyncio.run(run_test())
