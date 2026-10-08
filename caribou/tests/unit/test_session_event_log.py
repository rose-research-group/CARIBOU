"""The web session event log: token compaction, seq, and code-source records."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import pytest

from caribou.server.models import SessionCreateRequest, SessionStatus
from caribou.server.routes import websocket as ws_route
from caribou.server.session_manager import SessionManager
from caribou.server.session_persistence import load_persisted_sessions, save_session
from caribou.server.session_state import (
    SEQ_RESERVATION_BLOCK,
    _Session,
    append_session_event,
    backfill_event_seq,
)


def _session(tmp_path: Path, session_id: str = "session-log") -> _Session:
    config = SessionCreateRequest(
        mode="interactive",
        run_mode="full_system",
        agent_system="caribou",
        llm_backend="openai",
        sandbox_type="singularity",
        dataset_path="/tmp/example.h5ad",
    )
    return _Session(
        id=session_id,
        config=config,
        status=SessionStatus.running,
        current_agent="driver",
        current_turn=0,
        messages=[],
        artifacts=[],
        code_events=[],
        output_dir=tmp_path / session_id / "outputs",
        events=[],
        event_condition=asyncio.Condition(),
        stop_flag=threading.Event(),
        cancel_response_flag=threading.Event(),
        user_input_queue=queue.Queue(),
        created_at=datetime(2026, 7, 20, 12, 0, 0),
        updated_at=datetime(2026, 7, 20, 12, 1, 0),
    )


def _manager(session: _Session) -> SessionManager:
    manager = object.__new__(SessionManager)
    manager._sessions = {session.id: session}
    manager._deleted_session_ids = set()
    manager._lock = asyncio.Lock()
    manager._save_session = lambda _session: None
    return manager


def _event(session: _Session, event_type: str, data: dict, turn: int = 1) -> dict:
    return {
        "type": event_type,
        "session_id": session.id,
        "turn": turn,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }


def _assistant_turn(manager: SessionManager, session: _Session, turn: int, n_tokens: int) -> None:
    for i in range(n_tokens):
        manager.append_event(
            session, _event(session, "token", {"agent_name": "driver", "token": f"t{i}"}, turn)
        )
    manager.append_event(
        session,
        _event(
            session,
            "message_complete",
            {
                "message": {
                    "id": f"msg_{session.id}_{turn}",
                    "turn": turn,
                    "role": "assistant",
                    "agent_name": "driver",
                    "content": "done",
                }
            },
            turn,
        ),
    )
    action_id = f"action-{turn}"
    manager.append_event(
        session,
        _event(
            session,
            "code_submitted",
            {"action_id": action_id, "agent_name": "driver", "source": f"print({turn})"},
            turn,
        ),
    )
    manager.append_event(
        session,
        _event(
            session,
            "code_result",
            {"action_id": action_id, "agent_name": "driver", "stdout": str(turn), "success": True},
            turn,
        ),
    )
    manager.append_event(
        session, _event(session, "agent_switch", {"to_agent": "driver"}, turn)
    )


def test_structural_events_survive_long_sessions_and_tokens_are_compacted(tmp_path: Path) -> None:
    session = _session(tmp_path)
    manager = _manager(session)

    # Well past the old 5000-event cap that used to drop early structure.
    for turn in range(1, 61):
        _assistant_turn(manager, session, turn, n_tokens=200)

    types = [e["type"] for e in session.events]
    assert "token" not in types
    assert types.count("message_complete") == 60
    assert types.count("code_submitted") == 60
    assert types.count("code_result") == 60
    assert types.count("agent_switch") == 60
    seqs = [e["seq"] for e in session.events]
    assert seqs == sorted(set(seqs))
    assert seqs[0] == 201  # first message_complete, after 200 compacted tokens
    assert session.event_seq == 60 * 204


def test_tokens_of_the_in_progress_message_are_kept_for_reconnect(tmp_path: Path) -> None:
    session = _session(tmp_path)
    manager = _manager(session)
    _assistant_turn(manager, session, 1, n_tokens=3)
    for i in range(4):
        manager.append_event(
            session, _event(session, "token", {"agent_name": "driver", "token": f"p{i}"}, 2)
        )

    tokens = [e for e in session.events if e["type"] == "token"]
    assert [e["data"]["token"] for e in tokens] == ["p0", "p1", "p2", "p3"]
    assert [e["seq"] for e in session.events] == sorted(e["seq"] for e in session.events)


def test_code_event_record_keeps_submitted_source(tmp_path: Path) -> None:
    session = _session(tmp_path)
    manager = _manager(session)
    _assistant_turn(manager, session, 1, n_tokens=1)
    _assistant_turn(manager, session, 2, n_tokens=1)

    assert [c.source for c in session.code_events] == ["print(1)", "print(2)"]
    assert session.pending_code_sources == {}


def test_code_result_without_submission_raises_and_is_not_logged(tmp_path: Path) -> None:
    session = _session(tmp_path)
    manager = _manager(session)
    orphan = _event(
        session, "code_result", {"action_id": "never-submitted", "stdout": "", "success": True}
    )
    with pytest.raises(RuntimeError, match="no matching code_submitted"):
        manager.append_event(session, orphan)
    assert session.events == []
    assert session.code_events == []
    assert session.event_seq == 0


def test_event_seq_survives_restart_even_when_top_seqs_were_compacted(tmp_path: Path) -> None:
    session = _session(tmp_path)
    manager = _manager(session)
    _assistant_turn(manager, session, 1, n_tokens=5)
    for i in range(3):  # unfinished stream, then a structural event
        manager.append_event(
            session, _event(session, "token", {"agent_name": "driver", "token": f"p{i}"}, 2)
        )
    manager.append_event(session, _event(session, "user_message_queued", {}, 2))
    manager.append_event(
        session,
        _event(
            session,
            "message_complete",
            {"message": {"turn": 2, "role": "user", "agent_name": "", "content": "hi"}},
            2,
        ),
    )
    session.status = SessionStatus.idle
    last_seq = session.event_seq
    save_session(session, lambda _: False, tmp_path)

    loaded = load_persisted_sessions(tmp_path)[session.id]
    # Resumes from the reservation, never from the (compacted) log's max seq.
    assert loaded.event_seq == last_seq + SEQ_RESERVATION_BLOCK
    assert [e["seq"] for e in loaded.events] == [e["seq"] for e in session.events]


def test_interrupted_session_restart_event_continues_the_counter(tmp_path: Path) -> None:
    session = _session(tmp_path)
    manager = _manager(session)
    _assistant_turn(manager, session, 1, n_tokens=2)
    last_seq = session.event_seq
    save_session(session, lambda _: False, tmp_path)  # status is running

    loaded = load_persisted_sessions(tmp_path)[session.id]
    restart = loaded.events[-1]
    assert restart["data"]["reason"] == "server restarted"
    assert restart["seq"] == last_seq + SEQ_RESERVATION_BLOCK + 1
    assert loaded.event_seq == last_seq + SEQ_RESERVATION_BLOCK + 1


def test_legacy_session_file_events_are_backfilled_in_order(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.status = SessionStatus.idle
    save_session(session, lambda _: False, tmp_path)
    path = tmp_path / session.id / "session.json"
    record = json.loads(path.read_text())
    record["schema_version"] = "caribou.web_session.v4"
    del record["event_seq_reserved"]
    record["events"] = [
        _event(session, "token", {"token": "a"}),
        _event(session, "message_complete", {"message": {"turn": 1}}),
        _event(session, "agent_switch", {"to_agent": "x"}),
    ]
    path.write_text(json.dumps(record))

    loaded = load_persisted_sessions(tmp_path)[session.id]
    assert [e["seq"] for e in loaded.events] == [1, 2, 3]
    assert loaded.event_seq == 3


def test_backfill_rejects_inconsistent_logs() -> None:
    with pytest.raises(ValueError, match="only 1 of 2"):
        backfill_event_seq([{"type": "a", "seq": 1}, {"type": "b"}], 1)
    with pytest.raises(ValueError, match="does not follow"):
        backfill_event_seq([{"type": "a", "seq": 2}, {"type": "b", "seq": 2}], 2)
    with pytest.raises(ValueError, match="below the last event seq"):
        backfill_event_seq([{"type": "a", "seq": 1}, {"type": "b", "seq": 5}], 4)
    with pytest.raises(ValueError, match="event_seq is missing"):
        backfill_event_seq([{"type": "a", "seq": 1}], None)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, event: dict) -> None:
        self.sent.append(event)


def test_live_websocket_client_keeps_receiving_across_token_compaction(tmp_path: Path) -> None:
    async def run() -> None:
        session = _session(tmp_path)
        manager = _manager(session)
        _assistant_turn(manager, session, 1, n_tokens=2)
        # Mid-stream connect: replay what's there now.
        for i in range(3):
            manager._on_event(
                session, _event(session, "token", {"agent_name": "driver", "token": f"p{i}"}, 2)
            )
        websocket = _FakeWebSocket()
        replay = list(session.events)
        task = asyncio.create_task(
            ws_route._stream_events(websocket, session, replay[-1]["seq"])
        )
        await asyncio.sleep(0)

        # Live: more tokens, then message_complete compacts them out of the
        # log (shifting list positions), then structural events follow.
        manager._on_event(
            session, _event(session, "token", {"agent_name": "driver", "token": "p3"}, 2)
        )
        await asyncio.sleep(0.01)
        manager._on_event(
            session,
            _event(
                session,
                "message_complete",
                {"message": {"turn": 2, "role": "assistant", "agent_name": "driver", "content": "x"}},
                2,
            ),
        )
        manager._on_event(session, _event(session, "agent_switch", {"to_agent": "b"}, 2))
        manager._on_event(session, _event(session, "status_change", {"status": "stopped"}, 2))
        await asyncio.wait_for(task, timeout=2)

        assert [e["type"] for e in websocket.sent] == [
            "token",
            "message_complete",
            "agent_switch",
            "status_change",
        ]
        all_seqs = [e["seq"] for e in replay + websocket.sent]
        assert all_seqs == list(range(all_seqs[0], all_seqs[0] + len(all_seqs)))

    asyncio.run(run())


def test_unsaved_token_seqs_are_never_reused_after_a_crash(tmp_path: Path) -> None:
    """Tokens don't trigger saves; the reservation forces one before any seq
    passes what session.json reserves, so a restart can't hand them out again."""
    session = _session(tmp_path)
    manager = _manager(session)
    manager._save_session = lambda s: save_session(s, lambda _: False, tmp_path)

    async def run() -> int:
        manager._on_event(session, _event(session, "status_change", {"status": "running"}))
        for i in range(3 * SEQ_RESERVATION_BLOCK):
            manager._on_event(
                session, _event(session, "token", {"agent_name": "driver", "token": "x"}, 1)
            )
        return session.event_seq

    highest_seen = asyncio.run(run())
    # Crash: nothing else is saved. Reload from whatever is on disk.
    loaded = load_persisted_sessions(tmp_path)[session.id]
    assert loaded.events[-1]["data"]["reason"] == "server restarted"
    assert loaded.events[-1]["seq"] > highest_seen


def test_artifact_overwrite_replaces_record_by_path(tmp_path: Path) -> None:
    session = _session(tmp_path)
    manager = _manager(session)

    def artifact(path: str, mtime_ns: int, action_id) -> dict:
        return _event(
            session,
            "artifact",
            {
                "artifact": {
                    "filename": Path(path).name,
                    "path": path,
                    "type": "plot",
                    "mime_type": "image/png",
                    "size_bytes": mtime_ns,
                    "mtime_ns": mtime_ns,
                    "local_path": str(session.output_dir / path),
                    "turn": 1,
                    "action_id": action_id,
                }
            },
        )

    manager.append_event(session, artifact("qc.png", 1, "a:turn:1:block:1"))
    manager.append_event(session, artifact("figs/umap.png", 2, None))
    first_id = session.artifacts[0].id
    manager.append_event(session, artifact("qc.png", 3, "a:turn:2:block:1"))

    assert [a.path for a in session.artifacts] == ["qc.png", "figs/umap.png"]
    assert session.artifacts[0].mtime_ns == 3
    assert session.artifacts[0].action_id == "a:turn:2:block:1"
    assert session.artifacts[0].id == first_id
    assert session.artifacts[1].action_id is None


def _write_legacy_artifact_session(tmp_path: Path, create_file: bool) -> _Session:
    session = _session(tmp_path)
    session.status = SessionStatus.idle
    save_session(session, lambda _: False, tmp_path)
    path = tmp_path / session.id / "session.json"
    record = json.loads(path.read_text())
    outputs = tmp_path / session.id / "outputs"
    outputs.mkdir(parents=True)
    if create_file:
        (outputs / "qc.png").write_bytes(b"png")
    record["artifacts"] = [
        {
            "id": "art-1",
            "session_id": session.id,
            "turn": 1,
            "type": "plot",
            "filename": "qc.png",
            "mime_type": "image/png",
            "size_bytes": 3,
            "local_path": "/old/home/server_sessions/session-log/outputs/qc.png",
        }
    ]
    path.write_text(json.dumps(record))
    return session


def test_legacy_artifact_records_are_backfilled_from_disk(tmp_path: Path) -> None:
    session = _write_legacy_artifact_session(tmp_path, create_file=True)
    loaded = load_persisted_sessions(tmp_path)[session.id]
    record = loaded.artifacts[0]
    assert record.path == "qc.png"
    assert record.mtime_ns == (tmp_path / session.id / "outputs" / "qc.png").stat().st_mtime_ns
    assert record.action_id is None


def test_legacy_artifact_record_with_missing_file_fails_the_load(tmp_path: Path) -> None:
    session = _write_legacy_artifact_session(tmp_path, create_file=False)
    assert session.id not in load_persisted_sessions(tmp_path)


def test_fork_renumbers_retained_history_under_the_child_counter(
    tmp_path: Path, monkeypatch
) -> None:
    import caribou.server.session_manager as sm

    parent = _session(tmp_path, "parent")
    parent.status = SessionStatus.stopped
    parent_manager = _manager(parent)
    _assistant_turn(parent_manager, parent, 1, n_tokens=2)
    for _ in range(2):  # an unfinished stream: not retained in the fork
        parent_manager.append_event(
            parent, _event(parent, "token", {"agent_name": "driver", "token": "p"}, 2)
        )
    parent_events = [dict(e) for e in parent.events]
    parent_counter = parent.event_seq

    child = _session(tmp_path, "child")
    manager = _manager(parent)
    manager._sessions[child.id] = child

    async def no_recover(_session, _request):
        return None

    monkeypatch.setattr(
        sm, "load_checkpoint", lambda _d: {"checkpoint_id": "cp", "turn": 1, "complete": True}
    )
    monkeypatch.setattr(sm, "copy_output_tree", lambda *_a: None)
    monkeypatch.setattr(sm, "publish_checkpoint_pointer", lambda *_a: None)
    monkeypatch.setattr(sm.shutil, "copytree", lambda *_a, **_k: None)
    monkeypatch.setattr(manager, "_fork_work_items", lambda *_a: None)
    monkeypatch.setattr(manager, "_recover_session", no_recover)

    asyncio.run(manager._complete_fork(parent, child, None))

    seqs = [e["seq"] for e in child.events]
    assert seqs == list(range(1, len(seqs) + 1))
    assert child.event_seq == seqs[-1]
    assert all(e["session_id"] == "child" for e in child.events)
    types = [e["type"] for e in child.events]
    assert "token" not in types
    assert types.count("code_submitted") == 1 and types.count("code_result") == 1
    assert parent.events == parent_events
    assert parent.event_seq == parent_counter
