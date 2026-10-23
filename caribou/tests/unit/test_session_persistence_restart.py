"""Statuses written by a live runtime become stopped when the server reloads."""

from __future__ import annotations

import asyncio
import queue
import threading
from datetime import datetime
from pathlib import Path

from caribou.server.models import SessionCreateRequest, SessionStatus
from caribou.server.session_persistence import load_persisted_sessions, save_session
from caribou.server.session_state import _Session

SESSION_ID = "session-restart"


def _session(tmp_path: Path, status: SessionStatus) -> _Session:
    config = SessionCreateRequest(
        mode="interactive",
        run_mode="full_system",
        agent_system="caribou",
        llm_backend="openrouter",
        sandbox_type="singularity",
        dataset_path="/tmp/example.h5ad",
    )
    return _Session(
        id=SESSION_ID,
        config=config,
        status=status,
        current_agent="analyst",
        current_turn=2,
        messages=[],
        artifacts=[],
        code_events=[],
        output_dir=tmp_path / SESSION_ID / "outputs",
        events=[],
        event_condition=asyncio.Condition(),
        stop_flag=threading.Event(),
        cancel_response_flag=threading.Event(),
        user_input_queue=queue.Queue(),
        created_at=datetime(2026, 10, 23, 12, 0, 0),
        updated_at=datetime(2026, 10, 23, 12, 1, 0),
    )


def test_idle_session_reloads_as_stopped(tmp_path: Path) -> None:
    session = _session(tmp_path, SessionStatus.idle)
    save_session(session, lambda _: False, tmp_path)

    loaded = load_persisted_sessions(tmp_path)[SESSION_ID]

    assert loaded.status is SessionStatus.stopped
    assert loaded.events[-1]["type"] == "status_change"
    assert loaded.events[-1]["data"] == {
        "status": "stopped",
        "reason": "server restarted",
    }


def test_stopped_session_reloads_unchanged(tmp_path: Path) -> None:
    session = _session(tmp_path, SessionStatus.stopped)
    save_session(session, lambda _: False, tmp_path)

    loaded = load_persisted_sessions(tmp_path)[SESSION_ID]

    assert loaded.status is SessionStatus.stopped
    assert loaded.events == []
