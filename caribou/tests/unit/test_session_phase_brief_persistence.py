from __future__ import annotations

import asyncio
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

from caribou.server.models import SessionCreateRequest, SessionStatus
from caribou.server.session_persistence import load_persisted_sessions, save_session
from caribou.server.session_setup import resolve_model_info
from caribou.server.session_state import _Session


def _session(tmp_path: Path, **overrides) -> _Session:
    config = SessionCreateRequest(
        mode="interactive",
        run_mode="full_system",
        agent_system="caribou",
        llm_backend="openai",
        sandbox_type="singularity",
        dataset_path="/tmp/example.h5ad",
    )
    session = _Session(
        id="session-phase",
        config=config,
        status=SessionStatus.idle,
        current_agent="driver",
        current_turn=1,
        messages=[],
        artifacts=[],
        code_events=[],
        output_dir=tmp_path / "session-phase" / "outputs",
        events=[],
        event_condition=asyncio.Condition(),
        stop_flag=threading.Event(),
        cancel_response_flag=threading.Event(),
        user_input_queue=queue.Queue(),
        created_at=datetime(2026, 7, 20, 12, 0, 0),
        updated_at=datetime(2026, 7, 20, 12, 1, 0),
        resolved_model=resolve_model_info(config),
    )
    for key, value in overrides.items():
        setattr(session, key, value)
    return session


def test_phase_and_brief_round_trip(tmp_path: Path) -> None:
    frozen = {
        "schema_version": "caribou.session_brief.v1",
        "deliverable": "Ship the parser",
        "in_scope": ["parser"],
        "out_of_scope": [],
        "done_when": ["tests pass"],
        "precedent": [],
        "interface_delta": "none",
        "risks": [],
        "review_class": "routine",
        "source": None,
        "created_at": "2026-07-20T12:00:00Z",
        "created_by": "driver",
    }
    session = _session(tmp_path, phase="execution", brief=frozen)

    save_session(session, lambda _: False, tmp_path)
    record = json.loads((tmp_path / session.id / "session.json").read_text())
    assert record["schema_version"] == "caribou.web_session.v5"
    assert record["phase"] == "execution"
    assert record["brief"]["deliverable"] == "Ship the parser"

    loaded = load_persisted_sessions(tmp_path)[session.id]
    assert loaded.phase == "execution"
    assert loaded.brief is not None
    assert loaded.brief["deliverable"] == "Ship the parser"


def test_briefing_phase_persists(tmp_path: Path) -> None:
    session = _session(tmp_path, phase="briefing", brief=None)
    save_session(session, lambda _: False, tmp_path)
    loaded = load_persisted_sessions(tmp_path)[session.id]
    assert loaded.phase == "briefing"
    assert loaded.brief is None


def test_legacy_v3_session_without_phase_or_brief_loads_as_execution_no_brief(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    save_session(session, lambda _: False, tmp_path)

    record_path = tmp_path / session.id / "session.json"
    record = json.loads(record_path.read_text())
    record["schema_version"] = "caribou.web_session.v3"
    del record["phase"]
    del record["brief"]
    record_path.write_text(json.dumps(record))

    loaded = load_persisted_sessions(tmp_path)[session.id]
    assert loaded.phase == "execution"
    assert loaded.brief is None
