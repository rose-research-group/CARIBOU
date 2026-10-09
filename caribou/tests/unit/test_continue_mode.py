"""Step | Run continue mode, server side: default, persistence, create
request, SessionResponse, PATCH route, WebSocket message, the
`continue_mode_changed` event, fork/branch inheritance and the live getter
handed to the runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from caribou.server import streaming_runner
from caribou.server.models import (
    ContinueModeUpdateRequest,
    SessionCreateRequest,
    SessionForkRequest,
    SessionResponse,
    SessionStatus,
)
from caribou.server.routes import sessions as session_routes
from caribou.server.routes import websocket as websocket_route
from caribou.server.session_persistence import load_persisted_sessions, save_session

from .test_block_focus_messages import _FakeWebSocket
from .test_branch_api import _post, setup  # noqa: F401 — `setup` is a fixture
from .test_session_delete import _make_manager, _make_session
from .test_session_resume_fork_lifecycle import _manager, _stopped_session

SESSION_ID = "session-continue"


def _interactive_session(tmp_path: Path, manager, status="running"):
    session = _make_session(SESSION_ID, tmp_path / SESSION_ID / "outputs")
    session.config = session.config.model_copy(update={"mode": "interactive"})
    session.status = SessionStatus(status)
    manager._sessions[session.id] = session
    return session


def _set(manager, session_id, mode):
    return asyncio.run(manager.set_continue_mode(session_id, mode))


def _client(manager) -> TestClient:
    app = FastAPI()
    app.include_router(session_routes.router)
    return TestClient(app)


def _create_request(**overrides) -> SessionCreateRequest:
    return SessionCreateRequest(
        **{
            "mode": "interactive",
            "run_mode": "full_system",
            "agent_system": "caribou",
            "llm_backend": "openrouter",
            "model_name": "provider/model",
            "sandbox_type": "singularity",
            "dataset_path": "/tmp/example.h5ad",
            **overrides,
        }
    )


# --- Default -----------------------------------------------------------------


def test_interactive_session_defaults_to_step(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)

    assert session.continue_mode == "step"
    response = session.to_response()
    assert isinstance(response, SessionResponse)
    assert response.continue_mode == "step"
    assert json.loads(response.model_dump_json())["continue_mode"] == "step"


def test_create_request_defaults_to_step_and_accepts_run():
    assert _create_request().continue_mode == "step"
    assert _create_request(continue_mode="run").continue_mode == "run"
    with pytest.raises(ValidationError):
        _create_request(continue_mode="sprint")


@pytest.mark.parametrize("mode", ["step", "run"])
def test_create_session_starts_in_the_requested_mode(tmp_path, monkeypatch, mode):
    async def run():
        manager = _make_manager(tmp_path, monkeypatch)

        async def no_init(_session):
            return None

        monkeypatch.setattr(manager, "_initialize_session", no_init)
        monkeypatch.setattr("caribou.server.session_manager.SESSIONS_DIR", tmp_path)
        monkeypatch.setattr("caribou.server.session_manager.ENV_FILE", tmp_path / ".env")
        monkeypatch.setattr(
            "caribou.server.session_manager._create_session_logger",
            lambda *_: __import__("logging").getLogger("test-continue-mode"),
        )
        response = await manager.create_session(_create_request(continue_mode=mode))
        assert response.continue_mode == mode
        assert manager.get_session(response.id).continue_mode == mode
        # Persisted at creation, so a restart before the first event keeps it.
        saved = json.loads((tmp_path / response.id / "session.json").read_text())
        assert saved["continue_mode"] == mode
        assert saved["config"]["continue_mode"] == mode

    asyncio.run(run())


# --- Persistence -------------------------------------------------------------


def test_continue_mode_round_trips_through_session_json(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)
    session.continue_mode = "run"
    save_session(session, lambda _id: False, tmp_path)

    saved = json.loads((tmp_path / SESSION_ID / "session.json").read_text())
    assert saved["continue_mode"] == "run"
    assert load_persisted_sessions(tmp_path)[SESSION_ID].continue_mode == "run"


def test_files_without_continue_mode_load_as_step(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)
    session.continue_mode = "run"
    save_session(session, lambda _id: False, tmp_path)
    path = tmp_path / SESSION_ID / "session.json"
    data = json.loads(path.read_text())
    del data["continue_mode"]
    # A legacy config predates the field too.
    del data["config"]["continue_mode"]
    path.write_text(json.dumps(data))

    reloaded = load_persisted_sessions(tmp_path)[SESSION_ID]

    assert reloaded.continue_mode == "step"
    assert reloaded.config.continue_mode == "step"
    assert reloaded.to_response().continue_mode == "step"


def test_a_malformed_persisted_continue_mode_is_not_loaded(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)
    save_session(session, lambda _id: False, tmp_path)
    path = tmp_path / SESSION_ID / "session.json"
    data = json.loads(path.read_text())
    data["continue_mode"] = "sprint"
    path.write_text(json.dumps(data))

    assert SESSION_ID not in load_persisted_sessions(tmp_path)


# --- set_continue_mode -------------------------------------------------------


@pytest.mark.parametrize("status", ["initializing", "idle", "running", "stopped", "error", "recovering"])
def test_set_continue_mode_is_allowed_in_any_status(tmp_path, monkeypatch, status):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager, status=status)
    seq_before = session.events[-1]["seq"] if session.events else 0

    response = _set(manager, session.id, "run")

    assert session.continue_mode == "run"
    assert response.continue_mode == "run"
    event = session.events[-1]
    assert event["type"] == "continue_mode_changed"
    assert event["session_id"] == session.id
    assert event["turn"] == session.current_turn
    assert event["data"] == {"continue_mode": "run"}
    assert event["seq"] == seq_before + 1
    assert session.status == status
    # Persisted with the event.
    saved = json.loads((tmp_path / SESSION_ID / "session.json").read_text())
    assert saved["continue_mode"] == "run"


def test_set_continue_mode_is_idempotent(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)

    assert _set(manager, session.id, "step").continue_mode == "step"
    assert session.events == []

    _set(manager, session.id, "run")
    assert len(session.events) == 1
    assert _set(manager, session.id, "run").continue_mode == "run"
    assert len(session.events) == 1

    _set(manager, session.id, "step")
    assert [e["data"]["continue_mode"] for e in session.events] == ["run", "step"]
    assert [e["seq"] for e in session.events] == [1, 2]


def test_set_continue_mode_unknown_session(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    with pytest.raises(KeyError):
        _set(manager, "missing", "run")


@pytest.mark.parametrize("bad", ["sprint", None, 3, ""])
def test_set_continue_mode_refuses_bad_values_visibly(tmp_path, monkeypatch, bad):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)

    response = _set(manager, session.id, bad)

    assert session.continue_mode == "step"
    assert response.continue_mode == "step"
    [event] = session.events
    assert event["type"] == "error"
    assert event["data"]["code"] == "INVALID_CONTINUE_MODE"
    assert event["data"]["fatal"] is False
    assert repr(bad) in event["data"]["message"]


# --- PATCH /api/sessions/{id}/continue-mode ----------------------------------


def test_patch_continue_mode(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)
    monkeypatch.setattr(session_routes, "session_manager", manager)
    client = _client(manager)

    response = client.patch(
        f"/api/sessions/{SESSION_ID}/continue-mode", json={"continue_mode": "run"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == SESSION_ID
    assert body["continue_mode"] == "run"
    assert session.continue_mode == "run"
    assert session.events[-1]["type"] == "continue_mode_changed"

    # Idempotent: same body again is 200 and logs nothing new.
    again = client.patch(
        f"/api/sessions/{SESSION_ID}/continue-mode", json={"continue_mode": "run"}
    )
    assert again.status_code == 200
    assert again.json()["continue_mode"] == "run"
    assert len(session.events) == 1

    back = client.patch(
        f"/api/sessions/{SESSION_ID}/continue-mode", json={"continue_mode": "step"}
    )
    assert back.status_code == 200
    assert back.json()["continue_mode"] == "step"
    assert session.events[-1]["data"] == {"continue_mode": "step"}


def test_patch_continue_mode_unknown_session_is_404(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(session_routes, "session_manager", manager)
    client = _client(manager)

    response = client.patch(
        "/api/sessions/missing/continue-mode", json={"continue_mode": "run"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


@pytest.mark.parametrize(
    "body",
    [
        {"continue_mode": "sprint"},
        {"continue_mode": None},
        {},
        {"continue_mode": "run", "extra": 1},
        {"mode": "run"},
    ],
)
def test_patch_continue_mode_bad_body_is_422(tmp_path, monkeypatch, body):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)
    monkeypatch.setattr(session_routes, "session_manager", manager)
    client = _client(manager)

    response = client.patch(f"/api/sessions/{SESSION_ID}/continue-mode", json=body)

    assert response.status_code == 422, response.text
    assert session.continue_mode == "step"
    assert session.events == []


def test_update_request_model_rejects_extra_fields():
    assert ContinueModeUpdateRequest(continue_mode="run").continue_mode == "run"
    with pytest.raises(ValidationError):
        ContinueModeUpdateRequest(continue_mode="run", reason="x")


def test_get_session_and_list_carry_continue_mode(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)
    session.continue_mode = "run"
    monkeypatch.setattr(session_routes, "session_manager", manager)
    client = _client(manager)

    assert client.get(f"/api/sessions/{SESSION_ID}").json()["continue_mode"] == "run"
    [listed] = client.get("/api/sessions").json()
    assert listed["continue_mode"] == "run"


# --- WebSocket ---------------------------------------------------------------


class _RecordingManager:
    def __init__(self):
        self.calls = []

    async def set_continue_mode(self, session_id, continue_mode):
        self.calls.append((session_id, continue_mode))
        return None


@pytest.mark.parametrize(
    "message, expected",
    [
        ({"type": "set_continue_mode", "continue_mode": "run"}, "run"),
        ({"type": "set_continue_mode", "continue_mode": "step"}, "step"),
        # Passed on so the manager refuses it visibly (INVALID_CONTINUE_MODE).
        ({"type": "set_continue_mode", "continue_mode": "sprint"}, "sprint"),
        ({"type": "set_continue_mode"}, None),
    ],
)
def test_websocket_passes_set_continue_mode_to_manager(monkeypatch, message, expected):
    recorder = _RecordingManager()
    monkeypatch.setattr(websocket_route, "session_manager", recorder)

    class _Session:
        id = "session-ws"

    asyncio.run(websocket_route._receive_messages(_FakeWebSocket([message]), _Session()))

    assert recorder.calls == [("session-ws", expected)]


def test_websocket_set_continue_mode_end_to_end(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    session = _interactive_session(tmp_path, manager)
    monkeypatch.setattr(websocket_route, "session_manager", manager)

    asyncio.run(
        websocket_route._receive_messages(
            _FakeWebSocket(
                [
                    {"type": "set_continue_mode", "continue_mode": "run"},
                    {"type": "set_continue_mode", "continue_mode": "sprint"},
                ]
            ),
            session,
        )
    )

    assert session.continue_mode == "run"
    assert [e["type"] for e in session.events] == ["continue_mode_changed", "error"]
    assert session.events[0]["data"] == {"continue_mode": "run"}
    assert session.events[1]["data"]["code"] == "INVALID_CONTINUE_MODE"


# --- Fork and branch ---------------------------------------------------------


@pytest.mark.parametrize("mode", ["step", "run"])
def test_fork_copies_continue_mode(tmp_path, monkeypatch, mode):
    async def run():
        source = _stopped_session(tmp_path)
        source.continue_mode = mode
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
        assert response.continue_mode == mode
        child = manager.get_session(response.id)
        assert child.continue_mode == mode
        # Independent afterwards.
        await manager.set_continue_mode(child.id, "step" if mode == "run" else "run")
        assert source.continue_mode == mode

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["step", "run"])
def test_branch_copies_continue_mode(setup, mode):  # noqa: F811
    source, manager, _started, client = setup
    source.continue_mode = mode

    response = _post(client)

    assert response.status_code == 201, response.text
    assert response.json()["continue_mode"] == mode
    assert manager.get_session(response.json()["id"]).continue_mode == mode


# --- The getter handed to the runner -----------------------------------------


def test_launch_runner_hands_the_runner_a_live_getter(tmp_path, monkeypatch):
    async def run():
        manager = _make_manager(tmp_path, monkeypatch)
        manager._save_session = lambda _session: None
        session = _interactive_session(tmp_path, manager, status="idle")
        captured = {}

        async def fake_run_session_async(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(streaming_runner, "run_session_async", fake_run_session_async)

        assert await manager._launch_runner(session, [{"role": "user", "content": "go"}])
        await session.runner_task

        getter = captured["continue_mode_getter"]
        assert callable(getter)
        assert getter() == "step"
        await manager.set_continue_mode(session.id, "run")
        assert getter() == "run"
        await manager.set_continue_mode(session.id, "step")
        assert getter() == "step"

    asyncio.run(run())
