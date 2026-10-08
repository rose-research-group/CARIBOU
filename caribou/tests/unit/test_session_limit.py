"""Active-session limit (Settings: CARIBOU_MAX_ACTIVE_SESSIONS)."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from caribou.server.models import (
    SessionCreateRequest,
    SessionForkRequest,
    SessionResumeRequest,
    SessionStatus,
)
from caribou.server.routes import config as config_routes
from caribou.server.session_manager import SessionLimitReached
from caribou.server.session_state import read_max_active_sessions

from .test_session_resume_fork_lifecycle import _manager, _stopped_session

LIMIT_MESSAGE = (
    "Active session limit reached ({active} of {limit}). Stop a session or "
    "raise the limit in Settings."
)


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setattr("caribou.server.session_manager.ENV_FILE", path)
    monkeypatch.setattr(config_routes, "ENV_FILE", path)
    monkeypatch.delenv("CARIBOU_MAX_ACTIVE_SESSIONS", raising=False)
    # PATCH /settings load_dotenv()s the test .env into the process.
    saved = dict(os.environ)
    yield path
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def settings_client(env_file) -> TestClient:
    app = FastAPI()
    app.include_router(config_routes.router)
    return TestClient(app)


# -- Settings API ------------------------------------------------------------


def test_limit_is_unset_by_default(settings_client) -> None:
    assert settings_client.get("/api/settings").json()["max_active_sessions"] is None


def test_patch_sets_and_null_removes_the_limit(settings_client, env_file) -> None:
    response = settings_client.patch("/api/settings", json={"max_active_sessions": 3})
    assert response.status_code == 200
    assert "CARIBOU_MAX_ACTIVE_SESSIONS" in response.json()["updated"]
    assert settings_client.get("/api/settings").json()["max_active_sessions"] == 3
    assert read_max_active_sessions(env_file) == 3

    # Omitting the field leaves the limit alone.
    settings_client.patch("/api/settings", json={"ollama_model": "llama3"})
    assert read_max_active_sessions(env_file) == 3

    response = settings_client.patch("/api/settings", json={"max_active_sessions": None})
    assert response.status_code == 200
    assert settings_client.get("/api/settings").json()["max_active_sessions"] is None
    assert "CARIBOU_MAX_ACTIVE_SESSIONS" not in env_file.read_text()
    assert "CARIBOU_MAX_ACTIVE_SESSIONS" not in os.environ


@pytest.mark.parametrize("value", [0, -1, "3", 2.5, True])
def test_invalid_limits_are_422(settings_client, value) -> None:
    response = settings_client.patch("/api/settings", json={"max_active_sessions": value})
    assert response.status_code == 422


@pytest.mark.parametrize("raw", ["abc", "0"])
def test_a_corrupt_stored_limit_fails_loudly(env_file, raw) -> None:
    env_file.write_text(f"CARIBOU_MAX_ACTIVE_SESSIONS={raw}\n")
    with pytest.raises(ValueError, match="CARIBOU_MAX_ACTIVE_SESSIONS"):
        read_max_active_sessions(env_file)


# -- enforcement -------------------------------------------------------------


def _config(tmp_path: Path) -> SessionCreateRequest:
    return SessionCreateRequest(
        mode="interactive",
        agent_system="caribou",
        llm_backend="openrouter",
        model_name="provider/model",
        sandbox_type="singularity",
        dataset_path=str(tmp_path / "input.h5ad"),
    )


@pytest.fixture
def manager(tmp_path: Path, env_file, monkeypatch):
    source = _stopped_session(tmp_path)
    manager = _manager(source)

    async def no_init(_session):
        return None

    async def no_recover(_session, _request):
        return None

    async def no_fork(_source, _child, _request):
        return None

    monkeypatch.setattr(manager, "_initialize_session", no_init)
    monkeypatch.setattr(manager, "_recover_session", no_recover)
    monkeypatch.setattr(manager, "_complete_fork", no_fork)
    monkeypatch.setattr("caribou.server.session_manager.SESSIONS_DIR", tmp_path / "s")
    monkeypatch.setattr(
        "caribou.server.session_manager._create_session_logger",
        lambda *_: logging.getLogger("test_session_limit"),
    )
    return manager


def _live(manager, status=SessionStatus.idle) -> None:
    """Give the stopped source a live container in `status`."""
    source = manager.get_session("source-id")
    source.status = status
    source.sandbox_manager = object()


def test_no_limit_changes_nothing(manager, tmp_path) -> None:
    async def run():
        for _ in range(3):
            await manager.create_session(_config(tmp_path))

    asyncio.run(run())
    assert len(manager._sessions) == 4


def test_create_is_refused_before_any_directory_exists(manager, tmp_path, env_file) -> None:
    env_file.write_text("CARIBOU_MAX_ACTIVE_SESSIONS=1\n")
    _live(manager)

    with pytest.raises(SessionLimitReached) as caught:
        asyncio.run(manager.create_session(_config(tmp_path)))

    assert str(caught.value) == LIMIT_MESSAGE.format(active=1, limit=1)
    assert list(manager._sessions) == ["source-id"]
    assert not (tmp_path / "s").exists()


def test_stopped_and_errored_sessions_do_not_count(manager, tmp_path, env_file) -> None:
    env_file.write_text("CARIBOU_MAX_ACTIVE_SESSIONS=1\n")
    _live(manager, SessionStatus.stopped)
    asyncio.run(manager.create_session(_config(tmp_path)))
    assert len(manager._sessions) == 2


def test_concurrent_creates_cannot_both_pass(manager, tmp_path, env_file) -> None:
    env_file.write_text("CARIBOU_MAX_ACTIVE_SESSIONS=1\n")

    async def run():
        return await asyncio.gather(
            manager.create_session(_config(tmp_path)),
            manager.create_session(_config(tmp_path)),
            return_exceptions=True,
        )

    results = asyncio.run(run())
    refused = [r for r in results if isinstance(r, SessionLimitReached)]
    assert len(refused) == 1
    # The winner counts while initializing, before it has a sandbox (A6).
    started = [s for s in manager._sessions.values() if s.id != "source-id"]
    assert len(started) == 1 and started[0].sandbox_manager is None
    assert started[0].status == SessionStatus.initializing


def test_resume_is_refused_and_left_stopped(manager, env_file, tmp_path) -> None:
    env_file.write_text("CARIBOU_MAX_ACTIVE_SESSIONS=1\n")
    asyncio.run(manager.create_session(_config(tmp_path)))  # takes the slot

    with pytest.raises(SessionLimitReached):
        asyncio.run(
            manager.resume_session(
                "source-id", SessionResumeRequest(target_mode="interactive")
            )
        )
    source = manager.get_session("source-id")
    assert source.status == SessionStatus.stopped
    assert source.attempt_number == 1


def test_fork_is_refused_before_the_child_exists(manager, env_file, tmp_path) -> None:
    env_file.write_text("CARIBOU_MAX_ACTIVE_SESSIONS=1\n")
    _live(manager)

    with pytest.raises(SessionLimitReached):
        asyncio.run(
            manager.fork_session(
                "source-id",
                SessionForkRequest(name="fork", target_mode="interactive"),
            )
        )
    assert list(manager._sessions) == ["source-id"]
    assert not (tmp_path / "s").exists()


def test_branch_is_refused_with_409(tmp_path, env_file, monkeypatch) -> None:
    from caribou.server.routes import sessions as session_routes

    from .test_branch_api import _v2, _write_checkpoint
    from .test_blocks_route import _write_blocks

    source = _stopped_session(tmp_path)
    _write_blocks(source.output_dir.parent / "blocks.json", [_v2(1), _v2(2)])
    _write_checkpoint(source)
    manager = _manager(source)
    other = _stopped_session(tmp_path)
    other.id = "other-id"
    other.status = SessionStatus.recovering
    manager._sessions[other.id] = other
    monkeypatch.setattr(session_routes, "session_manager", manager)
    monkeypatch.setattr("caribou.server.session_manager.SESSIONS_DIR", tmp_path / "s")
    env_file.write_text("CARIBOU_MAX_ACTIVE_SESSIONS=1\n")
    app = FastAPI()
    app.include_router(session_routes.router)

    response = TestClient(app).post(
        "/api/sessions/source-id/blocks/blk-0002/branch",
        json={"instruction": "x", "restore_mode": "replay"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == LIMIT_MESSAGE.format(active=1, limit=1)
    assert set(manager._sessions) == {"source-id", "other-id"}
    assert not (tmp_path / "s").exists()


def test_create_route_maps_the_limit_to_409(manager, tmp_path, env_file, monkeypatch) -> None:
    from caribou.server.routes import sessions as session_routes

    env_file.write_text("CARIBOU_MAX_ACTIVE_SESSIONS=1\n")
    _live(manager, SessionStatus.running)
    monkeypatch.setattr(session_routes, "session_manager", manager)
    app = FastAPI()
    app.include_router(session_routes.router)

    response = TestClient(app).post(
        "/api/sessions", json=_config(tmp_path).model_dump(mode="json")
    )

    assert response.status_code == 409
    assert response.json()["detail"] == LIMIT_MESSAGE.format(active=1, limit=1)
