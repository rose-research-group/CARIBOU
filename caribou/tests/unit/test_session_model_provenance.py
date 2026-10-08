from __future__ import annotations

import asyncio
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import pytest

from caribou.server.models import SessionCreateRequest, SessionStatus
from caribou.core.python_environments import (
    PythonEnvironmentKind,
    ResolvedPythonEnvironment,
)
from caribou.server.session_persistence import load_persisted_sessions, save_session
from caribou.server.session_setup import resolve_model_info
from caribou.server.session_state import _Session


def _session(
    tmp_path: Path,
    *,
    backend: str,
) -> _Session:
    config = SessionCreateRequest(
        mode="auto",
        run_mode="full_system",
        agent_system="caribou",
        llm_backend=backend,
        sandbox_type="singularity",
        dataset_path="/tmp/example.h5ad",
        max_turns=1,
    )
    return _Session(
        id=f"session-{backend}",
        config=config,
        status=SessionStatus.idle,
        current_agent="analyst",
        current_turn=1,
        messages=[],
        artifacts=[],
        code_events=[],
        output_dir=tmp_path / f"session-{backend}" / "outputs",
        events=[],
        event_condition=asyncio.Condition(),
        stop_flag=threading.Event(),
        cancel_response_flag=threading.Event(),
        user_input_queue=queue.Queue(),
        created_at=datetime(2026, 7, 20, 12, 0, 0),
        updated_at=datetime(2026, 7, 20, 12, 1, 0),
        resolved_model=resolve_model_info(config),
    )


@pytest.mark.parametrize(
    ("backend", "expected_model", "expected_parameters"),
    [
        (
            "deepseek",
            "deepseek-v4-flash",
            {"thinking": False},
        ),
        (
            "deepseek-v4.1",
            "deepseek-flash",
            {"thinking": False},
        ),
        (
            "deepseek-thinking",
            "deepseek-v4-pro",
            {"thinking": True, "reasoning_effort": "high"},
        ),
    ],
)
def test_session_persistence_round_trips_exact_deepseek_identity(
    tmp_path: Path,
    backend: str,
    expected_model: str,
    expected_parameters: dict[str, object],
) -> None:
    session = _session(tmp_path, backend=backend)

    save_session(session, lambda _: False, tmp_path)

    record_path = tmp_path / session.id / "session.json"
    raw_record = json.loads(record_path.read_text(encoding="utf-8"))
    assert raw_record["resolved_model"] == {
        "provider": "deepseek",
        "model": expected_model,
        "parameters": expected_parameters,
    }

    loaded = load_persisted_sessions(tmp_path)[session.id]
    assert loaded.resolved_model is not None
    assert loaded.resolved_model.model == expected_model
    assert loaded.resolved_model.parameters == expected_parameters
    assert loaded.to_response().resolved_model == loaded.resolved_model


def test_legacy_session_without_model_record_stays_explicitly_unknown(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, backend="deepseek")
    save_session(session, lambda _: False, tmp_path)
    record_path = tmp_path / session.id / "session.json"
    raw_record = json.loads(record_path.read_text(encoding="utf-8"))
    raw_record.pop("resolved_model")
    record_path.write_text(json.dumps(raw_record), encoding="utf-8")

    loaded = load_persisted_sessions(tmp_path)[session.id]

    assert loaded.resolved_model is None
    assert loaded.model_name == ""


def test_session_persistence_round_trips_host_python_environment(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, backend="deepseek")
    session.python_environment = ResolvedPythonEnvironment(
        mode="host",
        path="/shared/envs/analysis",
        python_executable="/shared/envs/analysis/bin/python",
        kind=PythonEnvironmentKind.conda,
        python_version="3.12.4",
        fingerprint="abc123",
    )

    save_session(session, lambda _: False, tmp_path)
    loaded = load_persisted_sessions(tmp_path)[session.id]

    assert loaded.python_environment == session.python_environment
    assert loaded.to_response().python_environment.mode == "host"


def test_legacy_session_without_python_environment_uses_bundled_default(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, backend="deepseek")
    save_session(session, lambda _: False, tmp_path)
    record_path = tmp_path / session.id / "session.json"
    raw_record = json.loads(record_path.read_text(encoding="utf-8"))
    raw_record.pop("python_environment")
    record_path.write_text(json.dumps(raw_record), encoding="utf-8")

    loaded = load_persisted_sessions(tmp_path)[session.id]

    assert loaded.python_environment.mode == "bundled"
