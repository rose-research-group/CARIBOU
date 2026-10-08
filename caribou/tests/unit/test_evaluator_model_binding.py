from __future__ import annotations

import asyncio
import queue
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from caribou.server.models import (
    EvaluatorModelConfig,
    EvaluatorModelUpdateRequest,
    ResolvedModelInfo,
    SessionCreateRequest,
    SessionStatus,
)
from caribou.server.session_manager import SessionManager
from caribou.server.session_state import _Session


def _session(tmp_path: Path) -> _Session:
    config = SessionCreateRequest(
        mode="interactive",
        agent_system="caribou",
        llm_backend="chatgpt",
        model_name="worker-model",
        dataset_path=str(tmp_path / "dataset.h5ad"),
    )
    return _Session(
        id="session-evaluator",
        config=config,
        status=SessionStatus.idle,
        current_agent="worker",
        current_turn=2,
        messages=[],
        artifacts=[],
        code_events=[],
        output_dir=tmp_path / "outputs",
        events=[],
        event_condition=asyncio.Condition(),
        stop_flag=threading.Event(),
        cancel_response_flag=threading.Event(),
        user_input_queue=queue.Queue(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        llm_client=object(),
        model_name="worker-model",
        resolved_model=ResolvedModelInfo(provider="openai", model="worker-model"),
        evaluator_llm_client=object(),
        evaluator_model_name="worker-model",
        resolved_evaluator_model=ResolvedModelInfo(
            provider="openai", model="worker-model"
        ),
        initial_history=[{"role": "system", "content": "system"}],
        agent_system=object(),
    )


def _manager(session: _Session) -> SessionManager:
    manager = object.__new__(SessionManager)
    manager._sessions = {session.id: session}
    manager._deleted_session_ids = set()
    manager._lock = asyncio.Lock()
    manager._save_session = lambda _session: None
    return manager


def test_inherit_worker_cannot_mix_explicit_fields() -> None:
    with pytest.raises(ValueError, match="cannot declare"):
        EvaluatorModelConfig(mode="inherit_worker", llm_backend="chatgpt")


def test_evaluator_model_change_is_revisioned_and_traced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        session = _session(tmp_path)
        manager = _manager(session)
        evaluator_client = object()
        monkeypatch.setattr(
            "caribou.server.session_manager.build_evaluator_client",
            lambda *_args, **_kwargs: (
                evaluator_client,
                "judge-model",
                ResolvedModelInfo(provider="anthropic", model="judge-model"),
            ),
        )

        async def inline_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr(
            "caribou.server.session_manager.asyncio.to_thread", inline_to_thread
        )
        result = await manager.update_evaluator_model(
            session.id,
            EvaluatorModelUpdateRequest(
                expected_revision=1,
                selection=EvaluatorModelConfig(
                    mode="explicit",
                    llm_backend="claude",
                    model_name="judge-model",
                ),
                reason="  stronger final review  ",
            ),
        )

        assert result.revision == 2
        assert result.resolved_model is not None
        assert result.resolved_model.model == "judge-model"
        assert session.evaluator_llm_client is evaluator_client
        assert session.config.evaluator_model.mode == "explicit"
        changed = next(
            event
            for event in session.events
            if event["type"] == "evaluator_model_changed"
        )
        assert changed["data"]["reason"] == "stronger final review"
        assert any(message.role == "system" for message in session.messages)

        with pytest.raises(ValueError, match="refresh and retry"):
            await manager.update_evaluator_model(
                session.id,
                EvaluatorModelUpdateRequest(
                    expected_revision=1,
                    selection=EvaluatorModelConfig(),
                ),
            )

    asyncio.run(run())


def test_manual_evaluation_uses_evaluator_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        session = _session(tmp_path)
        manager = _manager(session)
        evaluator_client = object()
        session.evaluator_llm_client = evaluator_client
        session.evaluator_model_name = "judge-model"
        session.resolved_evaluator_model = ResolvedModelInfo(
            provider="anthropic", model="judge-model"
        )
        evaluator_agent = SimpleNamespace(name="evaluator")
        monkeypatch.setattr(
            "caribou.server.session_manager.resolve_evaluator_agent",
            lambda _system: (evaluator_agent, "test evaluator"),
        )
        captured: dict[str, object] = {}

        def fake_run_evaluation(**kwargs):
            captured.update(kwargs)
            return "assessment"

        monkeypatch.setattr(
            "caribou.server.session_manager.run_evaluation", fake_run_evaluation
        )

        async def inline_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr(
            "caribou.server.session_manager.asyncio.to_thread", inline_to_thread
        )
        result = await manager.evaluate_session(session.id)

        assert captured["llm_client"] is evaluator_client
        assert captured["model_name"] == "judge-model"
        assert result.provider == "anthropic"
        assert result.evaluator_model_revision == 1

    asyncio.run(run())
