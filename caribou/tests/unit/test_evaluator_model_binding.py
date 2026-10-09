from __future__ import annotations

import asyncio
import queue
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from caribou.server.models import (
    CodeEventRecord,
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


# --- work-item review evidence (web) ------------------------------------------


def _code_result(session: _Session, turn: int, action_id: str, block_id: str, source: str, **result) -> None:
    """Drive the manager's event intake the way the streaming runner does."""
    from .test_session_event_log import _event

    manager = _manager(session)
    manager.append_event(
        session,
        _event(session, "code_submitted", {"action_id": action_id, "block_id": block_id, "agent_name": "worker", "source": source}, turn),
    )
    data = {"action_id": action_id, "block_id": block_id, "agent_name": "worker", "stdout": "", "stderr": "", "success": True}
    data.update(result)
    manager.append_event(session, _event(session, "code_result", data, turn))


def test_code_event_records_carry_action_and_block_ids(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _code_result(session, 2, "s:turn:2:block:1", "blk-0001", "x = 1", stdout="done")

    [record] = session.code_events
    assert record.action_id == "s:turn:2:block:1"
    assert record.block_id == "blk-0001"
    assert record.source == "x = 1"
    # Records persisted before these fields existed read back as None.
    from caribou.server.models import CodeEventRecord

    legacy = CodeEventRecord(**{k: v for k, v in record.model_dump().items() if k not in ("action_id", "block_id")})
    assert legacy.action_id is None and legacy.block_id is None


def test_work_item_review_sends_block_evidence_from_code_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from caribou.execution.work_items import WorkItemPolicy

    from .test_blocks_route import _block, _write_blocks

    async def run() -> None:
        session = _session(tmp_path)
        session.agent_system = SimpleNamespace(work_item_policy=WorkItemPolicy())
        manager = _manager(session)
        store = manager._work_item_store(session)
        item = store.open("QC", "Compute QC", "worker", 1)
        store.close(item["id"], "QC computed", "worker", 2)
        _write_blocks(
            session.output_dir.parent / "blocks.json",
            [
                _block(1, session_id=session.id, work_item_id=item["id"], action_ids=["a1", "legacy"], failed_action_ids=["legacy"]),
                _block(2, session_id=session.id, work_item_id=99, action_ids=["other"]),
            ],
            session_id=session.id,
        )
        _code_result(session, 2, "a1", "blk-0001", "adata.var['mt']", stdout="ok")
        # A record from before action ids were kept cannot be joined.
        session.code_events.append(
            CodeEventRecord(session_id=session.id, turn=2, agent_name="worker", source="old", success=False)
        )
        monkeypatch.setattr(
            "caribou.server.session_manager.resolve_evaluator_agent",
            lambda _system: (SimpleNamespace(name="evaluator", get_full_prompt=lambda _p: "review"), "test"),
        )
        captured: dict[str, object] = {}

        def fake_run_evaluation(**kwargs):
            captured.update(kwargs)
            return '{"verdict": "approve", "assessment": "Code and output seen"}'

        monkeypatch.setattr("caribou.execution.evaluation.run_evaluation", fake_run_evaluation)

        async def inline_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr("caribou.server.session_manager.asyncio.to_thread", inline_to_thread)

        result = await manager.review_work_item(session.id, item["id"])

        payload = captured["payload"]
        assert payload["evidence"] is result["evidence"]
        [block] = payload["evidence"]["blocks"]
        assert block["block_id"] == "blk-0001"
        assert block["actions"][0] == {
            "action_id": "a1",
            "agent": "worker",
            "success": True,
            "source": "adata.var['mt']",
            "stdout": "ok",
            "stderr": "",
        }
        assert block["actions"][1]["missing_record"] is True
        assert block["actions"][1]["success"] is False
        assert payload["evidence"]["truncated"] is False
        assert "evidence field holds what was actually executed" in payload["instructions"]
        # The API model carries it to the UI.
        from caribou.server.models import WorkItemReviewResult

        validated = WorkItemReviewResult.model_validate(result)
        assert validated.evidence is not None
        assert validated.evidence.blocks[0].actions[1].missing_record is True
        assert validated.item.status == "Done"

    asyncio.run(run())


def test_work_item_review_without_a_blocks_index_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from caribou.execution.work_items import WorkItemPolicy

    async def run() -> None:
        session = _session(tmp_path)
        session.agent_system = SimpleNamespace(work_item_policy=WorkItemPolicy())
        manager = _manager(session)
        store = manager._work_item_store(session)
        item = store.open("QC", "Compute QC", "worker", 1)
        store.close(item["id"], "QC computed", "worker", 2)
        monkeypatch.setattr(
            "caribou.server.session_manager.resolve_evaluator_agent",
            lambda _system: (SimpleNamespace(name="evaluator", get_full_prompt=lambda _p: "review"), "test"),
        )
        monkeypatch.setattr(
            "caribou.execution.evaluation.run_evaluation",
            lambda **_kwargs: '{"verdict": "reject", "assessment": "nothing ran"}',
        )

        async def inline_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr("caribou.server.session_manager.asyncio.to_thread", inline_to_thread)

        result = await manager.review_work_item(session.id, item["id"])

        assert result["evidence"] == {"blocks": [], "truncated": False, "note": "no blocks index"}

    asyncio.run(run())
