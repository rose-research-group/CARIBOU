"""Keep the control-plane runner-event recorder in lockstep with the runner.

The runner emits event types as string literals; the control-plane recorder
raises on any type it does not know, which aborts the whole durable run.  These
tests parse ``execution/runner.py`` so a new emitted type cannot ship without a
matching recorder branch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import caribou.execution.runner as runner_module
from caribou.control.agent_workload import CARIBOU_AGENT_ADAPTER, _event_recorder
from caribou.control.store import ExperimentStore
from caribou.domain.enums import EventType, RunState
from caribou.domain.models import WorkItemChangedPayload
from caribou.domain.serialization import read_run_journal

from .test_control_agent_workload import _active_store


def _emitted_runner_event_types() -> set[str]:
    tree = ast.parse(Path(runner_module.__file__).read_text(encoding="utf-8"))
    emitted: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_emit_runner_event"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        value = keywords.get("event_type")
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            raise AssertionError(
                f"runner.py:{node.lineno} emits a non-literal event_type; the "
                "recorder coverage test cannot verify it"
            )
        emitted.add(value.value)
    return emitted


def _work_item_snapshot() -> dict[str, object]:
    return {
        "schema_version": "caribou.work_item.v2",
        "session_id": "run",
        "origin_run_id": "run",
        "id": 0,
        "title": "Cluster cells",
        "body": "Run Leiden clustering",
        "status": "In progress",
        "owner": "analyst",
        "created_turn": 1,
        "created_at": "2026-07-15T12:00:00Z",
        "completion_summary": None,
        "closed_turn": None,
        "closed_at": None,
        "completed_turn": None,
        "completed_at": None,
        "transitions": [
            {
                "kind": "opened",
                "actor": "analyst",
                "turn": 1,
                "timestamp": "2026-07-15T12:00:00Z",
                "from_status": None,
                "to_status": "In progress",
                "from_owner": None,
                "to_owner": "analyst",
            }
        ],
        "reviews": [],
        "notes": [],
        "latest_commit": "a" * 40,
        "opening_commit": "a" * 40,
    }


# One representative payload per runner event type, shaped like runner.py's.
_MINIMAL_PAYLOADS: dict[str, dict[str, object]] = {
    "turn_started": {"model_name": "test-model"},
    "assistant_message": {"role": "assistant", "content": "hello"},
    "rag_attempt": {"query": "leiden", "kind": "knowledge_query"},
    "rag_result": {
        "query": "leiden",
        "kind": "knowledge_query",
        "success": True,
        "content": "docs",
        "error": "",
    },
    "work_item_changed": {"item": _work_item_snapshot()},
    "agent_switch": {
        "from_agent": "analyst",
        "to_agent": "analyst",
        "command": "delegate_to_analyst",
    },
    "code_blocks_ignored": {
        "total_blocks_produced": 2,
        "executed_blocks": 1,
        "ignored_blocks": 1,
        "reason": "maximum one code block per provider turn",
    },
    "code_submitted": {
        "action_id": "run:turn:1:block:1",
        "source": "print(1)",
        "block_index": 1,
        "total_blocks": 1,
    },
    "code_result": {
        "action_id": "run:turn:1:block:1",
        "success": True,
        "status": "ok",
        "duration_ms": 5,
        "stdout": "1\n",
        "stderr": "",
        "block_index": 1,
        "total_blocks": 1,
    },
    "session_end": {
        "succeeded": True,
        "cancelled": False,
        "end_reason": "completed",
        "turns_completed": 1,
        "code_blocks_produced": 1,
        "code_exec_attempts": 1,
        "code_exec_failures": 0,
        "correction_count": 0,
        "started_at": "2026-07-15T12:00:00Z",
        "ended_at": "2026-07-15T12:00:01Z",
        "duration_seconds": 1.0,
    },
}


def _running_store(tmp_path: Path) -> tuple[ExperimentStore, str]:
    store, run_id, _ = _active_store(tmp_path, CARIBOU_AGENT_ADAPTER)
    store.transition_run(
        run_id,
        RunState.running,
        reason="runner-event coverage unit test initialized",
        actor="test-worker",
    )
    return store, run_id


def _event(run_id: str, event_type: str, payload: dict[str, object]) -> dict:
    return {
        "schema_version": "caribou.runner_event.v1",
        "event_type": event_type,
        "occurred_at": "2026-07-15T12:00:00Z",
        "run_id": run_id,
        "turn": 1,
        "agent_name": "analyst",
        "payload": payload,
    }


def test_runner_is_the_only_runner_event_emitter() -> None:
    # The scan below only covers runner.py; a second emitter must be added to
    # it rather than silently escaping the coverage check.
    package_root = Path(runner_module.__file__).resolve().parents[1]
    emitters = sorted(
        str(path.relative_to(package_root))
        for path in package_root.rglob("*.py")
        if "_emit_runner_event" in path.read_text(encoding="utf-8")
    )
    assert emitters == ["execution/runner.py"]


def test_ast_scan_finds_the_runner_event_types() -> None:
    emitted = _emitted_runner_event_types()
    # Guard against a broken parse passing vacuously.
    assert {"turn_started", "code_result", "session_end"} <= emitted


def test_every_runner_event_type_has_a_coverage_payload() -> None:
    missing = _emitted_runner_event_types() - set(_MINIMAL_PAYLOADS)
    assert not missing, (
        f"runner.py emits {sorted(missing)} but this test has no payload for "
        "them; add one and a matching _event_recorder branch"
    )


@pytest.mark.parametrize("event_type", sorted(_MINIMAL_PAYLOADS))
def test_event_recorder_handles_every_runner_event_type(
    tmp_path: Path, event_type: str
) -> None:
    assert event_type in _emitted_runner_event_types(), (
        f"{event_type} is no longer emitted by runner.py; drop its payload"
    )
    store, run_id = _running_store(tmp_path)
    before = store.run(run_id).event_sequence

    _event_recorder(store, run_id)(
        _event(run_id, event_type, _MINIMAL_PAYLOADS[event_type])
    )

    assert store.run(run_id).event_sequence > before


def test_work_item_changed_is_journaled_with_the_full_snapshot(
    tmp_path: Path,
) -> None:
    store, run_id = _running_store(tmp_path)
    snapshot = _work_item_snapshot()

    _event_recorder(store, run_id)(
        _event(run_id, "work_item_changed", {"item": snapshot})
    )

    # Reload from disk: the payload union is not discriminated, so prove the
    # journal round-trips into the right payload model.
    journal = read_run_journal(store.run_journal_path(run_id))
    event = journal.events[-1]
    assert event.event_type == EventType.work_item_changed
    assert isinstance(event.payload, WorkItemChangedPayload)
    assert event.payload.item_id == 0
    assert event.payload.status == "In progress"
    assert event.payload.owner == "analyst"
    assert event.payload.item == snapshot
    assert event.turn == 1


def test_work_item_changed_payload_rejects_snapshot_disagreement() -> None:
    with pytest.raises(ValueError, match="does not match snapshot"):
        WorkItemChangedPayload(
            item_id=1, status="In progress", owner="analyst", item=_work_item_snapshot()
        )
    snapshot = _work_item_snapshot()
    del snapshot["owner"]
    with pytest.raises(ValueError, match="missing 'owner'"):
        WorkItemChangedPayload(
            item_id=0, status="In progress", owner="analyst", item=snapshot
        )


def test_event_recorder_rejects_a_non_object_work_item(tmp_path: Path) -> None:
    store, run_id = _running_store(tmp_path)
    with pytest.raises(RuntimeError, match="'item' is not an object"):
        _event_recorder(store, run_id)(
            _event(run_id, "work_item_changed", {"item": "0"})
        )
