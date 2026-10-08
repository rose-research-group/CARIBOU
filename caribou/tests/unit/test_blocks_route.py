"""Web plumbing for workbench blocks: the REST route, fork copy, and event intake."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from caribou.execution.blocks import BlockError, load_blocks
from caribou.server.routes import sessions as session_routes

from .test_session_resume_fork_lifecycle import _manager, _stopped_session


def _block(index: int, session_id: str = "source-id", **overrides) -> dict:
    block = {
        "schema_version": "caribou.block.v1",
        "block_id": f"blk-{index:04d}",
        "session_id": session_id,
        "index": index,
        "work_item_id": 2,
        "attempt": 1,
        "implicit": False,
        "title": "QC",
        "kind": None,
        "agents": ["analyst"],
        "status": "ok",
        "turn_start": 3,
        "turn_end": 5,
        "action_ids": ["source-id:turn:3:block:1"],
        "failed_action_ids": [],
        "artifact_paths": ["figures/qc.png"],
        "created_at": "2026-10-08T12:00:00Z",
        "updated_at": "2026-10-08T12:05:00Z",
    }
    block.update(overrides)
    return block


def _write_blocks(path: Path, blocks: list, session_id: str = "source-id") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "caribou.block_index.v1",
                "session_id": session_id,
                "blocks": blocks,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def session(tmp_path: Path):
    return _stopped_session(tmp_path)


@pytest.fixture
def client(session, monkeypatch) -> TestClient:
    monkeypatch.setattr(session_routes, "session_manager", _manager(session))
    app = FastAPI()
    app.include_router(session_routes.router)
    # Return 500 responses instead of re-raising, so the malformed-file case
    # is observable as the status the frontend would see.
    return TestClient(app, raise_server_exceptions=False)


def _blocks_path(session) -> Path:
    return session.output_dir.parent / "blocks.json"


def test_unknown_session_is_404(client: TestClient) -> None:
    assert client.get("/api/sessions/nope/blocks").status_code == 404


def test_session_without_blocks_file_is_not_recorded(client: TestClient) -> None:
    response = client.get("/api/sessions/source-id/blocks")
    assert response.status_code == 200
    assert response.json() == {"recorded": False, "blocks": []}


def test_blocks_file_is_returned_validated(client: TestClient, session) -> None:
    blocks = [
        _block(1),
        _block(
            2,
            work_item_id=None,
            implicit=True,
            title="analyst (no work item)",
            status="running",
            artifact_paths=[],
        ),
    ]
    _write_blocks(_blocks_path(session), blocks)

    response = client.get("/api/sessions/source-id/blocks")

    assert response.status_code == 200
    assert response.json() == {"recorded": True, "blocks": blocks}


def test_malformed_blocks_file_is_500_not_empty(client: TestClient, session) -> None:
    _blocks_path(session).parent.mkdir(parents=True, exist_ok=True)
    _blocks_path(session).write_text("{not json", encoding="utf-8")
    assert client.get("/api/sessions/source-id/blocks").status_code == 500


def test_block_record_with_unknown_field_is_500(client: TestClient, session) -> None:
    _write_blocks(_blocks_path(session), [_block(1, surprise=True)])
    assert client.get("/api/sessions/source-id/blocks").status_code == 500


def test_block_record_with_bad_status_is_500(client: TestClient, session) -> None:
    _write_blocks(_blocks_path(session), [_block(1, status="done")])
    assert client.get("/api/sessions/source-id/blocks").status_code == 500


def _child(tmp_path: Path):
    child = _stopped_session(tmp_path)
    child.id = "child-id"
    child.output_dir = tmp_path / child.id / "outputs"
    child.output_dir.mkdir(parents=True, exist_ok=True)
    return child


def test_fork_copies_blocks_next_to_work_items(tmp_path: Path, session) -> None:
    _write_blocks(_blocks_path(session), [_block(1), _block(2)])
    child = _child(tmp_path)

    _manager(session)._fork_work_items(session, child)

    copied = load_blocks(child.output_dir.parent / "blocks.json")
    assert copied is not None
    assert copied["session_id"] == child.id
    assert [b["block_id"] for b in copied["blocks"]] == ["blk-0001", "blk-0002"]
    assert {b["session_id"] for b in copied["blocks"]} == {child.id}
    # The source file is untouched.
    assert load_blocks(_blocks_path(session))["session_id"] == session.id


def test_fork_without_source_blocks_copies_nothing(tmp_path: Path, session) -> None:
    child = _child(tmp_path)
    _manager(session)._fork_work_items(session, child)
    assert not (child.output_dir.parent / "blocks.json").exists()


def test_fork_with_malformed_source_blocks_raises(tmp_path: Path, session) -> None:
    _blocks_path(session).parent.mkdir(parents=True, exist_ok=True)
    _blocks_path(session).write_text("[]", encoding="utf-8")
    child = _child(tmp_path)
    with pytest.raises(BlockError, match="not a JSON object"):
        _manager(session)._fork_work_items(session, child)
    assert not (child.output_dir.parent / "blocks.json").exists()


def _event(event_type: str, data: dict) -> dict:
    return {
        "type": event_type,
        "session_id": "source-id",
        "turn": 3,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }


def test_process_event_accepts_block_events_and_block_ids(session) -> None:
    manager = _manager(session)
    action_id = "source-id:turn:3:block:1"

    manager.append_event(session, _event("block_changed", {"block": _block(1)}))
    manager.append_event(
        session,
        _event(
            "code_submitted",
            {"action_id": action_id, "source": "print(1)", "block_id": "blk-0001"},
        ),
    )
    manager.append_event(
        session,
        _event(
            "code_result",
            {
                "action_id": action_id,
                "agent_name": "analyst",
                "stdout": "1\n",
                "stderr": "",
                "success": True,
                "duration_ms": 4,
                "block_id": "blk-0001",
            },
        ),
    )
    manager.append_event(
        session,
        _event(
            "artifact",
            {
                "artifact": {
                    "type": "image",
                    "filename": "qc.png",
                    "path": "figures/qc.png",
                    "mtime_ns": 1,
                    "action_id": action_id,
                },
                "block_id": "blk-0001",
            },
        ),
    )

    assert [e["type"] for e in session.events] == [
        "block_changed",
        "code_submitted",
        "code_result",
        "artifact",
    ]
    assert len(session.code_events) == 1
    assert [a.path for a in session.artifacts] == ["figures/qc.png"]
