"""Branch API: POST /blocks/{block_id}/branch, GET /branches, lineage
persistence, the branch's first turn in both modes, and branch retry."""

from __future__ import annotations

import asyncio
import json
import queue
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from caribou.execution.session_recovery import CHECKPOINT_SCHEMA
from caribou.execution.user_input import UserTurn
from caribou.server.models import (
    BranchRestoreMode,
    RecoveryMode,
    RecoveryStatus,
    SessionMode,
    SessionResumeRequest,
    SessionStatus,
)
from caribou.server.routes import sessions as session_routes
from caribou.server.session_persistence import load_persisted_sessions, save_session

from .test_blocks_route import _block, _write_blocks
from .test_session_resume_fork_lifecycle import _manager, _stopped_session

FINGERPRINT = {
    "n_obs": 5,
    "n_vars": 3,
    "obs_keys": ["qc"],
    "var_keys": [],
    "obsm_keys": [],
    "layers_keys": [],
}


def _entry(checkpoint_id="checkpoint_e2", **overrides) -> dict:
    entry = {
        "turn": 4,
        "checkpoint_id": checkpoint_id,
        "checkpoint_complete": True,
        "fingerprint": FINGERPRINT,
        "work_items_commit": None,
    }
    entry.update(overrides)
    return entry


def _write_checkpoint(session, checkpoint_id="checkpoint_e2", **overrides) -> dict:
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "session_id": session.id,
        "turn": 4,
        "current_agent": "analyst",
        "pin_block_id": "blk-0002",
        "dataset": {"path": "dataset.h5ad", "kind": "working_anndata", "sha256": "x"},
        "fingerprint": FINGERPRINT,
        "ledger_base": "original_dataset",
        "history": [],
        "runner_state": {"turns_completed": 4, "action_ledger": []},
        "memory": None,
        "actions": [],
        "artifacts": [],
        "capture_error": None,
        "complete": True,
    }
    checkpoint.update(overrides)
    directory = session.output_dir.parent / ".checkpoints" / checkpoint_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "checkpoint.json").write_text(json.dumps(checkpoint))
    return checkpoint


def _v2(index: int, **overrides) -> dict:
    return _block(
        index,
        schema_version="caribou.block.v2",
        entry=overrides.pop("entry", _entry()),
        inherited_from=None,
        **overrides,
    )


@pytest.fixture
def setup(tmp_path: Path, monkeypatch):
    source = _stopped_session(tmp_path)
    source.name = "source"
    _write_blocks(
        source.output_dir.parent / "blocks.json",
        [_v2(1, entry=_entry("checkpoint_e1")), _v2(2, title="Normalize")],
    )
    _write_checkpoint(source)
    manager = _manager(source)
    started: list = []

    async def fake_complete(_source, child, checkpoint, block):
        started.append((child, checkpoint, block))

    monkeypatch.setattr(manager, "_complete_branch", fake_complete)
    monkeypatch.setattr("caribou.server.session_manager.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("caribou.server.session_manager.ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(
        "caribou.server.session_manager._create_session_logger", lambda *_: None
    )
    monkeypatch.setattr(session_routes, "session_manager", manager)
    app = FastAPI()
    app.include_router(session_routes.router)
    return source, manager, started, TestClient(app)


def _post(client, block_id="blk-0002", **body):
    payload = {"instruction": "try log1p", "restore_mode": "replay", **body}
    return client.post(f"/api/sessions/source-id/blocks/{block_id}/branch", json=payload)


def test_branch_creates_a_child_with_lineage(setup) -> None:
    source, manager, started, client = setup

    response = _post(client)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["parent_session_id"] == source.id
    assert body["forked_from_checkpoint_id"] == "checkpoint_e2"
    assert body["forked_from_block_id"] == "blk-0002"
    assert body["branch_restore_mode"] == "replay"
    assert body["branch_instruction"] == "try log1p"
    assert body["name"] == "source · branch from blk-0002"
    assert body["status"] == "recovering"
    assert body["mode"] == source.config.mode.value
    assert body["agent_system"] == source.config.agent_system
    child = manager.get_session(body["id"])
    assert child.current_turn == 4
    assert child.recovery_mode == RecoveryMode.literal_replay
    assert child.attempts[0]["kind"] == "branch"
    assert child.attempts[0]["source_block"]["title"] == "Normalize"
    assert len(started) == 1

    listing = client.get("/api/sessions/source-id/branches")
    assert listing.status_code == 200
    assert [item["session_id"] for item in listing.json()] == [child.id]
    assert listing.json()[0]["forked_from_block_id"] == "blk-0002"


def test_branches_lists_only_direct_branch_children_sorted(setup) -> None:
    _source, manager, _started, client = setup
    first = _post(client, name="first").json()["id"]
    second = _post(client, name="second").json()["id"]
    manager.get_session(second).created_at = manager.get_session(first).created_at.replace(
        year=2000
    )
    ids = [item["session_id"] for item in client.get("/api/sessions/source-id/branches").json()]
    assert ids == [second, first]
    assert client.get("/api/sessions/nope/branches").status_code == 404


@pytest.mark.parametrize(
    "mutate, body, detail",
    [
        (lambda s: setattr(s, "status", SessionStatus.running), {}, "is running"),
        (None, {"block_id": "blk-0009"}, "does not exist"),
        (
            "legacy",
            {"block_id": "blk-0003"},
            "Block blk-0003 was recorded before branching support; it has no entry checkpoint.",
        ),
        ("no_fingerprint", {}, "acknowledge_unverified"),
        ("ledger_restarted", {}, "ledger_base='restarted'"),
        ("incomplete", {"restore_mode": "checkpoint"}, "incomplete: write failed"),
        ("incomplete", {"restore_mode": "llm_regen"}, "incomplete"),
        ("missing_checkpoint", {}, "missing"),
    ],
)
def test_branch_preconditions_are_409(setup, mutate, body, detail) -> None:
    source, manager, started, client = setup
    if mutate == "legacy":
        blocks = [_v2(1), _v2(2), _block(3)]
        _write_blocks(source.output_dir.parent / "blocks.json", blocks)
    elif mutate == "no_fingerprint":
        _write_checkpoint(source, fingerprint=None)
    elif mutate == "ledger_restarted":
        _write_checkpoint(source, ledger_base="restarted")
    elif mutate == "incomplete":
        _write_checkpoint(source, complete=False, capture_error="write failed")
    elif mutate == "missing_checkpoint":
        _write_blocks(
            source.output_dir.parent / "blocks.json",
            [_v2(1), _v2(2, entry=_entry("checkpoint_gone"))],
        )
    elif mutate is not None:
        mutate(source)
    block_id = body.pop("block_id", "blk-0002")

    response = _post(client, block_id=block_id, **body)

    assert response.status_code == 409, response.text
    assert detail in response.json()["detail"]
    assert started == []
    assert list(manager._sessions) == [source.id]


def test_inherited_blocks_cannot_be_branched_from(setup) -> None:
    source, manager, started, client = setup
    inherited = _v2(1)
    inherited["inherited_from"] = {"session_id": "parent-id", "block_id": "blk-0001"}
    _write_blocks(source.output_dir.parent / "blocks.json", [inherited, _v2(2)])

    response = _post(client, block_id="blk-0001")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Block blk-0001 is inherited from parent-id; branch from it in that session."
    )
    assert started == []


def test_unverified_replay_is_allowed_when_acknowledged(setup) -> None:
    source, _manager_, started, client = setup
    _write_checkpoint(source, fingerprint=None)
    assert _post(client, acknowledge_unverified=True).status_code == 201
    assert len(started) == 1


def test_incomplete_checkpoint_still_allows_replay(setup) -> None:
    source, _manager_, _started, client = setup
    _write_checkpoint(source, complete=False, capture_error="write failed")
    assert _post(client).status_code == 201


def test_running_blocks_below_the_branch_point_are_409(setup) -> None:
    source, _manager_, _started, client = setup
    _write_blocks(
        source.output_dir.parent / "blocks.json",
        [_v2(1, status="running"), _v2(2)],
    )
    response = _post(client)
    assert response.status_code == 409
    assert "still running" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"instruction": "", "restore_mode": "replay"},
        {"instruction": "   ", "restore_mode": "replay"},
        {"instruction": "x" * 4001, "restore_mode": "replay"},
        {"instruction": "x", "restore_mode": "rewind"},
        {"instruction": "x", "restore_mode": "replay", "extra": 1},
        {"instruction": "x", "restore_mode": "replay", "name": "n" * 121},
    ],
)
def test_invalid_branch_bodies_are_422(setup, body) -> None:
    _source, _manager_, _started, client = setup
    response = client.post("/api/sessions/source-id/blocks/blk-0002/branch", json=body)
    assert response.status_code == 422


def test_unknown_session_is_404(setup) -> None:
    _source, _manager_, _started, client = setup
    response = client.post(
        "/api/sessions/nope/blocks/blk-0002/branch",
        json={"instruction": "x", "restore_mode": "replay"},
    )
    assert response.status_code == 404


def test_long_default_name_is_shortened_to_the_config_limit(setup) -> None:
    source, _manager_, _started, client = setup
    source.name = "s" * 200
    body = _post(client).json()
    assert len(body["name"]) == 120
    assert body["name"].endswith(" · branch from blk-0002")


def test_lineage_fields_survive_save_and_reload(setup, tmp_path) -> None:
    _source, manager, _started, client = setup
    child = manager.get_session(_post(client, restore_mode="checkpoint").json()["id"])
    sessions_dir = tmp_path / "persisted"
    save_session(child, lambda _id: False, sessions_dir)

    loaded = load_persisted_sessions(sessions_dir)[child.id]

    assert loaded.forked_from_block_id == "blk-0002"
    assert loaded.branch_restore_mode == BranchRestoreMode.checkpoint
    assert loaded.branch_instruction == "try log1p"
    assert loaded.to_response().branch_restore_mode == BranchRestoreMode.checkpoint


# -- the branch's first user turn (A7) -------------------------------------


def _launch_with_branch_turn(manager, session, monkeypatch):
    launched: dict = {}

    async def fake_launch(_session, history, *, resume_state=None, start_waiting=False):
        launched["history"] = history
        launched["start_waiting"] = start_waiting
        launched["queue"] = _session.user_input_queue
        return True

    monkeypatch.setattr(manager, "_launch_runner", fake_launch)
    session.resume_history = [{"role": "system", "content": "prompt"}]
    session.branch_restore_mode = BranchRestoreMode.replay
    asyncio.run(manager._launch_recovered_runner(session, branch_turn="BRANCH TEXT"))
    return launched


def test_interactive_branch_turn_is_preloaded_on_the_fresh_queue(tmp_path, monkeypatch) -> None:
    session = _stopped_session(tmp_path)
    manager = _manager(session)
    stale = session.user_input_queue

    launched = _launch_with_branch_turn(manager, session, monkeypatch)

    assert launched["start_waiting"] is True
    assert launched["queue"] is session.user_input_queue is not stale
    turn = session.user_input_queue.get_nowait()
    assert turn == UserTurn(content="BRANCH TEXT", block_id=None)
    with pytest.raises(queue.Empty):
        session.user_input_queue.get_nowait()
    assert all(item["role"] != "user" for item in launched["history"])
    assert "replay recovery" in launched["history"][-1]["content"]


def test_auto_branch_turn_is_the_last_history_message(tmp_path, monkeypatch) -> None:
    session = _stopped_session(tmp_path)
    session.config = session.config.model_copy(update={"mode": SessionMode.auto})
    manager = _manager(session)

    launched = _launch_with_branch_turn(manager, session, monkeypatch)

    assert launched["start_waiting"] is False
    assert launched["history"][-1] == {"role": "user", "content": "BRANCH TEXT"}
    assert session.user_input_queue.empty()
    user_messages = [m for m in session.messages if m.role == "user"]
    assert [m.content for m in user_messages] == ["BRANCH TEXT"]


# -- retry of a failed branch restore --------------------------------------


def _failed_branch_child(setup, monkeypatch, **body):
    source, manager, started, client = setup
    child = manager.get_session(_post(client, **body).json()["id"])
    # The child holds its own copy of E, as _complete_branch leaves it.
    _write_checkpoint(child, **body.pop("checkpoint", {}))
    child.attempts[0]["prepared"] = True
    child.status = SessionStatus.stopped
    child.recovery_status = RecoveryStatus.failed
    retried: list = []

    async def fake_retry(session, checkpoint, block):
        retried.append((session, checkpoint, block))

    monkeypatch.setattr(manager, "_retry_branch", fake_retry)
    return source, manager, child, retried, client


def test_retry_maps_recovery_modes_to_branch_modes(setup, monkeypatch) -> None:
    _source, manager, child, retried, client = _failed_branch_child(
        setup, monkeypatch, restore_mode="checkpoint"
    )

    response = client.post(
        f"/api/sessions/{child.id}/recovery/retry",
        json={"recovery_mode": "smart"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["branch_restore_mode"] == "llm_regen"
    assert child.recovery_mode == RecoveryMode.smart
    assert child.attempts[-1]["kind"] == "branch_retry"
    assert child.attempts[-1]["restore_mode"] == "llm_regen"
    session, checkpoint, block = retried[0]
    assert checkpoint["checkpoint_id"] == "checkpoint_e2"
    assert block["block_id"] == "blk-0002" and block["title"] == "Normalize"

    child.status = SessionStatus.stopped
    child.recovery_status = RecoveryStatus.failed
    response = client.post(
        f"/api/sessions/{child.id}/recovery/retry",
        json={"recovery_mode": "literal_replay", "acknowledge_replay_risk": True},
    )
    assert response.status_code == 202
    assert child.branch_restore_mode == BranchRestoreMode.replay
    assert child.recovery_mode == RecoveryMode.literal_replay


def test_retry_enforces_the_mode_requirements(setup, monkeypatch) -> None:
    source, manager, child, retried, client = _failed_branch_child(
        setup, monkeypatch, restore_mode="checkpoint"
    )
    _write_checkpoint(child, complete=False, capture_error="write failed")

    response = client.post(
        f"/api/sessions/{child.id}/recovery/retry", json={"recovery_mode": "smart"}
    )

    assert response.status_code == 409
    assert "llm_regen needs a complete entry checkpoint" in response.json()["detail"]
    assert child.branch_restore_mode == BranchRestoreMode.checkpoint
    assert child.status == SessionStatus.stopped
    assert retried == []


def test_retry_of_an_unacknowledged_unverified_replay_is_409(setup, monkeypatch) -> None:
    _source, _manager_, child, retried, client = _failed_branch_child(
        setup, monkeypatch, restore_mode="checkpoint"
    )
    _write_checkpoint(child, fingerprint=None)

    response = client.post(
        f"/api/sessions/{child.id}/recovery/retry",
        json={"recovery_mode": "literal_replay", "acknowledge_replay_risk": True},
    )

    assert response.status_code == 409
    assert "acknowledge_unverified" in response.json()["detail"]
    assert retried == []


def test_retry_branch_resets_outputs_then_restores(setup, monkeypatch) -> None:
    source, manager, _started, client = setup
    child = manager.get_session(_post(client, restore_mode="checkpoint").json()["id"])
    checkpoint = _write_checkpoint(child)
    (child.output_dir / "stale.txt").write_text("from the failed attempt")
    restored: list = []

    async def fake_restore(session, checkpoint_, block, skipped):
        restored.append((sorted(p.name for p in session.output_dir.iterdir()), skipped))

    monkeypatch.setattr(manager, "_restore_branch", fake_restore)
    child.branch_restore_mode = BranchRestoreMode.replay
    asyncio.run(manager._retry_branch(child, checkpoint, {"block_id": "blk-0002"}))

    assert restored == [([], [])]


def test_fork_and_branch_children_keep_the_source_brief(tmp_path) -> None:
    source = _stopped_session(tmp_path)
    source.brief = {"deliverable": "QC report"}
    source.phase = "execution"
    source.output_dir.mkdir(parents=True)
    (source.output_dir.parent / "brief.json").write_text('{"deliverable": "QC report"}')
    child = _stopped_session(tmp_path)
    child.id = "child-id"
    child.output_dir = tmp_path / "child-id" / "outputs"
    child.output_dir.mkdir(parents=True)

    _manager(source)._copy_brief(source, child)

    assert child.brief == {"deliverable": "QC report"}
    assert child.brief is not source.brief
    assert (child.output_dir.parent / "brief.json").read_text() == '{"deliverable": "QC report"}'


def test_retry_of_a_branch_whose_copy_phase_failed_is_409(setup, monkeypatch) -> None:
    source, manager, _started, client = setup
    child = manager.get_session(_post(client, restore_mode="checkpoint").json()["id"])
    checkpoint = _write_checkpoint(child)
    source_checkpoint = source.output_dir.parent / ".checkpoints" / "checkpoint_e2"
    assert source_checkpoint.is_dir()

    def broken_inherit(*_args, **_kwargs):
        raise RuntimeError("inherit failed")

    monkeypatch.setattr("caribou.server.session_manager.inherit_blocks", broken_inherit)
    import shutil

    shutil.rmtree(child.output_dir.parent / ".checkpoints")
    block = child.attempts[0]["source_block"]
    asyncio.run(manager.__class__._complete_branch(manager, source, child, checkpoint, block))
    assert child.recovery_status == RecoveryStatus.failed
    assert child.recovery_detail == "inherit failed"
    assert "prepared" not in child.attempts[0]

    response = client.post(
        f"/api/sessions/{child.id}/recovery/retry", json={"recovery_mode": "smart"}
    )
    assert response.status_code == 409
    assert "never finished copying" in response.json()["detail"]
