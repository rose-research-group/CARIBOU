from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from caribou.execution.session_recovery import (
    capture_checkpoint,
    literal_replay,
    load_checkpoint,
)


class ReplaySandbox:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def exec_code(self, source: str, timeout: int):
        self.sources.append(source)
        if source == "raise originally":
            return {"status": "error", "stdout": "", "stderr": "original failure"}
        return {"status": "ok", "stdout": "done", "stderr": ""}


def test_checkpoint_round_trips_runner_owned_action_ledger(tmp_path: Path) -> None:
    dataset = tmp_path / "input.h5ad"
    dataset.write_bytes(b"fixture")
    output_dir = tmp_path / "session" / "outputs"
    output_dir.mkdir(parents=True)
    session = SimpleNamespace(
        id="session-1",
        output_dir=output_dir,
        config=SimpleNamespace(dataset_path=str(dataset)),
        current_agent="analyst",
        current_turn=0,
        sandbox_manager=None,
        memory_manager=None,
        events=[],
        checkpoint_id=None,
        checkpoint_turn=None,
        checkpoint_healthy=False,
    )
    actions = [
        {
            "action_id": "session-1:1:1",
            "turn": 1,
            "agent_name": "analyst",
            "source": "raise originally",
            "recorded_result": {"success": False, "stderr": "original failure"},
        }
    ]

    captured = capture_checkpoint(
        session=session,
        history=[{"role": "system", "content": "policy"}],
        runner_state={
            "current_agent_name": "analyst",
            "turns_completed": 0,
            "action_ledger": actions,
        },
    )
    loaded = load_checkpoint(output_dir)

    assert loaded == captured
    assert loaded["actions"] == actions
    assert loaded["dataset"]["kind"] == "original_dataset"
    assert session.checkpoint_id == captured["checkpoint_id"]
    assert (output_dir.parent / ".checkpoints" / "latest.json").is_file()


def test_literal_replay_includes_originally_failed_attempts() -> None:
    sandbox = ReplaySandbox()
    progress: list[dict] = []
    checkpoint = {
        "actions": [
            {
                "source": "x = 1",
                "recorded_result": {"success": True},
            },
            {
                "source": "raise originally",
                "recorded_result": {"success": False},
            },
        ]
    }

    recovered, detail = literal_replay(
        sandbox=sandbox,
        checkpoint=checkpoint,
        emit=progress.append,
    )

    assert recovered is True
    assert "2 recorded code attempts" in detail
    assert sandbox.sources == ["x = 1", "raise originally"]
    assert [item["step"] for item in progress] == [1, 2]


def test_rolling_checkpoints_prune_superseded_unreferenced_versions(tmp_path: Path) -> None:
    dataset = tmp_path / "input.h5ad"
    dataset.write_bytes(b"fixture")
    output_dir = tmp_path / "session" / "outputs"
    output_dir.mkdir(parents=True)
    session = SimpleNamespace(
        id="session-rolling",
        output_dir=output_dir,
        config=SimpleNamespace(dataset_path=str(dataset)),
        current_agent="analyst",
        current_turn=0,
        sandbox_manager=None,
        memory_manager=None,
        events=[],
        attempts=[],
        forked_from_checkpoint_id=None,
        checkpoint_id=None,
        checkpoint_turn=None,
        checkpoint_healthy=False,
    )

    for _ in range(5):
        capture_checkpoint(
            session=session,
            history=[],
            runner_state={"current_agent_name": "analyst", "turns_completed": 0},
        )

    retained = list((output_dir.parent / ".checkpoints").glob("checkpoint_*/checkpoint.json"))
    assert len(retained) == 3
    assert load_checkpoint(output_dir)["checkpoint_id"] == session.checkpoint_id


def test_action_ledger_pairs_legacy_and_unified_action_ids() -> None:
    """A web session recorded before action_id was unified
    (`{session}:{turn}:{idx}`) and resumed afterwards
    (`{session}:turn:{turn}:block:{idx}`) still pairs every result with its
    submission: the ledger compares ids for equality and never parses them."""
    from caribou.execution.event_ids import make_action_id
    from caribou.execution.session_recovery import _action_ledger

    new_id = make_action_id("s1", 2, 1)
    events = [
        {"type": "code_submitted", "turn": 1, "data": {"action_id": "s1:1:1", "source": "a = 1"}},
        {"type": "code_submitted", "turn": 1, "data": {"action_id": "s1:1:2", "source": "b = 2"}},
        {"type": "code_result", "turn": 1, "data": {"action_id": "s1:1:2", "success": False, "stderr": "boom"}},
        {"type": "code_result", "turn": 1, "data": {"action_id": "s1:1:1", "success": True, "stdout": "ok"}},
        {"type": "code_submitted", "turn": 2, "data": {"action_id": new_id, "source": "c = 3"}},
        {"type": "code_result", "turn": 2, "data": {"action_id": new_id, "success": True}},
    ]

    ledger = _action_ledger(events, through_turn=2)

    assert [(item["action_id"], item["source"], item["recorded_result"]["success"]) for item in ledger] == [
        ("s1:1:1", "a = 1", True),
        ("s1:1:2", "b = 2", False),
        (new_id, "c = 3", True),
    ]
    assert ledger[1]["recorded_result"]["stderr"] == "boom"


def test_a_default_capture_is_unpinned_published_and_has_no_fingerprint(tmp_path: Path) -> None:
    dataset = tmp_path / "input.h5ad"
    dataset.write_bytes(b"fixture")
    output_dir = tmp_path / "session" / "outputs"
    output_dir.mkdir(parents=True)
    session = SimpleNamespace(
        id="session-default",
        output_dir=output_dir,
        config=SimpleNamespace(dataset_path=str(dataset)),
        current_agent="analyst",
        current_turn=0,
        sandbox_manager=None,
        memory_manager=None,
        events=[],
        checkpoint_id=None,
        checkpoint_turn=None,
        checkpoint_healthy=False,
    )

    captured = capture_checkpoint(
        session=session,
        history=[],
        runner_state={"current_agent_name": "analyst", "turns_completed": 0, "action_ledger": []},
    )

    assert captured["pin_block_id"] is None
    assert captured["fingerprint"] is None
    assert captured["ledger_base"] == "original_dataset"
    assert captured["actions"] == []
    assert session.checkpoint_id == captured["checkpoint_id"]
