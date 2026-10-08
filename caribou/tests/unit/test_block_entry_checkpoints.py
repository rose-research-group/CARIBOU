"""Block-entry capture in both agent loops (branch contract §1, A2, A4, A7).

The web loop calls `block_entry_checkpoint(history, runner_state, block_id)`
whenever the tracker creates a NEW block, before that block's first action
runs, and stores the result as the block's `entry`. The CLI loop records a
checkpoint-less entry.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import pytest

from caribou.execution.blocks import load_blocks
from caribou.execution.event_ids import make_action_id
from caribou.execution.user_input import UserTurn
from caribou.execution.work_items import WorkItemPolicy, WorkItemStore
from caribou.server.streaming_runner import run_session_sync

from .test_streaming_runner_interactive import FakeAgent, FakeAgentSystem, FakeSandbox
from .test_streaming_runner_step0 import RecordingLLM, _run_auto, _stub_rag


def _of(events, event_type):
    return [e["data"] for e in events if e["type"] == event_type]


class RecordingHook:
    """Stands in for session_manager's capture: records each call and
    returns a checkpoint-shaped dict."""

    def __init__(self, *, complete=True, fingerprint=None):
        self.calls: list[tuple[list, dict, str]] = []
        self.complete = complete
        self.fingerprint = fingerprint

    def __call__(self, history, runner_state, block_id):
        self.calls.append((history, runner_state, block_id))
        return {
            "checkpoint_id": f"checkpoint_{block_id}",
            "complete": self.complete,
            "fingerprint": self.fingerprint,
            "pin_block_id": block_id,
        }


def _store(tmp_path: Path) -> WorkItemStore:
    return WorkItemStore(
        tmp_path / "work-items",
        session_id="sess",
        policy=WorkItemPolicy(qc_mode="optional"),
    )


def test_hook_runs_once_per_new_block_before_its_first_action(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    store = _store(tmp_path)
    fingerprint = {
        "n_obs": 10,
        "n_vars": 5,
        "obs_keys": [],
        "var_keys": [],
        "obsm_keys": [],
        "layers_keys": [],
    }
    hook = RecordingHook(fingerprint=fingerprint)
    order: list[str] = []

    class OrderSandbox:
        def exec_code(self, code, timeout):
            order.append("exec")
            return {"status": "ok", "stdout": "", "stderr": ""}

    def recording_hook(history, runner_state, block_id):
        order.append(f"hook:{block_id}")
        return hook(history, runner_state, block_id)

    events = _run_auto(
        tmp_path,
        RecordingLLM(
            [
                "```python\na = 1\n```",
                # Same implicit block: no new capture.
                "```python\nb = 2\n```",
                'open_work_item "QC" "filter low quality cells"',
                "```python\nc = 3\n```",
            ]
        ),
        sandbox_manager=OrderSandbox(),
        work_item_store=store,
        block_entry_checkpoint=recording_hook,
        max_turns=4,
    )

    assert [call[2] for call in hook.calls] == ["blk-0001", "blk-0002"]
    assert order == ["hook:blk-0001", "exec", "exec", "hook:blk-0002", "exec"]

    index = load_blocks(tmp_path / "blocks.json")
    entries = {b["block_id"]: b["entry"] for b in index["blocks"]}
    assert entries["blk-0001"] == {
        "turn": 1,
        "checkpoint_id": "checkpoint_blk-0001",
        "checkpoint_complete": True,
        "fingerprint": fingerprint,
        "work_items_commit": None,
    }
    # The work item was opened (a commit) before blk-0002 began.
    assert entries["blk-0002"]["turn"] == 4
    assert entries["blk-0002"]["checkpoint_id"] == "checkpoint_blk-0002"
    assert entries["blk-0002"]["work_items_commit"] == store.head_commit()
    assert entries["blk-0002"]["work_items_commit"] is not None

    # The entry is sent through block_changed too.
    sent = [
        d["block"]["entry"]
        for d in _of(events, "block_changed")
        if d["block"]["block_id"] == "blk-0002"
    ]
    assert sent and all(entry == entries["blk-0002"] for entry in sent)


def test_hook_history_excludes_the_message_that_opens_the_block(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    hook = RecordingHook()
    _run_auto(
        tmp_path,
        RecordingLLM(
            [
                "```python\na = 1\n```",
                'open_work_item "QC" "filter low quality cells"',
                "Now QC.\n```python\nqc = 1\n```",
            ]
        ),
        work_item_store=_store(tmp_path),
        block_entry_checkpoint=hook,
        max_turns=3,
    )

    first_history, first_state, _ = hook.calls[0]
    # Only the initial system prompt and the action-space message: turn 1's
    # own assistant message is cut.
    assert all(m["role"] == "system" for m in first_history)
    assert not any("a = 1" in m["content"] for m in first_history)
    assert first_state["action_ledger"] == []

    history, state, block_id = hook.calls[1]
    assert block_id == "blk-0002"
    contents = [m["content"] for m in history]
    assert "```python\na = 1\n```" in contents
    assert any(c.startswith('open_work_item "QC"') for c in contents)
    # The message whose code opens blk-0002 is NOT the child's last message.
    assert not any("qc = 1" in c for c in contents)
    assert history[-1]["content"] != "Now QC.\n```python\nqc = 1\n```"
    assert [a["source"] for a in state["action_ledger"]] == ["a = 1"]
    assert state["action_ledger"][0]["recorded_result"]["success"] is True
    assert state["turns_completed"] == 3


def test_new_block_mid_message_cuts_the_whole_message_but_keeps_its_ledger(
    tmp_path, monkeypatch
):
    """A2's documented consequence: the message's first code block went to
    blk-0001 and its second opens blk-0002, so the message (and the first
    block's feedback) leaves the cut history, while the in-memory ledger
    still records the first block's attempt."""
    _stub_rag(monkeypatch)
    store = _store(tmp_path)

    class OpensItemOnFirstExec:
        """The first exec stands in for a REST ticket opened meanwhile; the
        tracker syncs before the second code block and opens blk-0002."""

        def __init__(self):
            self.calls = 0

        def exec_code(self, code, timeout):
            self.calls += 1
            if self.calls == 1:
                store.open("Ticket", "from a human", owner="driver", turn=1)
            return {"status": "ok", "stdout": "", "stderr": ""}

    hook = RecordingHook()
    message = "Two steps.\n```python\nfirst = 1\n```\n```python\nsecond = 2\n```"
    events = _run_auto(
        tmp_path,
        RecordingLLM([message]),
        sandbox_manager=OpensItemOnFirstExec(),
        work_item_store=store,
        block_entry_checkpoint=hook,
    )

    assert [s["block_id"] for s in _of(events, "code_submitted")] == [
        "blk-0001",
        "blk-0002",
    ]
    history0, _, _ = hook.calls[0]
    history1, state1, _ = hook.calls[1]
    assert history1 == history0
    assert not any(m["content"] == message for m in history1)
    assert [a["action_id"] for a in state1["action_ledger"]] == [
        make_action_id("sess", 1, 1)
    ]
    assert state1["action_ledger"][0]["recorded_result"] is not None


def test_hook_failure_is_a_runner_error(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    ran: list[str] = []

    class Sandbox:
        def exec_code(self, code, timeout):
            ran.append(code)
            return {"status": "ok", "stdout": "", "stderr": ""}

    def failing_hook(history, runner_state, block_id):
        raise FileNotFoundError("No recoverable dataset is available: x.h5ad")

    events = _run_auto(
        tmp_path,
        RecordingLLM(["```python\na = 1\n```"]),
        sandbox_manager=Sandbox(),
        block_entry_checkpoint=failing_hook,
    )

    errors = _of(events, "error")
    assert errors[-1]["code"] == "RUNNER_ERROR"
    assert "No recoverable dataset" in errors[-1]["message"]
    assert ran == []
    assert _of(events, "status_change")[-1]["status"] == "error"


def test_without_a_hook_the_entry_is_cli_style(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    _run_auto(tmp_path, RecordingLLM(["```python\na = 1\n```"]))
    (block,) = load_blocks(tmp_path / "blocks.json")["blocks"]
    assert block["entry"] == {
        "turn": 1,
        "checkpoint_id": None,
        "checkpoint_complete": False,
        "fingerprint": None,
        "work_items_commit": None,
    }


def test_incomplete_capture_is_recorded_as_incomplete(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    _run_auto(
        tmp_path,
        RecordingLLM(["```python\na = 1\n```"]),
        block_entry_checkpoint=RecordingHook(complete=False),
    )
    (block,) = load_blocks(tmp_path / "blocks.json")["blocks"]
    assert block["entry"]["checkpoint_complete"] is False
    assert block["entry"]["checkpoint_id"] == "checkpoint_blk-0001"


def test_focus_on_an_existing_block_takes_no_capture(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    hook = RecordingHook()
    stop = threading.Event()

    class ScriptedQueue(queue.Queue):
        def __init__(self, turns):
            super().__init__()
            self.turns = list(turns)

        def get(self, block=True, timeout=None):
            if not self.turns:
                stop.set()
                raise queue.Empty
            return self.turns.pop(0)

    driver = FakeAgent("driver")
    run_session_sync(
        session_id="sess",
        agent_system=FakeAgentSystem({"driver": driver}),
        driver_agent=driver,
        analysis_context="analysis",
        llm_client=RecordingLLM(["```python\na = 1\n```", "```python\nb = 2\n```"]),
        sandbox_manager=FakeSandbox(),
        history=[{"role": "system", "content": "You are the driver."}],
        is_auto=False,
        max_turns=10,
        model_name="fake",
        output_dir=tmp_path / "outputs",
        emit=lambda _event: None,
        stop_flag=stop,
        user_input_queue=ScriptedQueue([UserTurn("again", block_id="blk-0001")]),
        block_entry_checkpoint=hook,
    )
    assert [call[2] for call in hook.calls] == ["blk-0001"]
    (block,) = load_blocks(tmp_path / "blocks.json")["blocks"]
    assert len(block["action_ids"]) == 2


def test_resumed_ledger_keeps_old_actions_and_appends_new_ones(tmp_path, monkeypatch):
    """A4: the web loop's in-memory ledger is seeded from resume_state, so a
    resumed (or forked, or branched) session's ledger stays cumulative."""
    _stub_rag(monkeypatch)
    checkpoints: list[dict] = []
    _run_auto(
        tmp_path,
        RecordingLLM(["```python\na = 1\n```"]),
        checkpoint_callback=lambda _h, state: checkpoints.append(state),
    )
    resumed_state = checkpoints[-1]
    assert [a["source"] for a in resumed_state["action_ledger"]] == ["a = 1"]

    hook = RecordingHook()
    checkpoints.clear()
    _run_auto(
        tmp_path,
        RecordingLLM(['open_work_item "QC" "filter"', "```python\nb = 2\n```"]),
        work_item_store=_store(tmp_path),
        resume_state=resumed_state,
        block_entry_checkpoint=hook,
        checkpoint_callback=lambda _h, state: checkpoints.append(state),
        max_turns=3,
    )
    (call,) = hook.calls
    assert call[2] == "blk-0002"
    assert [a["source"] for a in call[1]["action_ledger"]] == ["a = 1"]
    assert [a["source"] for a in checkpoints[-1]["action_ledger"]] == [
        "a = 1",
        "b = 2",
    ]


def test_branch_of_a_branch_ledger_stays_cumulative(tmp_path, monkeypatch):
    """A4: a branch launched from a block-entry runner_state, and a branch of
    that branch, both keep every earlier attempt in order."""
    _stub_rag(monkeypatch)
    first = RecordingHook()
    _run_auto(
        tmp_path / "source",
        RecordingLLM(
            ["```python\na = 1\n```", 'open_work_item "QC" "f"', "```python\nb = 2\n```"]
        ),
        work_item_store=WorkItemStore(
            tmp_path / "source" / "work-items", session_id="src", policy=WorkItemPolicy()
        ),
        block_entry_checkpoint=first,
        max_turns=3,
        session_id="src",
    )
    branch_state = first.calls[1][1]
    assert [a["source"] for a in branch_state["action_ledger"]] == ["a = 1"]

    second = RecordingHook()
    _run_auto(
        tmp_path / "child",
        RecordingLLM(
            ["```python\nc = 3\n```", 'open_work_item "QC2" "g"', "```python\nd = 4\n```"]
        ),
        work_item_store=WorkItemStore(
            tmp_path / "child" / "work-items", session_id="child", policy=WorkItemPolicy()
        ),
        resume_state=branch_state,
        block_entry_checkpoint=second,
        max_turns=branch_state["turns_completed"] + 3,
        session_id="child",
    )
    grandchild_state = second.calls[-1][1]
    assert [a["source"] for a in grandchild_state["action_ledger"]] == ["a = 1", "c = 3"]
    assert [a["action_id"] for a in grandchild_state["action_ledger"]] == [
        make_action_id("src", 1, 1),
        make_action_id("child", 4, 1),
    ]


def test_preloaded_first_turn_is_the_first_input_of_a_recovered_runner(
    tmp_path, monkeypatch
):
    """A7: a branch pre-loads its first UserTurn on the queue; a recovered
    interactive runner (`start_waiting`) takes it before calling the LLM."""
    _stub_rag(monkeypatch)
    stop = threading.Event()
    user_queue: queue.Queue = queue.Queue()
    instruction = '[Branch from blk-0002 "QC" · replay]\nUse a stricter cutoff.'
    user_queue.put(UserTurn(content=instruction, block_id=None))
    llm = RecordingLLM(["Understood."])

    def emit(event):
        # The agent replied and the runner is waiting again: stop it.
        if (
            event["type"] == "status_change"
            and event["data"]["status"] == "idle"
            and llm.stream_requests
        ):
            stop.set()

    driver = FakeAgent("driver")
    run_session_sync(
        session_id="child",
        agent_system=FakeAgentSystem({"driver": driver}),
        driver_agent=driver,
        analysis_context="analysis",
        llm_client=llm,
        sandbox_manager=FakeSandbox(),
        history=[
            {"role": "system", "content": "You are the driver."},
            {"role": "assistant", "content": "earlier"},
        ],
        is_auto=False,
        max_turns=10,
        model_name="fake",
        output_dir=tmp_path / "outputs",
        emit=emit,
        stop_flag=stop,
        user_input_queue=user_queue,
        resume_state={"turns_completed": 3, "action_ledger": []},
        start_waiting=True,
    )
    assert len(llm.stream_requests) == 1
    request = [m["content"] for m in llm.stream_requests[0]]
    assert request.index(instruction) == request.index("earlier") + 1
    assert user_queue.empty()


def test_cli_loop_records_a_checkpointless_entry(tmp_path):
    from .test_execution_runner_hooks import RecordingSandbox, SequenceLlm, _run

    _run(tmp_path, SequenceLlm(["```python\na = 1\n```"]), RecordingSandbox())
    (block,) = load_blocks(tmp_path / "blocks.json")["blocks"]
    assert block["entry"] == {
        "turn": 1,
        "checkpoint_id": None,
        "checkpoint_complete": False,
        "fingerprint": None,
        "work_items_commit": None,
    }
