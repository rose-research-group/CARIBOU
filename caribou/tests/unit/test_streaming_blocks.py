"""Web loop: block_id on code and artifact events, block_changed events, and
blocks.json next to work-items/ (output_dir.parent)."""

from __future__ import annotations

from pathlib import Path

import pytest

from caribou.execution.blocks import BlockError, load_blocks
from caribou.execution.event_ids import make_action_id

from .test_streaming_artifacts import WritingSandbox, _write
from caribou.server.streaming_runner import run_session_sync

from .test_streaming_runner_interactive import FakeAgent, FakeAgentSystem, FakeSandbox
from .test_streaming_runner_step0 import RecordingLLM, _run_auto, _stub_rag


def _of(events, event_type):
    return [e["data"] for e in events if e["type"] == event_type]


def test_code_and_artifact_events_carry_the_block_id(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)

    def produce(out: Path):
        _write(out / "umap.png", "x", 1_000)

    events = _run_auto(
        tmp_path,
        RecordingLLM(["```python\npass\n```"]),
        session_id="sess",
        sandbox_manager=WritingSandbox(tmp_path / "outputs", [produce]),
    )

    action_id = make_action_id("sess", 1, 1)
    assert _of(events, "code_submitted")[0]["block_id"] == "blk-0001"
    assert _of(events, "code_result")[0]["block_id"] == "blk-0001"
    artifact = _of(events, "artifact")[0]
    assert artifact["block_id"] == "blk-0001"
    assert artifact["artifact"]["action_id"] == action_id

    types = [e["type"] for e in events]
    assert types.index("block_changed") < types.index("code_submitted")
    final = _of(events, "block_changed")[-1]["block"]
    assert final["status"] == "ok"
    assert final["artifact_paths"] == ["umap.png"]
    assert final["action_ids"] == [action_id]

    index = load_blocks(tmp_path / "blocks.json")
    assert index["blocks"] == [final]


def test_failed_sandbox_call_still_finishes_and_closes_the_block(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)

    class ExplodingSandbox:
        def exec_code(self, _code, timeout):
            raise RuntimeError("sandbox gone")

    events = _run_auto(
        tmp_path,
        RecordingLLM(["```python\npass\n```"]),
        sandbox_manager=ExplodingSandbox(),
    )

    assert _of(events, "code_result")[0]["block_id"] == "blk-0001"
    final = _of(events, "block_changed")[-1]["block"]
    assert final["status"] == "error"
    assert final["failed_action_ids"] == final["action_ids"]


def test_work_item_block_closes_when_the_item_is_done(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events = _run_auto(
        tmp_path,
        RecordingLLM(
            [
                'open_work_item "QC" "filter low quality cells"',
                "```python\npass\n```",
                'close_work_item 0 "filtered"',
            ]
        ),
        max_turns=3,
    )

    blocks = [d["block"] for d in _of(events, "block_changed")]
    assert {b["block_id"] for b in blocks} == {"blk-0001"}
    assert blocks[0]["work_item_id"] == 0
    assert blocks[0]["implicit"] is False
    assert blocks[-1]["status"] == "ok"
    done_index = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "work_item_changed" and e["data"]["item"]["status"] == "Done"
    )
    assert events[done_index + 1]["type"] == "block_changed"


def test_resumed_web_session_continues_block_ids(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    _run_auto(tmp_path, RecordingLLM(["```python\npass\n```"]), session_id="sess")
    events = _run_auto(
        tmp_path,
        RecordingLLM(["```python\npass\n```"]),
        session_id="sess",
        resume_state={"turns_completed": 1},
        max_turns=2,
    )
    assert _of(events, "code_submitted")[0]["block_id"] == "blk-0002"
    index = load_blocks(tmp_path / "blocks.json")
    assert [b["block_id"] for b in index["blocks"]] == ["blk-0001", "blk-0002"]


def test_malformed_blocks_file_fails_loudly(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    (tmp_path / "blocks.json").write_text("{not json")
    with pytest.raises(BlockError):
        _run_auto(tmp_path, RecordingLLM(["```python\npass\n```"]))


def test_user_message_from_the_queue_syncs_blocks_before_the_llm_call(
    tmp_path, monkeypatch
):
    """A human reject made through REST while the loop waits is applied as
    soon as the next user message is taken, before the LLM is called."""
    import queue
    import threading

    from caribou.execution.work_items import WorkItemPolicy, WorkItemStore

    _stub_rag(monkeypatch)
    store = WorkItemStore(
        tmp_path / "work-items",
        session_id="sess",
        policy=WorkItemPolicy(qc_mode="optional"),
    )
    stop = threading.Event()

    class ScriptedQueue(queue.Queue):
        """Each get runs a side effect (the REST call) and returns a message."""

        def __init__(self, script):
            super().__init__()
            self.script = list(script)

        def get(self, block=True, timeout=None):
            if not self.script:
                stop.set()
                raise queue.Empty
            side_effect, message = self.script.pop(0)
            if side_effect is not None:
                side_effect()
            return message

    def human_reject():
        store.record_review(0, turn=3, verdict="reject", assessment="redo")

    driver = FakeAgent("driver")
    events: list = []
    run_session_sync(
        session_id="sess",
        agent_system=FakeAgentSystem({"driver": driver}),
        driver_agent=driver,
        analysis_context="analysis",
        llm_client=RecordingLLM(
            [
                'open_work_item "QC" "filter low quality cells"',
                "```python\npass\n```",
                'close_work_item 0 "filtered"',
                "```python\npass\n```",
            ]
        ),
        sandbox_manager=FakeSandbox(),
        history=[{"role": "system", "content": "You are the driver."}],
        is_auto=False,
        max_turns=10,
        model_name="fake",
        output_dir=tmp_path / "outputs",
        emit=events.append,
        stop_flag=stop,
        work_item_store=store,
        user_input_queue=ScriptedQueue(
            [(None, "close it"), (human_reject, "redo it")]
        ),
    )

    user_redo = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "message_complete"
        and e["data"]["message"]["content"] == "redo it"
    )
    warned = events[user_redo + 1]
    assert warned["type"] == "block_changed"
    assert warned["data"]["block"]["block_id"] == "blk-0001"
    assert warned["data"]["block"]["status"] == "warn"

    submitted = _of(events, "code_submitted")
    assert [s["block_id"] for s in submitted] == ["blk-0001", "blk-0002"]
    index = load_blocks(tmp_path / "blocks.json")
    assert [(b["attempt"], b["status"]) for b in index["blocks"]] == [
        (1, "warn"),
        (2, "ok"),
    ]
