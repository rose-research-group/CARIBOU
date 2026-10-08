"""CLI/control loop: block attribution on RunnerEvents and blocks.json."""

from __future__ import annotations

import io

from rich.console import Console

from caribou.execution import runner
from caribou.execution.blocks import load_blocks

from .test_execution_runner_hooks import RecordingSandbox, SequenceLlm, _agents, _run


def _payloads(events, event_type):
    return [e["payload"] for e in events if e["event_type"] == event_type]


def test_code_events_carry_block_id_and_block_changed_precedes_submission(tmp_path):
    events = []
    _run(
        tmp_path,
        SequenceLlm(["```python\nprint('hello')\n```"]),
        RecordingSandbox(),
        durable_run_id="run_blocks",
        event_callback=events.append,
    )

    types = [e["event_type"] for e in events]
    assert types == [
        "turn_started",
        "assistant_message",
        "block_changed",
        "code_submitted",
        "code_result",
        "block_changed",
        "session_end",
    ]
    submitted = _payloads(events, "code_submitted")[0]
    result = _payloads(events, "code_result")[0]
    assert submitted["block_id"] == result["block_id"] == "blk-0001"

    opened, closed = _payloads(events, "block_changed")
    assert opened["block"]["status"] == "running"
    assert opened["block"]["implicit"] is True
    assert opened["block"]["session_id"] == "run_blocks"
    assert opened["block"]["action_ids"] == [submitted["action_id"]]
    assert closed["block"]["status"] == "ok"

    index = load_blocks(tmp_path / "blocks.json")
    assert index["session_id"] == "run_blocks"
    assert index["blocks"] == [closed["block"]]


def test_failed_code_closes_the_block_as_error(tmp_path):
    events = []
    _run(
        tmp_path,
        SequenceLlm(["```python\nraise RuntimeError()\n```"]),
        RecordingSandbox(status="error"),
        event_callback=events.append,
    )
    final = _payloads(events, "block_changed")[-1]["block"]
    assert final["status"] == "error"
    assert final["failed_action_ids"] == final["action_ids"]


def test_code_under_an_open_work_item_is_attributed_to_it(tmp_path):
    events = []
    _run(
        tmp_path,
        SequenceLlm(
            [
                'open_work_item "QC" "filter low quality cells"',
                "```python\nprint('qc')\n```",
                'close_work_item 0 "filtered"',
            ]
        ),
        RecordingSandbox(),
        max_turns=3,
        event_callback=events.append,
    )

    blocks = [p["block"] for p in _payloads(events, "block_changed")]
    assert {b["block_id"] for b in blocks} == {"blk-0001"}
    assert blocks[0]["work_item_id"] == 0
    assert blocks[0]["title"] == "QC"
    assert blocks[0]["attempt"] == 1
    # Closed by the work item reaching Done (optional QC), not by session end.
    assert blocks[-1]["status"] == "ok"
    close_index = next(
        i
        for i, e in enumerate(events)
        if e["event_type"] == "work_item_changed"
        and e["payload"]["item"]["status"] == "Done"
    )
    assert events[close_index + 1]["event_type"] == "block_changed"


def test_cli_terminal_output_is_unchanged_by_blocks(tmp_path):
    buffer = io.StringIO()
    agent_system, driver = _agents()
    runner.run_agent_session(
        console=Console(file=buffer, force_terminal=False, width=200),
        agent_system=agent_system,
        driver_agent=driver,
        analysis_context="bounded test",
        llm_client=SequenceLlm(["```python\nprint('hello')\n```"]),
        sandbox_manager=RecordingSandbox(),
        history=[{"role": "system", "content": "policy"}],
        is_auto=True,
        max_turns=1,
        output_dir=tmp_path,
    )
    assert "blk-" not in buffer.getvalue()
    assert "block_changed" not in buffer.getvalue()


def test_reused_output_dir_continues_block_ids_under_the_first_session(tmp_path):
    first, second = [], []
    _run(
        tmp_path,
        SequenceLlm(["```python\npass\n```"]),
        RecordingSandbox(),
        durable_run_id="run_one",
        event_callback=first.append,
    )
    _run(
        tmp_path,
        SequenceLlm(["```python\npass\n```"]),
        RecordingSandbox(),
        durable_run_id="run_two",
        event_callback=second.append,
    )

    assert _payloads(first, "code_submitted")[0]["block_id"] == "blk-0001"
    assert _payloads(second, "code_submitted")[0]["block_id"] == "blk-0002"
    assert _payloads(second, "block_changed")[-1]["block"]["session_id"] == "run_one"
    index = load_blocks(tmp_path / "blocks.json")
    assert index["session_id"] == "run_one"
    assert [b["block_id"] for b in index["blocks"]] == ["blk-0001", "blk-0002"]


def test_user_input_syncs_blocks_before_the_next_llm_call(tmp_path, monkeypatch):
    """A human reject made while the CLI waits for input is applied as soon as
    the input is read, before the next LLM call."""
    from caribou.execution.work_items import WorkItemStore

    # The reject goes through the runner's own store, as the CLI's
    # /review-work-item command does (ctx.work_items).
    stores: list = []

    class CapturingStore(WorkItemStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            stores.append(self)

    monkeypatch.setattr(runner, "WorkItemStore", CapturingStore)

    def human_reject():
        (store,) = stores
        store.record_review(0, turn=4, verdict="reject", assessment="redo")

    script = [(None, "close it"), (human_reject, "redo it"), (None, "exit")]

    def scripted_ask(*_args, **_kwargs):
        side_effect, answer = script.pop(0)
        if side_effect is not None:
            side_effect()
        return answer

    monkeypatch.setattr(runner.Prompt, "ask", scripted_ask)
    events = []
    agent_system, driver = _agents()
    runner.run_agent_session(
        console=Console(file=io.StringIO(), force_terminal=False),
        agent_system=agent_system,
        driver_agent=driver,
        analysis_context="bounded test",
        llm_client=SequenceLlm(
            [
                'open_work_item "QC" "filter low quality cells"',
                "```python\nprint('qc')\n```",
                'close_work_item 0 "filtered"',
                # Closing auto-continues (D1); this report then waits.
                "Closed; waiting for review.",
                "```python\nprint('qc again')\n```",
            ]
        ),
        sandbox_manager=RecordingSandbox(),
        history=[{"role": "system", "content": "policy"}],
        is_auto=False,
        max_turns=10,
        output_dir=tmp_path,
        durable_run_id="run_cli",
        event_callback=events.append,
    )

    assert script == []
    assistant = [
        i for i, e in enumerate(events) if e["event_type"] == "assistant_message"
    ]
    # The fifth LLM response is the one after "redo it".
    warned = [
        i
        for i, e in enumerate(events)
        if e["event_type"] == "block_changed"
        and e["payload"]["block"]["status"] == "warn"
    ]
    assert len(warned) == 1
    assert assistant[3] < warned[0] < assistant[4]
    assert events[warned[0]]["payload"]["block"]["block_id"] == "blk-0001"

    submitted = _payloads(events, "code_submitted")
    assert [s["block_id"] for s in submitted] == ["blk-0001", "blk-0002"]
    index = load_blocks(tmp_path / "blocks.json")
    assert [(b["attempt"], b["status"]) for b in index["blocks"]] == [
        (1, "warn"),
        (2, "ok"),
    ]
