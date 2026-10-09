"""Close-before-delegate, multi-command messages and the Step | Run continue
mode, in both agent loops (web `run_session_sync` and CLI
`run_agent_session`).

Modelled on session ac59f7b8, where one "please load the data" ran seven
stages with no user input: every work item was auto-transferred on
delegation and closed by the NEXT agent, `close_work_item` + `open_work_item`
in one message applied only the last line, and the last agent delegated to
itself.

- A: the message's trailing COMMAND BLOCK is applied in order (close + open
  in one message both apply).
- B: a delegation is refused while the delegating agent has an In-progress
  item it ran code under, and when it targets itself; "close then delegate"
  in one message switches without any transfer; a planning item the agent
  did no code work in still transfers.
- C: step mode (the default) pauses after a successful close with a "Stage
  closed" guidance message; run mode auto-continues; the web loop reads the
  mode live through `continue_mode_getter`.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from caribou.execution import runner
from caribou.execution.user_input import UserTurn
from caribou.server.streaming_runner import run_session_sync

from .test_execution_runner_hooks import RecordingSandbox
from .test_loop_feedback_fixes import (
    CliSequenceLlm,
    PopLLM,
    ScriptedQueue,
    _cli_agents,
    _guidance,
    _index,
    _no_runner_error,
)
from .test_streaming_runner_interactive import FakeAgent, FakeAgentSystem, FakeSandbox
from .test_streaming_runner_step0 import _stub_rag

OPEN_LOAD = 'open_work_item "Load" "load the data"'
LOAD_CODE = "```python\nprint('loaded')\n```"
CLOSE_LOAD = 'close_work_item 0 "loaded"'
OPEN_QC = 'open_work_item "QC" "filter low quality cells"'
STAGE_CLOSED_0 = (
    "Stage closed: work item #0 (block blk-0001). Review its gate, then send a "
    "message or Continue."
)
REFUSED_0 = (
    "Delegation refused: close work item #0 first — you did the work, so you "
    'write its completion summary (close_work_item 0 "<summary>"). Delegate in '
    "your next message."
)
REFUSED_SELF = "Delegation refused: you cannot delegate to yourself."


class HookedLLM(PopLLM):
    """PopLLM that runs `hooks[n]` right after its n-th provider call."""

    def __init__(self, responses, hooks=None):
        super().__init__(responses)
        self.hooks = dict(hooks or {})

    def _create(self, **kwargs):
        result = super()._create(**kwargs)
        hook = self.hooks.get(self.calls)
        if hook is not None:
            hook()
        return result


def _agents(*, qc_self_command: bool = False):
    qc_commands = {"delegate_to_input": SimpleNamespace(target_agent="input_agent")}
    if qc_self_command:
        qc_commands["delegate_to_QC_metrics"] = SimpleNamespace(
            target_agent="QC_metrics_agent"
        )
    qc = FakeAgent("QC_metrics_agent", qc_commands)
    input_agent = FakeAgent(
        "input_agent",
        {"delegate_to_QC_metrics": SimpleNamespace(target_agent="QC_metrics_agent")},
    )
    return FakeAgentSystem({"input_agent": input_agent, "QC_metrics_agent": qc}), input_agent


def _run_web(tmp_path: Path, responses, user_turns=(), *, llm=None, driver=None, **kwargs):
    agent_system, input_agent = _agents(qc_self_command=driver == "QC_metrics_agent")
    stop = threading.Event()
    events: list = []
    llm = llm or PopLLM(responses)
    run_session_sync(
        session_id="sess",
        agent_system=agent_system,
        driver_agent=agent_system.get_agent(driver) if driver else input_agent,
        analysis_context="analysis",
        llm_client=llm,
        sandbox_manager=FakeSandbox(),
        history=[{"role": "system", "content": "You are input_agent."}],
        is_auto=False,
        max_turns=50,
        model_name="fake",
        output_dir=tmp_path / "outputs",
        emit=events.append,
        stop_flag=stop,
        user_input_queue=ScriptedQueue(user_turns, stop),
        **kwargs,
    )
    return events, llm


def _items(events):
    return [e["data"]["item"] for e in events if e["type"] == "work_item_changed"]


def _switches(events):
    return [e["data"]["to_agent"] for e in events if e["type"] == "agent_switch"]


def _work_item_messages(events):
    return [
        e["data"]["content"]
        for e in events
        if e["type"] == "system_message" and e["data"].get("category") == "Work item"
    ]


def _user_messages(events):
    return [
        e["data"]["message"]["content"]
        for e in events
        if e["type"] == "message_complete" and e["data"]["message"]["role"] == "user"
    ]


def _idle_after(events, index):
    """True if the event right after `index` is the idle wait."""
    following = events[index + 1]
    return following["type"] == "status_change" and following["data"]["status"] == "idle"


# ---------------------------------------------------------------------------
# A. Command block
# ---------------------------------------------------------------------------


def test_web_close_and_open_in_one_message_both_apply(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        [
            OPEN_LOAD,
            f"Loaded.\n{LOAD_CODE}\n{CLOSE_LOAD}\n{OPEN_QC}",
            "Waiting.",
        ],
        continue_mode="run",
    )
    _no_runner_error(events)
    assert llm.calls == 3

    items = _items(events)
    assert [(i["id"], i["status"]) for i in items] == [
        (0, "In progress"),
        (0, "Done"),
        (1, "In progress"),
    ]
    assert items[2]["title"] == "QC" and items[2]["owner"] == "input_agent"
    # The close is reported before the open, each on its own.
    assert _work_item_messages(events) == [
        "Opened work item 0 (In progress) for input_agent.",
        "Work item 0 moved to Done.",
        "Opened work item 1 (In progress) for input_agent.",
    ]
    assert _guidance(events) == [
        "Continuing automatically after opening a work item.",
        "Continuing automatically after closing a work item and opening a work item.",
    ]


def test_web_a_stray_command_line_is_corrected_while_the_block_applies(
    tmp_path, monkeypatch
):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        [
            f"{OPEN_LOAD}\nI will now load the data.\n{LOAD_CODE}\n{OPEN_QC}",
            "Waiting.",
        ],
        continue_mode="run",
    )
    _no_runner_error(events)
    assert llm.calls == 2
    assert [i["title"] for i in _items(events)] == ["QC"]
    messages = _work_item_messages(events)
    assert messages[0] == "Opened work item 0 (In progress) for input_agent."
    assert messages[1].startswith("Work-item command not applied: 'open_work_item \"Load\"")
    assert "not in the command block" in messages[1]


# ---------------------------------------------------------------------------
# B. Close before delegating; no self-delegation
# ---------------------------------------------------------------------------


def test_web_close_then_delegate_in_one_message_switches_without_transfer(
    tmp_path, monkeypatch
):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        [OPEN_LOAD, f"{LOAD_CODE}\n{CLOSE_LOAD}\ndelegate_to_QC_metrics", "QC waiting."],
        continue_mode="run",
    )
    _no_runner_error(events)
    assert llm.calls == 3
    assert _switches(events) == ["QC_metrics_agent"]

    items = _items(events)
    assert [(i["id"], i["status"], i["owner"]) for i in items] == [
        (0, "In progress", "input_agent"),
        (0, "Done", "input_agent"),
    ]
    # The item was never reassigned: the one who did the work closed it.
    assert "reassigned" not in [t["kind"] for t in items[-1]["transitions"]]
    assert items[-1]["transitions"][-1]["actor"] == "input_agent"
    done = _index(
        events,
        lambda e: e["type"] == "work_item_changed" and e["data"]["item"]["status"] == "Done",
    )
    switch = _index(events, lambda e: e["type"] == "agent_switch")
    assert done < switch
    assert _guidance(events) == [
        "Continuing automatically after opening a work item.",
        "Continuing automatically after closing a work item and delegating to "
        "QC_metrics_agent (1 of 4).",
    ]


def test_web_delegation_with_an_open_worked_item_is_refused_until_closed(
    tmp_path, monkeypatch
):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        [
            OPEN_LOAD,
            f"{LOAD_CODE}\ndelegate_to_QC_metrics",  # refused: item 0 still open
            f"{CLOSE_LOAD}\ndelegate_to_QC_metrics",  # next message: close, then go
            "QC waiting.",
        ],
        continue_mode="run",
    )
    _no_runner_error(events)
    assert llm.calls == 4

    refused = _index(
        events,
        lambda e: e["type"] == "system_message" and e["data"]["content"] == REFUSED_0,
    )
    assert events[refused]["data"]["category"] == "Work item"
    switch = _index(events, lambda e: e["type"] == "agent_switch")
    assert refused < switch
    assert _switches(events) == ["QC_metrics_agent"]
    # The refusal auto-continued without consuming a handoff: the accepted
    # delegation is the first of four.
    assert _guidance(events) == [
        "Continuing automatically after opening a work item.",
        "Continuing automatically after a refused delegation.",
        "Continuing automatically after closing a work item and delegating to "
        "QC_metrics_agent (1 of 4).",
    ]
    items = _items(events)
    assert items[-1]["status"] == "Done" and items[-1]["owner"] == "input_agent"
    # The refused message's code still ran as input_agent, under item 0.
    submitted = [e["data"] for e in events if e["type"] == "code_submitted"]
    assert [(s["agent_name"], s["block_id"]) for s in submitted] == [
        ("input_agent", "blk-0001")
    ]


def test_web_planning_item_without_actions_still_transfers(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        ['open_work_item "Plan" "the pipeline"\ndelegate_to_QC_metrics', "QC waiting."],
        continue_mode="run",
    )
    _no_runner_error(events)
    assert llm.calls == 2
    assert _switches(events) == ["QC_metrics_agent"]
    items = _items(events)
    assert [(i["id"], i["owner"]) for i in items] == [
        (0, "input_agent"),
        (0, "QC_metrics_agent"),
    ]
    assert items[-1]["transitions"][-1]["kind"] == "reassigned"
    assert not any(m.startswith("Delegation refused") for m in _work_item_messages(events))


def test_web_self_delegation_is_refused(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        ["delegate_to_QC_metrics", "Noted; waiting."],
        driver="QC_metrics_agent",
        continue_mode="run",
    )
    _no_runner_error(events)
    assert llm.calls == 2
    assert _switches(events) == []
    assert _work_item_messages(events) == [REFUSED_SELF]
    assert _guidance(events) == ["Continuing automatically after a refused delegation."]


# ---------------------------------------------------------------------------
# C. Step | Run
# ---------------------------------------------------------------------------


def test_web_step_mode_pauses_after_a_close_and_run_mode_does_not(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    responses = [OPEN_LOAD, f"{LOAD_CODE}\n{CLOSE_LOAD}", "After the review."]

    # Step (the default): the open auto-continues, the close pauses, the
    # user's message is what starts the next turn.
    events, llm = _run_web(tmp_path / "step", responses, [UserTurn("Continue")])
    _no_runner_error(events)
    assert llm.calls == 3
    assert _guidance(events) == [
        "Continuing automatically after opening a work item.",
        STAGE_CLOSED_0,
    ]
    paused = _index(
        events,
        lambda e: e["type"] == "system_message" and e["data"]["content"] == STAGE_CLOSED_0,
    )
    assert _idle_after(events, paused)
    assert _user_messages(events) == ["Continue"]
    third = _index(
        events,
        lambda e: e["type"] == "message_complete"
        and e["data"]["message"]["content"] == "After the review.",
    )
    assert paused < third

    # Run: the close auto-continues; no user message is needed.
    events, llm = _run_web(tmp_path / "run", responses, continue_mode="run")
    _no_runner_error(events)
    assert llm.calls == 3
    assert _guidance(events) == [
        "Continuing automatically after opening a work item.",
        "Continuing automatically after closing a work item.",
    ]
    assert _user_messages(events) == []


def test_web_step_pause_after_close_then_delegate_waits_as_the_new_agent(
    tmp_path, monkeypatch
):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        [OPEN_LOAD, f"{LOAD_CODE}\n{CLOSE_LOAD}\ndelegate_to_QC_metrics", "QC starting."],
        [UserTurn("Continue")],
    )
    _no_runner_error(events)
    assert llm.calls == 3
    assert _switches(events) == ["QC_metrics_agent"]
    switch = _index(events, lambda e: e["type"] == "agent_switch")
    paused = _index(
        events,
        lambda e: e["type"] == "system_message" and e["data"]["content"] == STAGE_CLOSED_0,
    )
    assert switch < paused and _idle_after(events, paused)
    qc_turn = _index(
        events,
        lambda e: e["type"] == "message_complete"
        and e["data"]["message"]["agent_name"] == "QC_metrics_agent",
    )
    assert paused < qc_turn


def test_web_step_pause_names_no_block_for_an_item_closed_without_code(
    tmp_path, monkeypatch
):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path, [OPEN_LOAD, CLOSE_LOAD, "Waiting."], [UserTurn("Continue")]
    )
    _no_runner_error(events)
    assert llm.calls == 3
    assert _guidance(events)[-1] == (
        "Stage closed: work item #0 (no code block). Review its gate, then send a "
        "message or Continue."
    )


def test_web_live_getter_change_takes_effect_at_the_next_decision(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    mode = ["run"]
    llm = HookedLLM(
        [
            OPEN_LOAD,
            CLOSE_LOAD,  # run: auto-continues
            OPEN_QC,  # the toggle flips to step right after this call
            'close_work_item 1 "filtered"',  # step: pauses
            "After the review.",
        ],
        hooks={3: lambda: mode.__setitem__(0, "step")},
    )
    events, llm = _run_web(
        tmp_path,
        [],
        [UserTurn("Continue")],
        llm=llm,
        continue_mode="run",  # ignored while a getter is given
        continue_mode_getter=lambda: mode[0],
    )
    _no_runner_error(events)
    assert llm.calls == 5
    assert _guidance(events) == [
        "Continuing automatically after opening a work item.",
        "Continuing automatically after closing a work item.",
        "Continuing automatically after opening a work item.",
        "Stage closed: work item #1 (no code block). Review its gate, then send a "
        "message or Continue.",
    ]
    assert _user_messages(events) == ["Continue"]


def test_web_rejects_an_unknown_continue_mode(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    with pytest.raises(ValueError, match="continue_mode must be one of step, run"):
        _run_web(tmp_path, ["unused"], continue_mode="sometimes")


def test_web_getter_returning_an_unknown_mode_fails_loudly(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path, [OPEN_LOAD, CLOSE_LOAD], continue_mode_getter=lambda: "pause"
    )
    errors = [e["data"] for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert "continue_mode must be one of step, run" in errors[0]["message"]


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------


def _run_cli(tmp_path, monkeypatch, responses, answers, **kwargs):
    script = list(answers)

    def scripted_ask(*_args, **_kwargs):
        return script.pop(0)

    monkeypatch.setattr(runner.Prompt, "ask", scripted_ask)
    agent_system, driver = _cli_agents()
    events: list = []
    buffer = io.StringIO()
    llm = CliSequenceLlm(responses)
    result = runner.run_agent_session(
        console=Console(file=buffer, force_terminal=False, width=200),
        agent_system=agent_system,
        driver_agent=driver,
        analysis_context="bounded test",
        llm_client=llm,
        sandbox_manager=RecordingSandbox(),
        history=[{"role": "system", "content": "policy"}],
        is_auto=False,
        max_turns=50,
        output_dir=tmp_path,
        durable_run_id="run_cli",
        event_callback=events.append,
        **kwargs,
    )
    assert script == []
    return result, events, llm, buffer.getvalue()


def test_cli_delegation_is_refused_until_the_worked_item_is_closed(
    tmp_path, monkeypatch
):
    result, events, llm, output = _run_cli(
        tmp_path,
        monkeypatch,
        [
            OPEN_LOAD,
            f"{LOAD_CODE}\ndelegate_to_QC_metrics",  # refused
            f"{CLOSE_LOAD}\ndelegate_to_QC_metrics",  # close, then go
            "QC waiting.",
        ],
        ["exit"],
        continue_mode="run",
    )
    assert llm.calls == 4
    assert REFUSED_0 in output
    assert "Continuing automatically after a refused delegation." in output
    assert (
        "Continuing automatically after closing a work item and delegating to "
        "QC_metrics_agent (1 of 4)." in output
    )
    assert [e["payload"]["to_agent"] for e in events if e["event_type"] == "agent_switch"] == [
        "QC_metrics_agent"
    ]
    items = [e["payload"]["item"] for e in events if e["event_type"] == "work_item_changed"]
    assert items[-1]["status"] == "Done" and items[-1]["owner"] == "input_agent"
    assert result.current_agent_name == "QC_metrics_agent"
    assert result.end_reason == "user_exit"


def test_cli_close_and_open_in_one_message_both_apply(tmp_path, monkeypatch):
    result, events, llm, output = _run_cli(
        tmp_path,
        monkeypatch,
        [OPEN_LOAD, f"{LOAD_CODE}\n{CLOSE_LOAD}\n{OPEN_QC}", "Waiting."],
        ["exit"],
        continue_mode="run",
    )
    assert llm.calls == 3
    items = [e["payload"]["item"] for e in events if e["event_type"] == "work_item_changed"]
    assert [(i["id"], i["status"]) for i in items] == [
        (0, "In progress"),
        (0, "Done"),
        (1, "In progress"),
    ]
    assert "Continuing automatically after closing a work item and opening a work item." in output


def test_cli_step_mode_pauses_after_a_close_and_run_mode_does_not(tmp_path, monkeypatch):
    responses = [OPEN_LOAD, f"{LOAD_CODE}\n{CLOSE_LOAD}", "After the review."]

    # Step (the default): the close waits at the prompt; "Continue" goes on.
    result, events, llm, output = _run_cli(
        tmp_path / "step", monkeypatch, list(responses), ["Continue", "exit"]
    )
    assert llm.calls == 3
    assert STAGE_CLOSED_0 in output
    assert "Continuing automatically after closing a work item." not in output
    assert result.end_reason == "user_exit"

    # Run: no pause after the close.
    result, events, llm, output = _run_cli(
        tmp_path / "run", monkeypatch, list(responses), ["exit"], continue_mode="run"
    )
    assert llm.calls == 3
    assert "Stage closed" not in output
    assert "Continuing automatically after closing a work item." in output


def test_cli_rejects_an_unknown_continue_mode(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="continue_mode must be one of step, run"):
        _run_cli(tmp_path, monkeypatch, [], [], continue_mode="sometimes")
