"""Session-feedback fixes in both agent loops (web `run_session_sync` and CLI
`run_agent_session`):

- D1: auto-continue after a work-item open/close or a delegation; only
  delegations consume the budget of 4 handoffs, which a real user message
  resets;
- D2: inside one message, code runs first, then the work-item command (which
  may be the last line after prose), then the delegation;
- D3: code runs and is recorded as the agent that WROTE it; only then does the
  delegation switch agents;
- D4: a UserTurn with a block_id focuses that block until the handoff.

Modelled on session 53c1efa5: input_agent writes prose, a code block and a
last-line `delegate_to_QC_metrics`.
"""

from __future__ import annotations

import io
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from caribou.agents.AgentSystem import Agent, AgentSystem, Command
from caribou.execution import runner
from caribou.execution.blocks import load_blocks
from caribou.execution.user_input import UserTurn
from caribou.server.streaming_runner import run_session_sync

from .test_execution_runner_hooks import RecordingSandbox
from .test_streaming_runner_interactive import FakeAgent, FakeAgentSystem, FakeSandbox
from .test_streaming_runner_step0 import _stub_rag

INPUT_MESSAGE = (
    "The data is loaded; handing over to QC.\n"
    "```python\nprint('loaded')\n```\n"
    "delegate_to_QC_metrics"
)
QC_CODE = "```python\nprint('qc')\n```"


class PopLLM:
    """Streams one scripted response per provider call; a call past the end
    of the script fails the test loudly."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if not self.responses:
            raise AssertionError(f"unexpected provider call #{self.calls + 1}")
        self.calls += 1
        text = self.responses.pop(0)
        if kwargs.get("stream"):
            return [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
                )
            ]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )


class ScriptedQueue(queue.Queue):
    """Hands out the scripted UserTurns (or raw items) in order, then stops
    the session."""

    def __init__(self, items, stop: threading.Event):
        super().__init__()
        self.items = list(items)
        self.stop = stop

    def get(self, block=True, timeout=None):
        if not self.items:
            self.stop.set()
            raise queue.Empty
        return self.items.pop(0)


def _web_agents():
    qc = FakeAgent(
        "QC_metrics_agent",
        {"delegate_to_input": SimpleNamespace(target_agent="input_agent")},
    )
    input_agent = FakeAgent(
        "input_agent",
        {
            "delegate_to_QC_metrics": SimpleNamespace(
                target_agent="QC_metrics_agent"
            )
        },
    )
    return (
        FakeAgentSystem({"input_agent": input_agent, "QC_metrics_agent": qc}),
        input_agent,
    )


def _run_web(tmp_path: Path, responses, user_turns=()):
    _stub_agents, driver = _web_agents()
    stop = threading.Event()
    events: list = []
    llm = PopLLM(responses)
    run_session_sync(
        session_id="sess",
        agent_system=_stub_agents,
        driver_agent=driver,
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
    )
    return events, llm


def _types(events):
    return [e["type"] for e in events]


def _index(events, predicate):
    return next(i for i, e in enumerate(events) if predicate(e))


def _guidance(events):
    return [
        e["data"]["content"]
        for e in events
        if e["type"] == "system_message"
        and e["data"].get("category") == "Runner guidance"
    ]


def _no_runner_error(events):
    errors = [e["data"] for e in events if e["type"] == "error"]
    assert errors == []


# ---------------------------------------------------------------------------
# Web loop
# ---------------------------------------------------------------------------


def test_web_code_runs_as_its_author_before_the_delegation(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, llm = _run_web(tmp_path, [INPUT_MESSAGE, QC_CODE])
    _no_runner_error(events)

    assert llm.calls == 2
    submitted = [e["data"] for e in events if e["type"] == "code_submitted"]
    results = [e["data"] for e in events if e["type"] == "code_result"]
    assert [s["agent_name"] for s in submitted] == [
        "input_agent",
        "QC_metrics_agent",
    ]
    assert [r["agent_name"] for r in results] == ["input_agent", "QC_metrics_agent"]

    # Strict order inside input_agent's message: its code, then the switch,
    # then the automatic continue, then QC's turn.
    first_result = _index(events, lambda e: e["type"] == "code_result")
    switch = _index(events, lambda e: e["type"] == "agent_switch")
    guidance = _index(
        events,
        lambda e: e["type"] == "system_message"
        and e["data"].get("category") == "Runner guidance",
    )
    qc_message = _index(
        events,
        lambda e: e["type"] == "message_complete"
        and e["data"]["message"]["agent_name"] == "QC_metrics_agent",
    )
    assert first_result < switch < guidance < qc_message
    assert events[switch]["data"]["from_agent"] == "input_agent"
    assert events[switch]["data"]["to_agent"] == "QC_metrics_agent"
    assert _guidance(events) == [
        "Continuing automatically after delegating to QC_metrics_agent (1 of 4)."
    ]
    # No user message was needed between the two agent turns.
    assert not any(
        e["type"] == "message_complete" and e["data"]["message"]["role"] == "user"
        for e in events
    )

    blocks = load_blocks(tmp_path / "blocks.json")["blocks"]
    by_id = {b["block_id"]: b for b in blocks}
    assert by_id[submitted[0]["block_id"]]["agents"] == ["input_agent"]
    qc_block = by_id[submitted[1]["block_id"]]
    assert qc_block["title"] == "QC metrics (from input_agent)"
    assert qc_block["agents"] == ["QC_metrics_agent"]


def test_web_prose_then_last_line_close_closes_and_auto_continues(
    tmp_path, monkeypatch
):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        [
            'open_work_item "QC" "filter low quality cells"',
            "Filtering is finished and saved.\n"
            "```python\nprint('filter')\n```\n"
            'close_work_item 0 "filtered"',
            "All done for now.",
        ],
    )
    _no_runner_error(events)

    assert llm.calls == 3
    items = [e["data"]["item"] for e in events if e["type"] == "work_item_changed"]
    assert items[-1]["id"] == 0
    assert items[-1]["status"] == "Done"
    assert _guidance(events) == [
        "Continuing automatically after opening a work item.",
        "Continuing automatically after closing a work item.",
    ]
    # The close is applied after the message's code ran (D2), and the code is
    # attributed to the item it closes.
    code_result = _index(events, lambda e: e["type"] == "code_result")
    done = _index(
        events,
        lambda e: e["type"] == "work_item_changed"
        and e["data"]["item"]["status"] == "Done",
    )
    assert code_result < done
    block = load_blocks(tmp_path / "blocks.json")["blocks"][0]
    assert block["work_item_id"] == 0
    assert block["status"] == "ok"


def test_web_opens_and_closes_do_not_consume_the_handoff_budget(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    opens = [f'open_work_item "Step {n}" "Do step {n}"' for n in range(6)]
    events, llm = _run_web(tmp_path, opens + ["waiting"])
    _no_runner_error(events)

    # Six opens in a row (more than the budget of four) all auto-continue.
    assert llm.calls == 7
    assert _guidance(events) == [
        "Continuing automatically after opening a work item."
    ] * 6


def test_web_delegations_consume_the_budget_and_a_user_message_resets_it(
    tmp_path, monkeypatch
):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        [
            # input_agent: a command directly before the delegation line.
            'open_work_item "Load" "load the data"\ndelegate_to_QC_metrics',
            "delegate_to_input",  # QC_metrics_agent
            "delegate_to_QC_metrics",  # input_agent
            "delegate_to_input",  # QC_metrics_agent
            "delegate_to_QC_metrics",  # input_agent: 5th handoff, budget spent
            "delegate_to_input",  # QC_metrics_agent, after the user's message
            "waiting",  # input_agent
        ],
        user_turns=[UserTurn("keep going")],
    )
    _no_runner_error(events)

    assert llm.calls == 7
    assert _guidance(events) == [
        "Continuing automatically after opening a work item and delegating to "
        "QC_metrics_agent (1 of 4).",
        "Continuing automatically after delegating to input_agent (2 of 4).",
        "Continuing automatically after delegating to QC_metrics_agent (3 of 4).",
        "Continuing automatically after delegating to input_agent (4 of 4).",
        "Automatic continuation paused after 4 handoffs. Waiting for your next message.",
        "Continuing automatically after delegating to input_agent (1 of 4).",
    ]
    # The fifth delegation still switched agents; only the auto-continue
    # paused, so the user's message went to QC_metrics_agent.
    switches = [e["data"]["to_agent"] for e in events if e["type"] == "agent_switch"]
    assert switches == [
        "QC_metrics_agent",
        "input_agent",
        "QC_metrics_agent",
        "input_agent",
        "QC_metrics_agent",
        "input_agent",
    ]
    paused = _index(
        events,
        lambda e: e["type"] == "system_message"
        and e["data"]["content"].startswith("Automatic continuation paused"),
    )
    assert events[paused + 1]["type"] == "status_change"
    assert events[paused + 1]["data"]["status"] == "idle"
    # The open before the delegation line was applied (and transferred).
    opened = [e["data"]["item"] for e in events if e["type"] == "work_item_changed"]
    assert opened[0]["title"] == "Load"
    assert opened[0]["owner"] == "input_agent"
    assert opened[1]["owner"] == "QC_metrics_agent"


def test_web_focus_reopens_a_closed_block_until_the_handoff(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, llm = _run_web(
        tmp_path,
        [
            "```python\nprint('first')\n```",  # blk-0001, implicit
            'open_work_item "Next" "next step"',
            "```python\nprint('next')\n```",  # blk-0002; closes blk-0001
            # Feedback about blk-0001: fix it, then hand off.
            "Fixing it.\n```python\nprint('fix')\n```\ndelegate_to_QC_metrics",
            "QC has nothing to run yet.",
        ],
        user_turns=[
            UserTurn("start the next step"),
            UserTurn("[block blk-0001] redo this", block_id="blk-0001"),
        ],
    )
    _no_runner_error(events)
    assert llm.calls == 5

    submitted = [e["data"] for e in events if e["type"] == "code_submitted"]
    assert [s["block_id"] for s in submitted] == ["blk-0001", "blk-0002", "blk-0001"]

    def blk1_status(i):
        return (
            events[i]["type"] == "block_changed"
            and events[i]["data"]["block"]["block_id"] == "blk-0001"
        )

    closed_before = max(
        i
        for i in range(len(events))
        if blk1_status(i)
        and events[i]["data"]["block"]["status"] == "ok"
        and i < _index(events, lambda e: e["type"] == "code_submitted"
                       and e["data"]["action_id"] == submitted[2]["action_id"])
    )
    fix_submitted = _index(
        events,
        lambda e: e["type"] == "code_submitted"
        and e["data"]["action_id"] == submitted[2]["action_id"],
    )
    reopened = fix_submitted - 1
    assert blk1_status(reopened)
    assert events[reopened]["data"]["block"]["status"] == "running"
    assert closed_before < reopened
    switch = _index(events, lambda e: e["type"] == "agent_switch")
    reclosed = [
        i
        for i in range(fix_submitted, switch)
        if blk1_status(i) and events[i]["data"]["block"]["status"] == "ok"
    ]
    assert len(reclosed) == 1

    blocks = {b["block_id"]: b for b in load_blocks(tmp_path / "blocks.json")["blocks"]}
    assert blocks["blk-0001"]["status"] == "ok"
    assert submitted[2]["action_id"] in blocks["blk-0001"]["action_ids"]
    assert submitted[2]["agent_name"] == "input_agent"


def test_web_queue_rejects_a_plain_string(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, _llm = _run_web(
        tmp_path, ["Waiting for instructions."], user_turns=["plain text"]
    )
    errors = [e["data"] for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "RUNNER_ERROR"
    assert "must be a UserTurn" in errors[0]["message"]


def test_web_briefing_reply_with_a_block_id_raises(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    _agents, driver = _web_agents()
    stop = threading.Event()
    events: list = []
    run_session_sync(
        session_id="sess",
        agent_system=_agents,
        driver_agent=driver,
        analysis_context="analysis",
        llm_client=PopLLM([]),
        sandbox_manager=FakeSandbox(),
        history=[{"role": "system", "content": "You are input_agent."}],
        is_auto=False,
        max_turns=5,
        model_name="fake",
        output_dir=tmp_path / "outputs",
        emit=events.append,
        stop_flag=stop,
        user_input_queue=ScriptedQueue(
            [UserTurn("brief me", block_id="blk-0001")], stop
        ),
        phase="briefing",
    )
    errors = [e["data"] for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert "briefing reply cannot target a block" in errors[0]["message"]


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------


class CliSequenceLlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        if not self.responses:
            raise AssertionError(f"unexpected provider call #{self.calls + 1}")
        self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.responses.pop(0)))
            ]
        )


def _cli_agents():
    input_agent = Agent(
        name="input_agent",
        prompt="Load the data.",
        commands={
            "delegate_to_QC_metrics": Command(
                "delegate_to_QC_metrics", "QC_metrics_agent", "Run QC"
            )
        },
        code_samples={},
    )
    qc = Agent(
        name="QC_metrics_agent",
        prompt="Compute QC.",
        commands={
            "delegate_to_input": Command("delegate_to_input", "input_agent", "Back")
        },
        code_samples={},
    )
    return (
        AgentSystem(
            global_policy="Be accurate",
            agents={"input_agent": input_agent, "QC_metrics_agent": qc},
        ),
        input_agent,
    )


def _run_cli(tmp_path, monkeypatch, responses, answers):
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
    )
    assert script == []
    return result, events, llm, buffer.getvalue()


def test_cli_code_runs_as_its_author_before_the_delegation(tmp_path, monkeypatch):
    result, events, llm, output = _run_cli(
        tmp_path, monkeypatch, [INPUT_MESSAGE, QC_CODE], ["exit"]
    )

    assert llm.calls == 2
    assert result.current_agent_name == "QC_metrics_agent"
    types = [e["event_type"] for e in events]
    assert types == [
        "turn_started",
        "assistant_message",
        "block_changed",
        "code_submitted",
        "code_result",
        "agent_switch",
        "turn_started",
        "assistant_message",
        "block_changed",  # input_agent's implicit block closes
        "block_changed",  # QC's new block opens
        "code_submitted",
        "code_result",
        "block_changed",  # session end closes QC's block
        "session_end",
    ]
    submitted = [e for e in events if e["event_type"] == "code_submitted"]
    results = [e for e in events if e["event_type"] == "code_result"]
    assert [e["agent_name"] for e in submitted] == ["input_agent", "QC_metrics_agent"]
    assert [e["agent_name"] for e in results] == ["input_agent", "QC_metrics_agent"]
    assert events[1]["agent_name"] == "input_agent"
    assert events[7]["agent_name"] == "QC_metrics_agent"
    assert (
        "Continuing automatically after delegating to QC_metrics_agent (1 of 4)."
        in output
    )

    blocks = {b["block_id"]: b for b in load_blocks(tmp_path / "blocks.json")["blocks"]}
    assert blocks[submitted[0]["payload"]["block_id"]]["agents"] == ["input_agent"]
    qc_block = blocks[submitted[1]["payload"]["block_id"]]
    assert qc_block["title"] == "QC metrics (from input_agent)"
    assert qc_block["agents"] == ["QC_metrics_agent"]


def test_cli_last_line_close_and_opens_do_not_consume_the_budget(
    tmp_path, monkeypatch
):
    opens = [f'open_work_item "Step {n}" "Do step {n}"' for n in range(4)]
    close_after_prose = (
        "Step 0 is finished.\n```python\nprint('done')\n```\n"
        'close_work_item 0 "finished"'
    )
    result, events, llm, output = _run_cli(
        tmp_path,
        monkeypatch,
        opens + [close_after_prose, "Nothing more to do."],
        ["exit"],
    )

    # Four opens and a close: five auto-continues, none counted, no pause.
    assert llm.calls == 6
    items = [
        e["payload"]["item"] for e in events if e["event_type"] == "work_item_changed"
    ]
    assert items[-1]["id"] == 0 and items[-1]["status"] == "Done"
    code_result = next(
        i for i, e in enumerate(events) if e["event_type"] == "code_result"
    )
    done = next(
        i
        for i, e in enumerate(events)
        if e["event_type"] == "work_item_changed"
        and e["payload"]["item"]["status"] == "Done"
    )
    assert code_result < done
    assert output.count("Continuing automatically after opening a work item.") == 4
    assert "Continuing automatically after closing a work item." in output
    assert "of 4)" not in output
    assert "Automatic continuation paused" not in output
    assert result.end_reason == "user_exit"


def test_cli_delegations_consume_the_budget_and_a_user_message_resets_it(
    tmp_path, monkeypatch
):
    result, events, llm, output = _run_cli(
        tmp_path,
        monkeypatch,
        [
            "delegate_to_QC_metrics",  # input_agent
            "delegate_to_input",  # QC_metrics_agent
            "delegate_to_QC_metrics",  # input_agent
            "delegate_to_input",  # QC_metrics_agent
            "delegate_to_QC_metrics",  # input_agent: 5th handoff, budget spent
            "delegate_to_input",  # QC_metrics_agent, after the user's message
            "Done for now.",  # input_agent
        ],
        ["keep going", "exit"],
    )

    assert llm.calls == 7
    for n, target in enumerate(
        ["QC_metrics_agent", "input_agent", "QC_metrics_agent", "input_agent"], start=1
    ):
        assert (
            f"Continuing automatically after delegating to {target} ({n} of 4)."
            in output
        )
    assert (
        "Automatic continuation paused after 4 handoffs. Waiting for your next message."
        in output
    )
    assert output.count("Automatic continuation paused") == 1
    assert "Continuing automatically after delegating to input_agent (1 of 4)." in output
    assert [e["payload"]["to_agent"] for e in events if e["event_type"] == "agent_switch"] == [
        "QC_metrics_agent",
        "input_agent",
        "QC_metrics_agent",
        "input_agent",
        "QC_metrics_agent",
        "input_agent",
    ]
    assert result.current_agent_name == "input_agent"
    assert result.end_reason == "user_exit"
