"""Reruns go to the owner (contract B.1, web loop).

Modelled on session ef4aea40: the user's "Request changes" on a block of a
work item owned by QC_metrics_agent was delivered to the agent that
happened to be current (doublet_agent), which could only refuse it. Now a
focused user turn on another agent's work item switches to that owner
before the LLM is called, with the same mechanics as a delegation switch,
and the owner's code lands in a new attempt of the item.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import SimpleNamespace

from caribou.execution.blocks import load_blocks
from caribou.execution.user_input import UserTurn
from caribou.execution.work_items import WorkItemPolicy, WorkItemStore
from caribou.server.streaming_runner import run_session_sync

from .test_loop_feedback_fixes import PopLLM, _guidance, _index, _no_runner_error
from .test_streaming_runner_interactive import FakeAgent, FakeAgentSystem, FakeSandbox
from .test_streaming_runner_step0 import _stub_rag

QC_CODE = "```python\nprint('qc')\n```"
FIX_CODE = "```python\nprint('fix')\n```"
SENT_BACK = "[Workbench · block blk-0001 \"QC filtering\" · work item #0] Work item #0 sent back: redo"

# input_agent delegates to QC; QC opens an item, then in one message runs
# code (blk-0001, item 0 attempt 1), closes the item (blk-0001 closes) and
# hands back, the command directly before the delegation line; input_agent
# then waits for the user. Every message but the last auto-continues.
QC_FINISH = QC_CODE + '\nclose_work_item 0 "filtered"\ndelegate_to_input'
QC_ROUND_TRIP = [
    "delegate_to_QC_metrics",  # input_agent
    'open_work_item "QC filtering" "filter low quality cells"',  # QC
    QC_FINISH,  # QC
    "Waiting for review.",  # input_agent
]


class RecordingPopLLM(PopLLM):
    """PopLLM that also keeps every request's message list."""

    def __init__(self, responses):
        super().__init__(responses)
        self.requests: list[list[dict]] = []

    def _create(self, **kwargs):
        self.requests.append([dict(m) for m in kwargs["messages"]])
        return super()._create(**kwargs)


class ScriptedQueue(queue.Queue):
    """Each get runs a side effect on the store (e.g. the REST human
    reject) and hands out a UserTurn; an empty script stops the session."""

    def __init__(self, script, stop: threading.Event, store: WorkItemStore):
        super().__init__()
        self.script = list(script)
        self.stop = stop
        self.store = store

    def get(self, block=True, timeout=None):
        if not self.script:
            self.stop.set()
            raise queue.Empty
        side_effect, turn = self.script.pop(0)
        if side_effect is not None:
            side_effect(self.store)
        return turn


def _agents(*names):
    qc = FakeAgent(
        "QC_metrics_agent",
        {"delegate_to_input": SimpleNamespace(target_agent="input_agent")},
    )
    input_agent = FakeAgent(
        "input_agent",
        {"delegate_to_QC_metrics": SimpleNamespace(target_agent="QC_metrics_agent")},
    )
    agents = {"input_agent": input_agent, "QC_metrics_agent": qc}
    for name in names:
        agents[name] = FakeAgent(name)
    return FakeAgentSystem(agents), input_agent


def _run(tmp_path: Path, responses, script, *, qc_mode="optional", extra_agents=()):
    store = WorkItemStore(
        tmp_path / "work-items",
        session_id="sess",
        policy=WorkItemPolicy(qc_mode=qc_mode),
    )
    agent_system, driver = _agents(*extra_agents)
    stop = threading.Event()
    events: list = []
    llm = RecordingPopLLM(responses)
    run_session_sync(
        session_id="sess",
        agent_system=agent_system,
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
        work_item_store=store,
        user_input_queue=ScriptedQueue(script, stop, store),
    )
    return events, llm, store


def _switches(events):
    return [e["data"] for e in events if e["type"] == "agent_switch"]


def _submitted(events):
    return [e["data"] for e in events if e["type"] == "code_submitted"]


def test_a_focused_turn_on_another_agents_item_returns_it_to_the_owner(
    tmp_path, monkeypatch
):
    _stub_rag(monkeypatch)
    events, llm, store = _run(
        tmp_path,
        QC_ROUND_TRIP + [FIX_CODE],  # by QC_metrics_agent, then waits
        [
            (
                lambda store: store.record_review(
                    0, turn=6, verdict="reject", assessment="redo"
                ),
                UserTurn(SENT_BACK, block_id="blk-0001"),
            )
        ],
    )
    _no_runner_error(events)
    assert llm.calls == 5

    switches = _switches(events)
    assert len(switches) == 3
    returned = switches[2]
    assert returned == {
        "from_agent": "input_agent",
        "to_agent": "QC_metrics_agent",
        "command": "returned_for_changes",
        "reason": "work item #0 sent back",
    }
    assert _guidance(events)[-1] == "Returned to QC_metrics_agent for work item #0."

    # Order: the user's message, the switch and its prompt messages, the
    # guidance, then the owner's turn starts and its code runs.
    user_msg = _index(
        events,
        lambda e: e["type"] == "message_complete"
        and e["data"]["message"]["content"] == SENT_BACK,
    )
    switch = _index(events, lambda e: e["data"] is returned)
    prompt = _index(
        events,
        lambda e: e["type"] == "system_message"
        and e["data"].get("category") == "Agent prompt"
        and e["data"]["content"].startswith("You are QC_metrics_agent.")
        and e["turn"] == events[switch]["turn"],
    )
    guidance = _index(
        events,
        lambda e: e["type"] == "system_message"
        and e["data"]["content"] == "Returned to QC_metrics_agent for work item #0.",
    )
    running = _index(
        events,
        lambda e: e["type"] == "status_change"
        and e["data"]["status"] == "running"
        and e["data"]["reason"] == "turn 5 — QC_metrics_agent",
    )
    fix = _index(
        events, lambda e: e["type"] == "code_submitted" and "fix" in e["data"]["source"]
    )
    assert user_msg < switch < prompt < guidance < running < fix
    assert events[switch]["turn"] == events[user_msg]["turn"] == 5
    # No handoff report and no work-item transfer: the owner keeps the item.
    assert llm.calls == 5
    item = store.read(0)
    assert item["owner"] == "QC_metrics_agent"
    # (A human reject under optional QC is recorded as a reopen.)
    assert [t["kind"] for t in item["transitions"]] == ["opened", "closed", "reopened"]

    # The owner's code lands in a new attempt of the item, as the owner.
    submitted = _submitted(events)
    assert [(s["agent_name"], s["block_id"]) for s in submitted] == [
        ("QC_metrics_agent", "blk-0001"),
        ("QC_metrics_agent", "blk-0002"),
    ]
    blocks = {b["block_id"]: b for b in load_blocks(tmp_path / "blocks.json")["blocks"]}
    assert (blocks["blk-0001"]["attempt"], blocks["blk-0001"]["status"]) == (1, "ok")
    assert blocks["blk-0001"]["action_ids"] == [submitted[0]["action_id"]]
    assert blocks["blk-0002"]["work_item_id"] == 0
    assert blocks["blk-0002"]["attempt"] == 2
    assert blocks["blk-0002"]["agents"] == ["QC_metrics_agent"]
    assert blocks["blk-0002"]["action_ids"] == [submitted[1]["action_id"]]
    assert blocks["blk-0002"]["title"] == "QC filtering"


def test_the_owner_sees_the_return_note_and_its_prompt(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, llm, store = _run(
        tmp_path,
        QC_ROUND_TRIP + ["Fixed."],
        [
            (
                lambda store: store.record_review(
                    0, turn=6, verdict="reject", assessment="redo"
                ),
                UserTurn(SENT_BACK, block_id="blk-0001"),
            )
        ],
    )
    _no_runner_error(events)
    request = llm.requests[-1]
    assert request[1]["role"] == "system"
    assert request[1]["content"].startswith("You are QC_metrics_agent.")
    contents = [m["content"] for m in request]
    assert SENT_BACK in contents
    assert "Returned to QC_metrics_agent for work item #0." in contents
    assert contents.index(SENT_BACK) < contents.index(
        "Returned to QC_metrics_agent for work item #0."
    )


def test_a_focused_turn_on_the_current_agents_own_item_does_not_switch(
    tmp_path, monkeypatch
):
    _stub_rag(monkeypatch)
    # QC never hands back, so it is still current when the user replies.
    events, llm, store = _run(
        tmp_path,
        QC_ROUND_TRIP[:2]
        + [QC_CODE + '\nclose_work_item 0 "filtered"', "Closed; waiting.", FIX_CODE],
        [
            (
                lambda store: store.record_review(
                    0, turn=5, verdict="reject", assessment="redo"
                ),
                UserTurn(SENT_BACK, block_id="blk-0001"),
            )
        ],
    )
    _no_runner_error(events)
    assert llm.calls == 5
    assert [s["to_agent"] for s in _switches(events)] == ["QC_metrics_agent"]
    assert not any("Returned to" in g for g in _guidance(events))
    assert [(s["agent_name"], s["block_id"]) for s in _submitted(events)] == [
        ("QC_metrics_agent", "blk-0001"),
        ("QC_metrics_agent", "blk-0002"),
    ]


def test_a_focused_turn_on_an_implicit_block_does_not_switch(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, llm, store = _run(
        tmp_path,
        [
            "delegate_to_QC_metrics",
            QC_CODE + "\ndelegate_to_input",  # QC: implicit blk-0001, then back
            "Waiting.",
            FIX_CODE,  # input_agent, into the focused implicit block
        ],
        [(None, UserTurn("[block blk-0001] tweak this", block_id="blk-0001"))],
    )
    _no_runner_error(events)
    assert llm.calls == 4
    assert [s["to_agent"] for s in _switches(events)] == [
        "QC_metrics_agent",
        "input_agent",
    ]
    assert [(s["agent_name"], s["block_id"]) for s in _submitted(events)] == [
        ("QC_metrics_agent", "blk-0001"),
        ("input_agent", "blk-0001"),
    ]


def test_an_unfocused_turn_never_switches(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    events, llm, store = _run(
        tmp_path,
        QC_ROUND_TRIP + ["Noted."],
        [
            (
                lambda store: store.record_review(
                    0, turn=6, verdict="reject", assessment="redo"
                ),
                UserTurn("please redo QC"),
            )
        ],
    )
    _no_runner_error(events)
    assert [s["to_agent"] for s in _switches(events)] == [
        "QC_metrics_agent",
        "input_agent",
    ]
    assert events[-1]["type"] == "status_change"


def test_an_owner_that_is_not_an_agent_fails_loudly(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)

    def reject_and_hand_to_a_ghost(store):
        store.record_review(0, turn=4, verdict="reject", assessment="redo")
        store.transfer(0, "QC_metrics_agent", "ghost_agent", 4)

    events, llm, store = _run(
        tmp_path,
        QC_ROUND_TRIP + ["never reached"],
        [(reject_and_hand_to_a_ghost, UserTurn(SENT_BACK, block_id="blk-0001"))],
    )
    assert llm.calls == 4
    errors = [e["data"] for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["fatal"] is True
    assert "owned by 'ghost_agent', which is not an agent in this system" in errors[0][
        "message"
    ]
    assert events[-1]["data"] == {"status": "error", "reason": errors[0]["message"]}


def test_required_qc_evaluator_reject_then_sent_back_goes_to_the_owner(
    tmp_path, monkeypatch
):
    """The evaluator's reject puts the item back In progress with its owner;
    the user's "send back" turn on the rejected block then reaches that
    owner, whose fix opens attempt 2."""
    _stub_rag(monkeypatch)
    events, llm, store = _run(
        tmp_path,
        QC_ROUND_TRIP + [FIX_CODE + '\nclose_work_item 0 "fixed"', "Resubmitted."],
        [
            (
                lambda store: store.record_review(
                    0,
                    evaluator="evaluator_agent",
                    turn=6,
                    verdict="reject",
                    assessment="no evidence",
                ),
                UserTurn(SENT_BACK, block_id="blk-0001"),
            )
        ],
        qc_mode="required",
        extra_agents=("evaluator_agent",),
    )
    _no_runner_error(events)
    assert llm.calls == 6
    assert _switches(events)[-1]["command"] == "returned_for_changes"
    blocks = load_blocks(tmp_path / "blocks.json")["blocks"]
    assert [(b["work_item_id"], b["attempt"], b["status"]) for b in blocks] == [
        (0, 1, "ok"),
        (0, 2, "ok"),
    ]
    assert store.read(0)["status"] == "In review"
