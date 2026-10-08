from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import SimpleNamespace

from caribou.execution.session_brief import BriefPolicy
from caribou.server.streaming_runner import run_session_sync

from .test_streaming_runner_interactive import FakeAgent, FakeAgentSystem, FakeSandbox


class FakeLLM:
    """Each call returns one whole response as a single streamed chunk."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        self.calls += 1
        text = self.responses[min(self.calls - 1, len(self.responses) - 1)]

        class Chunk:
            def __init__(self, token):
                self.choices = [SimpleNamespace(delta=SimpleNamespace(content=token))]

        return [Chunk(text)]


def _two_message_history() -> list[dict]:
    return [
        {"role": "system", "content": "**GLOBAL POLICY**: Be accurate"},
        {"role": "system", "content": "You are the driver."},
    ]


def _agents():
    driver = FakeAgent("driver")
    return FakeAgentSystem({"driver": driver}), driver


def _run(
    tmp_path: Path,
    llm,
    *,
    user_input_queue,
    brief_decision_queue,
    events: list,
):
    agent_system, driver = _agents()
    stop_flag = threading.Event()
    run_session_sync(
        session_id="s",
        agent_system=agent_system,
        driver_agent=driver,
        analysis_context="analysis",
        llm_client=llm,
        sandbox_manager=FakeSandbox(),
        history=_two_message_history(),
        is_auto=False,
        max_turns=5,
        model_name="fake",
        output_dir=tmp_path / "outputs",
        emit=events.append,
        stop_flag=stop_flag,
        user_input_queue=user_input_queue,
        brief_decision_queue=brief_decision_queue,
        phase="briefing",
        brief_policy=BriefPolicy(enabled=True, mode="context"),
    )
    return stop_flag


def test_web_briefing_accepted_freezes_and_transitions_to_execution(tmp_path) -> None:
    user_input_queue: queue.Queue = queue.Queue()
    brief_decision_queue: queue.Queue = queue.Queue()
    events: list = []

    user_input_queue.put("Please build a QC pipeline.")
    brief_decision_queue.put({"decision": "accept"})

    llm = FakeLLM(
        [
            '```brief\n{"deliverable": "QC pipeline", "in_scope": ["QC"], '
            '"out_of_scope": [], "done_when": ["QC report produced"]}\n```',
            "end_session",
        ]
    )
    stop_flag = _run(
        tmp_path,
        llm,
        user_input_queue=user_input_queue,
        brief_decision_queue=brief_decision_queue,
        events=events,
    )
    stop_flag.set()

    assert (tmp_path / "outputs" / ".." / "brief.json").resolve().exists()
    event_types = [ev["type"] for ev in events]
    assert "brief_draft" in event_types
    assert "brief_accepted" in event_types
    assert "phase_change" in event_types
    phase_events = [ev for ev in events if ev["type"] == "phase_change"]
    assert phase_events[0]["data"]["phase"] == "execution"


def test_web_briefing_declined_stops_the_session(tmp_path) -> None:
    user_input_queue: queue.Queue = queue.Queue()
    brief_decision_queue: queue.Queue = queue.Queue()
    events: list = []

    user_input_queue.put("Never mind.")

    llm = FakeLLM(["end_session"])
    _run(
        tmp_path,
        llm,
        user_input_queue=user_input_queue,
        brief_decision_queue=brief_decision_queue,
        events=events,
    )

    assert not (tmp_path / "brief.json").exists()
    stopped = [
        ev
        for ev in events
        if ev["type"] == "status_change" and ev["data"].get("reason") == "briefing_declined"
    ]
    assert stopped


def test_web_briefing_repairs_an_invalid_block_before_accepting(tmp_path) -> None:
    user_input_queue: queue.Queue = queue.Queue()
    brief_decision_queue: queue.Queue = queue.Queue()
    events: list = []

    user_input_queue.put("Please build a QC pipeline.")
    user_input_queue.put("")  # let the agent retry after the repair feedback
    brief_decision_queue.put({"decision": "accept"})

    llm = FakeLLM(
        [
            '```brief\n{"deliverable": "QC pipeline", "in_scope": ["QC"], '
            '"out_of_scope": [], "done_when": []}\n```',
            'Fixed:\n\n```brief\n{"deliverable": "QC pipeline", "in_scope": ["QC"], '
            '"out_of_scope": [], "done_when": ["QC report produced"]}\n```',
            "end_session",
        ]
    )
    _run(
        tmp_path,
        llm,
        user_input_queue=user_input_queue,
        brief_decision_queue=brief_decision_queue,
        events=events,
    )

    assert "QC report produced" in (tmp_path / "brief.json").read_text()
    assert llm.calls == 3


def test_web_briefing_reject_sends_feedback_and_continues(tmp_path) -> None:
    user_input_queue: queue.Queue = queue.Queue()
    brief_decision_queue: queue.Queue = queue.Queue()
    events: list = []

    user_input_queue.put("Please build a QC pipeline.")
    brief_decision_queue.put({"decision": "reject", "reason": "Too broad."})
    user_input_queue.put("")
    brief_decision_queue.put({"decision": "accept"})

    llm = FakeLLM(
        [
            '```brief\n{"deliverable": "QC pipeline", "in_scope": ["QC", "everything"], '
            '"out_of_scope": [], "done_when": ["done"]}\n```',
            '```brief\n{"deliverable": "QC pipeline", "in_scope": ["QC"], '
            '"out_of_scope": [], "done_when": ["QC report produced"]}\n```',
            "end_session",
        ]
    )
    _run(
        tmp_path,
        llm,
        user_input_queue=user_input_queue,
        brief_decision_queue=brief_decision_queue,
        events=events,
    )

    assert "QC report produced" in (tmp_path / "brief.json").read_text()
    feedback = [
        ev
        for ev in events
        if ev["type"] == "system_message" and "Too broad" in ev["data"].get("content", "")
    ]
    assert feedback
