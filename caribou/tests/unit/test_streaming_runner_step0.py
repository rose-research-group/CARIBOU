"""Web loop: unified action_id, brief goal in ambient state, and handoff
report failures propagating (step-0 fixes)."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

from caribou.execution.event_ids import make_action_id
from caribou.execution.session_brief import BriefPolicy
from caribou.execution.user_input import UserTurn
from caribou.server.streaming_runner import run_session_sync

from .test_streaming_runner_interactive import FakeAgent, FakeAgentSystem, FakeSandbox


class RecordingLLM:
    """Streams one scripted response per turn and records every request.

    Non-streaming calls (the handoff report) return `report` or raise it.
    """

    def __init__(self, responses, *, report="handoff summary"):
        self.responses = list(responses)
        self.report = report
        self.stream_requests: list[list[dict]] = []
        self.report_requests: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if not kwargs.get("stream"):
            self.report_requests.append(kwargs["messages"])
            if isinstance(self.report, BaseException):
                raise self.report
            message = SimpleNamespace(content=self.report)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        self.stream_requests.append(kwargs["messages"])
        index = min(len(self.stream_requests) - 1, len(self.responses) - 1)
        text = self.responses[index]
        return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])]


class FakeReportMemory:
    def __init__(self):
        self.reports: list[tuple[str, str]] = []

    def build_context(self, working_history):
        return [dict(item) for item in working_history]

    def add_report(self, agent_name, report):
        self.reports.append((agent_name, report))

    def update_agent_prompt(self, _prompt):
        return None


def _stub_rag(monkeypatch):
    rag_stub = ModuleType("caribou.execution.rag_client")
    rag_stub.get_rag_client = lambda _console: None
    monkeypatch.setitem(__import__("sys").modules, "caribou.execution.rag_client", rag_stub)


def _run_auto(tmp_path: Path, llm, *, agents=None, driver=None, **kwargs):
    if agents is None:
        driver = FakeAgent("driver")
        agents = FakeAgentSystem({"driver": driver})
    events: list = []
    run_session_sync(
        session_id=kwargs.pop("session_id", "sess"),
        agent_system=agents,
        driver_agent=driver,
        analysis_context="analysis",
        llm_client=llm,
        sandbox_manager=kwargs.pop("sandbox_manager", FakeSandbox()),
        history=[
            {"role": "system", "content": "**GLOBAL POLICY**: Be accurate"},
            {"role": "system", "content": "You are the driver."},
        ],
        is_auto=True,
        max_turns=kwargs.pop("max_turns", 1),
        model_name="fake",
        output_dir=tmp_path / "outputs",
        emit=events.append,
        stop_flag=threading.Event(),
        **kwargs,
    )
    return events


def _state_blocks(messages: list[dict]) -> list[str]:
    return [
        m["content"]
        for m in messages
        if m["role"] == "system" and "WORK ITEMS" in m["content"]
    ]


def test_code_events_use_the_shared_action_id_format(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    llm = RecordingLLM(["```python\nx = 1\n```\n```python\ny = 2\n```"])

    events = _run_auto(tmp_path, llm, session_id="sess-9")

    submitted = [ev["data"]["action_id"] for ev in events if ev["type"] == "code_submitted"]
    results = [ev["data"]["action_id"] for ev in events if ev["type"] == "code_result"]
    expected = [make_action_id("sess-9", 1, 1), make_action_id("sess-9", 1, 2)]
    assert submitted == expected
    assert results == expected


def test_pre_frozen_brief_goal_is_in_every_turns_state_block(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    llm = RecordingLLM(["thinking", "still thinking"])

    _run_auto(
        tmp_path,
        llm,
        max_turns=2,
        brief={"deliverable": "Annotated UMAP of all clusters"},
    )

    assert len(llm.stream_requests) == 2
    for messages in llm.stream_requests:
        (block,) = _state_blocks(messages)
        assert block.startswith("BRIEF GOAL: Annotated UMAP of all clusters\n")


def test_no_brief_means_no_goal_line(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    llm = RecordingLLM(["thinking"])

    _run_auto(tmp_path, llm)

    (block,) = _state_blocks(llm.stream_requests[0])
    assert "BRIEF GOAL" not in block


def test_brief_accepted_in_the_interview_sets_the_goal(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    driver = FakeAgent("driver")
    agents = FakeAgentSystem({"driver": driver})
    user_input_queue: queue.Queue = queue.Queue()
    brief_decision_queue: queue.Queue = queue.Queue()
    user_input_queue.put(UserTurn("Please build a QC pipeline."))
    brief_decision_queue.put({"decision": "accept"})
    llm = RecordingLLM(
        [
            '```brief\n{"deliverable": "QC pipeline", "in_scope": ["QC"], '
            '"out_of_scope": [], "done_when": ["QC report produced"]}\n```',
            "end_session",
        ]
    )
    stop_flag = threading.Event()

    run_session_sync(
        session_id="s",
        agent_system=agents,
        driver_agent=driver,
        analysis_context="analysis",
        llm_client=llm,
        sandbox_manager=FakeSandbox(),
        history=[
            {"role": "system", "content": "**GLOBAL POLICY**: Be accurate"},
            {"role": "system", "content": "You are the driver."},
        ],
        is_auto=False,
        max_turns=5,
        model_name="fake",
        output_dir=tmp_path / "outputs",
        emit=lambda _ev: None,
        stop_flag=stop_flag,
        user_input_queue=user_input_queue,
        brief_decision_queue=brief_decision_queue,
        phase="briefing",
        brief_policy=BriefPolicy(enabled=True, mode="context"),
    )
    stop_flag.set()

    # Request 0 is the briefing interview; request 1 is execution turn 1.
    (block,) = _state_blocks(llm.stream_requests[1])
    assert block.startswith("BRIEF GOAL: QC pipeline\n")


def _delegating_agents():
    coder = FakeAgent("coder")
    planner = FakeAgent("planner", {"delegate_to_coder": SimpleNamespace(target_agent="coder")})
    return FakeAgentSystem({"planner": planner, "coder": coder}), planner


def test_handoff_report_is_recorded_on_delegation(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    agents, planner = _delegating_agents()
    llm = RecordingLLM(["delegate_to_coder"])
    report_memory = FakeReportMemory()

    events = _run_auto(
        tmp_path, llm, agents=agents, driver=planner, report_memory=report_memory
    )

    assert report_memory.reports == [("planner", "handoff summary")]
    assert not [ev for ev in events if ev["type"] == "error"]


def test_failed_handoff_report_stops_the_session(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    agents, planner = _delegating_agents()
    llm = RecordingLLM(["delegate_to_coder"], report=RuntimeError("report provider down"))
    report_memory = FakeReportMemory()

    events = _run_auto(
        tmp_path, llm, agents=agents, driver=planner, report_memory=report_memory, max_turns=3
    )

    errors = [ev for ev in events if ev["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"]["code"] == "RUNNER_ERROR"
    assert errors[0]["data"]["fatal"] is True
    assert "report provider down" in errors[0]["data"]["message"]
    assert events[-1]["type"] == "status_change"
    assert events[-1]["data"]["status"] == "error"
    assert report_memory.reports == []
    # The session did not go on to a second provider turn.
    assert len(llm.stream_requests) == 1


def test_empty_handoff_report_stops_the_session(tmp_path, monkeypatch):
    _stub_rag(monkeypatch)
    agents, planner = _delegating_agents()
    llm = RecordingLLM(["delegate_to_coder"], report=None)
    report_memory = FakeReportMemory()

    events = _run_auto(
        tmp_path, llm, agents=agents, driver=planner, report_memory=report_memory
    )

    errors = [ev for ev in events if ev["type"] == "error"]
    assert len(errors) == 1
    assert "returned no content" in errors[0]["data"]["message"]
    assert report_memory.reports == []
