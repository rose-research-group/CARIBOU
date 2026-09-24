import queue
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from caribou.server.streaming_runner import (
    _stream_tokens,
    run_session_sync,
)


class FakeAgent:
    def __init__(self, name, commands=None):
        self.name = name
        self.commands = commands or {}
        self.is_rag_enabled = False

    def get_full_prompt(self, _):
        return f"You are {self.name}."


class FakeAgentSystem:
    def __init__(self, agents):
        self.agents = agents

    def get_agent(self, name):
        return self.agents.get(name)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        self.calls += 1
        text = self.responses.pop(0)

        class Chunk:
            def __init__(self, token):
                self.choices = [SimpleNamespace(delta=SimpleNamespace(content=token))]

        return [Chunk(text)]


class FakeSandbox:
    def exec_code(self, _code, timeout):
        return {"status": "ok", "stdout": "", "stderr": ""}


class FakeNonStreamingLLM:
    def __init__(self, text):
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        message = SimpleNamespace(content=self.text, role="assistant")
        choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
        return SimpleNamespace(choices=[choice])


class FakeFailingStreamingLLM:
    def __init__(self, *, fail_after_token=False):
        self.fail_after_token = fail_after_token
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)

        if kwargs.get("stream"):

            def chunks():
                if self.fail_after_token:
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(delta=SimpleNamespace(content="partial"))
                        ]
                    )
                raise RuntimeError("incomplete chunked read")

            return chunks()

        message = SimpleNamespace(content="fallback response", role="assistant")
        choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
        return SimpleNamespace(choices=[choice])


def test_stream_tokens_accepts_non_streaming_openai_compatible_response():
    llm = FakeNonStreamingLLM("loaded")

    assert list(
        _stream_tokens(llm, "llama3", [{"role": "user", "content": "load"}])
    ) == ["loaded"]


def test_stream_failure_does_not_issue_a_second_provider_request():
    llm = FakeFailingStreamingLLM()

    with pytest.raises(RuntimeError, match="incomplete chunked read"):
        list(_stream_tokens(llm, "deepseek-chat", [{"role": "user", "content": "hi"}]))

    assert len(llm.calls) == 1
    assert llm.calls[0]["stream"] is True


def test_partial_stream_failure_does_not_replay_the_prompt():
    llm = FakeFailingStreamingLLM(fail_after_token=True)

    stream = _stream_tokens(llm, "deepseek-chat", [{"role": "user", "content": "hi"}])

    assert next(stream) == "partial"
    with pytest.raises(RuntimeError, match="incomplete chunked read"):
        next(stream)
    assert len(llm.calls) == 1


def test_interactive_delegation_waits_for_user_before_next_agent_turn(
    tmp_path: Path, monkeypatch
):
    rag_stub = ModuleType("caribou.execution.rag_client")
    rag_stub.get_rag_client = lambda _console: None
    monkeypatch.setitem(
        __import__("sys").modules, "caribou.execution.rag_client", rag_stub
    )

    coder = FakeAgent("coder")
    planner = FakeAgent(
        "planner",
        {"delegate_to_coder": SimpleNamespace(target_agent="coder")},
    )
    agent_system = FakeAgentSystem({"planner": planner, "coder": coder})
    llm = FakeLLM(["delegate_to_coder", "coder should not run yet"])
    stop_flag = threading.Event()
    user_input_queue = queue.Queue()
    events = []
    idle_seen = threading.Event()

    def emit(event):
        events.append(event)
        if event["type"] == "status_change" and event["data"].get("status") == "idle":
            idle_seen.set()

    thread = threading.Thread(
        target=run_session_sync,
        kwargs={
            "session_id": "session-1",
            "agent_system": agent_system,
            "driver_agent": planner,
            "analysis_context": "analysis",
            "llm_client": llm,
            "sandbox_manager": FakeSandbox(),
            "history": [{"role": "user", "content": "start"}],
            "is_auto": False,
            "max_turns": 10,
            "model_name": "fake",
            "output_dir": tmp_path / "outputs",
            "emit": emit,
            "stop_flag": stop_flag,
            "user_input_queue": user_input_queue,
        },
    )
    thread.start()

    assert idle_seen.wait(timeout=2)
    assert llm.calls == 1
    assert [
        event["data"]["message"]["content"]
        for event in events
        if event["type"] == "message_complete"
    ] == ["delegate_to_coder"]
    assert any(event["type"] == "agent_switch" for event in events)

    stop_flag.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_interactive_work_item_transfers_and_blocks_end_session(
    tmp_path: Path, monkeypatch
):
    rag_stub = ModuleType("caribou.execution.rag_client")
    rag_stub.get_rag_client = lambda _console: None
    monkeypatch.setitem(
        __import__("sys").modules, "caribou.execution.rag_client", rag_stub
    )

    coder = FakeAgent("coder")
    planner = FakeAgent(
        "planner",
        {"delegate_to_coder": SimpleNamespace(target_agent="coder")},
    )
    agent_system = FakeAgentSystem({"planner": planner, "coder": coder})
    llm = FakeLLM(
        [
            'open_work_item "Implement" "Build and test it"',
            "delegate_to_coder",
            "end_session",
        ]
    )
    stop_flag = threading.Event()
    user_input_queue = queue.Queue()
    user_input_queue.put("continue")
    events = []
    second_idle = threading.Event()
    idle_count = 0

    def emit(event):
        nonlocal idle_count
        events.append(event)
        if event["type"] == "status_change" and event["data"].get("status") == "idle":
            idle_count += 1
            if idle_count == 2:
                second_idle.set()

    thread = threading.Thread(
        target=run_session_sync,
        kwargs={
            "session_id": "session-work-items",
            "agent_system": agent_system,
            "driver_agent": planner,
            "analysis_context": "analysis",
            "llm_client": llm,
            "sandbox_manager": FakeSandbox(),
            "history": [{"role": "user", "content": "start"}],
            "is_auto": False,
            "max_turns": 10,
            "model_name": "fake",
            "output_dir": tmp_path / "outputs",
            "emit": emit,
            "stop_flag": stop_flag,
            "user_input_queue": user_input_queue,
        },
    )
    thread.start()

    assert second_idle.wait(timeout=5)
    changes = [
        event["data"]["item"]
        for event in events
        if event["type"] == "work_item_changed"
    ]
    assert changes[0]["id"] == 0
    assert changes[-1]["owner"] == "coder"
    assert any(
        event["type"] == "system_message"
        and "end_session refused" in event["data"].get("content", "")
        for event in events
    )
    # Every system_message must carry a distinct, non-empty id. The
    # frontend dedupes chat messages by id (upsertMessage); with none set,
    # every system message after the first collided on an identical
    # `undefined` id and was silently dropped, hiding work-item feedback
    # (and everything else routed through system_message) from the user.
    system_message_ids = [
        event["data"].get("id")
        for event in events
        if event["type"] == "system_message"
    ]
    assert len(system_message_ids) >= 2
    assert all(system_message_ids)
    assert len(set(system_message_ids)) == len(system_message_ids)
    assert not any(
        event["type"] == "status_change"
        and event["data"].get("reason") == "agent_requested_end"
        for event in events
    )

    stop_flag.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_open_work_item_auto_continues_once_then_waits(
    tmp_path: Path, monkeypatch
):
    rag_stub = ModuleType("caribou.execution.rag_client")
    rag_stub.get_rag_client = lambda _console: None
    monkeypatch.setitem(
        __import__("sys").modules, "caribou.execution.rag_client", rag_stub
    )

    planner = FakeAgent("planner")
    llm = FakeLLM(
        [
            'open_work_item "First" "Do the first thing"',
            'open_work_item "Second" "Do the second thing"',
        ]
    )
    stop_flag = threading.Event()
    events = []
    idle_seen = threading.Event()

    def emit(event):
        events.append(event)
        if event["type"] == "status_change" and event["data"].get("status") == "idle":
            idle_seen.set()

    thread = threading.Thread(
        target=run_session_sync,
        kwargs={
            "session_id": "session-auto-continue",
            "agent_system": FakeAgentSystem({"planner": planner}),
            "driver_agent": planner,
            "analysis_context": "analysis",
            "llm_client": llm,
            "sandbox_manager": FakeSandbox(),
            "history": [{"role": "user", "content": "start"}],
            "is_auto": False,
            "max_turns": 10,
            "model_name": "fake",
            "output_dir": tmp_path / "outputs",
            "emit": emit,
            "stop_flag": stop_flag,
            "user_input_queue": queue.Queue(),
        },
    )
    thread.start()

    assert idle_seen.wait(timeout=5)
    # The first open auto-continued into a second provider turn, but that second
    # open was bounded by the one-per-user-message budget and therefore waited.
    assert llm.calls == 2
    opened_items = [
        event["data"]["item"]
        for event in events
        if event["type"] == "work_item_changed"
    ]
    assert [item["id"] for item in opened_items] == [0, 1]
    assert any(
        event["type"] == "system_message"
        and "Continuing automatically after opening a work item."
        in event["data"].get("content", "")
        for event in events
    )

    stop_flag.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_turn_checkpoint_contains_complete_failed_action_ledger(
    tmp_path: Path, monkeypatch
):
    rag_stub = ModuleType("caribou.execution.rag_client")
    rag_stub.get_rag_client = lambda _console: None
    monkeypatch.setitem(
        __import__("sys").modules, "caribou.execution.rag_client", rag_stub
    )

    class FailingSandbox:
        def exec_code(self, _code, timeout):
            return {"status": "error", "stdout": "", "stderr": "expected failure"}

    analyst = FakeAgent("analyst")
    checkpoints = []
    run_session_sync(
        session_id="session-ledger",
        agent_system=FakeAgentSystem({"analyst": analyst}),
        driver_agent=analyst,
        analysis_context="analysis",
        llm_client=FakeLLM(["```python\nraise ValueError('boom')\n```"]),
        sandbox_manager=FailingSandbox(),
        history=[{"role": "user", "content": "start"}],
        is_auto=True,
        max_turns=1,
        model_name="fake",
        output_dir=tmp_path / "outputs",
        emit=lambda _event: None,
        stop_flag=threading.Event(),
        checkpoint_callback=lambda history, state: checkpoints.append((history, state)),
    )

    ledger = checkpoints[-1][1]["action_ledger"]
    assert ledger[0]["source"] == "raise ValueError('boom')"
    assert ledger[0]["recorded_result"]["success"] is False
    assert ledger[0]["recorded_result"]["stderr"] == "expected failure"
