from __future__ import annotations

import io

from rich.console import Console

from caribou.execution import runner
from caribou.execution.session_brief import BriefPolicy

from .test_execution_runner_hooks import RecordingSandbox, SequenceLlm, _agents


class ScriptedPrompt:
    """Replaces `rich.prompt.Prompt.ask` with a scripted answer sequence."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, dict]] = []

    def ask(self, prompt_text: str = "", **kwargs) -> str:
        self.calls.append((prompt_text, kwargs))
        return self.answers.pop(0)


def _two_message_history() -> list[dict]:
    return [
        {"role": "system", "content": "**GLOBAL POLICY**: Be accurate"},
        {"role": "system", "content": "You are the driver."},
    ]


def _run_interactive(tmp_path, llm, prompt, *, brief_policy=None, **kwargs):
    agent_system, driver = _agents()
    return runner.run_agent_session(
        console=Console(file=io.StringIO(), force_terminal=False),
        agent_system=agent_system,
        driver_agent=driver,
        analysis_context="bounded test",
        llm_client=llm,
        sandbox_manager=RecordingSandbox(),
        history=kwargs.pop("history", _two_message_history()),
        is_auto=False,
        max_turns=1,
        output_dir=tmp_path,
        brief_policy=brief_policy,
        **kwargs,
    )


def test_briefing_phase_accepted_freezes_brief_and_pins_it_to_history(
    tmp_path, monkeypatch
) -> None:
    prompt = ScriptedPrompt(
        [
            "Please build a QC pipeline.",  # briefing: opening user turn
            "accept",  # briefing: accept the proposed brief
            "n",  # execution: "Continue anyway?" after end_session -> end the session
        ]
    )
    monkeypatch.setattr(runner, "Prompt", prompt)

    llm = SequenceLlm(
        [
            'Here is my proposal:\n\n```brief\n{"deliverable": "QC pipeline", '
            '"in_scope": ["QC"], "out_of_scope": [], '
            '"done_when": ["QC report produced"]}\n```',
            "end_session",
        ]
    )
    result = _run_interactive(
        tmp_path,
        llm,
        prompt,
        brief_policy=BriefPolicy(enabled=True, mode="context"),
    )

    assert result.end_reason == "agent_finished"
    brief_path = tmp_path / "brief.json"
    assert brief_path.exists()
    assert "QC pipeline" in brief_path.read_text()


def test_briefing_phase_never_runs_in_auto_mode(tmp_path, monkeypatch) -> None:
    agent_system, driver = _agents()
    prompt = ScriptedPrompt([])  # must never be consulted
    monkeypatch.setattr(runner, "Prompt", prompt)

    llm = SequenceLlm(["end_session"])
    result = runner.run_agent_session(
        console=Console(file=io.StringIO(), force_terminal=False),
        agent_system=agent_system,
        driver_agent=driver,
        analysis_context="bounded test",
        llm_client=llm,
        sandbox_manager=RecordingSandbox(),
        history=_two_message_history(),
        is_auto=True,
        max_turns=1,
        output_dir=tmp_path,
        brief_policy=BriefPolicy(enabled=True, mode="context"),
    )

    assert not (tmp_path / "brief.json").exists()
    assert result.end_reason == "agent_finished"
    assert prompt.calls == []


def test_briefing_phase_declined_ends_the_session_without_freezing(
    tmp_path, monkeypatch
) -> None:
    prompt = ScriptedPrompt(["exit"])  # human exits during briefing
    monkeypatch.setattr(runner, "Prompt", prompt)

    llm = SequenceLlm(["should not be called"])
    result = _run_interactive(
        tmp_path,
        llm,
        prompt,
        brief_policy=BriefPolicy(enabled=True, mode="context"),
    )

    assert result.end_reason == "briefing_declined"
    assert result.succeeded is False
    assert llm.calls == 0
    assert not (tmp_path / "brief.json").exists()


def test_briefing_phase_repairs_an_invalid_block_before_accepting(
    tmp_path, monkeypatch
) -> None:
    prompt = ScriptedPrompt(
        [
            "Please build a QC pipeline.",
            "",  # after the validation-failure feedback, let the agent retry
            "accept",
            "n",  # "Continue anyway?" after end_session
        ]
    )
    monkeypatch.setattr(runner, "Prompt", prompt)

    llm = SequenceLlm(
        [
            # Missing done_when -> validation failure -> repair loop.
            '```brief\n{"deliverable": "QC pipeline", "in_scope": ["QC"], '
            '"out_of_scope": [], "done_when": []}\n```',
            'Understood, revising:\n\n```brief\n{"deliverable": "QC pipeline", '
            '"in_scope": ["QC"], "out_of_scope": [], '
            '"done_when": ["QC report produced"]}\n```',
            "end_session",
        ]
    )
    result = _run_interactive(
        tmp_path,
        llm,
        prompt,
        brief_policy=BriefPolicy(enabled=True, mode="context"),
    )

    assert result.end_reason == "agent_finished"
    assert "QC report produced" in (tmp_path / "brief.json").read_text()
    assert llm.calls == 3  # invalid attempt, repaired attempt, then turn 1


def test_briefing_phase_seed_item_mode_opens_a_work_item(tmp_path, monkeypatch) -> None:
    prompt = ScriptedPrompt(
        [
            "Please build a QC pipeline.",
            "accept",
            # seed_item mode opens item #0 owned by the driver, so
            # end_session is refused (correctly) rather than prompting
            # "Continue anyway?" — exit via the ordinary chat prompt instead.
            "exit",
        ]
    )
    monkeypatch.setattr(runner, "Prompt", prompt)

    llm = SequenceLlm(
        [
            '```brief\n{"deliverable": "QC pipeline", "in_scope": ["QC"], '
            '"out_of_scope": [], "done_when": ["QC report produced"]}\n```',
            "end_session",
        ]
    )
    _run_interactive(
        tmp_path,
        llm,
        prompt,
        brief_policy=BriefPolicy(enabled=True, mode="seed_item"),
    )

    from caribou.execution.work_items import WorkItemPolicy, WorkItemStore

    store = WorkItemStore(
        tmp_path / "work-items", session_id="ignored", policy=WorkItemPolicy()
    )
    items = store.list()
    assert len(items) == 1
    assert items[0]["title"] == "QC pipeline"


def test_brief_repl_command_shows_the_frozen_brief(tmp_path, capsys) -> None:
    from caribou.execution.user_commands import USER_COMMANDS, UserCommandContext
    from caribou.execution.work_items import WorkItemPolicy, WorkItemStore

    store = WorkItemStore(
        tmp_path / "work-items", session_id="s", policy=WorkItemPolicy()
    )
    (tmp_path / "brief.json").write_text('{"deliverable": "Ship the parser"}')
    console = Console(force_terminal=False, width=120, record=True)
    ctx = UserCommandContext(
        console=console,
        run_id="s",
        turn=1,
        history=[],
        artifacts=None,
        agent_system=None,
        current_agent=None,
        llm_client=None,
        model_name="fake",
        memory_manager=None,
        report_memory=None,
        benchmark_modules=None,
        sandbox_manager=None,
        output_dir=tmp_path,
        evaluator_runtime=None,
        work_items=store,
    )
    USER_COMMANDS["/brief"].handler("", ctx)
    captured = console.export_text()
    assert "Ship the parser" in captured


def test_brief_repl_command_reports_no_brief_when_none_frozen(tmp_path) -> None:
    from caribou.execution.user_commands import USER_COMMANDS, UserCommandContext
    from caribou.execution.work_items import WorkItemPolicy, WorkItemStore

    store = WorkItemStore(
        tmp_path / "work-items", session_id="s", policy=WorkItemPolicy()
    )
    console = Console(force_terminal=False, width=120, record=True)
    ctx = UserCommandContext(
        console=console,
        run_id="s",
        turn=1,
        history=[],
        artifacts=None,
        agent_system=None,
        current_agent=None,
        llm_client=None,
        model_name="fake",
        memory_manager=None,
        report_memory=None,
        benchmark_modules=None,
        sandbox_manager=None,
        output_dir=tmp_path,
        evaluator_runtime=None,
        work_items=store,
    )
    USER_COMMANDS["/brief"].handler("", ctx)
    assert "No brief is frozen" in console.export_text()
