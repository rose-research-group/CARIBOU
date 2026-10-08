from __future__ import annotations

import copy
import io
import time
from dataclasses import is_dataclass
from pathlib import Path
from typing import Callable

import pytest
from rich.console import Console

from caribou.execution import runner

from .test_execution_runner_hooks import RecordingSandbox, SequenceLlm, _agents


AUTO_CONTINUE_MESSAGE = "Please continue with the next step."


class DelayedSequenceLlm(SequenceLlm):
    def __init__(self, responses: list[str], *, delay_seconds: float) -> None:
        super().__init__(responses)
        self.delay_seconds = delay_seconds

    def _create(self, **kwargs):
        time.sleep(self.delay_seconds)
        return super()._create(**kwargs)


def _initial_history() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Be accurate."},
        {"role": "system", "content": "Drive the bounded analysis."},
        {"role": "user", "content": "Complete the deterministic workflow."},
    ]


def _run_session(
    tmp_path: Path,
    *,
    llm: SequenceLlm,
    sandbox: RecordingSandbox,
    history: list[dict[str, str]],
    max_turns: int = 3,
    event_callback: Callable[[runner.RunnerEvent], None] | None = None,
    should_checkpoint: Callable[[], bool] | None = None,
    checkpoint_callback: (
        Callable[[runner.AgentSessionCheckpointState], None] | None
    ) = None,
    resume_state: runner.AgentSessionCheckpointState | None = None,
) -> runner.AgentSessionResult:
    agent_system, driver = _agents()
    return runner.run_agent_session(
        console=Console(file=io.StringIO(), force_terminal=False),
        agent_system=agent_system,
        driver_agent=driver,
        analysis_context="bounded checkpoint test",
        llm_client=llm,
        sandbox_manager=sandbox,
        history=history,
        is_auto=True,
        max_turns=max_turns,
        output_dir=tmp_path,
        durable_run_id="run_" + "c" * 32,
        event_callback=event_callback,
        should_checkpoint=should_checkpoint,
        checkpoint_callback=checkpoint_callback,
        resume_state=resume_state,
    )


def _checkpoint_after_turn_two(
    tmp_path: Path,
    *,
    provider_delay_seconds: float = 0.0,
) -> tuple[
    runner.AgentSessionCheckpointState,
    list[dict[str, str]],
    SequenceLlm,
    RecordingSandbox,
    list[runner.RunnerEvent],
    runner.AgentSessionResult,
]:
    responses = [
        "delegate_to_coder",
        "```python\nprint('checkpointed action')\n```",
        "end_session",
    ]
    llm = (
        DelayedSequenceLlm(responses, delay_seconds=provider_delay_seconds)
        if provider_delay_seconds
        else SequenceLlm(responses)
    )
    sandbox = RecordingSandbox()
    history = _initial_history()
    events: list[runner.RunnerEvent] = []
    checkpoints: list[runner.AgentSessionCheckpointState] = []
    boundary_checks = 0

    def should_checkpoint() -> bool:
        nonlocal boundary_checks
        boundary_checks += 1
        return boundary_checks == 2

    def checkpoint_callback(state: runner.AgentSessionCheckpointState) -> None:
        # The callback itself observes the declared boundary: every turn-two
        # effect and the normal auto-continue input are present, while turn
        # three has not started.
        assert history[-1] == {
            "role": "user",
            "content": AUTO_CONTINUE_MESSAGE,
        }
        assert events[-1]["event_type"] == "code_result"
        assert max(event["turn"] for event in events) == 2
        checkpoints.append(state)

    result = _run_session(
        tmp_path,
        llm=llm,
        sandbox=sandbox,
        history=history,
        event_callback=events.append,
        should_checkpoint=should_checkpoint,
        checkpoint_callback=checkpoint_callback,
    )

    assert boundary_checks == 2
    assert len(checkpoints) == 1
    return checkpoints[0], history, llm, sandbox, events, result


def test_checkpoint_stops_at_completed_turn_boundary_before_next_provider_call(
    tmp_path: Path,
) -> None:
    state, history, llm, sandbox, events, result = _checkpoint_after_turn_two(tmp_path)

    assert is_dataclass(state)
    assert result.succeeded is False
    assert result.cancelled is False
    assert result.end_reason == "checkpointed"
    assert result.turns_completed == result.final_turn == 2
    assert llm.calls == 2
    assert sandbox.calls == ["print('checkpointed action')"]
    assert not any(
        event["event_type"] == "turn_started" and event["turn"] == 3 for event in events
    )
    assert events[-1]["event_type"] == "session_end"
    assert events[-1]["turn"] == 2
    assert events[-1]["payload"]["end_reason"] == "checkpointed"
    assert history[-1] == {
        "role": "user",
        "content": AUTO_CONTINUE_MESSAGE,
    }

    assert state.current_agent_name == "coder"
    assert state.turns_completed == 2
    assert state.next_turn == 3
    assert state.code_blocks_produced == 1
    assert state.code_exec_attempts == 1
    assert state.code_exec_failures == 0
    assert state.consecutive_exec_failures == 0
    assert state.consecutive_no_action == 0
    assert state.correction_count == 0
    assert state.elapsed_seconds >= 0
    assert [action["type"] for action in state.action_space_past_actions] == [
        "agent_switch",
        "code_execution",
    ]


def test_resume_restores_turn_agent_history_actions_and_matches_clean_control(
    tmp_path: Path,
) -> None:
    (
        state,
        checkpoint_history,
        source_llm,
        source_sandbox,
        _,
        _,
    ) = _checkpoint_after_turn_two(tmp_path / "source", provider_delay_seconds=0.01)
    history_at_checkpoint = copy.deepcopy(checkpoint_history)
    action_updates_at_checkpoint = sum(
        message["content"].startswith("ACTION SPACE UPDATE:")
        for message in history_at_checkpoint
    )

    resumed_llm = SequenceLlm(["end_session"])
    resumed_sandbox = RecordingSandbox()
    resumed_events: list[runner.RunnerEvent] = []
    resumed_result = _run_session(
        tmp_path / "resumed",
        llm=resumed_llm,
        sandbox=resumed_sandbox,
        history=checkpoint_history,
        event_callback=resumed_events.append,
        resume_state=state,
    )

    control_history = _initial_history()
    control_llm = SequenceLlm(
        [
            "delegate_to_coder",
            "```python\nprint('checkpointed action')\n```",
            "end_session",
        ]
    )
    control_sandbox = RecordingSandbox()
    control_result = _run_session(
        tmp_path / "control",
        llm=control_llm,
        sandbox=control_sandbox,
        history=control_history,
    )

    assert resumed_result.succeeded is True
    assert resumed_result.end_reason == "agent_finished"
    assert resumed_result.current_agent_name == "coder"
    assert resumed_result.turns_completed == resumed_result.final_turn == 3
    assert resumed_events[0]["event_type"] == "turn_started"
    assert resumed_events[0]["turn"] == 3
    assert resumed_events[0]["agent_name"] == "coder"
    assert all(event["turn"] >= 3 for event in resumed_events)

    assert resumed_llm.calls == 1
    # WS-2 appends the ambient work-item state block to every outgoing
    # request after context assembly; it's derived, not part of the
    # checkpointed history itself (see render_work_item_state).
    assert resumed_llm.request_kwargs[0]["messages"] == history_at_checkpoint + [
        {"role": "system", "content": "WORK ITEMS: none open."}
    ]
    assert source_llm.calls + resumed_llm.calls == control_llm.calls == 3
    assert source_sandbox.calls + resumed_sandbox.calls == control_sandbox.calls
    assert resumed_sandbox.calls == []
    assert checkpoint_history == control_history
    assert (
        sum(
            message["content"].startswith("ACTION SPACE UPDATE:")
            for message in checkpoint_history
        )
        == action_updates_at_checkpoint
    )

    comparable_fields = (
        "succeeded",
        "cancelled",
        "end_reason",
        "turns_completed",
        "code_blocks_produced",
        "code_exec_attempts",
        "code_exec_failures",
        "correction_count",
        "current_agent_name",
        "final_turn",
    )
    assert {field: getattr(resumed_result, field) for field in comparable_fields} == {
        field: getattr(control_result, field) for field in comparable_fields
    }
    assert resumed_result.duration_seconds >= round(state.elapsed_seconds, 2)


def test_resume_max_turns_is_a_logical_total_not_a_fresh_attempt_budget(
    tmp_path: Path,
) -> None:
    state, checkpoint_history, _, _, _, _ = _checkpoint_after_turn_two(
        tmp_path / "source"
    )
    llm = SequenceLlm(
        [
            "```python\nprint('turn three only')\n```",
            "```python\nraise AssertionError('turn four must not run')\n```",
        ]
    )
    sandbox = RecordingSandbox()
    events: list[runner.RunnerEvent] = []

    result = _run_session(
        tmp_path / "resumed",
        llm=llm,
        sandbox=sandbox,
        history=checkpoint_history,
        max_turns=3,
        event_callback=events.append,
        resume_state=state,
    )

    assert llm.calls == 1
    assert sandbox.calls == ["print('turn three only')"]
    assert result.end_reason == "max_turns_reached"
    assert result.turns_completed == result.final_turn == 3
    assert result.code_blocks_produced == 2
    assert result.code_exec_attempts == 2
    assert not any(event["turn"] == 4 for event in events)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("next_turn", 2),
        ("current_agent_name", "missing-agent"),
        ("code_exec_attempts", -1),
        ("code_exec_failures", 2),
        ("action_space_past_actions", ("not-an-action-record",)),
    ],
)
def test_invalid_resume_state_fails_before_provider_or_sandbox_side_effects(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    state, checkpoint_history, _, _, _, _ = _checkpoint_after_turn_two(
        tmp_path / "source"
    )
    tampered = copy.deepcopy(state)
    object.__setattr__(tampered, field, invalid_value)
    history_before_resume = copy.deepcopy(checkpoint_history)
    llm = SequenceLlm(["must not be called"])
    sandbox = RecordingSandbox()
    events: list[runner.RunnerEvent] = []

    with pytest.raises(ValueError):
        _run_session(
            tmp_path / f"invalid-{field}",
            llm=llm,
            sandbox=sandbox,
            history=checkpoint_history,
            event_callback=events.append,
            resume_state=tampered,
        )

    assert llm.calls == 0
    assert sandbox.calls == []
    assert events == []
    assert checkpoint_history == history_before_resume
