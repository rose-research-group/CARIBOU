# caribou/execution/runner.py
from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, cast

from rich.console import Console
from rich.prompt import Prompt

# --- Project-specific Imports ---
try:
    from caribou.agents.AgentSystem import Agent, AgentSystem
    from caribou.core.io_helpers import (
        display,
        extract_python_code_blocks,
        format_execute_response,
    )
    from caribou.core.sandbox_management import SandboxReplUnavailableError
    from caribou.execution.MemoryManager import MemoryManager
    from caribou.execution.ActionSpace import AgentActionSpace
    from caribou.execution.artifacts import SessionArtifacts
    from caribou.execution.agent_management import (
        _extract_possible_actions,
        _apply_agent_switch,
    )
    from caribou.execution.benchmark_runner import run_benchmark
    from caribou.execution.message_utils import (
        detect_delegation,
        detect_end_session,
        detect_rag,
        extract_labeled_block,
        _count_code_blocks,
        _code_preview,
    )
    from caribou.execution.path_utils import _init_paths, get_default_runs_dir
    from caribou.execution.report_generation import (
        AgentReportMemory,
        _write_session_report,
        _generate_agent_report,
    )
    from caribou.execution.user_commands import (
        USER_COMMANDS,
        UserCommandContext,
        dispatch_user_command,
    )
    from caribou.execution.evaluation import EvaluatorRuntime
    from caribou.execution.work_items import (
        WorkItemError,
        WorkItemPolicy,
        WorkItemStore,
        parse_work_item_command,
        render_work_item_prompt,
    )
    from caribou.execution.work_item_runtime import (
        apply_command as apply_work_item_command,
        end_session_block,
        freeze_brief,
        render_brief_pin,
        render_work_item_state,
        stall_report,
        transfer_on_delegation,
    )
    from caribou.execution.session_brief import (
        BriefParseError,
        BriefPolicy,
        SessionBrief,
        parse_brief_block,
        render_briefing_prompt,
    )
except ImportError as e:
    print(f"Failed to import a required CARIBOU module: {e}", file=sys.stderr)
    sys.exit(1)


def get_rag_client(console: Console):
    """Load the optional RAG stack only when a runner action requires it."""
    from caribou.execution.rag_client import get_rag_client as create_rag_client

    return create_rag_client(console)


# Consecutive no-action / code-exec failures we tolerate before ending an auto run.
MAX_CONSECUTIVE_NO_ACTION = 3
MAX_CONSECUTIVE_EXEC_FAILURES = 5
# Retry policy for transient LLM errors.
_LLM_RETRY_ATTEMPTS = 3
_LLM_RETRY_BASE_DELAY = 2.0
_LLM_RETRY_MAX_DELAY = 4.0


class RunnerEvent(TypedDict):
    """Transport-neutral event emitted synchronously by the legacy runner."""

    schema_version: str
    event_type: str
    occurred_at: str
    run_id: str
    turn: int
    agent_name: str
    payload: Dict[str, object]


RunnerEventCallback = Callable[[RunnerEvent], None]
LlmAttemptCallback = Callable[[Dict[str, object]], None]
CancellationCheck = Callable[[], bool]
CheckpointCheck = Callable[[], bool]


@dataclass(frozen=True)
class AgentSessionCheckpointState:
    """JSON-serializable state captured only at a completed-turn boundary.

    This is deliberately narrower than a Python-process snapshot. The full message
    history is stored as its own checkpoint component; this record restores the
    runner cursor, current agent, counters, and action-space state needed to make
    the next model call without replaying a completed turn.
    """

    schema_version: str
    current_agent_name: str
    turns_completed: int
    next_turn: int
    code_blocks_produced: int
    code_exec_attempts: int
    code_exec_failures: int
    consecutive_exec_failures: int
    consecutive_no_action: int
    correction_count: int
    action_space_past_actions: tuple[Dict[str, object], ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if self.schema_version != "caribou.agent_session_checkpoint_state.v1":
            raise ValueError("unsupported agent checkpoint state schema")
        if not self.current_agent_name.strip():
            raise ValueError("checkpoint current agent must be non-empty")
        integer_fields = (
            self.turns_completed,
            self.next_turn,
            self.code_blocks_produced,
            self.code_exec_attempts,
            self.code_exec_failures,
            self.consecutive_exec_failures,
            self.consecutive_no_action,
            self.correction_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_fields
        ):
            raise ValueError("checkpoint counters must be nonnegative integers")
        if self.next_turn != self.turns_completed + 1:
            raise ValueError("checkpoint next turn must follow turns_completed")
        if self.code_exec_failures > self.code_exec_attempts:
            raise ValueError("checkpoint failures exceed execution attempts")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or self.elapsed_seconds < 0
            or not math.isfinite(self.elapsed_seconds)
        ):
            raise ValueError(
                "checkpoint elapsed seconds must be finite and nonnegative"
            )
        for action in self.action_space_past_actions:
            if not isinstance(action, dict):
                raise ValueError("checkpoint action-space entries must be objects")


CheckpointCallback = Callable[[AgentSessionCheckpointState], None]


@dataclass(frozen=True)
class AgentSessionResult:
    """Immutable terminal summary returned by :func:`run_agent_session`."""

    schema_version: str
    run_id: str
    succeeded: bool
    cancelled: bool
    end_reason: str
    turns_completed: int
    code_blocks_produced: int
    code_exec_attempts: int
    code_exec_failures: int
    correction_count: int
    current_agent_name: str
    final_turn: int
    started_at: str
    ended_at: str
    duration_seconds: float


class _LlmCallCancelled(Exception):
    """Internal control-flow signal for cooperative cancellation during an LLM call."""


_UNSUCCESSFUL_END_REASONS = frozenset(
    {
        "cancelled",
        "checkpointed",
        "llm_error",
        "max_turns_reached",
        "stuck_no_action",
        "stuck_code_failures",
        "timeout",
        "briefing_declined",
        "work_item_stalled",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _emit_runner_event(
    callback: Optional[RunnerEventCallback],
    *,
    event_type: str,
    run_id: str,
    turn: int,
    agent_name: str,
    payload: Optional[Dict[str, object]] = None,
) -> None:
    if callback is None:
        return
    callback(
        {
            "schema_version": "caribou.runner_event.v1",
            "event_type": event_type,
            "occurred_at": _utc_now(),
            "run_id": run_id,
            "turn": turn,
            "agent_name": agent_name,
            "payload": dict(payload or {}),
        }
    )


def _cancellation_requested(should_cancel: Optional[CancellationCheck]) -> bool:
    return should_cancel is not None and should_cancel()


def _retry_backoff_cancelled(
    delay: float, should_cancel: Optional[CancellationCheck]
) -> bool:
    """Wait for a retry without making cooperative cancellation unresponsive."""
    if should_cancel is None:
        time.sleep(delay)
        return False
    deadline = time.monotonic() + delay
    while True:
        if should_cancel():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))


def _provider_value(value: object, name: str) -> object | None:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _provider_text(value: object | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _provider_count(value: object | None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _provider_cost(value: object | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _provider_success_observation(response: object) -> Dict[str, object]:
    usage = _provider_value(response, "usage")
    prompt_details = _provider_value(usage, "prompt_tokens_details")
    completion_details = _provider_value(usage, "completion_tokens_details")
    cost_details = _provider_value(usage, "cost_details")
    cached_tokens = _provider_count(_provider_value(prompt_details, "cached_tokens"))
    if cached_tokens is None:
        cached_tokens = _provider_count(
            _provider_value(usage, "prompt_cache_hit_tokens")
        )
    choices = _provider_value(response, "choices")
    first_choice = (
        choices[0] if isinstance(choices, (list, tuple)) and choices else None
    )
    return {
        "outcome": "succeeded",
        "response_id": _provider_text(_provider_value(response, "id")),
        "request_id": _provider_text(_provider_value(response, "_request_id"))
        or _provider_text(_provider_value(response, "request_id")),
        "response_model": _provider_text(_provider_value(response, "model")),
        "upstream_provider": _provider_text(_provider_value(response, "provider")),
        "system_fingerprint": _provider_text(
            _provider_value(response, "system_fingerprint")
        ),
        "finish_reason": _provider_text(_provider_value(first_choice, "finish_reason")),
        "prompt_tokens": _provider_count(_provider_value(usage, "prompt_tokens")),
        "completion_tokens": _provider_count(
            _provider_value(usage, "completion_tokens")
        ),
        "total_tokens": _provider_count(_provider_value(usage, "total_tokens")),
        "cached_tokens": cached_tokens,
        "cache_miss_tokens": _provider_count(
            _provider_value(usage, "prompt_cache_miss_tokens")
        ),
        "reasoning_tokens": _provider_count(
            _provider_value(completion_details, "reasoning_tokens")
        ),
        "cost_usd": _provider_cost(_provider_value(usage, "cost")),
        "upstream_cost_usd": _provider_cost(
            _provider_value(cost_details, "upstream_inference_cost")
        ),
        "failure_type": None,
        "http_status_code": None,
    }


def _provider_failure_observation(error: Exception) -> Dict[str, object]:
    status_code = _provider_value(error, "status_code")
    return {
        "outcome": "failed",
        "response_id": None,
        "request_id": _provider_text(_provider_value(error, "request_id")),
        "response_model": None,
        "upstream_provider": None,
        "system_fingerprint": None,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cached_tokens": None,
        "cache_miss_tokens": None,
        "reasoning_tokens": None,
        "cost_usd": None,
        "upstream_cost_usd": None,
        "failure_type": type(error).__name__,
        "http_status_code": (
            status_code
            if isinstance(status_code, int)
            and not isinstance(status_code, bool)
            and status_code > 0
            else None
        ),
    }


def _record_llm_attempt(
    callback: Optional[LlmAttemptCallback],
    *,
    observation: Dict[str, object],
    turn: int,
    agent_name: str,
    attempt: int,
    maximum_attempts: int,
    started_at: str,
    started_monotonic: float,
    requested_model: str,
) -> None:
    if callback is None:
        return
    ended_at = _utc_now()
    duration_ms = max(0, round((time.monotonic() - started_monotonic) * 1000))
    callback(
        {
            "turn": turn,
            "agent_name": agent_name,
            "attempt": attempt,
            "maximum_attempts": maximum_attempts,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_ms": duration_ms,
            "requested_model": requested_model,
            **observation,
        }
    )


def _call_llm_with_retry(
    *,
    console: Console,
    llm_client: object,
    model_name: str,
    messages: List[Dict[str, str]],
    turn: int = 1,
    agent_name: str = "agent",
    llm_attempt_callback: Optional[LlmAttemptCallback] = None,
    should_cancel: Optional[CancellationCheck] = None,
    retry_attempts: int = _LLM_RETRY_ATTEMPTS,
    retry_base_delay: float = _LLM_RETRY_BASE_DELAY,
    retry_max_delay: float = _LLM_RETRY_MAX_DELAY,
    request_timeout_seconds: float | None = None,
    max_output_tokens: int | None = None,
) -> Optional[str]:
    """
    Call the LLM with exponential backoff on transient errors. Returns the
    assistant message string, or None if all retries are exhausted.

    When the optional cancellation hook is used, cancellation raises the private
    ``_LlmCallCancelled`` control-flow signal for ``run_agent_session`` to handle.
    """
    if retry_attempts < 1 or retry_base_delay < 0 or retry_max_delay < retry_base_delay:
        raise ValueError("invalid LLM retry policy")
    if request_timeout_seconds is not None and request_timeout_seconds <= 0:
        raise ValueError("request timeout must be greater than zero")
    if max_output_tokens is not None and (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise ValueError("max_output_tokens must be a positive integer")
    last_failure_type = "provider error"
    for attempt in range(1, retry_attempts + 1):
        if _cancellation_requested(should_cancel):
            raise _LlmCallCancelled
        attempt_started_at = _utc_now()
        attempt_started_monotonic = time.monotonic()
        try:
            request_options: Dict[str, object] = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.0,
            }
            if request_timeout_seconds is not None:
                request_options["timeout"] = request_timeout_seconds
            if max_output_tokens is not None:
                # The external providers currently supported by the experiment
                # workload use the OpenAI-compatible ``max_tokens`` request key.
                request_options["max_tokens"] = max_output_tokens
            resp = cast(Any, llm_client).chat.completions.create(**request_options)
        except _LlmCallCancelled:
            raise
        except Exception as e:  # noqa: BLE001 — SDKs raise varied exception types
            _record_llm_attempt(
                llm_attempt_callback,
                observation=_provider_failure_observation(e),
                turn=turn,
                agent_name=agent_name,
                attempt=attempt,
                maximum_attempts=retry_attempts,
                started_at=attempt_started_at,
                started_monotonic=attempt_started_monotonic,
                requested_model=model_name,
            )
            last_failure_type = type(e).__name__
            if attempt >= retry_attempts:
                break
            delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
            console.print(
                f"[yellow]LLM API error (attempt {attempt}/{retry_attempts}; "
                f"type {last_failure_type}). "
                f"Retrying in {delay:.1f}s…[/yellow]"
            )
            if _retry_backoff_cancelled(delay, should_cancel):
                raise _LlmCallCancelled
        else:
            _record_llm_attempt(
                llm_attempt_callback,
                observation=_provider_success_observation(resp),
                turn=turn,
                agent_name=agent_name,
                attempt=attempt,
                maximum_attempts=retry_attempts,
                started_at=attempt_started_at,
                started_monotonic=attempt_started_monotonic,
                requested_model=model_name,
            )
            if _cancellation_requested(should_cancel):
                raise _LlmCallCancelled
            return resp.choices[0].message.content
    console.print(
        f"[red]LLM API error after {retry_attempts} attempts "
        f"(type {last_failure_type}).[/red]"
    )
    return None


# --- Type Hinting & Base Classes ---
class SandboxManager:
    """Abstract base class for sandbox interaction."""

    def start_container(self) -> bool:
        raise NotImplementedError

    def stop_container(self) -> None:
        raise NotImplementedError

    def exec_code(self, code: str, timeout: int) -> dict:
        raise NotImplementedError


def _run_briefing_phase(
    *,
    console: Console,
    llm_client: object,
    model_name: str,
    history: List[Dict[str, str]],
    driver_agent: Agent,
    llm_attempt_callback: Optional[LlmAttemptCallback] = None,
    llm_retry_attempts: int = _LLM_RETRY_ATTEMPTS,
    llm_retry_base_delay: float = _LLM_RETRY_BASE_DELAY,
    llm_retry_max_delay: float = _LLM_RETRY_MAX_DELAY,
    max_output_tokens: Optional[int] = None,
) -> Tuple[Optional[SessionBrief], bool]:
    """Run the interactive briefing conversation (WS-5.3) until the human
    accepts a brief or ends the session. Interactive/CLI only — auto mode
    never calls this (see the docstring on `run_agent_session`'s
    `brief_policy` parameter).

    Mutates `history` in place, appending every turn exactly like the main
    turn loop does. Not resumable: a crash mid-briefing loses the
    conversation and the session must be started over — briefing runs
    entirely before any checkpoint boundary exists, so
    `AgentSessionCheckpointState` is untouched by this function.

    Returns `(brief, ended)`: `brief` is the accepted `SessionBrief`, or
    `None` if the session was ended during briefing instead (`ended=True` in
    that case — the caller must stop, not fall through to execution).
    """
    console.print(
        "[bold cyan]Briefing phase — describe what you want before any work "
        "starts. The agent will propose a brief for you to accept, reject, "
        "or edit.[/bold cyan]"
    )
    while True:
        user_input = Prompt.ask("\n[bold]You[/bold]", default="").strip()
        if user_input.lower() in {"exit", "quit"}:
            console.print("[bold yellow]Ending session during briefing.[/bold yellow]")
            return None, True
        if user_input:
            history.append({"role": "user", "content": user_input})
            display(console, "user", user_input)

        msg = _call_llm_with_retry(
            console=console,
            llm_client=llm_client,
            model_name=model_name,
            messages=history,
            turn=0,
            agent_name=driver_agent.name,
            llm_attempt_callback=llm_attempt_callback,
            retry_attempts=llm_retry_attempts,
            retry_base_delay=llm_retry_base_delay,
            retry_max_delay=llm_retry_max_delay,
            max_output_tokens=max_output_tokens,
        )
        if msg is None:
            console.print(
                "[red]Briefing LLM call failed after retries. Ending session.[/red]"
            )
            return None, True
        history.append({"role": "assistant", "content": msg})
        display(console, f"assistant ({driver_agent.name})", msg)

        if detect_end_session(msg):
            console.print(
                "[yellow]Agent requested end_session during briefing.[/yellow]"
            )
            return None, True

        block = extract_labeled_block(msg, "brief")
        if block is None:
            continue

        try:
            draft = parse_brief_block(block, created_by=driver_agent.name)
        except BriefParseError as exc:
            feedback = f"[SYSTEM] {exc.feedback}"
            history.append({"role": "system", "content": feedback})
            display(console, "system", feedback)
            continue

        console.print("\n[bold]Proposed brief:[/bold]")
        console.print(render_brief_pin(draft))
        choice = Prompt.ask(
            "Accept this brief?",
            choices=["accept", "reject", "edit"],
            default="accept",
        )
        if choice == "accept":
            return draft, False
        if choice == "reject":
            reason = Prompt.ask(
                "Why? (sent back to the agent to revise)",
                default="Please revise and re-propose.",
            )
            feedback = f"[SYSTEM] Brief rejected by the human: {reason}"
            history.append({"role": "system", "content": feedback})
            display(console, "system", feedback)
            continue
        # edit: the human supplies corrected JSON directly, bypassing the
        # agent for this round — the harness still validates it the same way.
        console.print("Paste corrected JSON for the brief block:")
        edited_raw = Prompt.ask("JSON")
        try:
            edited = parse_brief_block(edited_raw, created_by="human")
        except BriefParseError as exc:
            console.print(f"[red]{exc.feedback}[/red]")
            history.append(
                {
                    "role": "system",
                    "content": f"[SYSTEM] Human edit was invalid: {exc.feedback}",
                }
            )
            continue
        return edited, False


# --- Core Runner Functions ---
def run_agent_session(
    *,
    console: Console,
    agent_system: AgentSystem,
    driver_agent: Agent,
    analysis_context: str,
    llm_client: object,
    sandbox_manager: SandboxManager,
    history: List[Dict[str, str]],
    is_auto: bool,
    compress_memory: bool = False,
    max_turns: int = 1,
    model_name: str = "gpt-5.2",
    model_parameters: Optional[Dict[str, object]] = None,
    benchmark_modules: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    make_report: bool = False,
    agent_report_memory: bool = False,
    durable_run_id: Optional[str] = None,
    should_cancel: Optional[CancellationCheck] = None,
    event_callback: Optional[RunnerEventCallback] = None,
    llm_attempt_callback: Optional[LlmAttemptCallback] = None,
    should_checkpoint: Optional[CheckpointCheck] = None,
    checkpoint_callback: Optional[CheckpointCallback] = None,
    resume_state: Optional[AgentSessionCheckpointState] = None,
    timeout_seconds: Optional[float] = None,
    max_consecutive_no_action: int = MAX_CONSECUTIVE_NO_ACTION,
    max_consecutive_exec_failures: int = MAX_CONSECUTIVE_EXEC_FAILURES,
    llm_retry_attempts: int = _LLM_RETRY_ATTEMPTS,
    llm_retry_base_delay: float = _LLM_RETRY_BASE_DELAY,
    llm_retry_max_delay: float = _LLM_RETRY_MAX_DELAY,
    max_output_tokens: int | None = None,
    evaluator_runtime: Optional[EvaluatorRuntime] = None,
    brief_policy: Optional["BriefPolicy"] = None,
    brief: Optional["SessionBrief"] = None,
) -> AgentSessionResult:
    """
    Main driver for agent execution sessions, passing output_dir for benchmark saving.

    `brief_policy`/`brief` (WS-5): if `brief` is already supplied (a
    pre-authored, frozen brief — the auto-mode `--brief <path>` case), it is
    frozen immediately and no briefing conversation runs, regardless of
    `brief_policy`. Otherwise, if `brief_policy.enabled` and this is an
    interactive run resuming from nothing (`resume_state is None`), a
    briefing conversation runs before the first execution turn (WS-5.3); see
    `_run_briefing_phase`. In every other case (auto with no `--brief`,
    `brief_policy` disabled, or resuming a session already past briefing),
    execution starts immediately with no brief, exactly as before WS-5.
    """
    if durable_run_id is not None and not durable_run_id.strip():
        raise ValueError("durable_run_id must be non-empty when provided")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if max_consecutive_no_action < 1 or max_consecutive_exec_failures < 1:
        raise ValueError("consecutive failure limits must be positive")
    if max_output_tokens is not None and (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise ValueError("max_output_tokens must be a positive integer")
    if evaluator_runtime is None:
        evaluator_runtime = EvaluatorRuntime(
            llm_client=llm_client,
            model_name=model_name,
            provider="worker-provider",
            selection={"mode": "inherit_worker"},
        )
    if (should_checkpoint is None) != (checkpoint_callback is None):
        raise ValueError(
            "should_checkpoint and checkpoint_callback must be supplied together"
        )
    if resume_state is not None and not isinstance(
        resume_state, AgentSessionCheckpointState
    ):
        raise ValueError("resume_state must be an AgentSessionCheckpointState")
    if resume_state is not None:
        # Frozen dataclasses can still be corrupted through low-level mutation or
        # untrusted deserialization. Reconstruct before any provider/sandbox work.
        resume_state = replace(resume_state)

    _init_paths(output_dir)
    run_id = durable_run_id or f"run_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    default_runs_dir = get_default_runs_dir()
    artifacts_dir = (
        output_dir if output_dir else (default_runs_dir / "session_notes" / run_id)
    )
    artifacts = SessionArtifacts(run_id=run_id, base_dir=artifacts_dir)
    work_item_policy = getattr(agent_system, "work_item_policy", WorkItemPolicy())
    evaluator_agent_name = getattr(agent_system, "evaluator_agent_name", None)
    # Work items are constructed in both interactive and auto runs; `is_auto`
    # only gates human-confirmation points elsewhere (end_session prompts),
    # not whether work-item state exists. Auto runs need ambient state (WS-2)
    # and stall detection (WS-4) at least as much as interactive ones, since
    # there's no human watching to notice an abandoned item.
    work_items = WorkItemStore(
        artifacts_dir / "work-items",
        session_id=run_id,
        policy=work_item_policy,
        origin_run_id=run_id,
    )
    frozen_brief: Optional["SessionBrief"] = brief
    ended_during_briefing = False
    should_run_briefing = (
        frozen_brief is None
        and brief_policy is not None
        and brief_policy.enabled
        and not is_auto
        and resume_state is None
        and len(history) > 1
    )
    if len(history) > 1:
        if should_run_briefing:
            briefing_appendix = render_briefing_prompt()
            if briefing_appendix not in history[1].get("content", ""):
                history[1]["content"] = (
                    history[1].get("content", "") + briefing_appendix
                )
            frozen_brief, ended_during_briefing = _run_briefing_phase(
                console=console,
                llm_client=llm_client,
                model_name=model_name,
                history=history,
                driver_agent=driver_agent,
                llm_attempt_callback=llm_attempt_callback,
                llm_retry_attempts=llm_retry_attempts,
                llm_retry_base_delay=llm_retry_base_delay,
                llm_retry_max_delay=llm_retry_max_delay,
                max_output_tokens=max_output_tokens,
            )
        if not ended_during_briefing:
            prompt_appendix = render_work_item_prompt(work_item_policy)
            if prompt_appendix not in history[1].get("content", ""):
                history[1]["content"] = (
                    history[1].get("content", "") + prompt_appendix
                )
    if frozen_brief is not None and not ended_during_briefing:
        seed_item = freeze_brief(
            frozen_brief,
            brief_path=artifacts_dir / "brief.json",
            store=work_items,
            brief_mode=(brief_policy.mode if brief_policy is not None else "context"),
            owner=driver_agent.name,
            turn=0,
        )
        history.append(
            {"role": "system", "content": render_brief_pin(frozen_brief)}
        )
        if seed_item is not None:
            console.print(
                f"[green]Seeded work item {seed_item['id']}: "
                f"{seed_item['title']}[/green]"
            )

    if agent_report_memory and compress_memory:
        console.print(
            "[yellow]Agent report memory enabled; disabling episodic compression for this session.[/yellow]"
        )
        compress_memory = False

    memory_manager: Optional[MemoryManager] = None
    if compress_memory:
        console.print("[bold cyan]🧠 Adaptive context memory is enabled.[/bold cyan]")
        memory_manager = MemoryManager(
            llm_client=llm_client, model_name=model_name, initial_history=history
        )

    report_memory: Optional[AgentReportMemory] = None
    current_agent_history_start = 0
    if agent_report_memory:
        base_globals = [history[0]] if history else []
        agent_prompt_content = history[1]["content"] if len(history) > 1 else ""
        report_memory = AgentReportMemory(base_globals, agent_prompt_content)
        current_agent_history_start = min(len(history), 2)

    current_agent = driver_agent
    if resume_state is not None:
        restored_agent = agent_system.get_agent(resume_state.current_agent_name)
        if restored_agent is None:
            raise ValueError(
                "checkpoint current agent is absent from the frozen blueprint"
            )
        current_agent = restored_agent
    action_space = AgentActionSpace(current_agent.name)
    action_space.set_possible_actions(_extract_possible_actions(current_agent))
    if resume_state is None:
        action_init_msg = action_space.to_message()
        history.append({"role": "system", "content": action_init_msg})
        if memory_manager:
            memory_manager.add_message("system", action_init_msg)
        if agent_report_memory:
            current_agent_history_start = min(len(history), 2)
    else:
        action_space.past_actions = [
            dict(action) for action in resume_state.action_space_past_actions
        ]

    # --- Display the initial context provided by the CLI ---
    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if role in ["system", "user"]:
            display(console, role, content)

    turns_completed = resume_state.turns_completed if resume_state is not None else 0
    final_turn = turns_completed
    code_block_count = (
        resume_state.code_blocks_produced if resume_state is not None else 0
    )
    code_exec_attempts = (
        resume_state.code_exec_attempts if resume_state is not None else 0
    )
    code_exec_failures = (
        resume_state.code_exec_failures if resume_state is not None else 0
    )
    consecutive_failures = (
        resume_state.consecutive_exec_failures if resume_state is not None else 0
    )
    consecutive_no_action = (
        resume_state.consecutive_no_action if resume_state is not None else 0
    )
    correction_count = resume_state.correction_count if resume_state is not None else 0
    # A nonzero consecutive-failure count proves that a prior provider turn left
    # an unresolved execution failure. The exact historical turn is unnecessary
    # after resume: the next provider turn is necessarily later than the captured
    # completed-turn boundary.
    pending_correction_turn = (
        resume_state.turns_completed
        if resume_state is not None and consecutive_failures
        else None
    )
    prior_elapsed_seconds = (
        resume_state.elapsed_seconds if resume_state is not None else 0.0
    )
    session_start_ts = _utc_now()
    session_start_time = time.monotonic()
    session_deadline = None
    if timeout_seconds is not None:
        session_deadline = session_start_time + max(
            0.0, timeout_seconds - prior_elapsed_seconds
        )

    def session_stop_reason() -> str | None:
        if _cancellation_requested(should_cancel):
            return "cancelled"
        if session_deadline is not None and time.monotonic() >= session_deadline:
            return "timeout"
        return None

    session_end_reason = "completed"
    last_code_snippet: str | None = None
    # At most one automatic "continue" per user message, consumed after the agent
    # opens a work item so it can proceed with the work instead of stopping.
    auto_continue_budget = 1

    def checkpoint_at_completed_turn() -> bool:
        if should_checkpoint is None or not should_checkpoint():
            return False
        assert checkpoint_callback is not None
        checkpoint_callback(
            AgentSessionCheckpointState(
                schema_version="caribou.agent_session_checkpoint_state.v1",
                current_agent_name=current_agent.name,
                turns_completed=turns_completed,
                next_turn=turns_completed + 1,
                code_blocks_produced=code_block_count,
                code_exec_attempts=code_exec_attempts,
                code_exec_failures=code_exec_failures,
                consecutive_exec_failures=consecutive_failures,
                consecutive_no_action=consecutive_no_action,
                correction_count=correction_count,
                action_space_past_actions=tuple(
                    dict(action) for action in action_space.past_actions
                ),
                elapsed_seconds=(
                    prior_elapsed_seconds + time.monotonic() - session_start_time
                ),
            )
        )
        return True

    if not is_auto:
        try:
            import readline

            def _complete_user_command(text: str, state: int) -> Optional[str]:
                options = [name for name in USER_COMMANDS if name.startswith(text)]
                return options[state] if state < len(options) else None

            readline.set_completer(_complete_user_command)
            readline.parse_and_bind("tab: complete")
        except ImportError:
            pass  # readline isn't available on this platform; no tab-completion.

    while True:
        if ended_during_briefing:
            session_end_reason = "briefing_declined"
            break
        stop_reason = session_stop_reason()
        if stop_reason is not None:
            session_end_reason = stop_reason
            break
        if is_auto and turns_completed >= max_turns:
            console.print(
                "[bold green]Auto run finished: Max turns reached.[/bold green]"
            )
            session_end_reason = "max_turns_reached"
            break
        turn = turns_completed + 1
        final_turn = turn
        console.print(f"\n[bold]LLM call (turn {turn})…[/bold]")
        _emit_runner_event(
            event_callback,
            event_type="turn_started",
            run_id=run_id,
            turn=turn,
            agent_name=current_agent.name,
            payload={"model_name": model_name},
        )

        if report_memory:
            working_history = history[current_agent_history_start:]
            context_to_send = report_memory.build_context(working_history)
        elif memory_manager:
            context_to_send = memory_manager.get_context()
        else:
            context_to_send = history
        # Claude requires that assistant messages don't end with trailing whitespace
        cleaned_context = []
        for context_message in context_to_send:
            cleaned_msg = context_message.copy()
            if "content" in cleaned_msg and isinstance(cleaned_msg["content"], str):
                cleaned_msg["content"] = cleaned_msg["content"].rstrip()
            cleaned_context.append(cleaned_msg)

        # Ambient work-item state (WS-2): a view recomputed from the store
        # each turn, appended after context assembly so every memory
        # strategy sees current state without memory-subsystem changes.
        state_block = render_work_item_state(work_items, current_agent.name)
        if state_block:
            cleaned_context.append({"role": "system", "content": state_block})

        # Item-level stall detection (WS-4): surface a stuck item by id
        # rather than only tripping the turn-count breakers, which halt the
        # run without saying which item stalled it.
        stalled = stall_report(
            work_items,
            current_agent.name,
            turn=turn,
            stall_turns=work_item_policy.stall_turns,
        )
        if stalled is not None:
            if stalled["idle_turns"] >= work_item_policy.stall_halt_turns:
                halt_note = (
                    f"Halted: work item {stalled['item_id']} "
                    f"('{stalled['title']}') stalled for {stalled['idle_turns']} turns."
                )
                work_items.note(stalled["item_id"], "runner", turn, halt_note)
                console.print(f"[bold red]{halt_note}[/bold red]")
                session_end_reason = "work_item_stalled"
                break
            nudge = (
                f"[SYSTEM] Work item {stalled['item_id']} "
                f"('{stalled['title']}') has had no status change for "
                f"{stalled['idle_turns']} turns. Close it, transfer it, or "
                "explain why it is still open."
            )
            cleaned_context.append({"role": "system", "content": nudge})

        try:
            request_timeout_seconds = (
                max(0.001, session_deadline - time.monotonic())
                if session_deadline is not None
                else None
            )
            msg = _call_llm_with_retry(
                console=console,
                llm_client=llm_client,
                model_name=model_name,
                messages=cleaned_context,
                turn=turn,
                agent_name=current_agent.name,
                llm_attempt_callback=llm_attempt_callback,
                should_cancel=lambda: session_stop_reason() is not None,
                retry_attempts=llm_retry_attempts,
                retry_base_delay=llm_retry_base_delay,
                retry_max_delay=llm_retry_max_delay,
                request_timeout_seconds=request_timeout_seconds,
                max_output_tokens=max_output_tokens,
            )
        except _LlmCallCancelled:
            session_end_reason = session_stop_reason() or "cancelled"
            break
        if msg is None:
            session_end_reason = "llm_error"
            break
        if not msg.strip():
            # A blank completion (e.g. a truncated or empty provider turn) is
            # treated as no action, not a fatal error: downstream event
            # recording requires non-empty message content, and this turn
            # must still flow through the existing no-action feedback path
            # below rather than crash the whole run on an unhandled
            # validation error.
            msg = "[SYSTEM: provider returned an empty response]"

        history.append({"role": "assistant", "content": msg})
        if memory_manager:
            memory_manager.add_message("assistant", msg)
        display(console, f"assistant ({current_agent.name})", msg)
        turns_completed += 1
        _emit_runner_event(
            event_callback,
            event_type="assistant_message",
            run_id=run_id,
            turn=turn,
            agent_name=current_agent.name,
            payload={"role": "assistant", "content": msg},
        )

        stop_reason = session_stop_reason()
        if stop_reason is not None:
            session_end_reason = stop_reason
            break

        blocks_found = _count_code_blocks(msg)
        if blocks_found:
            code_block_count += blocks_found

        # Track whether any substantive action fires this turn.
        _action_fired = False
        _delegated = False
        _opened_work_item = False

        # --- Enforced work-item commands ---
        work_command = parse_work_item_command(msg)
        work_result = apply_work_item_command(
            work_items, msg, owner=current_agent.name, turn=turn
        )
        if work_result is not None:
            _action_fired = True
            if (
                work_command is not None
                and work_command.name == "open_work_item"
                and work_result.success
            ):
                _opened_work_item = True
            work_feedback = "[SYSTEM] " + work_result.feedback
            history.append({"role": "system", "content": work_feedback})
            if memory_manager:
                memory_manager.add_message("system", work_feedback)
            action_space.add_action(
                "work_item_command",
                work_feedback,
                status="ok" if work_result.success else "error",
            )
            if work_result.changed_item is not None:
                _emit_runner_event(
                    event_callback,
                    event_type="work_item_changed",
                    run_id=run_id,
                    turn=turn,
                    agent_name=current_agent.name,
                    payload={"item": work_result.changed_item},
                )

        # --- End session handling ---
        # Only end session if there's no delegation command also present
        # (prevents premature exit when LLM outputs both delegation and end_session)
        has_delegation = detect_delegation(msg) is not None
        end_session_refused = False
        if detect_end_session(msg):
            blocking = end_session_block(work_items, current_agent.name)
            if blocking:
                end_session_refused = True
                blocked_feedback = (
                    "[SYSTEM] end_session refused. Current owner "
                    f"{current_agent.name} still owns non-Done work items: "
                    + ", ".join(
                        f"{item['id']} ({item['status']}, owner={item['owner']})"
                        for item in blocking
                    )
                    + ". Close or transfer them before ending the session."
                )
                history.append({"role": "system", "content": blocked_feedback})
                if memory_manager:
                    memory_manager.add_message("system", blocked_feedback)
                action_space.add_action(
                    "end_session_refused", blocked_feedback, status="error"
                )
        if (
            detect_end_session(msg)
            and _count_code_blocks(msg) == 0
            and not has_delegation
            and not end_session_refused
        ):
            if is_auto:
                console.print(
                    "[yellow]Agent requested end_session. Ending auto run.[/yellow]"
                )
                session_end_reason = "agent_finished"
                break
            else:
                user_choice = Prompt.ask(
                    "Agent requested end_session. Continue anyway?",
                    choices=["y", "n"],
                    default="n",
                ).lower()
                if user_choice == "n":
                    console.print(
                        "[bold yellow]Ending session at agent request.[/bold yellow]"
                    )
                    session_end_reason = "agent_finished"
                    break

        # --- RAG handling ---
        query_from_re = detect_rag(msg)
        if query_from_re and current_agent.is_rag_enabled:
            normalized_rag_query = " ".join(query_from_re.split()).casefold()
            duplicate_rag_query = any(
                action.get("type") == "rag_query"
                and action.get("status") == "ok"
                and action.get("meta", {}).get("agent") == current_agent.name
                and action.get("meta", {}).get("normalized_query")
                == normalized_rag_query
                for action in action_space.past_actions
            )
            _emit_runner_event(
                event_callback,
                event_type="rag_attempt",
                run_id=run_id,
                turn=turn,
                agent_name=current_agent.name,
                payload={"query": query_from_re, "kind": "knowledge_query"},
            )
            if duplicate_rag_query:
                duplicate_error = "duplicate successful RAG query suppressed"
                duplicate_feedback = (
                    f"[SYSTEM] {duplicate_error}: {query_from_re!r}. The result is "
                    "already present in the conversation. Use it and continue with a "
                    "different action."
                )
                console.print(f"[yellow]{duplicate_feedback}[/yellow]")
                history.append({"role": "system", "content": duplicate_feedback})
                if memory_manager:
                    memory_manager.add_message("system", duplicate_feedback)
                action_space.add_action(
                    "rag_query_duplicate",
                    f"Suppressed duplicate knowledge query: {query_from_re}",
                    status="error",
                    meta={
                        "agent": current_agent.name,
                        "query": query_from_re,
                        "normalized_query": normalized_rag_query,
                    },
                )
                _emit_runner_event(
                    event_callback,
                    event_type="rag_result",
                    run_id=run_id,
                    turn=turn,
                    agent_name=current_agent.name,
                    payload={
                        "query": query_from_re,
                        "kind": "knowledge_query",
                        "success": False,
                        "content": "",
                        "error": duplicate_error,
                    },
                )
            else:
                _action_fired = True
                console.print(
                    f"[yellow]🔍 Triggering RAG query: {query_from_re}[/yellow]"
                )
                rag_error: Optional[Exception] = None
                try:
                    rag_client = get_rag_client(console)
                    retrieved_docs = rag_client.query(query_from_re)
                except Exception as rag_exc:  # noqa: BLE001 — surface, don't swallow
                    rag_error = rag_exc
                    console.print(f"[red] RAG query failed: {rag_exc} [/red]")
                    rag_err = (
                        f"[SYSTEM] RAG query for '{query_from_re}' failed: {rag_exc}. "
                        f"Proceed without retrieved context."
                    )
                    history.append({"role": "system", "content": rag_err})
                    if memory_manager:
                        memory_manager.add_message("system", rag_err)
                    retrieved_docs = None
                _emit_runner_event(
                    event_callback,
                    event_type="rag_result",
                    run_id=run_id,
                    turn=turn,
                    agent_name=current_agent.name,
                    payload={
                        "query": query_from_re,
                        "kind": "knowledge_query",
                        "success": bool(retrieved_docs),
                        "content": retrieved_docs or "",
                        "error": str(rag_error) if rag_error else "",
                    },
                )
                if retrieved_docs:
                    console.print("[green] RAG query successful. [/green]")
                    feedback = (
                        f"RAG RESULT for {query_from_re!r} (retrieval complete):\n"
                        f"{retrieved_docs}\n\n"
                        "Continue the task using this result. Do not repeat the same "
                        "query unless new information is required."
                    )
                    console.print(feedback)
                    if memory_manager:
                        memory_manager.add_message("system", feedback)
                    history.append({"role": "system", "content": feedback})
                else:
                    console.print("[red] RAG query unsuccessful. [/red]")

                action_space.add_action(
                    "rag_query",
                    f"Retrieved knowledge for query: {query_from_re}",
                    status="ok" if retrieved_docs else "error",
                    meta={
                        "agent": current_agent.name,
                        "query": query_from_re,
                        "normalized_query": normalized_rag_query,
                    },
                )

            rag_action_msg = action_space.to_message()
            history.append({"role": "system", "content": rag_action_msg})
            if memory_manager:
                memory_manager.add_message("system", rag_action_msg)

            if duplicate_rag_query:
                console.print(
                    "[yellow]Repeated retrieval is not counted as progress.[/yellow]"
                )
            elif retrieved_docs:
                # A successful first retrieval is a substantive action.
                _action_fired = True
            else:
                # A failed first retrieval still records an attempted action.
                _action_fired = True

            stop_reason = session_stop_reason()
            if stop_reason is not None:
                session_end_reason = stop_reason
                break

        # Agent delegation command (e.g. "delegate_to_coder"), emitted by the
        # LLM's own generated message and matched against the blueprint's
        # Agent.commands. Unrelated to the human-typed REPL commands (/todo,
        # /evaluate, ...) dispatched later in the interactive prompt loop via
        # caribou.execution.user_commands — no human input is involved here.
        cmd = detect_delegation(msg)
        if cmd and cmd in current_agent.commands:
            target_agent_name = current_agent.commands[cmd].target_agent
            new_agent = agent_system.get_agent(target_agent_name)
            previous_agent_name = current_agent.name
            if new_agent is not None:
                try:
                    transferred_items = transfer_on_delegation(
                        work_items,
                        previous_agent_name,
                        target_agent_name,
                        turn=turn,
                        evaluator_agent_name=evaluator_agent_name,
                    )
                except WorkItemError as exc:
                    transfer_feedback = (
                        f"[SYSTEM] Delegation refused because work-item transfer "
                        f"failed: {exc}"
                    )
                    history.append({"role": "system", "content": transfer_feedback})
                    if memory_manager:
                        memory_manager.add_message("system", transfer_feedback)
                    action_space.add_action(
                        "work_item_transfer", transfer_feedback, status="error"
                    )
                    new_agent = None
                else:
                    for transferred_item in transferred_items:
                        _emit_runner_event(
                            event_callback,
                            event_type="work_item_changed",
                            run_id=run_id,
                            turn=turn,
                            agent_name=previous_agent_name,
                            payload={"item": transferred_item},
                        )
            if new_agent:
                _action_fired = True
                if report_memory:
                    agent_history_slice = history[current_agent_history_start:]
                    agent_report = _generate_agent_report(
                        console,
                        llm_client=llm_client,
                        model_name=model_name,
                        agent_name=current_agent.name,
                        history_slice=agent_history_slice,
                    )
                    if agent_report:
                        report_memory.add_report(current_agent.name, agent_report)
                        history.append(
                            {
                                "role": "system",
                                "content": f"Agent report from {current_agent.name}:\n{agent_report}",
                            }
                        )
                    current_agent_history_start = len(history)
                routing_message = f"🔄 Routing to '{target_agent_name}' via {cmd}"
                current_agent = new_agent
                # Global policy lives in the pinned first system message; skip re-embedding here.
                system_prompt = current_agent.get_full_prompt(None, work_item_policy)
                prompt_with_context = system_prompt + "\n\n" + analysis_context
                console.print(f"[yellow]{routing_message}[/yellow]")
                history.append(
                    {
                        "role": "assistant",
                        "content": f"🔄 Routing to **{target_agent_name}** (command `{cmd}`)",
                    }
                )
                if memory_manager:
                    memory_manager.add_message("assistant", routing_message)
                _apply_agent_switch(
                    new_agent_prompt=system_prompt,
                    analysis_context=analysis_context,
                    history=history,
                    memory_manager=memory_manager,
                    action_space=action_space,
                    new_agent=new_agent,
                )
                if report_memory:
                    report_memory.update_agent_prompt(prompt_with_context)
                    current_agent_history_start = len(history)
                _emit_runner_event(
                    event_callback,
                    event_type="agent_switch",
                    run_id=run_id,
                    turn=turn,
                    agent_name=current_agent.name,
                    payload={
                        "from_agent": previous_agent_name,
                        "to_agent": target_agent_name,
                        "command": cmd,
                    },
                )
                _delegated = True

        stop_reason = session_stop_reason()
        if stop_reason is not None:
            session_end_reason = stop_reason
            break

        produced_code_blocks = extract_python_code_blocks(msg)
        if produced_code_blocks:
            _action_fired = True
            ignored_block_count = max(0, len(produced_code_blocks) - 1)
            if ignored_block_count:
                ignored_feedback = (
                    f"[SYSTEM] Ignored {ignored_block_count} additional complete "
                    "Python code block(s) from this provider turn. CARIBOU executes "
                    "at most the first complete Python block per provider turn so "
                    "the model can observe its result before proposing another "
                    "action. Emit at most one Python code block on the next turn."
                )
                history.append({"role": "system", "content": ignored_feedback})
                if memory_manager:
                    memory_manager.add_message("system", ignored_feedback)
                action_space.add_action(
                    "code_blocks_ignored",
                    f"Ignored {ignored_block_count} additional code block(s).",
                    status="error",
                    meta={
                        "total_blocks_produced": len(produced_code_blocks),
                        "executed_blocks": 1,
                        "ignored_blocks": ignored_block_count,
                    },
                )
                _emit_runner_event(
                    event_callback,
                    event_type="code_blocks_ignored",
                    run_id=run_id,
                    turn=turn,
                    agent_name=current_agent.name,
                    payload={
                        "total_blocks_produced": len(produced_code_blocks),
                        "executed_blocks": 1,
                        "ignored_blocks": ignored_block_count,
                        "reason": "maximum one code block per provider turn",
                    },
                )
            code_blocks = produced_code_blocks[:1]
            total_blocks = len(code_blocks)
            rag_short_circuit = False
            cancelled_during_actions = False
            sandbox_permanently_unavailable = False
            for idx, code in enumerate(code_blocks, start=1):
                if session_stop_reason() is not None:
                    cancelled_during_actions = True
                    break
                last_code_snippet = code
                console.print("[cyan]Executing code in sandbox…[/cyan]")
                action_id = f"{run_id}:turn:{turn}:block:{idx}"
                _emit_runner_event(
                    event_callback,
                    event_type="code_submitted",
                    run_id=run_id,
                    turn=turn,
                    agent_name=current_agent.name,
                    payload={
                        "action_id": action_id,
                        "source": code,
                        "block_index": idx,
                        "total_blocks": total_blocks,
                    },
                )
                code_started = time.monotonic()
                code_timeout = 600
                if session_deadline is not None:
                    code_timeout = max(
                        1,
                        min(600, math.ceil(session_deadline - time.monotonic())),
                    )
                try:
                    exec_result = sandbox_manager.exec_code(code, timeout=code_timeout)
                except SandboxReplUnavailableError as sandbox_error:
                    # A prior turn's execution timeout or cancellation can
                    # invalidate the stateful REPL; the sandbox layer then
                    # deliberately raises on the next call rather than
                    # silently starting a fresh, state-losing REPL. Every
                    # subsequent call is guaranteed to fail identically, so
                    # record it as an ordinary execution failure (for the
                    # audit trail) but end the session immediately below
                    # rather than crashing the worker or burning further
                    # provider turns on retries that cannot possibly succeed.
                    sandbox_permanently_unavailable = True
                    exec_result = {
                        "status": "error",
                        "stdout": "",
                        "stderr": (
                            f"Sandbox execution unavailable: {sandbox_error}. "
                            "A previous turn's execution timeout or "
                            "cancellation left the REPL unusable; in-memory "
                            "state from earlier turns cannot be recovered."
                        ),
                        "images": [],
                    }
                code_duration_ms = max(0, int((time.monotonic() - code_started) * 1000))
                code_exec_attempts += 1
                if exec_result.get("status") != "ok":
                    code_exec_failures += 1
                    consecutive_failures += 1
                    pending_correction_turn = turn
                else:
                    if (
                        pending_correction_turn is not None
                        and turn > pending_correction_turn
                    ):
                        correction_count += 1
                    pending_correction_turn = None
                    consecutive_failures = 0
                feedback = format_execute_response(
                    exec_result, output_dir if output_dir else get_default_runs_dir()
                )
                if total_blocks > 1:
                    feedback = feedback.replace(
                        "Code execution result:",
                        f"Code execution result (block {idx}/{total_blocks}):",
                        1,
                    )
                if memory_manager:
                    memory_manager.add_message("system", feedback)
                    if exec_result.get("status") == "ok":
                        memory_manager.add_pivotal_code(code)
                action_label = "Ran code block"
                if total_blocks > 1:
                    action_label = f"Ran code block {idx}/{total_blocks}"
                action_space.add_action(
                    "code_execution",
                    f"{action_label}:\n{_code_preview(code)}",
                    status=exec_result.get("status"),
                )
                summary_msg = action_space.to_message()
                history.append({"role": "system", "content": summary_msg})
                if memory_manager:
                    memory_manager.add_message("system", summary_msg)
                history.append({"role": "assistant", "content": feedback})
                display(console, "code execution result", feedback)
                _emit_runner_event(
                    event_callback,
                    event_type="code_result",
                    run_id=run_id,
                    turn=turn,
                    agent_name=current_agent.name,
                    payload={
                        "action_id": action_id,
                        "success": exec_result.get("status") == "ok",
                        "status": str(exec_result.get("status", "unknown")),
                        "duration_ms": code_duration_ms,
                        "stdout": str(exec_result.get("stdout", "")),
                        "stderr": str(exec_result.get("stderr", "")),
                        "block_index": idx,
                        "total_blocks": total_blocks,
                    },
                )

                # Try the RAG-based signature-repair lookup before deciding whether
                # to halt: a failure that reaches the consecutive-failure threshold
                # is exactly the failure most in need of a repair attempt, and must
                # not be skipped in favor of the plain halt below. This is bounded
                # to at most one threshold's worth of extra rescues — a lexical
                # corpus match can hit on an unrelated error, so repair must not be
                # able to indefinitely postpone the stuck-run safety net.
                stderr = exec_result.get("stderr", "")
                if (
                    stderr
                    and current_agent.is_rag_enabled
                    and not sandbox_permanently_unavailable
                    and consecutive_failures <= max_consecutive_exec_failures * 2
                ):
                    func_error_patterns = [
                        r"(\w+)\(.*\) missing \d+ required positional argument",  # TypeError missing arguments
                        r"NameError: name '(\w+)' is not defined",  # NameError
                        r"AttributeError: .* has no attribute '(\w+)'",  # AttributeError
                        r"'(\w+)\(.*\) got an unexpected keyword argument",  # Unexpected keyword argument
                    ]

                    function_name = ""
                    retrieved_docs = ""

                    for pat in func_error_patterns:
                        match = re.search(pat, stderr)
                        if match:
                            function_name = next(
                                (group for group in match.groups() if group), ""
                            )
                            break

                    if function_name:
                        console.print(
                            f"[yellow]🔍 Incorrect function signature detected: {function_name}, function database search...[/yellow]"
                        )
                        rag_client = get_rag_client(console)
                        retrieved_docs = rag_client.retrieve_function(function_name)
                        if retrieved_docs:
                            console.print(
                                "[green] Query successful - Function signature found. [/green]"
                            )
                            feedback += (
                                f"\n {function_name} produced an error. The correct function signature for "
                                f"{function_name} is:\n{retrieved_docs}"
                            )
                            history.append({"role": "system", "content": feedback})
                            rag_short_circuit = True
                            break
                        else:
                            print(
                                "Error Query unsuccessful - Function signature does not exist in the current database."
                            )
                if session_stop_reason() is not None:
                    cancelled_during_actions = True
                    break
            if cancelled_during_actions:
                session_end_reason = session_stop_reason() or "cancelled"
                break
            if sandbox_permanently_unavailable:
                console.print(
                    "[bold red]Auto run halted: the sandbox REPL is "
                    "unavailable and cannot recover mid-session.[/bold red]"
                )
                escalation_msg = (
                    "[SYSTEM] The sandbox REPL is unavailable after a prior "
                    "execution timeout or cancellation. Ending this auto run "
                    "so a human can inspect the state."
                )
                history.append({"role": "system", "content": escalation_msg})
                if memory_manager:
                    memory_manager.add_message("system", escalation_msg)
                session_end_reason = "stuck_code_failures"
                break
            if rag_short_circuit:
                if checkpoint_at_completed_turn():
                    session_end_reason = "checkpointed"
                    break
                continue
            # Escalate if the same code path keeps failing — don't loop forever.
            if is_auto and consecutive_failures >= max_consecutive_exec_failures:
                console.print(
                    f"[bold red]Auto run halted: {consecutive_failures} consecutive "
                    f"code execution failures — likely stuck on the same error.[/bold red]"
                )
                escalation_msg = (
                    f"[SYSTEM] {consecutive_failures} consecutive code execution failures. "
                    f"The current approach is not working — ending this auto run so a human "
                    f"can inspect the state."
                )
                history.append({"role": "system", "content": escalation_msg})
                if memory_manager:
                    memory_manager.add_message("system", escalation_msg)
                session_end_reason = "stuck_code_failures"
                break

        # In auto mode, delegation immediately hands execution to the new agent.
        # In interactive mode, the user gets control back after the handoff.
        if _delegated and is_auto:
            if checkpoint_at_completed_turn():
                session_end_reason = "checkpointed"
                break
            continue

        if is_auto and not _action_fired:
            # No action was taken this turn — give the LLM explicit feedback so it
            # doesn't silently repeat the same output indefinitely.
            consecutive_no_action += 1
            if consecutive_no_action >= max_consecutive_no_action:
                # The agent is stuck emitting non-actionable text. End the run
                # rather than burn tokens looping on the same feedback message.
                console.print(
                    f"[bold red]Auto run halted: agent produced no action for "
                    f"{consecutive_no_action} consecutive turns.[/bold red]"
                )
                session_end_reason = "stuck_no_action"
                break
            rag_hint = ""
            if current_agent.is_rag_enabled:
                rag_hint = (
                    " To query the knowledge base write `query_rag_<topic>` "
                    "(with angle brackets) on its own line, e.g. `query_rag_<Celltyping API>`."
                )
            no_action_msg = (
                f"[SYSTEM] No action was recognised in your last message "
                f"(no Python code block, no delegation command, no RAG query).{rag_hint} "
                f"Please either write executable Python code in a ```python ... ``` block, "
                f"issue a delegation command, or use a RAG query. "
                f"Do not output plain text descriptions of what you intend to do. "
                f"(Attempt {consecutive_no_action}/{max_consecutive_no_action} — the run "
                f"will halt after {max_consecutive_no_action} consecutive no-action turns.)"
            )
            console.print(f"[red]{no_action_msg}[/red]")
            history.append({"role": "system", "content": no_action_msg})
            if memory_manager:
                memory_manager.add_message("system", no_action_msg)
        elif _action_fired:
            consecutive_no_action = 0

        if is_auto:
            if benchmark_modules:
                stop_reason = session_stop_reason()
                if stop_reason is not None:
                    session_end_reason = stop_reason
                    break
                # Determine output_dir for benchmark
                bench_output_dir = output_dir if output_dir else get_default_runs_dir()
                result_str = run_benchmark(
                    console,
                    sandbox_manager,
                    benchmark_modules[0],
                    is_auto=True,
                    metadata={"name": "auto"},
                    agent_name=current_agent.name,
                    code_snippet=last_code_snippet,
                    output_dir=bench_output_dir,
                )
                if memory_manager:
                    memory_manager.add_message("system", result_str)
                history.append({"role": "system", "content": result_str})
                display(console, "user", result_str)
                stop_reason = session_stop_reason()
                if stop_reason is not None:
                    session_end_reason = stop_reason
                    break

            # Add a user message to continue the conversation in auto mode
            auto_continue_msg = "Please continue with the next step."
            history.append({"role": "user", "content": auto_continue_msg})
            if memory_manager:
                memory_manager.add_message("user", auto_continue_msg)

            if checkpoint_at_completed_turn():
                session_end_reason = "checkpointed"
                break

            console.print(
                f"[yellow]Auto-continuing... {turns_completed}/{max_turns} turns complete.[/yellow]"
            )
            continue

        # Interactive mode: after opening a work item, let the agent keep going
        # for one extra turn instead of stopping to wait for the user.
        if not is_auto and _opened_work_item and auto_continue_budget > 0:
            auto_continue_budget -= 1
            history.append(
                {"role": "user", "content": "Please continue with the next step."}
            )
            if memory_manager:
                memory_manager.add_message(
                    "user", "Please continue with the next step."
                )
            continue

        # Interactive mode: prompt user for next action
        while True:
            stop_reason = session_stop_reason()
            if stop_reason is not None:
                session_end_reason = stop_reason
                break
            prompt_text = (
                "\n[bold]Next message ('/help' for commands, 'exit' to quit)[/bold]"
            )
            try:
                user_input = Prompt.ask(prompt_text, default="").strip()
            except (EOFError, KeyboardInterrupt):
                user_input = "exit"

            if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
                console.print("[bold yellow]Exiting session.[/bold yellow]")
                session_end_reason = "user_exit"
                break

            # --- User-typed REPL commands (not agent delegation, see comment
            # above the detect_delegation() call earlier in this function) ---
            command_ctx = UserCommandContext(
                console=console,
                run_id=run_id,
                turn=turn,
                history=history,
                artifacts=artifacts,
                agent_system=agent_system,
                current_agent=current_agent,
                llm_client=llm_client,
                model_name=model_name,
                memory_manager=memory_manager,
                report_memory=report_memory,
                benchmark_modules=benchmark_modules,
                sandbox_manager=sandbox_manager,
                output_dir=output_dir,
                evaluator_runtime=evaluator_runtime,
                work_items=cast(WorkItemStore, work_items),
            )
            if dispatch_user_command(user_input, command_ctx):
                continue

            if user_input:
                if memory_manager:
                    memory_manager.add_message("user", user_input)
                history.append({"role": "user", "content": user_input})
                display(console, "user", user_input)
                auto_continue_budget = 1
            break

        # if we broke out of the inner prompt loop due to exit, stop the session
        if session_end_reason == "user_exit":
            break

    session_end_ts = _utc_now()
    duration_seconds = round(
        prior_elapsed_seconds + time.monotonic() - session_start_time, 6
    )

    if make_report:
        session_stats: dict[str, object] = {
            "mode": "auto" if is_auto else "interactive",
            "driver_agent": driver_agent.name,
            "model": model_name,
            "model_parameters": dict(model_parameters or {}),
            "evaluator_model": evaluator_runtime.model_name,
            "evaluator_provider": evaluator_runtime.provider,
            "evaluator_model_revision": evaluator_runtime.revision,
            "evaluator_selection": dict(evaluator_runtime.selection),
            "agent_turns": turns_completed,
            "code_blocks_produced": code_block_count,
            "code_exec_attempts": code_exec_attempts,
            "code_exec_failures": code_exec_failures,
            "correction_count": correction_count,
            "session_start": session_start_ts,
            "session_end": session_end_ts,
            "duration_seconds": duration_seconds,
            "max_turns_requested": max_turns if is_auto else None,
            "end_reason": session_end_reason,
        }
        report_output_dir = output_dir if output_dir else get_default_runs_dir()
        _write_session_report(
            console, output_dir=report_output_dir, stats=session_stats
        )

    result = AgentSessionResult(
        schema_version="caribou.agent_session_result.v1",
        run_id=run_id,
        succeeded=session_end_reason not in _UNSUCCESSFUL_END_REASONS,
        cancelled=session_end_reason == "cancelled",
        end_reason=session_end_reason,
        turns_completed=turns_completed,
        code_blocks_produced=code_block_count,
        code_exec_attempts=code_exec_attempts,
        code_exec_failures=code_exec_failures,
        correction_count=correction_count,
        current_agent_name=current_agent.name,
        final_turn=final_turn,
        started_at=session_start_ts,
        ended_at=session_end_ts,
        duration_seconds=duration_seconds,
    )
    _emit_runner_event(
        event_callback,
        event_type="session_end",
        run_id=run_id,
        turn=final_turn,
        agent_name=current_agent.name,
        payload={
            "succeeded": result.succeeded,
            "cancelled": result.cancelled,
            "end_reason": result.end_reason,
            "turns_completed": result.turns_completed,
            "code_blocks_produced": result.code_blocks_produced,
            "code_exec_attempts": result.code_exec_attempts,
            "code_exec_failures": result.code_exec_failures,
            "correction_count": result.correction_count,
            "started_at": result.started_at,
            "ended_at": result.ended_at,
            "duration_seconds": result.duration_seconds,
        },
    )
    return result
