"""
User-typed REPL commands for the interactive `caribou run` session.

These are commands the *human* types at the interactive prompt (`/work-items`,
`/evaluate`, ...). They are unrelated to agent delegation commands
(`delegate_to_<agent>`), which are emitted by the LLM itself and dispatched in
runner.py by matching `detect_delegation()` against `Agent.commands` from the
blueprint JSON — that mechanism has no human input in the loop at all.

Dispatch is exact-match on the first whitespace-delimited token (after
lowercasing), not prefix matching.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel

from caribou.agents.AgentSystem import Agent, AgentSystem
from caribou.execution.artifacts import SessionArtifacts
from caribou.execution.benchmark_runner import run_benchmark
from caribou.execution.evaluation import (
    EvaluationContextTooLarge,
    EvaluatorRuntime,
    evaluation_response_metadata,
    evaluate_work_item,
    build_evaluation_payload,
    resolve_evaluator_agent,
    run_evaluation,
)
from caribou.execution.MemoryManager import MemoryManager
from caribou.execution.path_utils import get_default_runs_dir
from caribou.execution.report_generation import AgentReportMemory
from caribou.execution.token_utils import estimate_messages_tokens
from caribou.execution.work_items import WorkItemError, WorkItemStore


@dataclass
class UserCommandContext:
    """Everything a REPL command handler needs, gathered fresh each time the
    interactive prompt loop asks for the next user turn (current_agent can
    change between turns via delegation, so this must not be built once and
    reused)."""

    console: Console
    run_id: str
    turn: int
    history: List[Dict[str, str]]
    artifacts: SessionArtifacts
    agent_system: AgentSystem
    current_agent: Agent
    llm_client: object
    model_name: str
    memory_manager: Optional[MemoryManager]
    report_memory: Optional[AgentReportMemory]
    benchmark_modules: Optional[List[str]]
    sandbox_manager: object
    output_dir: Optional[Path]
    evaluator_runtime: EvaluatorRuntime
    work_items: WorkItemStore


@dataclass
class UserCommand:
    name: str  # canonical form, e.g. "/evaluate"
    aliases: Tuple[str, ...]
    help: str
    handler: Callable[[str, "UserCommandContext"], None]


def _cmd_evaluate(_arg: str, ctx: UserCommandContext) -> None:
    console = ctx.console
    evaluator_agent, source = resolve_evaluator_agent(ctx.agent_system)
    console.print(f"[dim]Using {source}.[/dim]")

    payload = build_evaluation_payload(
        run_id=ctx.run_id,
        turn=ctx.turn,
        active_agent=ctx.current_agent.name,
        history=ctx.history,
        todos=[
            {
                "id": t.id,
                "text": t.text,
                "status": t.status,
                "added_by": t.added_by,
                "turn": t.turn,
            }
            for t in ctx.artifacts.list_todos()
        ],
    )

    console.print("[cyan]Evaluating run...[/cyan]")
    provider_receipt: Dict[str, object] = {}
    try:
        assessment = run_evaluation(
            evaluator_agent=evaluator_agent,
            llm_client=ctx.evaluator_runtime.llm_client,
            model_name=ctx.evaluator_runtime.model_name,
            payload=payload,
            response_callback=lambda response: provider_receipt.update(
                evaluation_response_metadata(response)
            ),
        )
    except EvaluationContextTooLarge as exc:
        console.print(f"[red]{exc} Evaluation aborted; nothing was sent.[/red]")
        return

    console.print(Panel(assessment, title="Run Evaluation", border_style="cyan"))

    report_output_dir = ctx.output_dir if ctx.output_dir else get_default_runs_dir()
    report_dir = report_output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        report_dir / f"evaluation_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
    )
    report_path.write_text(
        json.dumps(
            {
                "run_id": ctx.run_id,
                "turn": ctx.turn,
                "evaluator_agent": evaluator_agent.name,
                "model": ctx.evaluator_runtime.model_name,
                "provider": ctx.evaluator_runtime.provider,
                "evaluator_model_revision": ctx.evaluator_runtime.revision,
                "provider_receipt": provider_receipt,
                "assessment": assessment,
            },
            indent=2,
        )
    )
    console.print(f"[bold green]✓ Evaluation saved to:[/bold green] {report_path}")


def _cmd_evaluator(arg: str, ctx: UserCommandContext) -> None:
    """Inspect or change the evaluator-only model binding."""

    tokens = shlex.split(arg)
    if not tokens or tokens[0] == "status":
        ctx.console.print(
            f"Evaluator model: {ctx.evaluator_runtime.provider}/"
            f"{ctx.evaluator_runtime.model_name} "
            f"(revision {ctx.evaluator_runtime.revision})"
        )
        return
    if tokens[0] != "model":
        ctx.console.print(
            "[yellow]Usage: /evaluator status | /evaluator model "
            "--llm PROVIDER --model MODEL_ID [--reason TEXT][/yellow]"
        )
        return
    values: Dict[str, str] = {}
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token not in {"--llm", "--model", "--reason"} or index + 1 >= len(tokens):
            ctx.console.print("[red]Invalid evaluator model arguments.[/red]")
            return
        values[token] = tokens[index + 1]
        index += 2
    backend = values.get("--llm", "").strip()
    model = values.get("--model", "").strip()
    if not backend or not model:
        ctx.console.print("[red]--llm and --model are required.[/red]")
        return
    reason = values.get("--reason", "").strip() or None
    if reason is not None and len(reason) > 1000:
        ctx.console.print("[red]--reason cannot exceed 1000 characters.[/red]")
        return
    from caribou.server.session_setup import build_model_client, resolve_model_binding

    try:
        client, exact_model = build_model_client(backend=backend, model_name=model)
        resolved = resolve_model_binding(
            backend=backend, model_name=model, resolved_model_name=exact_model
        )
    except Exception as exc:
        ctx.console.print(f"[red]Could not configure evaluator model: {exc}[/red]")
        return
    previous = ctx.evaluator_runtime.model_name
    ctx.evaluator_runtime.llm_client = client
    ctx.evaluator_runtime.model_name = exact_model
    ctx.evaluator_runtime.provider = resolved.provider if resolved else backend
    ctx.evaluator_runtime.selection = {
        "mode": "explicit",
        "llm_backend": backend,
        "model_name": model,
    }
    ctx.evaluator_runtime.revision += 1
    system_message = (
        "[SYSTEM — CONFIGURATION CHANGE] Evaluator model changed "
        f"from {previous} to {exact_model}. Configuration revision: "
        f"{ctx.evaluator_runtime.revision - 1} → {ctx.evaluator_runtime.revision}. "
        f"Reason: {reason or 'not provided'}. Worker model unchanged."
    )
    ctx.history.append({"role": "system", "content": system_message})
    if ctx.memory_manager:
        ctx.memory_manager.add_message("system", system_message)
    ctx.console.print(f"[green]{system_message}[/green]")


def _format_state_value(value: object) -> str:
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _print_memory_state(console: Console, state: Dict[str, object], label: str) -> None:
    breakdown = state.get("context_breakdown", {})
    top_lines = [
        f"{k}: {_format_state_value(v)}"
        for k, v in state.items()
        if k != "context_breakdown"
    ]
    breakdown_lines = [f"  {k}: {_format_state_value(v)}" for k, v in breakdown.items()]
    body = (
        "\n".join(top_lines)
        + "\n\n-- context breakdown --\n"
        + "\n".join(breakdown_lines)
    )
    console.print(Panel(body, title=f"Memory State ({label})", border_style="magenta"))


def _cmd_memory(_arg: str, ctx: UserCommandContext) -> None:
    console = ctx.console
    if ctx.memory_manager is not None:
        _print_memory_state(console, ctx.memory_manager.get_state(), "episodic")
    elif ctx.report_memory is not None:
        _print_memory_state(console, ctx.report_memory.get_state(), "agent_report")
    else:
        tokens = estimate_messages_tokens(ctx.history)
        console.print(
            Panel(
                f"strategy: full\nmessages: {len(ctx.history)}\nestimated_tokens: {tokens}",
                title="Memory State (full)",
                border_style="magenta",
            )
        )


def _cmd_brief(_arg: str, ctx: UserCommandContext) -> None:
    # brief.json lives beside the work-items store (WS-5.3 step 1: written
    # to `<output_dir>/brief.json`, and work_items.root is
    # `<output_dir>/work-items`) — no separate path needs threading through
    # UserCommandContext just for this.
    brief_path = ctx.work_items.root.parent / "brief.json"
    if not brief_path.exists():
        ctx.console.print(
            "[yellow]No brief is frozen for this session.[/yellow]"
        )
        return
    ctx.console.print(
        Panel(brief_path.read_text(encoding="utf-8"), title="Session Brief", border_style="cyan")
    )


def _cmd_work_items(_arg: str, ctx: UserCommandContext) -> None:
    items = ctx.work_items.list()
    if not items:
        ctx.console.print("[yellow]No work items.[/yellow]")
        return
    lines = []
    for item in items:
        completed = "—"
        if item.get("completed_at"):
            completed = f"T{item['completed_turn']} @ {item['completed_at']}"
        lines.append(
            f"{item['id']}  {item['status']}  {item['owner']}  {item['title']}\n"
            f"  created T{item['created_turn']} @ {item['created_at']}  "
            f"completed {completed}"
        )
    ctx.console.print(Panel("\n".join(lines), title="Work Items", border_style="cyan"))


def _cmd_work_item(arg: str, ctx: UserCommandContext) -> None:
    if not arg.strip().isdigit():
        ctx.console.print("[yellow]Usage: /work-item <id>[/yellow]")
        return
    try:
        item = ctx.work_items.read(int(arg.strip()))
    except WorkItemError as exc:
        ctx.console.print(f"[red]{exc}[/red]")
        return
    ctx.console.print(
        Panel(
            json.dumps(item, indent=2),
            title=f"Work Item {item['id']}",
            border_style="cyan",
        )
    )


def _cmd_review_work_item(arg: str, ctx: UserCommandContext) -> None:
    if not arg.strip().isdigit():
        ctx.console.print("[yellow]Usage: /review-work-item <id>[/yellow]")
        return
    try:
        evaluator_agent, source = resolve_evaluator_agent(ctx.agent_system)
        ctx.console.print(f"[dim]Using {source}.[/dim]")
        result = evaluate_work_item(
            store=ctx.work_items,
            item_id=int(arg.strip()),
            run_id=ctx.run_id,
            turn=ctx.turn,
            evaluator_agent=evaluator_agent,
            llm_client=ctx.evaluator_runtime.llm_client,
            model_name=ctx.evaluator_runtime.model_name,
        )
    except (
        Exception
    ) as exc:  # provider errors are durably recorded by evaluate_work_item
        ctx.console.print(f"[red]Work-item review failed: {exc}[/red]")
        return
    ctx.console.print(
        Panel(
            str(result["assessment"]),
            title=f"Work Item Review: {result['verdict']}",
            border_style="cyan",
        )
    )


def _cmd_artifacts(_arg: str, ctx: UserCommandContext) -> None:
    base_dir = ctx.artifacts.base_dir
    files = sorted(
        p for p in base_dir.rglob("*") if p.is_file() and ".git" not in p.parts
    )
    if not files:
        ctx.console.print(f"[yellow]No artifact files found under {base_dir}[/yellow]")
        return
    lines = [f"{p.relative_to(base_dir)}  ({p.stat().st_size:,} bytes)" for p in files]
    ctx.console.print(
        Panel("\n".join(lines), title=f"Artifacts ({base_dir})", border_style="green")
    )


def _cmd_benchmark(_arg: str, ctx: UserCommandContext) -> None:
    if not ctx.benchmark_modules:
        ctx.console.print(
            "[yellow]No benchmark modules were specified at startup.[/yellow]"
        )
        return
    bench_output_dir = ctx.output_dir if ctx.output_dir else get_default_runs_dir()
    for bm_module in ctx.benchmark_modules:
        run_benchmark(
            ctx.console,
            ctx.sandbox_manager,
            bm_module,
            is_auto=False,
            output_dir=bench_output_dir,
        )


def _cmd_help(_arg: str, ctx: UserCommandContext) -> None:
    lines = [cmd.help for cmd in USER_COMMANDS.values()]
    lines.append("/exit, /quit (also: exit, quit) — end the session")
    ctx.console.print(
        Panel("\n".join(lines), title="Available Commands", border_style="blue")
    )


USER_COMMANDS: Dict[str, UserCommand] = {}
_ALIASES: Dict[str, str] = {}


def _register(command: UserCommand) -> None:
    USER_COMMANDS[command.name] = command
    for token in (command.name,) + command.aliases:
        _ALIASES[token.lower()] = command.name


_register(
    UserCommand(
        name="/evaluator",
        aliases=(),
        help="/evaluator status | /evaluator model --llm PROVIDER --model MODEL_ID [--reason TEXT]",
        handler=_cmd_evaluator,
    )
)
_register(
    UserCommand(
        name="/brief",
        aliases=(),
        help="/brief — show this session's frozen brief, if any",
        handler=_cmd_brief,
    )
)
_register(
    UserCommand(
        name="/work-items",
        aliases=(),
        help="/work-items — list work items",
        handler=_cmd_work_items,
    )
)
_register(
    UserCommand(
        name="/work-item",
        aliases=(),
        help="/work-item <id> — inspect one work item",
        handler=_cmd_work_item,
    )
)
_register(
    UserCommand(
        name="/review-work-item",
        aliases=(),
        help="/review-work-item <id> — evaluate a closed work item",
        handler=_cmd_review_work_item,
    )
)
_register(
    UserCommand(
        name="/artifacts",
        aliases=(),
        help="/artifacts — list files produced by this run",
        handler=_cmd_artifacts,
    )
)
_register(
    UserCommand(
        name="/benchmark",
        aliases=("benchmark",),
        help="/benchmark — run the benchmark module(s) configured for this session",
        handler=_cmd_benchmark,
    )
)
_register(
    UserCommand(
        name="/memory",
        aliases=(),
        help="/memory — show the current context/memory usage breakdown",
        handler=_cmd_memory,
    )
)
_register(
    UserCommand(
        name="/evaluate",
        aliases=(),
        help="/evaluate — send the full run context to an evaluator agent for review",
        handler=_cmd_evaluate,
    )
)
_register(
    UserCommand(
        name="/help",
        aliases=(),
        help="/help — list available commands",
        handler=_cmd_help,
    )
)


def _tokenize(raw_input: str) -> Tuple[str, str]:
    stripped = raw_input.strip()
    if not stripped:
        return "", ""
    parts = stripped.split(None, 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def dispatch_user_command(raw_input: str, ctx: UserCommandContext) -> bool:
    """Exact-match the first token against the command registry and run its
    handler. Returns True if the input was a recognized command (the caller
    should re-prompt), False if it should be treated as a normal chat
    message."""
    first_token, remainder = _tokenize(raw_input)
    canonical = _ALIASES.get(first_token.lower())
    if canonical is None:
        return False
    USER_COMMANDS[canonical].handler(remainder, ctx)
    return True
