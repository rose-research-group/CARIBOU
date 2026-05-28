# caribou/execution/runner.py
from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.prompt import Prompt

# --- Project-specific Imports ---
try:
    from caribou.agents.AgentSystem import Agent, AgentSystem
    from caribou.core.io_helpers import display, extract_code_blocks, extract_python_code_blocks, format_execute_response
    from caribou.execution.MemoryManager import MemoryManager
    from caribou.execution.ActionSpace import AgentActionSpace
    from caribou.execution.artifacts import SessionArtifacts
    from caribou.execution.agent_management import _extract_possible_actions, _apply_agent_switch
    from caribou.execution.benchmark_runner import run_benchmark
    from caribou.execution.message_utils import (
        detect_delegation,
        detect_end_session,
        detect_rag,
        _extract_artifacts_from_msg,
        _count_code_blocks,
        _code_preview,
    )
    from caribou.execution.path_utils import _init_paths, get_default_runs_dir
    from caribou.execution.rag_client import get_rag_client
    from caribou.execution.report_generation import (
        AgentReportMemory,
        _write_session_report,
        _generate_agent_report,
    )
    from caribou.execution.ui_helpers import _render_todos
except ImportError as e:
    print(f"Failed to import a required CARIBOU module: {e}", file=sys.stderr)
    sys.exit(1)


# --- Type Hinting & Base Classes ---
class SandboxManager:
    """Abstract base class for sandbox interaction."""
    def start_container(self) -> bool:
        raise NotImplementedError

    def stop_container(self) -> None:
        raise NotImplementedError

    def exec_code(self, code: str, timeout: int) -> dict:
        raise NotImplementedError


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
    benchmark_modules: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    make_report: bool = False,
    agent_report_memory: bool = False,
):
    """
    Main driver for agent execution sessions, passing output_dir for benchmark saving.
    """
    _init_paths(output_dir)

    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    default_runs_dir = get_default_runs_dir()
    artifacts_dir = output_dir if output_dir else (default_runs_dir / "session_notes" / run_id)
    artifacts = SessionArtifacts(run_id=run_id, base_dir=artifacts_dir)

    if agent_report_memory and compress_memory:
        console.print("[yellow]Agent report memory enabled; disabling episodic compression for this session.[/yellow]")
        compress_memory = False

    memory_manager: Optional[MemoryManager] = None
    if compress_memory:
        console.print("[bold cyan]🧠 Adaptive context memory is enabled.[/bold cyan]")
        memory_manager = MemoryManager(llm_client=llm_client, model_name=model_name, initial_history=history)

    report_memory: Optional[AgentReportMemory] = None
    current_agent_history_start = 0
    if agent_report_memory:
        base_globals = [history[0]] if history else []
        agent_prompt_content = history[1]["content"] if len(history) > 1 else ""
        report_memory = AgentReportMemory(base_globals, agent_prompt_content)
        current_agent_history_start = min(len(history), 2)

    action_space = AgentActionSpace(driver_agent.name)
    action_space.set_possible_actions(_extract_possible_actions(driver_agent))
    action_init_msg = action_space.to_message()
    history.append({"role": "system", "content": action_init_msg})
    if memory_manager:
        memory_manager.add_message("system", action_init_msg)
    if agent_report_memory:
        current_agent_history_start = min(len(history), 2)

    # --- Display the initial context provided by the CLI ---
    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if role in ["system", "user"]:
            display(console, role, content)

    current_agent = driver_agent
    turns_completed = 0
    code_block_count = 0
    code_exec_attempts = 0
    code_exec_failures = 0
    consecutive_failures = 0
    correction_count = 0
    session_start_ts = datetime.utcnow()
    session_start_time = time.time()
    session_end_reason = "completed"
    last_code_snippet: str | None = None

    while True:
        if is_auto and turns_completed >= max_turns:
            console.print("[bold green]Auto run finished: Max turns reached.[/bold green]")
            session_end_reason = "max_turns_reached"
            break
        turn = turns_completed + 1
        console.print(f"\n[bold]LLM call (turn {turn})…[/bold]")

        if report_memory:
            working_history = history[current_agent_history_start:]
            context_to_send = report_memory.build_context(working_history)
        elif memory_manager:
            context_to_send = memory_manager.get_context()
        else:
            context_to_send = history
        # Claude requires that assistant messages don't end with trailing whitespace
        cleaned_context = []
        for msg in context_to_send:
            cleaned_msg = msg.copy()
            if "content" in cleaned_msg and isinstance(cleaned_msg["content"], str):
                cleaned_msg["content"] = cleaned_msg["content"].rstrip()
            cleaned_context.append(cleaned_msg)

        try:
            resp = llm_client.chat.completions.create(
                model=model_name,
                messages=cleaned_context,
                temperature=0.0,
            )
            msg = resp.choices[0].message.content
        except Exception as e:
            console.print(f"[red]LLM API error: {e}[/red]")
            session_end_reason = "llm_error"
            break

        history.append({"role": "assistant", "content": msg})
        if memory_manager:
            memory_manager.add_message("assistant", msg)
        display(console, f"assistant ({current_agent.name})", msg)
        turns_completed += 1

        blocks_found = _count_code_blocks(msg)
        if blocks_found:
            code_block_count += blocks_found

        # --- Artifact extraction (notes, TODOs) ---
        extracted_notes, extracted_todos = _extract_artifacts_from_msg(msg)
        if extracted_notes:
            for note in extracted_notes:
                artifacts.add_note(note, current_agent.name, turn)
                note_msg = f"Captured note (turn {turn}, agent {current_agent.name}): {note}"
                history.append({"role": "system", "content": note_msg})
                if memory_manager:
                    memory_manager.add_message("system", note_msg)
            action_space.add_action("note_logged", f"Logged {len(extracted_notes)} note(s).", status="ok")
        if extracted_todos:
            for todo_text in extracted_todos:
                item = artifacts.add_todo(todo_text, current_agent.name, turn)
                todo_msg = f"TODO added (#{item.id}) by {current_agent.name}: {item.text}"
                history.append({"role": "system", "content": todo_msg})
                if memory_manager:
                    memory_manager.add_message("system", todo_msg)
            action_space.add_action("todo_logged", f"Logged {len(extracted_todos)} TODO(s).", status="ok")

        # --- End session handling ---
        # Only end session if there's no delegation command also present
        # (prevents premature exit when LLM outputs both delegation and end_session)
        has_delegation = detect_delegation(msg) is not None
        if detect_end_session(msg) and _count_code_blocks(msg) == 0 and not has_delegation:
            if is_auto:
                console.print("[yellow]Agent requested end_session. Ending auto run.[/yellow]")
                session_end_reason = "agent_finished"
                break
            else:
                user_choice = Prompt.ask(
                    "Agent requested end_session. Continue anyway?",
                    choices=["y", "n"],
                    default="n",
                ).lower()
                if user_choice == "n":
                    console.print("[bold yellow]Ending session at agent request.[/bold yellow]")
                    session_end_reason = "agent_finished"
                    break

        # Track whether any substantive action fires this turn (for loop-detection feedback)
        _action_fired = False
        _delegated = False

        # --- RAG handling ---
        query_from_re = detect_rag(msg)
        if query_from_re and current_agent.is_rag_enabled:
            _action_fired = True
            console.print(f"[yellow]🔍 Triggering RAG query: {query_from_re}[/yellow]")
            rag_client = get_rag_client(console)
            retrieved_docs = rag_client.query(query_from_re)
            if retrieved_docs:
                console.print(f"[green] RAG query successful. [/green]")
                feedback = retrieved_docs
                console.print(feedback)
                if memory_manager:
                    memory_manager.add_message("system", feedback)
                history.append({"role": "system", "content": feedback})
            else:
                console.print(f"[red] RAG query unsuccessful. [/red]")

        cmd = detect_delegation(msg)
        if cmd and cmd in current_agent.commands:
            _action_fired = True
            target_agent_name = current_agent.commands[cmd].target_agent
            new_agent = agent_system.get_agent(target_agent_name)
            if new_agent:
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
                        history.append({"role": "system", "content": f"Agent report from {current_agent.name}:\n{agent_report}"})
                    current_agent_history_start = len(history)
                routing_message = f"🔄 Routing to '{target_agent_name}' via {cmd}"
                current_agent = new_agent
                # Global policy lives in the pinned first system message; skip re-embedding here.
                system_prompt = current_agent.get_full_prompt(None)
                prompt_with_context = system_prompt + "\n\n" + analysis_context
                console.print(f"[yellow]{routing_message}[/yellow]")
                history.append({"role": "assistant", "content": f"🔄 Routing to **{target_agent_name}** (command `{cmd}`)"})
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
                _delegated = True

        code_blocks = extract_code_blocks(msg)
        if code_blocks:
            _action_fired = True
            total_blocks = len(code_blocks)
            rag_short_circuit = False
            for idx, (language, code) in enumerate(code_blocks, start=1):
                last_code_snippet = code
                console.print("[cyan]Executing code in sandbox…[/cyan]")
                exec_code = f"%%R\n{code}" if language == "r" else code
                exec_result = sandbox_manager.exec_code(exec_code, timeout=600)
                code_exec_attempts += 1
                if exec_result.get("status") != "ok":
                    code_exec_failures += 1
                    consecutive_failures += 1
                    if consecutive_failures == 2:
                        correction_count += 1
                else:
                    consecutive_failures = 0
                feedback = format_execute_response(exec_result, output_dir if output_dir else get_default_runs_dir())
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

                stderr = exec_result.get("stderr", "")
                if stderr and current_agent.is_rag_enabled:
                    func_error_patterns = [
                        r"(\w+)\(.*\) missing \d+ required positional argument",  # TypeError missing arguments
                        r"NameError: name '(\w+)' is not defined",  # NameError
                        r"AttributeError: .* has no attribute '(\w+)'",  # AttributeError
                        r"'(\w+)\(.*\) got an unexpected keyword argument"  # Unexpected keyword argument
                    ]

                    function_name = ""
                    retrieved_docs = ""

                    for pat in func_error_patterns:
                        match = re.search(pat, stderr)
                        if match:
                            function_name = [g for g in match.groups() if g]
                            break

                    if function_name:
                        function_name = function_name[0]
                        console.print(
                            f"[yellow]🔍 Incorrect function signature detected: {function_name}, function database search...[/yellow]"
                        )
                        rag_client = get_rag_client(console)
                        retrieved_docs = rag_client.retrieve_function(function_name)
                        if retrieved_docs:
                            console.print("[green] Query successful - Function signature found. [/green]")
                            feedback += (
                                f"\n {function_name} produced an error. The correct function signature for "
                                f"{function_name} is:\n{retrieved_docs}"
                            )
                            history.append({"role": "system", "content": feedback})
                            rag_short_circuit = True
                            break
                        else:
                            print("Error Query unsuccessful - Function signature does not exist in the current database.")
            if rag_short_circuit:
                continue

        # If delegation happened (with or without a code block), let the new agent
        # reply immediately rather than waiting for user input.
        if _delegated:
            continue

        if is_auto and not _action_fired:
            # No action was taken this turn — give the LLM explicit feedback so it
            # doesn't silently repeat the same output indefinitely.
            rag_hint = ""
            if current_agent.is_rag_enabled:
                rag_hint = (
                    " To query the knowledge base write `query_rag_<topic>` "
                    "(with angle brackets) on its own line, e.g. `query_rag_<Celltyping API>`."
                )
            no_action_msg = (
                f"[SYSTEM] No action was recognised in your last message "
                f"(no Python code block, no R code block, no delegation command, no RAG query).{rag_hint} "
                f"Please either write executable Python code in a ```python ... ``` block, "
                f"executable R code in a ```r ... ``` block, "
                f"issue a delegation command, or use a RAG query. "
                f"Do not output plain text descriptions of what you intend to do."
            )
            console.print(f"[red]{no_action_msg}[/red]")
            history.append({"role": "system", "content": no_action_msg})
            if memory_manager:
                memory_manager.add_message("system", no_action_msg)

        if is_auto:
            if benchmark_modules:
                # Determine output_dir for benchmark
                bench_output_dir = output_dir if output_dir else get_default_runs_dir()
                result_str = run_benchmark(
                    console, sandbox_manager, benchmark_modules[0],
                    is_auto=True, metadata={"name": "auto"}, agent_name=current_agent.name,
                    code_snippet=last_code_snippet, output_dir=bench_output_dir
                )
                if memory_manager:
                    memory_manager.add_message("system", result_str)
                history.append({"role": "system", "content": result_str})
                display(console, "user", result_str)

            # Add a user message to continue the conversation in auto mode
            auto_continue_msg = "Please continue with the next step."
            history.append({"role": "user", "content": auto_continue_msg})
            if memory_manager:
                memory_manager.add_message("user", auto_continue_msg)

            console.print(f"[yellow]Auto-continuing... {turns_completed}/{max_turns} turns complete.[/yellow]")
            continue

        # Interactive mode: prompt user for next action
        while True:
            prompt_text = "\n[bold]Next message ('benchmark' to run selected benchmark, 'exit' to quit)[/bold]"
            try:
                user_input = Prompt.ask(prompt_text, default="").strip()
            except (EOFError, KeyboardInterrupt):
                user_input = "exit"

            if user_input.lower() in {"exit", "quit"}:
                console.print("[bold yellow]Exiting session.[/bold yellow]")
                session_end_reason = "user_exit"
                break

            # --- Quick commands for TODO management ---
            if user_input.lower().startswith("/todo"):
                todo_text = user_input[len("/todo"):].strip()
                if todo_text:
                    item = artifacts.add_todo(todo_text, "user", turn)
                    msg = f"TODO added (#{item.id}) by user: {item.text}"
                    history.append({"role": "system", "content": msg})
                    if memory_manager:
                        memory_manager.add_message("system", msg)
                    console.print(f"[green]Added TODO #[/green]{item.id}: {item.text}")
                else:
                    console.print("[yellow]Usage: /todo <task>[/yellow]")
                continue

            if user_input.lower().startswith("/done"):
                parts = user_input.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    todo_id = int(parts[1])
                    item = artifacts.complete_todo(todo_id)
                    if item:
                        msg = f"TODO completed (#{item.id}) by user"
                        history.append({"role": "system", "content": msg})
                        if memory_manager:
                            memory_manager.add_message("system", msg)
                        console.print(f"[green]Marked TODO #[/green]{todo_id} as done")
                    else:
                        console.print(f"[yellow]No TODO found with id {todo_id}[/yellow]")
                else:
                    console.print("[yellow]Usage: /done <id>[/yellow]")
                continue

            if user_input.lower() in {"/todos", "todos"}:
                todo_items = [
                    {
                        "id": t.id,
                        "text": t.text,
                        "status": t.status,
                        "added_by": t.added_by,
                        "turn": t.turn,
                    }
                    for t in artifacts.list_todos()
                ]
                _render_todos(console, todo_items)
                continue

            if user_input.lower() == "benchmark":
                if benchmark_modules:
                    bench_output_dir = output_dir if output_dir else get_default_runs_dir()
                    for bm_module in benchmark_modules:
                        run_benchmark(console, sandbox_manager, bm_module, is_auto=False, output_dir=bench_output_dir)
                    continue
                else:
                    console.print("[yellow]No benchmark modules were specified at startup.[/yellow]")
                    continue

            if user_input:
                if memory_manager:
                    memory_manager.add_message("user", user_input)
                history.append({"role": "user", "content": user_input})
                display(console, "user", user_input)
            break

        # if we broke out of the inner prompt loop due to exit, stop the session
        if session_end_reason == "user_exit":
            break

    session_end_ts = datetime.utcnow()
    duration_seconds = round(time.time() - session_start_time, 2)

    if make_report:
        session_stats = {
            "mode": "auto" if is_auto else "interactive",
            "driver_agent": driver_agent.name,
            "model": model_name,
            "agent_turns": turns_completed,
            "code_blocks_produced": code_block_count,
            "code_exec_attempts": code_exec_attempts,
            "code_exec_failures": code_exec_failures,
            "correction_count": correction_count,
            "session_start": session_start_ts.isoformat(),
            "session_end": session_end_ts.isoformat(),
            "duration_seconds": duration_seconds,
            "max_turns_requested": max_turns if is_auto else None,
            "end_reason": session_end_reason,
        }
        report_output_dir = output_dir if output_dir else get_default_runs_dir()
        _write_session_report(console, output_dir=report_output_dir, stats=session_stats)
