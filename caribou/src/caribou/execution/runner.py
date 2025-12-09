# caribou/execution/runner.py
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table

# --- Project-specific Imports ---
try:
    from caribou.config import CARIBOU_HOME
    from caribou.agents.AgentSystem import Agent, AgentSystem
    from caribou.core.io_helpers import display, extract_python_code, format_execute_response
    from caribou.rag.RetrievalAugmentedGeneration import RetrievalAugmentedGeneration
    from caribou.execution.MemoryManager import MemoryManager
    from caribou.execution.ActionSpace import AgentActionSpace
    from caribou.execution.artifacts import SessionArtifacts
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

# --- Constants and Path Setup ---
_DELEG_RE = re.compile(r"delegate_to_([A-Za-z0-9_]+)")
_RAG_RE = re.compile(r"query_rag_<([^>]+)>")

# Default output directories if --output-dir is NOT specified
_DEFAULT_RUNS_DIR = CARIBOU_HOME / "runs"
_DEFAULT_SNIPPET_DIR = _DEFAULT_RUNS_DIR / "snippets"
_DEFAULT_BENCHMARK_LEDGER_PATH = _DEFAULT_RUNS_DIR / f"benchmark_history_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.jsonl"
_CODE_BLOCK_RE = re.compile(r"```(?:python)?[ \t]*\n[\s\S]*?\n```", re.MULTILINE)

# --- Lazily initialize RAG ---
_RAG_SINGLETON = None
def get_rag_client(console: Console) -> RetrievalAugmentedGeneration:
    global _RAG_SINGLETON
    if _RAG_SINGLETON is None:
        console.print("[cyan]Initializing RAG model (this may take a moment)...[/cyan]")
        _RAG_SINGLETON = RetrievalAugmentedGeneration()
    return _RAG_SINGLETON

def _init_paths(output_dir: Optional[Path] = None):
    """Ensure output directories exist before writing."""
    snippet_dir = output_dir / "snippets" if output_dir else _DEFAULT_SNIPPET_DIR
    ledger_path = output_dir / "benchmark_results.jsonl" if output_dir else _DEFAULT_BENCHMARK_LEDGER_PATH
    
    snippet_dir.mkdir(exist_ok=True, parents=True)
    ledger_path.parent.mkdir(exist_ok=True, parents=True)

# --- Helper Functions ---
def detect_delegation(msg: str) -> Optional[str]:
    """Return the *full* command name (e.g. 'delegate_to_coder') if present."""
    m = _DELEG_RE.search(msg)
    return f"delegate_to_{m.group(1)}" if m else None

def detect_rag(msg: str) -> Optional[str]:
    """Return the *partial* RAG command if present."""
    m = _RAG_RE.search(msg)
    return m.group(1) if m else None

def _dump_code_snippet(run_id: str, code: str, output_dir: Optional[Path] = None) -> str:
    """Write <run_id>.py under the appropriate snippets dir and return the relative path."""
    base_output_dir = output_dir if output_dir else _DEFAULT_RUNS_DIR
    snippet_dir = base_output_dir / "snippets"
    snippet_dir.mkdir(exist_ok=True, parents=True) # Ensure it exists
    
    snippet_path = snippet_dir / f"{run_id}.py"
    snippet_path.write_text(code, encoding="utf-8")
    # Return path relative to the main output directory for consistency in the log
    return str(snippet_path.relative_to(base_output_dir))

def _save_benchmark_record(*, run_id: str, results: dict, meta: dict, code: str | None, output_dir: Optional[Path] = None):
    """Append a JSONL record for the benchmark run to the correct ledger file."""
    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run": run_id,
        "dataset": meta.get("name"),
        "results": results,
    }
    if code:
        record["code_path"] = _dump_code_snippet(run_id, code, output_dir)
        
    # Determine the correct ledger path
    ledger_path = output_dir / "benchmark_results.jsonl" if output_dir else _DEFAULT_BENCHMARK_LEDGER_PATH
    
    with ledger_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

def _render_todos(console: Console, todos: List[dict]) -> None:
    """Pretty-print todo list to the console."""
    if not todos:
        console.print("[dim]No TODOs recorded yet.[/dim]")
        return

    table = Table(title="TODOs")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Text")
    table.add_column("Added By", style="green")
    table.add_column("Turn", style="yellow")
    for item in todos:
        status = "[green]✓[/green]" if item.get("status") == "done" else "[yellow]·[/yellow]"
        table.add_row(str(item.get("id")), status, item.get("text", ""), item.get("added_by", ""), str(item.get("turn", "")))
    console.print(table)


def _extract_artifacts_from_msg(msg: str) -> Tuple[List[str], List[str]]:
    """Return (notes, todos) extracted from assistant content."""
    notes: List[str] = []
    todos: List[str] = []

    # Code fences for bulk capture
    fence_patterns = [
        (r"```notes\n([\s\S]*?)```", notes),
        (r"```todo\n([\s\S]*?)```", todos),
        (r"```todos\n([\s\S]*?)```", todos),
    ]
    for pattern, bucket in fence_patterns:
        for m in re.finditer(pattern, msg, flags=re.IGNORECASE):
            content = m.group(1).strip()
            if not content:
                continue
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            for ln in lines:
                bucket.append(ln)

    for raw_line in msg.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("NOTE:"):
            notes.append(line[len("NOTE:"):].strip())
            continue
        if upper.startswith("TODO:"):
            todos.append(line[len("TODO:"):].strip())
            continue
        if line.startswith("- [ ]"):
            todos.append(line[len("- [ ]"):].strip())
            continue
        if line.startswith("- [x]") or line.startswith("- [X]"):
            todos.append(line[len("- [x]"):].strip())

    return notes, todos

def _count_code_blocks(msg: str) -> int:
    """Count fenced code blocks in an assistant message."""
    if not msg:
        return 0
    return len(_CODE_BLOCK_RE.findall(msg))

def _write_session_report(
    console: Console,
    *,
    output_dir: Optional[Path],
    stats: Dict[str, object],
) -> Optional[Path]:
    """Persist session statistics to disk."""
    report_dir = output_dir if output_dir else (_DEFAULT_RUNS_DIR / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"session_report_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
    try:
        report_path.write_text(json.dumps(stats, indent=2))
        console.print(f"[bold green]✓ Session report saved to:[/bold green] {report_path}")
        return report_path
    except Exception as exc:
        console.print(f"[yellow]Warning: Failed to write session report: {exc}[/yellow]")
        return None
    
# --- Core Runner Functions ---
def run_benchmark(
    console: Console,
    mgr: SandboxManager,
    benchmark_module: Path,
    *,
    is_auto: bool,
    output_dir: Optional[Path] = None,
    metadata: Optional[Dict] = None,
    agent_name: Optional[str] = None,
    code_snippet: Optional[str] = None,
) -> str:
    """
    Execute a benchmark module inside the sandbox.
    In auto mode, saves results and returns a result string for the history.
    In interactive mode, prints results to the console.
    """
    console.print(f"\n[bold cyan]Running benchmark module: {benchmark_module.name}[/bold cyan]")
    autometric_base_path = benchmark_module.parent / "AutoMetric.py"
    try:
        with open(autometric_base_path, "r") as f:
            autometric_code = f.read()
        with open(benchmark_module, "r") as f:
            benchmark_code = f.read()
    except FileNotFoundError as e:
        err = f"Benchmark module or AutoMetric.py not found: {e}"
        console.print(f"[red]{err}[/red]")
        return err if is_auto else ""

    code_to_execute = f"# --- Code from AutoMetric.py ---\n{autometric_code}\n# --- Code from {benchmark_module.name} ---\n{benchmark_code}"
    console.print("[cyan]Executing benchmark code...[/cyan]")
    
    try:
        exec_result = mgr.exec_code(code_to_execute, timeout=300)

        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        stdout = exec_result.get("stdout", "")
        result_dict = {}
        try:
            result_dict = json.loads(stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as e:
            console.print(f"[yellow]Warning: Could not parse JSON from stdout: {e}[/yellow]")

        if exec_result.get("status") == "ok" and isinstance(result_dict, dict):
            for key, value in result_dict.items():
                table.add_row(str(key), str(value))
            if is_auto:
                _save_benchmark_record(
                    run_id=f"{benchmark_module.stem}:{agent_name}:{int(time.time())}",
                    results=result_dict,
                    meta=metadata if metadata else {},
                    code=code_snippet,
                    output_dir=output_dir,
                )
        else:
            error_message = exec_result.get("stderr") or "An unknown error occurred."
            table.add_row("Error", error_message)
        
        console.print(table)
        return "Benchmark results:\n" + json.dumps(result_dict)

    except Exception as exc:
        err_msg = f"Benchmark execution failed: {exc}"
        console.print(f"[red]{err_msg}[/red]")
        return err_msg

# --- Helpers to keep memory/history in sync ---
def _extract_possible_actions(agent: Agent) -> List[Dict[str, str]]:
    actions: List[Dict[str, str]] = [{"name": "continue", "detail": "Continue reasoning and generate next step."}]
    if getattr(agent, "is_rag_enabled", False):
        actions.append({
            "name": "query_rag_<topic>",
            "detail": "Retrieve context from knowledge base for a specific topic or function (replace <topic> accordingly).",
        })
    for cmd_name, cmd in getattr(agent, "commands", {}).items():
        detail = f"Delegate via command '{cmd_name}'"
        target = getattr(cmd, "target_agent", None)
        if target:
            detail += f" to agent '{target}'"
        actions.append({"name": cmd_name, "detail": detail})
    return actions


def _code_preview(code: str, max_chars: int = 200, max_lines: int = 4) -> str:
    """Return a short, meaningful preview of a code block."""
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    snippet = "\n".join(lines[:max_lines])
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3] + "..."
    return snippet or "(empty code block)"


def _apply_agent_switch(
    *,
    new_agent_prompt: str,
    analysis_context: str,
    history: List[Dict[str, str]],
    memory_manager: Optional[MemoryManager],
    action_space: Optional[AgentActionSpace],
    new_agent: Agent,
) -> None:
    """
    Ensure both the raw history and the memory manager reflect the current agent prompt.
    Replace the second system message and append a short reminder so identity survives summarization.
    """
    updated_prompt = {"role": "system", "content": new_agent_prompt + "\n\n" + analysis_context}

    if len(history) >= 2 and history[1].get("role") == "system":
        history[1] = updated_prompt
    else:
        history.insert(1, updated_prompt)

    if memory_manager:
        memory_manager.update_system_prompt(updated_prompt["content"])
        reminder = {
            "role": "system",
            "content": "REMINDER: You are now following the above agent system prompt; stay in that role.",
        }
        history.append(reminder)
        memory_manager.add_message(reminder["role"], reminder["content"])

    if action_space:
        action_space.agent_name = new_agent.name
        action_space.set_possible_actions(_extract_possible_actions(new_agent))
        action_space.add_action(
            "agent_switch",
            f"Switched to agent '{new_agent.name}' via delegation.",
            status="ok",
            meta={"prompt_refreshed": True},
        )
        summary_msg = action_space.to_message()
        history.append({"role": "system", "content": summary_msg})
        if memory_manager:
            memory_manager.add_message("system", summary_msg)


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
    model_name: str = "gpt-4.1",
    benchmark_modules: Optional[List[Path]] = None,
    output_dir: Optional[Path] = None, # <-- ADDED output_dir parameter
    make_report: bool = False,
):
    """
    Main driver for agent execution sessions, passing output_dir for benchmark saving.
    """
    from rich.prompt import Prompt
    _init_paths(output_dir)

    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    artifacts_dir = output_dir if output_dir else (_DEFAULT_RUNS_DIR / "session_notes" / run_id)
    artifacts = SessionArtifacts(run_id=run_id, base_dir=artifacts_dir)

    memory_manager: Optional[MemoryManager] = None
    if compress_memory:
        console.print("[bold cyan]🧠 Adaptive context memory is enabled.[/bold cyan]")
        memory_manager = MemoryManager(llm_client=llm_client, model_name=model_name, initial_history=history)
    
    action_space = AgentActionSpace(driver_agent.name)
    action_space.set_possible_actions(_extract_possible_actions(driver_agent))
    action_init_msg = action_space.to_message()
    history.append({"role": "system", "content": action_init_msg})
    if memory_manager:
        memory_manager.add_message("system", action_init_msg)

    # --- Display the initial context provided by the CLI ---
    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if role in ["system", "user"]:
            display(console, role, content)
            
    current_agent = driver_agent
    turns_completed = 0
    code_block_count = 0
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
        
        if memory_manager:
            context_to_send = memory_manager.get_context()
        else:
            context_to_send = history

        try:
            resp = llm_client.chat.completions.create(
                model=model_name,
                messages=context_to_send,
                temperature=0.7,
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

        # --- RAG handling ---
        query_from_re = detect_rag(msg)
        if query_from_re and current_agent.is_rag_enabled:
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
            target_agent_name = current_agent.commands[cmd].target_agent
            new_agent = agent_system.get_agent(target_agent_name)
            if new_agent:
                routing_message = f"🔄 Routing to '{target_agent_name}' via {cmd}"
                current_agent = new_agent
                # Global policy lives in the pinned first system message; skip re-embedding here.
                system_prompt = current_agent.get_full_prompt(None)
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

        code = extract_python_code(msg)
        if code:
            last_code_snippet = code
            console.print("[cyan]Executing code in sandbox…[/cyan]")
            exec_result = sandbox_manager.exec_code(code, timeout=300)
            feedback = format_execute_response(exec_result, output_dir if output_dir else _DEFAULT_RUNS_DIR)
            if memory_manager:
                memory_manager.add_message("system", feedback)
                if exec_result.get("status") == "ok":
                    memory_manager.add_pivotal_code(code)
            action_space.add_action(
                "code_execution",
                f"Ran code block:\n{_code_preview(code)}",
                status=exec_result.get("status"),
            )
            summary_msg = action_space.to_message()
            history.append({"role": "system", "content": summary_msg})
            if memory_manager:
                memory_manager.add_message("system", summary_msg)
            history.append({"role": "assistant", "content": feedback})
            display(console, "code execution result", feedback)

            stderr = exec_result.get('stderr', '')
            if stderr and current_agent.is_rag_enabled:
                func_error_patterns = [
                r"(\w+)\(.*\) missing \d+ required positional argument", # TypeError missing arguments
                r"NameError: name '(\w+)' is not defined",             # NameError
                r"AttributeError: .* has no attribute '(\w+)'",       # AttributeError
                r"'(\w+)\(.*\) got an unexpected keyword argument"         # Unexpected keyword argument
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
                    console.print(f"[yellow]🔍 Incorrect function signature detected: {function_name}, function database search...[/yellow]")
                    rag_client = get_rag_client(console)
                    retrieved_docs = rag_client.retrieve_function(function_name)
                    if retrieved_docs:
                        console.print(f"[green] Query successful - Function signature found. [/green]")
                        feedback += f"\n {function_name} produced an error. The correct function signature for {function_name} is:\n{retrieved_docs}"
                        history.append({"role": "system", "content": feedback})
                        continue
                    else:
                        print(f"Error Query unsuccessful - Function signature does not exist in the current database.")
                 

        if is_auto:
            if benchmark_modules:
                result_str = run_benchmark(
                    console, sandbox_manager, benchmark_modules[0],
                    is_auto=True, metadata={"name": "auto"}, agent_name=current_agent.name, code_snippet=last_code_snippet, output_dir=output_dir
                )
                if memory_manager:
                    memory_manager.add_message("system", result_str)
                history.append({"role": "system", "content": result_str})
                display(console, "user", result_str)
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
                    for bm_module in benchmark_modules:
                        run_benchmark(console, sandbox_manager, bm_module, is_auto=False, output_dir=output_dir)
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
            "session_start": session_start_ts.isoformat(),
            "session_end": session_end_ts.isoformat(),
            "duration_seconds": duration_seconds,
            "max_turns_requested": max_turns if is_auto else None,
            "end_reason": session_end_reason,
        }
        _write_session_report(console, output_dir=output_dir, stats=session_stats)
