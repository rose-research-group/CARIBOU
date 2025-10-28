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
):
    """
    Main driver for agent execution sessions, passing output_dir for benchmark saving.
    """
    from rich.prompt import Prompt
    _init_paths(output_dir)

    memory_manager: Optional[MemoryManager] = None
    if compress_memory:
        console.print("[bold cyan]🧠 Adaptive context memory is enabled.[/bold cyan]")
        memory_manager = MemoryManager(llm_client=llm_client, model_name=model_name, initial_history=history)
    
    # --- Display the initial context provided by the CLI ---
    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if role in ["system", "user"]:
            display(console, role, content)
            
    current_agent = driver_agent
    turn = 0
    last_code_snippet: str | None = None

    
    while True:
        turn += 1
        if is_auto and turn > max_turns:
            console.print("[bold green]Auto run finished: Max turns reached.[/bold green]")
            break

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
            break
        
        history.append({"role": "assistant", "content": msg})
        if memory_manager:
            memory_manager.add_message("assistant", msg)
        display(console, f"assistant ({current_agent.name})", msg)  

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
                system_prompt = (current_agent.get_full_prompt(agent_system.global_policy) + "\n\n" + analysis_context)
                console.print(f"[yellow]{routing_message}[/yellow]")
                history.append({"role": "assistant", "content": f"🔄 Routing to **{target_agent_name}** (command `{cmd}`)"})
                if memory_manager:
                    memory_manager.add_message("assistant", routing_message)
                    memory_manager.update_system_prompt(system_prompt)
                # We replace the last system prompt with the new one for the new agent
                history.insert(1, {"role": "system", "content": system_prompt}) # agent prompt is second message
                # Remove the old system prompt to avoid confusion
                if len(history) > 2 and history[2].get("role") == "system":
                    history.pop(2)

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
            console.print(f"[yellow]Auto-continuing... {turn}/{max_turns} turns complete.[/yellow]")
        else:
            while True:
                prompt_text = "\n[bold]Next message ('benchmark' to run selected benchmark, 'exit' to quit)[/bold]"
                try:
                    user_input = Prompt.ask(prompt_text, default="").strip()
                except (EOFError, KeyboardInterrupt):
                    user_input = "exit"

                if user_input.lower() in {"exit", "quit"}:
                    console.print("[bold yellow]Exiting session.[/bold yellow]")
                    return

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
