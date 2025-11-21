# caribou/cli/run_cli.py
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import List, Tuple, Optional, cast, Dict, Any
import subprocess
import json
from datetime import datetime

import typer
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from dotenv import load_dotenv

from caribou.config import DEFAULT_AGENT_DIR, ENV_FILE, CARIBOU_HOME

# --------------------------------------------------------------------------------------
# Constants & Package Paths
# --------------------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_AGENTS_DIR = PACKAGE_ROOT / "agents"
PACKAGE_DATASETS_DIR = PACKAGE_ROOT / "datasets"
PACKAGE_AUTO_METRICS_DIR = PACKAGE_ROOT / "auto_metrics"

SANDBOX_DATA_PATH = "/workspace/dataset.h5ad"
SANDBOX_REF_DATA_PATH = "/workspace/reference.h5ad"

# --------------------------------------------------------------------------------------
# Typer App
# --------------------------------------------------------------------------------------

run_app = typer.Typer(
    name="run",
    help="Run an agent system. Prompts for configuration if not provided via flags.",
    no_args_is_help=True,
)

# --------------------------------------------------------------------------------------
# Context Object
# --------------------------------------------------------------------------------------

class AppContext:
    """
    Mutable context shared across subcommands. Built once and attached to ctx.obj.
    """
    def __init__(self) -> None:
        self.console: Console = Console()
        self.agent_system: "AgentSystem" | None = None
        self.driver_agent_name: str | None = None
        self.analysis_context: str | None = None
        self.sandbox_manager: "SandboxManager" | None = None
        self.llm_client: object | None = None
        self.model_name: str | None = None
        self.initial_history: List[dict] | None = None
        self.dataset_path: Path | None = None
        self.reference_dataset_path: Optional[Path] = None
        self.resources: List[Tuple[Path, str]] = []
        self.sandbox_details: dict = {}
        self.compress_memory: bool = False
        self.output_dir: Optional[Path] = None
        self.parent_params: Dict[str, Any] = {}

# --------------------------------------------------------------------------------------
# Prompt Helpers
# --------------------------------------------------------------------------------------

def _prompt_for_driver(console: Console, system: "AgentSystem") -> str:
    """Prompt to select a driver agent from the loaded system."""
    console.print("[bold]Select a driver agent:[/bold]")
    agents = list(system.agents.keys())
    if not agents:
        raise typer.BadParameter("No agents found in the loaded AgentSystem.")
    return Prompt.ask("Enter the name of the driver agent", choices=agents, default=agents[0])

def _prompt_for_benchmark_module(console: Console) -> Optional[Path]:
    """
    Prompt to select a benchmark module (optional). Returns a Path or None.
    Only .py files in PACKAGE_AUTO_METRICS_DIR are listed, excluding sentinels.
    """
    console.print("[bold]Select a benchmark module (optional):[/bold]")

    if not PACKAGE_AUTO_METRICS_DIR.exists():
        console.print(f"[yellow]No benchmark directory found at {PACKAGE_AUTO_METRICS_DIR}.[/yellow]")
        return None

    modules = sorted(
        [
            m for m in PACKAGE_AUTO_METRICS_DIR.glob("*.py")
            if m.name not in {"__init__.py", "AutoMetric.py"}
        ],
        key=lambda p: p.name.lower(),
    )

    if not modules:
        console.print("[yellow]No benchmark modules found.[/yellow]")
        return None

    for i, mod in enumerate(modules, 1):
        console.print(f"  [cyan]{i}[/cyan]: {mod.name}")

    skip_index_display = len(modules) + 1
    console.print(f"  [cyan]{skip_index_display}[/cyan]: Skip")

    choices = [str(i) for i in range(1, skip_index_display + 1)]
    choice_str = Prompt.ask(
        "Enter the number of your choice",
        choices=choices,
        default=str(skip_index_display),
    )
    choice_idx = int(choice_str) - 1

    if choice_idx == len(modules):  # Skip selected
        return None
    return modules[choice_idx]

# --------------------------------------------------------------------------------------
# Core Runner
# --------------------------------------------------------------------------------------
def _setup_and_run_session(
    context: AppContext,
    history: list,
    is_auto: bool,
    max_turns: int,
    benchmark_modules: Optional[List[Path]] = None,
) -> None:
    """
    Start, run, and stop the sandbox session with proper resource handling and logging.
    Manages output saving based on whether context.output_dir is set.
    """
    from caribou.execution.runner import run_agent_session, SandboxManager
    from caribou.agents.AgentSystem import AgentSystem
    from caribou.core.io_helpers import save_chat_history_as_json, save_chat_history_as_notebook
    import shutil

    sandbox_manager = cast(SandboxManager, context.sandbox_manager)
    console = context.console
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if context.output_dir:
        host_output_path = context.output_dir
        host_output_path.mkdir(parents=True, exist_ok=True) # Ensure it exists
        console.print(f"Session outputs will be saved to specified directory: [cyan]{host_output_path}[/cyan]")
    else:
        # Default behavior: create a timestamped temporary directory
        temp_output_dir = CARIBOU_HOME / "runs" / "session_outputs"
        host_output_path = temp_output_dir / f"run_{timestamp}"
        host_output_path.mkdir(parents=True, exist_ok=True)
        console.print(f"Session outputs will be staged in temporary directory: [cyan]{host_output_path}[/cyan]")

    console.print("[cyan]Starting sandbox...[/cyan]")
    
    details = context.sandbox_details
    dataset_path = cast(Path, context.dataset_path)

    if details.get("is_exec_mode") and hasattr(sandbox_manager, "set_data"):
        all_resources = [(dataset_path, SANDBOX_DATA_PATH)] + list(context.resources)
        sandbox_manager.set_data(all_resources, host_output_path)

    if not sandbox_manager.start_container():
        console.print("[bold red]Failed to start sandbox container.[/bold red]")
        raise typer.Exit(1)
    
    try:
        if not details.get("is_exec_mode"):
            copy_cmd = details["copy_cmd"]
            handle = details["handle"]
            copy_cmd(str(dataset_path), f"{handle}:{SANDBOX_DATA_PATH}")
            for host_path, container_path in context.resources:
                copy_cmd(str(host_path), f"{handle}:{container_path}")

        run_agent_session(
            console=console,
            agent_system=cast(AgentSystem, context.agent_system),
            driver_agent=cast(AgentSystem, context.agent_system).get_agent(cast(str, context.driver_agent_name)),
            analysis_context=cast(str, context.analysis_context),
            llm_client=cast(object, context.llm_client),
            sandbox_manager=sandbox_manager,
            history=history,
            is_auto=is_auto,
            max_turns=max_turns,
            benchmark_modules=benchmark_modules,
            model_name=cast(str, context.model_name),
            compress_memory=context.compress_memory,
            output_dir=host_output_path if context.output_dir else None,
        )
    finally:
        auto_save_mode = context.output_dir is not None
        is_exec_mode = details.get("is_exec_mode")

        # --- Gather output file info in a backend-agnostic way ---
        output_files_info = []
        if hasattr(sandbox_manager, "list_output_files"):
            try:
                output_files_info = sandbox_manager.list_output_files() or []
            except Exception as e:
                console.print(f"[yellow]Warning: Could not list output files: {e}[/yellow]")

        # Fallback for exec-mode when outputs already live on host
        if not output_files_info and is_exec_mode and host_output_path.exists():
            output_files_info = [
                {"name": f.name, "size": f"{f.stat().st_size / 1e6:.2f} MB"}
                for f in host_output_path.iterdir()
                if f.is_file()
            ]

        if output_files_info:
            if auto_save_mode:
                if is_exec_mode:
                    console.print(f"[bold green]✓ Session outputs saved in:[/bold green] {host_output_path}")
                else:
                    files_to_copy = [f["name"] for f in output_files_info]
                    sandbox_manager.retrieve_output_files(host_output_path, files_to_copy)
            else:
                console.print("\n[bold]Review generated files:[/bold]")
                from rich.table import Table
                table = Table(title="Generated Output Files")
                table.add_column("Index", style="cyan")
                table.add_column("Filename")
                table.add_column("Size", style="yellow")
                for i, f in enumerate(output_files_info, 1):
                    table.add_row(str(i), f.get("name", ""), f.get("size", ""))
                console.print(table)

                if is_exec_mode:
                    keep = Prompt.ask(
                        f"\nDo you want to keep the output directory and its contents?",
                        choices=["y", "n"],
                        default="y",
                    ).lower() == "y"
                    if not keep and not auto_save_mode:
                        shutil.rmtree(host_output_path, ignore_errors=True)
                        console.print(f"[dim]Removed temporary output directory.[/dim]")
                    else:
                        console.print(f"[bold green]✓ Session outputs saved in:[/bold green] {host_output_path}")
                else:
                    if Prompt.ask("\n[bold]Do you want to save these files?[/bold]", choices=["y", "n"], default="y").lower() == "y":
                        files_to_copy = [f["name"] for f in output_files_info if "name" in f]
                        sandbox_manager.retrieve_output_files(host_output_path, files_to_copy)
                    else:
                        console.print("Generated files will be discarded.")
                        shutil.rmtree(host_output_path, ignore_errors=True)
        else:
            console.print("\n[dim]No output files were generated.[/dim]")
            if host_output_path.exists() and not context.output_dir:
                shutil.rmtree(host_output_path, ignore_errors=True)

        console.print("[cyan]Stopping sandbox...[/cyan]")
        sandbox_manager.stop_container()

        save_chat = auto_save_mode or \
                    (not is_auto and Prompt.ask("\n[bold]Do you want to save the chat history?[/bold]", choices=["y", "n"], default="y").lower() == "y")

        if save_chat:
            log_dir = host_output_path if auto_save_mode else (CARIBOU_HOME / "runs" / "chat_logs")
            log_dir.mkdir(parents=True, exist_ok=True) # Ensure log dir exists, especially if output_dir was used

            if auto_save_mode:
                # Default to notebook format when using --output-dir
                save_format = "notebook"
                file_extension = ".ipynb"
                save_path = log_dir / f"chat_log_{timestamp}{file_extension}"
            else: # Interactive prompt for format and location
                save_format = Prompt.ask("Save chat log format", choices=["json", "notebook"], default="notebook")
                file_extension = ".ipynb" if save_format == "notebook" else ".json"
                default_path = log_dir / f"interactive_chat_{timestamp}{file_extension}"
                save_path_str = Prompt.ask("Enter the save path for the chat log", default=str(default_path))
                save_path = Path(save_path_str).expanduser()

            if save_format == "notebook":
                save_chat_history_as_notebook(console, history, save_path)
            else:
                save_chat_history_as_json(console, history, save_path)

# --------------------------------------------------------------------------------------
# Initialization (shared for callback and subcommands)
# --------------------------------------------------------------------------------------

def initialize_context(
    context: AppContext,
    *,
    blueprint: Optional[Path],
    driver_agent: Optional[str],
    dataset: Optional[Path],
    reference_dataset: Optional[Path],
    resources_dir: Optional[Path],
    llm_backend: Optional[str],
    ollama_host: str,
    sandbox: Optional[str],
    force_refresh: bool,
    compress_memory: bool,
    output_dir: Optional[Path], # <-- ADDED
) -> None:
    """
    Build out the AppContext with all shared resources and configuration.
    Prompts for any missing values not supplied by flags.
    """
    from caribou.agents.AgentSystem import AgentSystem
    from caribou.core.io_helpers import collect_resources, prompt_for_file
    from caribou.core.sandbox_management import init_docker, init_singularity_exec
    from caribou.datasets.czi_datasets import get_datasets_dir
    from openai import OpenAI

    load_dotenv(dotenv_path=ENV_FILE)

    console = context.console
    context.compress_memory = compress_memory
    context.output_dir = output_dir # <-- Store output dir

    # ---- Agent System Blueprint ----
    if blueprint is None:
        blueprint = prompt_for_file(console, DEFAULT_AGENT_DIR, PACKAGE_AGENTS_DIR, ".json", "Agent System Blueprint")
    context.agent_system = AgentSystem.load_from_json(str(blueprint))

    # ---- Driver Agent ----
    if driver_agent is None:
        driver_agent = _prompt_for_driver(console, context.agent_system)
    if driver_agent not in context.agent_system.agents:
        raise typer.BadParameter(f"Driver agent '{driver_agent}' not found.")
    context.driver_agent_name = driver_agent

    # ---- Datasets ----
    if dataset is None:
        dataset = prompt_for_file(console, get_datasets_dir(), PACKAGE_DATASETS_DIR, ".h5ad", "Primary Dataset")
    context.dataset_path = dataset

    if reference_dataset is None and output_dir is None: # Only prompt if not in auto-save mode
        if Prompt.ask("Do you want to add a reference dataset?", choices=["y", "n"], default="n").lower() == "y":
            reference_dataset = prompt_for_file(console, get_datasets_dir(), PACKAGE_DATASETS_DIR, ".h5ad", "Reference Dataset")
    context.reference_dataset_path = reference_dataset

    # ---- Sandbox Backend ----
    if sandbox is None:
        sandbox = Prompt.ask("Choose a sandbox backend", choices=["docker", "singularity"], default="docker")

    console.print(f"[cyan]Initializing sandbox backend: {sandbox}[/cyan]")
    script_dir = Path(__file__).resolve().parent

    if sandbox == "docker":
        manager_class, handle, copy_cmd, exec_endpoint, status_endpoint = init_docker(
            script_dir, subprocess, console, force_refresh=force_refresh
        )
        is_exec_mode = False
    elif sandbox == "singularity":
        manager_class, handle, copy_cmd, exec_endpoint, status_endpoint = init_singularity_exec(
            script_dir, SANDBOX_DATA_PATH, subprocess, console, force_refresh=force_refresh
        )
        is_exec_mode = True
    else:
        raise typer.BadParameter(f"Unknown sandbox type '{sandbox}'.")

    context.sandbox_manager = manager_class()
    context.sandbox_details = {
        "handle": handle,
        "copy_cmd": copy_cmd,
        "is_exec_mode": is_exec_mode,
        "exec_endpoint": exec_endpoint,
        "status_endpoint": status_endpoint,
    }

    # ---- LLM Backend ----
    if llm_backend is None:
        llm_backend = Prompt.ask("Choose an LLM backend", choices=["chatgpt", "ollama", "deepseek"], default="chatgpt")

    console.print(f"[cyan]Initializing LLM backend: {llm_backend}[/cyan]")

    if llm_backend == "chatgpt":
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[bold red]Error: OPENAI_API_KEY not set. Use 'caribou config set-openai-key'.[/bold red]")
            raise typer.Exit(1)
        context.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        context.model_name = "gpt-4o"

    elif llm_backend == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY"):
            console.print("[bold red]Error: DEEPSEEK_API_KEY not set. Use 'caribou config set-deepseek-key'.[/bold red]")
            raise typer.Exit(1)
        context.llm_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        context.model_name = "deepseek-chat"
    elif llm_backend == "ollama":
        if ollama_host == "http://localhost:11434" and output_dir is None:
            ollama_host = Prompt.ask("Enter the Ollama base URL", default="http://localhost:11434")
        from caribou.core.ollama_wrapper import OllamaClient
        context.llm_client = OllamaClient(host=ollama_host)
        context.model_name = "llama3"
    else:
        raise typer.BadParameter(f"Unknown LLM backend '{llm_backend}'.")

    # ---- Additional Resources ----
    context.resources = collect_resources(console, resources_dir) if resources_dir else []
    if context.reference_dataset_path:
        context.resources.append((context.reference_dataset_path, SANDBOX_REF_DATA_PATH))

    # ---- Analysis Context & Initial History ----
    analysis_context_str = f"Primary dataset path: **{SANDBOX_DATA_PATH}**\n"
    if context.reference_dataset_path:
        analysis_context_str += f"Reference dataset path: **{SANDBOX_REF_DATA_PATH}**\n"
    analysis_context_str += "\n**IMPORTANT**: Please save all generated output files (plots, .h5ad, .csv) to the `/workspace/outputs/` directory."
    context.analysis_context = textwrap.dedent(analysis_context_str)
    
    driver = context.agent_system.get_agent(context.driver_agent_name)
    system_prompt = (driver.get_full_prompt(context.agent_system.global_policy) + "\n\n" + context.analysis_context)
    context.initial_history = [
        {"role": "system", "content": f"**GLOBAL POLICY**: {context.agent_system.global_policy}\n"},
        {"role": "system", "content": system_prompt},
    ]

# --------------------------------------------------------------------------------------
# Helpers to merge options
# --------------------------------------------------------------------------------------

def _merge(parent: Dict[str, Any], **child: Any) -> Dict[str, Any]:
    """Merge options, child takes precedence if not None."""
    merged = dict(parent)
    for k, v in child.items():
        if v is not None:
            if isinstance(v, bool) and not v: 
                if k not in parent or parent[k] is False:
                    merged[k] = v
            else:
                 merged[k] = v
    return merged

def _extract_common_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Pick only the shared options for initialize_context."""
    keys = [
        "blueprint", "driver_agent", "dataset", "reference_dataset",
        "resources_dir", "llm_backend", "ollama_host", "sandbox",
        "force_refresh", "compress_memory", "output_dir", # <-- ADDED
    ]
    out = {k: params.get(k, None) for k in keys}
    out["force_refresh"] = bool(out.get("force_refresh", False))
    out["compress_memory"] = bool(out.get("compress_memory", False))
    if out.get("ollama_host") is None: out["ollama_host"] = "http://localhost:11434"
    return out

# --------------------------------------------------------------------------------------
# Callback
# --------------------------------------------------------------------------------------

@run_app.callback(invoke_without_command=True)
def main_run_callback(
    ctx: typer.Context,
    blueprint: Optional[Path] = typer.Option(None, "--blueprint", "-bp", help="Path to the agent system JSON blueprint.", readable=True),
    driver_agent: Optional[str] = typer.Option(None, "--driver-agent", "-d", help="Name of the agent to start with."),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-ds", help="Path to the primary dataset file (.h5ad).", readable=True),
    reference_dataset: Optional[Path] = typer.Option(None, "--reference-dataset", "-ref", help="Path to an optional reference dataset file (.h5ad).", readable=True),
    resources_dir: Optional[Path] = typer.Option(None, "--resources", help="Path to a directory of resource files to mount.", exists=True, file_okay=False),
    llm_backend: Optional[str] = typer.Option(None, "--llm", help="LLM backend: 'chatgpt', 'ollama', or 'deepseek'."),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host", help="Base URL for Ollama backend."),
    sandbox: Optional[str] = typer.Option(None, "--sandbox", help="Sandbox backend: 'docker' or 'singularity'."),
    force_refresh: bool = typer.Option(False, "--force-refresh", help="Force refresh/rebuild of the sandbox environment."),
    compress_memory: bool = typer.Option(False, "--compress-memory", help="Enable episodic summarization to manage long-term context."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to save ALL outputs (files and logs) non-interactively.", file_okay=False, writable=True, resolve_path=True),
) -> None:
    """
    Captures top-level flags and defaults to interactive mode if no subcommand.
    """
    app_context = getattr(ctx, "obj", None)
    if app_context is None:
        app_context = AppContext()
        ctx.obj = app_context

    parent_params = _extract_common_kwargs(locals())
    parent_params.pop("ctx", None)
    app_context.parent_params = parent_params

    if ctx.invoked_subcommand is not None:
        return

    # No subcommand: behave like default "interactive"
    initialize_context(
        app_context,
        **parent_params,
    )

    console = app_context.console
    console.print("\n[bold blue]🚀 Starting Interactive Mode...[/bold blue]")
    benchmark_module = _prompt_for_benchmark_module(console)
    history = list(app_context.initial_history or [])
    history.append({"role": "user", "content": "Beginning interactive session. What is the plan?"})
    _setup_and_run_session(
        context=app_context,
        history=history,
        is_auto=False,
        max_turns=-1,
        benchmark_modules=[benchmark_module] if benchmark_module else None,
    )

# --------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------

@run_app.command("interactive")
def run_interactive(
    ctx: typer.Context,
    # duplicate the shared options to allow flags AFTER the subcommand
    blueprint: Path = typer.Option(None, "--blueprint", "-bp", help="Path to the agent system JSON blueprint.", readable=True),
    driver_agent: str = typer.Option(None, "--driver-agent", "-d", help="Name of the agent to start with."),
    dataset: Path = typer.Option(None, "--dataset", "-ds", help="Path to the primary dataset file (.h5ad).", readable=True),
    reference_dataset: Path = typer.Option(None, "--reference-dataset", "-ref", help="Path to an optional reference dataset file (.h5ad).", readable=True),
    resources_dir: Path = typer.Option(None, "--resources", help="Path to a directory of resource files to mount.", exists=True, file_okay=False),
    llm_backend: str = typer.Option(None, "--llm", help="LLM backend to use: 'chatgpt', 'ollama', or 'deepseek'."),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host", help="Base URL for Ollama backend."),
    sandbox: str = typer.Option(None, "--sandbox", help="Sandbox backend to use: 'docker' or 'singularity'."),
    force_refresh: bool = typer.Option(False, "--force-refresh", help="Force refresh/rebuild of the sandbox environment."),
    compress_memory: bool = typer.Option(False, "--compress-memory", help="Enable episodic summarization to manage long-term context."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to save ALL outputs (files and logs) non-interactively.", file_okay=False, writable=True, resolve_path=True),
) -> None:
    """
    Run the agent system in a manual, interactive chat session.
    Prompts for a benchmark (optional) BEFORE launching the session.
    """
    context: AppContext = cast(AppContext, ctx.obj)
    if context is None:
        context = AppContext()
        ctx.obj = context

    # Merge callback (parent) flags with subcommand flags; subcommand values take precedence when provided
    parent = getattr(context, "parent_params", {})
    merged = _merge(parent, **_extract_common_kwargs(locals()))
    initialize_context(context, **merged)

    console = context.console
    console.print("\n[bold blue]🚀 Starting Interactive Mode...[/bold blue]")
    benchmark_module = _prompt_for_benchmark_module(console)
    history = list(context.initial_history or [])
    history.append({"role": "user", "content": "Beginning interactive session. What is the plan?"})
    _setup_and_run_session(
        context=context,
        history=history,
        is_auto=False,
        max_turns=-1,
        benchmark_modules=[benchmark_module] if benchmark_module else None,
    )

@run_app.command("auto")
def run_auto(
    ctx: typer.Context,
    # shared options so flags AFTER subcommand work
    blueprint: Path = typer.Option(None, "--blueprint", "-bp", help="Path to the agent system JSON blueprint.", readable=True),
    driver_agent: str = typer.Option(None, "--driver-agent", "-d", help="Name of the agent to start with."),
    dataset: Path = typer.Option(None, "--dataset", "-ds", help="Path to the primary dataset file (.h5ad).", readable=True),
    reference_dataset: Path = typer.Option(None, "--reference-dataset", "-ref", help="Path to an optional reference dataset file (.h5ad).", readable=True),
    resources_dir: Path = typer.Option(None, "--resources", help="Path to a directory of resource files to mount.", exists=True, file_okay=False),
    llm_backend: str = typer.Option(None, "--llm", help="LLM backend to use: 'chatgpt', 'ollama', or 'deepseek'."),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host", help="Base URL for Ollama backend."),
    sandbox: str = typer.Option(None, "--sandbox", help="Sandbox backend to use: 'docker' or 'singularity'."),
    force_refresh: bool = typer.Option(False, "--force-refresh", help="Force refresh/rebuild of the sandbox environment."),
    compress_memory: bool = typer.Option(False, "--compress-memory", help="Enable episodic summarization to manage long-term context."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to save ALL outputs (files and logs) non-interactively.", file_okay=False, writable=True, resolve_path=True),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Initial prompt for the auto run."),
    turns: Optional[int] = typer.Option(None, "--turns", "-t", help="Number of turns to run automatically."),
    benchmark_module: Optional[Path] = typer.Option(None, "--benchmark-module", "-bm", help="Path to the auto metric script.", readable=True, exists=True),
) -> None:
    """
    Run the agent system automatically for a set number of turns.
    Prompts for a benchmark (optional) BEFORE launching the session if not provided.
    """
    context: AppContext = cast(AppContext, ctx.obj)
    if context is None:
        context = AppContext()
        ctx.obj = context

    # Merge parent and subcommand flags
    parent = getattr(context, "parent_params", {})
    merged = _merge(parent, **_extract_common_kwargs(locals()))
    initialize_context(context, **merged)

    console = context.console
    
    if context.output_dir is None:
        if prompt is None:
            prompt = Prompt.ask("Enter the initial prompt", default="Analyze this dataset.")
        if turns is None:
            turns = IntPrompt.ask("Enter the number of turns", default=3)
        if benchmark_module is None:
            benchmark_module = _prompt_for_benchmark_module(console)
    else: # If output_dir IS set, require prompt and turns via flags
        if prompt is None:
            console.print("[bold red]Error: --prompt is required when using --output-dir for automated runs.[/bold red]")
            raise typer.Exit(1)
        if turns is None:
            turns = 3 # Default turns if not specified in pure auto mode
            console.print(f"[yellow]--turns not specified, defaulting to {turns}.[/yellow]")

    console.print(f"\n[bold green]🚀 Starting Automated Mode for {turns} turns...[/bold green]")

    history = list(context.initial_history or [])
    history.append({"role": "user", "content": str(prompt)})

    _setup_and_run_session(
        context=context,
        history=history,
        is_auto=True,
        max_turns=int(turns),
        benchmark_modules=[benchmark_module] if benchmark_module else None,
    )
