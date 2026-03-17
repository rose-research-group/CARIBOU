# caribou/cli/utils_cli.py
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Import from the central config to know where chat logs are stored by default
from caribou.config import CARIBOU_HOME
from caribou.core.io_helpers import split_message_by_fence

utils_app = typer.Typer(
    name="utils",
    help="Utility commands for managing CARIBOU artifacts like chat logs.",
    no_args_is_help=True
)

console = Console()
LOG_DIR = CARIBOU_HOME / "runs" / "chat_logs"


@utils_app.command("refresh-sif")
def refresh_sif() -> None:
    """
    Force re-download the Singularity SIF used by CARIBOU.
    """
    try:
        from caribou.sandbox import benchmarking_sandbox_management_singularity as sing
    except Exception as exc:
        console.print(f"[bold red]Error importing Singularity manager: {exc}[/bold red]")
        raise typer.Exit(1)

    console.print("[yellow]Refreshing Singularity sandbox SIF...[/yellow]")
    if not sing.pull_sif_if_needed(force_pull=True):
        console.print("[bold red]Failed to refresh the Singularity SIF.[/bold red]")
        raise typer.Exit(1)
    console.print("[bold green]Singularity SIF refreshed.[/bold green]")

def _convert_history_to_notebook(history_path: Path, output_path: Path):
    """
    Parses an CARIBOU chat log and converts it into a Jupyter Notebook (.ipynb).
    """
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error: Could not read or parse the history file at {history_path}.[/bold red]\n{e}")
        raise typer.Exit(1)

    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11" # This can be made more dynamic if needed
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    for message in history:
        role = message.get("role")
        content = message.get("content", "")
        
        # We are primarily interested in the agent's responses
        if role and "assistant" in role:
            parts = split_message_by_fence(content)
            for kind, part in parts:
                if kind == "code":
                    cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": part,
                    }
                else:
                    cell = {"cell_type": "markdown", "metadata": {}, "source": part}
                notebook["cells"].append(cell)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        console.print(f"[bold green]✓ Successfully converted chat log to notebook:[/bold green] {output_path}")
    except Exception as e:
        console.print(f"[bold red]Error writing notebook file: {e}[/bold red]")
        raise typer.Exit(1)

@utils_app.command("convert-to-notebook")
def convert_to_notebook(
    chat_log: Path = typer.Argument(
        ...,
        help="Path to the interactive chat log JSON file to convert.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the output .ipynb file. Defaults to the same name as the input file.",
        writable=True,
        resolve_path=True,
    ),
):
    """
    Converts an CARIBOU interactive chat log into an executable Jupyter Notebook.
    
    This command parses the JSON log file, extracts all Python code blocks generated
    by the assistant, and arranges them into code cells. The explanatory text
    between code blocks is converted into markdown cells, creating a clean,
    reproducible protocol of the agent session.
    """
    if not chat_log.name.startswith("interactive_chat_"):
        console.print(f"[yellow]Warning: The input file '{chat_log.name}' does not look like a standard CARIBOU chat log.[/yellow]")

    output_path = output
    if output_path is None:
        # Default to the same name as the input file, but with an .ipynb extension
        output_path = chat_log.with_suffix(".ipynb")
    
    # Ensure the output path has the correct extension
    if output_path.suffix != ".ipynb":
        output_path = output_path.with_suffix(".ipynb")

    console.print(f"Converting [cyan]{chat_log.name}[/cyan] to Jupyter Notebook...")
    _convert_history_to_notebook(chat_log, output_path)
