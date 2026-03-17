from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from typing import Optional
import re
import json
import sys
from pathlib import Path
from typing import Tuple, List
import textwrap
import base64
from datetime import datetime

_FENCE_RE = re.compile(
    r'^[ \t]*```(?:python)?[ \t]*\n'   # opening fence, with optional "python"
    r'([\s\S]*?)'                     # capture all lines
    r'^[ \t]*```[ \t]*$',             # closing fence
    re.MULTILINE
)

def extract_python_code_blocks(txt: str) -> List[str]:
    """Return all fenced code blocks in order, or an empty list if absent.

    Handles:
    * ```python ... ```
    * ``` ... ``` (no language tag)
    * Leading indentation before fences (common in Markdown transcripts)
    """
    if not txt:
        return []
    code_blocks = _FENCE_RE.findall(txt)
    if not code_blocks:
        return []
    cleaned_blocks = []
    for block in code_blocks:
        cleaned = textwrap.dedent(block).strip()
        if cleaned:
            cleaned_blocks.append(cleaned)
    return cleaned_blocks


def extract_python_code(txt: str) -> Optional[str]:
    """Return all fenced code blocks concatenated, or None if absent."""
    code_blocks = extract_python_code_blocks(txt)
    if not code_blocks:
        return None
    full_script = "\n\n".join(code_blocks)
    return full_script if full_script else None


def split_message_by_fence(txt: str) -> List[Tuple[str, str]]:
    """Return ordered (kind, content) parts for text/code fences."""
    if not txt:
        return []
    parts: List[Tuple[str, str]] = []
    last_end = 0
    for match in _FENCE_RE.finditer(txt):
        if match.start() > last_end:
            text = txt[last_end:match.start()].strip()
            if text:
                parts.append(("text", text))
        code = textwrap.dedent(match.group(1)).strip()
        if code:
            parts.append(("code", code))
        last_end = match.end()
    if last_end < len(txt):
        text = txt[last_end:].strip()
        if text:
            parts.append(("text", text))
    return parts

# Rich display wrappers

def _panel(console: Console, title: str, content: str, style: str):
    """A helper to print a consistent rich Panel."""
    # The content is now wrapped in Markdown for better formatting
    from rich.markdown import Markdown
    console.print(Panel(Markdown(content, code_theme="monokai"), title=title, border_style=style, highlight=True))

def display(console: Console, role: str, content: str):
    """
    Formats and displays messages, correctly separating all code blocks from text
    and showing the current agent's name in the panel title.
    """
    role_styles = {
        "system": "dim blue",
        "user": "cyan",
        "assistant": "green",
        "code execution result": "magenta"
    }
    
    # Default title and style
    title = role.upper()
    style = role_styles.get(role, "white")

    # Special handling for dynamic assistant roles like "assistant (coder)"
    if role.startswith("assistant (") and role.endswith(")"):
        style = role_styles["assistant"]
        match = re.search(r'\((.*?)\)', role)
        agent_name = match.group(1) if match else "agent"
        title = f"ASSISTANT ({agent_name})"
    
    # Logic to separate text and code for assistant messages
    if "assistant" in role.lower():
        parts = split_message_by_fence(content)
        if not parts:
            _panel(console, title, content, style)
            return
        for kind, chunk in parts:
            if kind == "text":
                _panel(console, title, chunk, style)
            else:
                console.print(
                    Panel(
                        Syntax(chunk, "python", theme="monokai", line_numbers=True),
                        title=f"{title} (code)",
                        border_style=style,
                    )
                )
    else:
        _panel(console, title, content, style)

def select_dataset(console, dataset_dir) -> Tuple[Path, dict]:
    if not dataset_dir.exists():
        console.print(f"[red]Datasets dir not found: {dataset_dir}[/red]")
        sys.exit(1)
    items = [
        (p, json.loads(p.with_suffix(".json").read_text()))
        for p in dataset_dir.glob("*.h5ad")
        if p.with_suffix(".json").exists()
    ]
    if not items:
        console.print("[red]No datasets found.[/red]")
        sys.exit(1)
    tbl = Table(title="Datasets")
    tbl.add_column("Idx", justify="right")
    tbl.add_column("Name")
    tbl.add_column("Cells", justify="right")
    for i, (p, meta) in enumerate(items, 1):
        tbl.add_row(str(i), meta.get("dataset_title", p.stem), str(meta.get("cell_count", "?")))
    console.print(tbl)
    idx = int(Prompt.ask("Choose index", choices=[str(i) for i in range(1, len(items) + 1)])) - 1
    return items[idx]

def get_initial_prompt(console) -> str:
    console.print("[bold cyan]Enter the initial user prompt (Ctrl+D to finish):[/bold cyan]")
    try:
        txt = sys.stdin.read().strip()
    except EOFError:
        txt = ""
    if not txt:
        console.print("[red]Empty prompt – aborting.[/red]")
        sys.exit(1)
    return txt

def collect_resources(console, sandbox_sources_dir) -> List[Tuple[Path, str]]:
    console.print("\n[bold cyan]Optional: paths to bind inside sandbox[/bold cyan] (blank line to finish)")
    res: List[Tuple[Path, str]] = []
    while True:
        p = Prompt.ask("Path", default="").strip()
        if not p:
            break
        path = Path(p).expanduser().resolve()
        if not path.exists():
            console.print(f"[yellow]Path does not exist: {path}[/yellow]")
            continue
        res.append((path, f"{sandbox_sources_dir}/{path.name}"))
    return res

def load_bp_json(console) -> Path:
    """
    Try to find a blueprint JSON file from common locations.
    If multiple are found, prompt user to choose or enter manual path.
    """
    search_paths = [
        Path.home() / "Caribou" / "cli" / "agents",
        Path.cwd() / "cli" / "agents",
        Path.cwd() / "agents"
    ]

    # Search for JSON files in known paths
    for path in search_paths:
        if path.is_dir():
            json_files = list(path.rglob("*.json"))
            if json_files:
                choices = [f.name for f in json_files]
                choices.append("manual")

                choice = Prompt.ask(
                    "Select a blueprint JSON file or choose 'manual' to enter path",
                    choices=choices,
                    default="system_blueprint.json"
                )
                if choice == "manual":
                    break  # jump to manual path section
                selected = path / choice
                if selected.exists():
                    return selected

    # Manual fallback
    user_path = Prompt.ask(
        "Please provide absolute or relative path to blueprint JSON",
        default="~/system_blueprint.json"
    )
    bp = Path(user_path).expanduser()

    if not bp.exists():
        console.print(f"[red]Blueprint file not found at: {bp}[/red]")
        sys.exit(1)

    return bp

def format_execute_response(resp: dict, output_dir) -> str:
    lines = ["Code execution result:"]
    if resp.get("final_status") != "ok":
        lines.append(f"[status: {resp.get('status')}]")
    #if the key outputs in in resp we get the second dictionary
    if 'outputs' in resp:
        outputs = resp['outputs']
        resp = outputs[1]
    stdout, stderr, text = resp.get("stdout", ""), resp.get("stderr", ""), resp.get("text", "")
    error = False
    if resp.get("type") == "error":
        error = resp.get("evalue", "")
        traceback = resp.get("traceback", "")
        if traceback:
            error += "\n" + traceback
    if text and not error:
        lines += ["--- TEXT ---", text[:1500]]
    if stdout:
        lines += ["--- STDOUT ---", stdout[:1500]]
    if stderr:
        lines += ["--- STDERR ---", stderr[:1500]]
    if error:
        lines += ["--- ERROR ---", error[:1500]]
    img_paths = []
    for b64 in resp.get("images", []):
        fname = output_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        fname.parent.mkdir(exist_ok=True, parents=True)
        with open(fname, "wb") as f:
            f.write(base64.b64decode(b64))
        img_paths.append(str(fname))
    if img_paths:
        lines.append("Saved images: " + ", ".join(img_paths))
    return "\n".join(lines)

def prompt_for_file(
    console: Console, user_dir: Path, package_dir: Path, extension: str, prompt_title: str
) -> Path:
    """
    Generic helper to find files, or prompt for a custom path if none are suitable.
    """
    console.print(f"[bold]Select {prompt_title}:[/bold]")
    found_files = []
    seen_filenames = set()
    if user_dir.exists():
        for file_path in sorted(list(user_dir.glob(f"**/*{extension}"))):
            if file_path.name not in seen_filenames:
                found_files.append({"path": file_path, "label": "User"})
                seen_filenames.add(file_path.name)
    if package_dir.exists():
        for file_path in sorted(list(package_dir.glob(f"**/*{extension}"))):
            if file_path.name not in seen_filenames:
                found_files.append({"path": file_path, "label": "Package"})
                seen_filenames.add(file_path.name)
    
    # Display any found files
    for i, file_info in enumerate(found_files, 1):
        console.print(f"  [cyan]{i}[/cyan]: {file_info['path'].name} [yellow]({file_info['label']})[/yellow]")

    # Add the custom path option at the end of the list
    custom_path_option_index = len(found_files) + 1
    console.print(f"  [cyan]{custom_path_option_index}[/cyan]: Provide a custom file path...")
    
    choices = [str(i) for i in range(1, custom_path_option_index + 1)]
    choice_str = Prompt.ask("Enter the number of your choice", choices=choices)
    choice_idx = int(choice_str) - 1

    if choice_idx == len(found_files):  # User selected the "Provide custom path" option
        while True:
            custom_path_str = Prompt.ask(f"Enter the path to your {extension} file").strip()
            if not custom_path_str:
                console.print("[yellow]Path cannot be empty. Please try again.[/yellow]")
                continue
            
            custom_path = Path(custom_path_str).expanduser().resolve()
            if custom_path.exists() and custom_path.is_file():
                return custom_path
            else:
                console.print(f"[bold red]Error: File not found at '{custom_path}'. Please check the path and try again.[/bold red]")
    else:  # User selected a pre-listed file
        return found_files[choice_idx]['path']

def save_chat_history_as_json(console: Console, history: list, file_path: Path):
    """Saves the interactive chat history to a user-specified JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        console.print(f"\n[bold green]✓ Chat history saved to:[/bold green] {file_path}")
    except Exception as e:
        console.print(f"\n[bold red]Error saving chat history: {e}[/bold red]")

def save_chat_history_as_notebook(console: Console, history: list, file_path: Path):
    """Parses an CARIBOU chat log and converts it into a Jupyter Notebook (.ipynb)."""
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": { "name": "python", "version": "3.11" }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    for message in history:
        if message.get("role") and "assistant" in message.get("role", ""):
            content = message.get("content", "")
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
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        console.print(f"\n[bold green]✓ Notebook saved to:[/bold green] {file_path}")
    except Exception as e:
        console.print(f"\n[bold red]Error saving notebook: {e}[/bold red]")
