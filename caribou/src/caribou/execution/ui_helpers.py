"""
User interface helpers for displaying information in the console.

This module handles:
- Rendering TODO lists in rich tables
- Other console display utilities
"""
from __future__ import annotations

from typing import List

from rich.console import Console
from rich.table import Table


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
        table.add_row(
            str(item.get("id")),
            status,
            item.get("text", ""),
            item.get("added_by", ""),
            str(item.get("turn", ""))
        )
    console.print(table)
