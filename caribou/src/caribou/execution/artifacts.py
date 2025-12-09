"""
Session artifact helpers for persisting agent notes and TODOs during runs.

Files created per run_id (under the provided base_dir):
  - notes.md      : human-readable notes with timestamps and agent names
  - notes.ndjson  : structured append-only note log
  - todos.json    : structured todo list with status

Use append-only writes for notes to reduce loss if the run stops unexpectedly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class TodoItem:
    id: int
    text: str
    status: str  # "open" | "done"
    added_by: str
    turn: int
    created_at: str
    completed_at: Optional[str] = None


class SessionArtifacts:
    """Persist notes and TODOs for a single run."""

    def __init__(self, run_id: str, base_dir: Path) -> None:
        self.run_id = run_id
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.notes_md_path = self.base_dir / "notes.md"
        self.notes_ndjson_path = self.base_dir / "notes.ndjson"
        self.todos_json_path = self.base_dir / "todos.json"

        self._todos: List[TodoItem] = self._load_todos()

    # --- Notes ---
    def add_note(self, text: str, author: str, turn: int) -> None:
        timestamp = _utc_now()
        header = f"## Turn {turn} - {author} @ {timestamp}\n"
        content = text.strip()

        with self.notes_md_path.open("a", encoding="utf-8") as fh:
            fh.write(header)
            fh.write(content + "\n\n")

        note_record = {
            "run": self.run_id,
            "ts": timestamp,
            "turn": turn,
            "author": author,
            "note": content,
        }
        with self.notes_ndjson_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(note_record) + "\n")

    # --- Todos ---
    def add_todo(self, text: str, author: str, turn: int) -> TodoItem:
        next_id = (max((t.id for t in self._todos), default=0) + 1)
        item = TodoItem(
            id=next_id,
            text=text.strip(),
            status="open",
            added_by=author,
            turn=turn,
            created_at=_utc_now(),
        )
        self._todos.append(item)
        self._write_todos()
        return item

    def complete_todo(self, todo_id: int, status: str = "done") -> Optional[TodoItem]:
        for t in self._todos:
            if t.id == todo_id:
                t.status = status
                t.completed_at = _utc_now()
                self._write_todos()
                return t
        return None

    def list_todos(self) -> List[TodoItem]:
        return list(self._todos)

    # --- Internal IO helpers ---
    def _load_todos(self) -> List[TodoItem]:
        if not self.todos_json_path.exists():
            return []
        try:
            raw = json.loads(self.todos_json_path.read_text(encoding="utf-8"))
            return [TodoItem(**item) for item in raw]
        except Exception:
            # If file is corrupt, start fresh but do not crash the run.
            return []

    def _write_todos(self) -> None:
        payload = [asdict(t) for t in self._todos]
        # Write atomically by using a temp file
        tmp_path = self.todos_json_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.todos_json_path)
