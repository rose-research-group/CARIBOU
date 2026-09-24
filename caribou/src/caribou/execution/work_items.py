"""Enforced, Git-backed work items shared by CLI and web execution.

The public identifier is a small per-run integer.  Git commit identifiers are
kept as provenance only, so agents and operators never need to pass SHAs.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


WORK_ITEM_SCHEMA = "caribou.work_item.v2"
WORK_ITEM_INDEX_SCHEMA = "caribou.work_item_index.v2"
WORK_ITEM_STATUSES = ("Backlog", "Ready", "In progress", "In review", "Done")
QcMode = Literal["optional", "required"]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


class WorkItemError(ValueError):
    """Base class for user-visible work-item failures."""


class WorkItemNotFound(WorkItemError):
    pass


class WorkItemConflict(WorkItemError):
    pass


class WorkItemPersistenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkItemPolicy:
    qc_mode: QcMode = "optional"
    stall_turns: int = 4
    stall_halt_turns: int = 8

    @classmethod
    def from_dict(cls, raw: object) -> "WorkItemPolicy":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("work_item_policy must be an object")
        qc_mode = raw.get("qc_mode", "optional")
        if qc_mode not in {"optional", "required"}:
            raise ValueError(
                "work_item_policy.qc_mode must be 'optional' or 'required'"
            )
        stall_turns = raw.get("stall_turns", 4)
        stall_halt_turns = raw.get("stall_halt_turns", 8)
        if not isinstance(stall_turns, int) or stall_turns < 1:
            raise ValueError("work_item_policy.stall_turns must be a positive integer")
        if not isinstance(stall_halt_turns, int) or stall_halt_turns <= stall_turns:
            raise ValueError(
                "work_item_policy.stall_halt_turns must be a positive integer "
                "greater than stall_turns"
            )
        return cls(
            qc_mode=qc_mode,
            stall_turns=stall_turns,
            stall_halt_turns=stall_halt_turns,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qc_mode": self.qc_mode,
            "stall_turns": self.stall_turns,
            "stall_halt_turns": self.stall_halt_turns,
        }


@dataclass(frozen=True)
class WorkItemCommand:
    name: Literal[
        "open_work_item", "close_work_item", "list_work_items", "read_work_item"
    ]
    item_id: Optional[int] = None
    title: str = ""
    body: str = ""
    completion_summary: str = ""


@dataclass(frozen=True)
class WorkItemCommandResult:
    feedback: str
    success: bool
    changed_item: Optional[Dict[str, Any]] = None


def parse_work_item_command(message: str) -> Optional[WorkItemCommand]:
    """Parse one exact command-only assistant message.

    Quoting follows shell-like rules solely for tokenization; nothing is ever
    passed through a shell.  Messages containing prose, code, or extra lines do
    not become commands accidentally.
    """
    if not message or len(message.splitlines()) != 1:
        return None
    try:
        tokens = shlex.split(message.strip())
    except ValueError:
        return None
    if not tokens:
        return None
    name = tokens[0]
    if name == "list_work_items" and len(tokens) == 1:
        return WorkItemCommand(name="list_work_items")
    if name == "read_work_item" and len(tokens) == 2 and tokens[1].isdigit():
        return WorkItemCommand(name="read_work_item", item_id=int(tokens[1]))
    if name == "open_work_item" and len(tokens) == 3:
        title, body = tokens[1], tokens[2]
        if title.strip() and body.strip():
            return WorkItemCommand(name="open_work_item", title=title, body=body)
    if name == "close_work_item" and len(tokens) == 3 and tokens[1].isdigit():
        if tokens[2].strip():
            return WorkItemCommand(
                name="close_work_item",
                item_id=int(tokens[1]),
                completion_summary=tokens[2],
            )
    return None


class WorkItemStore:
    """A per-run, commit-backed work-item ledger."""

    _lock_registry_guard = threading.Lock()
    _locks: Dict[str, threading.RLock] = {}

    def __init__(
        self,
        work_items_dir: Path,
        *,
        session_id: str,
        policy: WorkItemPolicy,
        origin_run_id: Optional[str] = None,
    ) -> None:
        self.root = Path(work_items_dir)
        self.items_dir = self.root / "items"
        self.session_id = session_id
        self.origin_run_id = origin_run_id or session_id
        self.policy = policy
        # Invalidated by every mutating call (`_commit`); safe because all
        # reads and writes happen under `self._lock`, and a store is meant to
        # be held as a per-session singleton (see WS-0) rather than
        # constructed fresh per call, which would make this cache observe
        # stale state written by another instance.
        self._index_cache: Optional[Dict[str, Any]] = None
        lock_key = str(self.root.resolve())
        with self._lock_registry_guard:
            self._lock = self._locks.setdefault(lock_key, threading.RLock())
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self.items_dir.mkdir(parents=True, exist_ok=True)
            if not (self.root / ".git").exists():
                self._git("init", "--quiet")

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.root,
                check=check,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "GIT_AUTHOR_NAME": "CARIBOU",
                    "GIT_AUTHOR_EMAIL": "caribou@localhost",
                    "GIT_COMMITTER_NAME": "CARIBOU",
                    "GIT_COMMITTER_EMAIL": "caribou@localhost",
                },
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            raise WorkItemPersistenceError(
                f"work-item Git operation failed: {detail}"
            ) from exc

    def _has_head(self) -> bool:
        result = self._git("rev-parse", "--verify", "HEAD", check=False)
        return result.returncode == 0

    def _read_head_json(self, relative_path: str, default: Any = None) -> Any:
        if not self._has_head():
            return default
        result = self._git("show", f"HEAD:{relative_path}", check=False)
        if result.returncode != 0:
            return default
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise WorkItemPersistenceError(
                f"committed work-item data is invalid: {relative_path}"
            ) from exc

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _restore_committed_tree(self) -> None:
        if self._has_head():
            self._git("restore", "--source=HEAD", "--staged", "--worktree", "--", ".")

    def _commit(self, message: str, index: Dict[str, Any], item: Dict[str, Any]) -> str:
        self._restore_committed_tree()
        self._atomic_json(self.root / "index.json", index)
        self._atomic_json(self.items_dir / f"{item['id']}.json", item)
        self._git("add", "index.json", f"items/{item['id']}.json")
        self._git("commit", "--quiet", "-m", message)
        self._index_cache = None
        return self._git("rev-parse", "HEAD").stdout.strip()

    def _index(self) -> Dict[str, Any]:
        if self._index_cache is None:
            self._index_cache = self._read_head_json(
                "index.json",
                {
                    "schema_version": WORK_ITEM_INDEX_SCHEMA,
                    "session_id": self.session_id,
                    "origin_run_id": self.origin_run_id,
                    "forked_from_session_id": None,
                    "qc_mode": self.policy.qc_mode,
                    "stall_turns": self.policy.stall_turns,
                    "stall_halt_turns": self.policy.stall_halt_turns,
                    "next_id": 0,
                    "items": [],
                },
            )
        index = dict(self._index_cache)
        # Reconcile the full committed policy, not just qc_mode: a partial
        # reconciliation would silently reset any field it skips back to
        # WorkItemPolicy()'s defaults on every _index() call, discarding
        # whatever the blueprint or a prior commit set.
        committed_mode = index.get("qc_mode")
        if committed_mode in {"optional", "required"}:
            self.policy = WorkItemPolicy(
                qc_mode=committed_mode,
                stall_turns=index.get("stall_turns", self.policy.stall_turns),
                stall_halt_turns=index.get(
                    "stall_halt_turns", self.policy.stall_halt_turns
                ),
            )
        return index

    def _item(self, item_id: int) -> Dict[str, Any]:
        item = self._read_head_json(f"items/{item_id}.json")
        if item is None:
            raise WorkItemNotFound(f"work item {item_id} was not found")
        return item

    @staticmethod
    def _summary(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: item.get(key)
            for key in (
                "id",
                "title",
                "status",
                "owner",
                "created_turn",
                "created_at",
                "completed_turn",
                "completed_at",
            )
        }

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            index = self._index()
            return [dict(value) for value in index.get("items", [])]

    def read(self, item_id: int) -> Dict[str, Any]:
        with self._lock:
            item = dict(self._item(item_id))
            log = self._git(
                "log", "--format=%H", "--", f"items/{item_id}.json", check=False
            )
            commits = [line for line in log.stdout.splitlines() if line]
            item["latest_commit"] = commits[0] if commits else None
            item["opening_commit"] = commits[-1] if commits else None
            return item

    def _update_index(self, index: Dict[str, Any], item: Dict[str, Any]) -> None:
        summaries = [
            summary
            for summary in index.get("items", [])
            if summary.get("id") != item["id"]
        ]
        summaries.append(self._summary(item))
        summaries.sort(key=lambda value: int(value["id"]))
        index["items"] = summaries

    def open(self, title: str, body: str, owner: str, turn: int) -> Dict[str, Any]:
        if not title.strip() or not body.strip():
            raise WorkItemConflict("work-item title and body must be non-empty")
        with self._lock:
            index = self._index()
            item_id = int(index.get("next_id", 0))
            timestamp = utc_now()
            item = {
                "schema_version": WORK_ITEM_SCHEMA,
                "session_id": index.get("session_id", self.session_id),
                "origin_run_id": index.get("origin_run_id", self.origin_run_id),
                "id": item_id,
                "title": title.strip(),
                "body": body.strip(),
                "status": "In progress",
                "owner": owner,
                "created_turn": turn,
                "created_at": timestamp,
                "completion_summary": None,
                "closed_turn": None,
                "closed_at": None,
                "completed_turn": None,
                "completed_at": None,
                "transitions": [
                    {
                        "kind": "opened",
                        "actor": owner,
                        "turn": turn,
                        "timestamp": timestamp,
                        "from_status": None,
                        "to_status": "In progress",
                        "from_owner": None,
                        "to_owner": owner,
                    }
                ],
                "reviews": [],
                "notes": [],
            }
            index["next_id"] = item_id + 1
            self._update_index(index, item)
            self._commit(f"work-item {item_id}: opened", index, item)
            return self.read(item_id)

    def close(
        self, item_id: int, summary: str, actor: str, turn: int
    ) -> Dict[str, Any]:
        if not summary.strip():
            raise WorkItemConflict("completion summary must be non-empty")
        with self._lock:
            index = self._index()
            item = self._item(item_id)
            if item["owner"] != actor:
                raise WorkItemConflict(
                    f"work item {item_id} is owned by {item['owner']}, not {actor}"
                )
            if item["status"] != "In progress":
                raise WorkItemConflict(
                    f"work item {item_id} cannot close from {item['status']}"
                )
            timestamp = utc_now()
            destination = "Done" if self.policy.qc_mode == "optional" else "In review"
            item["status"] = destination
            item["completion_summary"] = summary.strip()
            item["closed_turn"] = turn
            item["closed_at"] = timestamp
            if destination == "Done":
                item["completed_turn"] = turn
                item["completed_at"] = timestamp
            item["transitions"].append(
                {
                    "kind": "closed",
                    "actor": actor,
                    "turn": turn,
                    "timestamp": timestamp,
                    "from_status": "In progress",
                    "to_status": destination,
                    "from_owner": actor,
                    "to_owner": actor,
                }
            )
            self._update_index(index, item)
            self._commit(f"work-item {item_id}: closed to {destination}", index, item)
            return self.read(item_id)

    def transfer(
        self, item_id: int, from_owner: str, to_owner: str, turn: int
    ) -> Dict[str, Any]:
        with self._lock:
            index = self._index()
            item = self._item(item_id)
            if item["owner"] != from_owner:
                raise WorkItemConflict(
                    f"work item {item_id} is owned by {item['owner']}, not {from_owner}"
                )
            if item["status"] == "Done":
                raise WorkItemConflict(f"work item {item_id} is already Done")
            timestamp = utc_now()
            item["owner"] = to_owner
            item["transitions"].append(
                {
                    "kind": "reassigned",
                    "actor": from_owner,
                    "turn": turn,
                    "timestamp": timestamp,
                    "from_status": item["status"],
                    "to_status": item["status"],
                    "from_owner": from_owner,
                    "to_owner": to_owner,
                }
            )
            self._update_index(index, item)
            self._commit(
                f"work-item {item_id}: reassigned {from_owner} to {to_owner}",
                index,
                item,
            )
            return self.read(item_id)

    def transfer_active(
        self, from_owner: str, to_owner: str, turn: int
    ) -> List[Dict[str, Any]]:
        """Hand off every in-progress item the delegating agent owns.

        This is the harness-side ownership rule: an agent delegates with a bare
        ``delegate_to_<agent>`` command and every active work item it owns
        follows to the receiving agent automatically. Agents never need to know
        or emit item ids. Returns the updated items (possibly empty).
        """
        transferred: List[Dict[str, Any]] = []
        for summary in self.list():
            if summary.get("owner") != from_owner:
                continue
            if summary.get("status") != "In progress":
                continue
            transferred.append(
                self.transfer(int(summary["id"]), from_owner, to_owner, turn)
            )
        return transferred

    def note(self, item_id: int, actor: str, turn: int, text: str) -> Dict[str, Any]:
        """Record a non-status-changing note on an item (e.g. a stall halt)."""
        with self._lock:
            index = self._index()
            item = self._item(item_id)
            timestamp = utc_now()
            item.setdefault("notes", []).append(
                {"actor": actor, "turn": turn, "timestamp": timestamp, "text": text}
            )
            self._update_index(index, item)
            self._commit(f"work-item {item_id}: note", index, item)
            return self.read(item_id)

    def blocking_for_owner(self, owner: str) -> List[Dict[str, Any]]:
        return [
            item
            for item in self.list()
            if item.get("owner") == owner and item.get("status") != "Done"
        ]

    @staticmethod
    def last_transition_turn(item: Dict[str, Any]) -> int:
        """The turn of an item's most recent transition, derived on read.

        Not persisted as a separate field — it would duplicate
        ``transitions[-1]["turn"]`` and could drift from it.
        """
        transitions = item.get("transitions") or []
        if not transitions:
            return int(item.get("created_turn", 0))
        return int(transitions[-1]["turn"])

    def copy_to(
        self, dst_dir: Path, *, child_session_id: str, forked_from_session_id: str
    ) -> "WorkItemStore":
        """Copy this store's full on-disk state (including `.git` history)
        to `dst_dir` for a forked session, recording provenance in the
        child's index.

        Held under `self._lock` for the whole copy so a concurrent write to
        this store cannot be caught mid-commit; the working tree always
        matches HEAD when unlocked (commits are atomic via `os.replace`), so
        a lock-protected `shutil.copytree` yields a valid child `.git`.
        """
        dst_dir = Path(dst_dir)
        with self._lock:
            if dst_dir.exists():
                raise WorkItemConflict(f"fork destination already exists: {dst_dir}")
            shutil.copytree(self.root, dst_dir)
        child = WorkItemStore(
            dst_dir,
            session_id=child_session_id,
            policy=self.policy,
            origin_run_id=self.origin_run_id,
        )
        with child._lock:
            index = child._index()
            index["session_id"] = child_session_id
            index["forked_from_session_id"] = forked_from_session_id
            child._atomic_json(child.root / "index.json", index)
            for item_path in child.items_dir.glob("*.json"):
                item = json.loads(item_path.read_text(encoding="utf-8"))
                item["session_id"] = child_session_id
                child._atomic_json(item_path, item)
            child._git("add", "index.json", "items")
            child._git(
                "commit",
                "--quiet",
                "-m",
                f"fork: recorded from session {forked_from_session_id}",
            )
            child._index_cache = None
        return child

    def review_diff(self, item_id: int) -> str:
        with self._lock:
            item = self.read(item_id)
            opening, latest = item.get("opening_commit"), item.get("latest_commit")
            if not opening or not latest or opening == latest:
                return (
                    self._git(
                        "show", "--format=", latest, "--", f"items/{item_id}.json"
                    ).stdout
                    if latest
                    else ""
                )
            return self._git(
                "diff", opening, latest, "--", f"items/{item_id}.json"
            ).stdout

    def record_review(
        self,
        item_id: int,
        *,
        evaluator: str,
        turn: int,
        verdict: Optional[Literal["approve", "reject"]],
        assessment: str,
        provider_receipt: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            index = self._index()
            item = self._item(item_id)
            if evaluator == item["owner"]:
                raise WorkItemConflict(
                    f"work item {item_id} cannot be reviewed by its own owner "
                    f"({evaluator})"
                )
            if self.policy.qc_mode == "required" and item["status"] != "In review":
                raise WorkItemConflict(
                    f"required-QC review needs In review status, found {item['status']}"
                )
            if self.policy.qc_mode == "optional" and item["status"] != "Done":
                raise WorkItemConflict(
                    f"optional-QC review needs Done status, found {item['status']}"
                )
            timestamp = utc_now()
            review = {
                "evaluator": evaluator,
                "turn": turn,
                "timestamp": timestamp,
                "verdict": verdict,
                "assessment": assessment,
                "provider_receipt": dict(provider_receipt or {}),
                "error": error,
            }
            item["reviews"].append(review)
            if not error and self.policy.qc_mode == "required":
                previous = item["status"]
                destination = "Done" if verdict == "approve" else "In progress"
                item["status"] = destination
                if destination == "Done":
                    item["completed_turn"] = turn
                    item["completed_at"] = timestamp
                item["transitions"].append(
                    {
                        "kind": "reviewed",
                        "actor": evaluator,
                        "turn": turn,
                        "timestamp": timestamp,
                        "from_status": previous,
                        "to_status": destination,
                        "from_owner": item["owner"],
                        "to_owner": item["owner"],
                    }
                )
            self._update_index(index, item)
            label = "failed" if error else str(verdict)
            self._commit(f"work-item {item_id}: review {label}", index, item)
            return self.read(item_id)


def render_work_item_prompt(policy: WorkItemPolicy) -> str:
    close_result = "Done" if policy.qc_mode == "optional" else "In review"
    return (
        "\n\nWork items are enforced in this run. Use exactly one command on "
        "a standalone line, with no prose, Markdown, or code in the same message:\n"
        '- `open_work_item "<title>" "<body>"`\n'
        '- `close_work_item <id> "<completion summary>"`\n'
        "- `list_work_items`\n"
        "- `read_work_item <id>`\n"
        "Delegating to another agent automatically transfers your in-progress "
        "work items to that agent; you do not need to mention an item id. "
        f"Closing moves the item to {close_result}. You cannot use `end_session` "
        "while you own a work item that is not Done."
    )


def execute_work_item_command(
    store: WorkItemStore,
    command: WorkItemCommand,
    *,
    owner: str,
    turn: int,
) -> WorkItemCommandResult:
    """Apply one parsed command with transport-neutral feedback."""
    try:
        if command.name == "open_work_item":
            item = store.open(command.title, command.body, owner, turn)
            feedback = (
                f"Opened work item {item['id']} ({item['status']}) for {item['owner']}."
            )
            return WorkItemCommandResult(feedback, True, item)
        if command.name == "close_work_item":
            if command.item_id is None:
                raise WorkItemConflict("close_work_item needs an item id")
            item = store.close(command.item_id, command.completion_summary, owner, turn)
            feedback = f"Work item {item['id']} moved to {item['status']}."
            return WorkItemCommandResult(feedback, True, item)
        if command.name == "list_work_items":
            return WorkItemCommandResult(
                "Work items: " + json.dumps(store.list(), sort_keys=True), True
            )
        if command.item_id is None:
            raise WorkItemConflict("read_work_item needs an item id")
        return WorkItemCommandResult(
            "Work item: " + json.dumps(store.read(command.item_id), sort_keys=True),
            True,
        )
    except WorkItemError as exc:
        return WorkItemCommandResult(f"Work-item command refused: {exc}", False)
