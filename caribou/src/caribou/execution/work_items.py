"""Enforced, Git-backed work items shared by CLI and web execution.

The public identifier is a small per-run integer.  Git commit identifiers are
kept as provenance only, so agents and operators never need to pass SHAs.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# v3 adds "anchor" (null for agent-opened items). A v2 item has no "anchor"
# key; `read` reports it as null, since a v2 item was never anchored.
# "reruns" (the id of the Done / In review item of the same owner and title
# that a new item redoes, else null) was added within v3: an item written
# before it has no "reruns" key and reads as null.
WORK_ITEM_SCHEMA = "caribou.work_item.v3"
WORK_ITEM_INDEX_SCHEMA = "caribou.work_item_index.v2"
WORK_ITEM_STATUSES = ("Backlog", "Ready", "In progress", "In review", "Done")
QcMode = Literal["optional", "required"]
# The evaluator (and transition actor) name for every review a person makes.
HUMAN_REVIEWER = "user"
_ANCHOR_KEYS = ("block_id", "action_id", "artifact_path")
# A whole line that is nothing but a delegation command.
_DELEGATION_LINE_RE = re.compile(r"^delegate_to_[A-Za-z0-9_]+$")
# Characters dropped when two titles are compared for the reruns link.
_TITLE_NOISE_RE = re.compile(r"[\W_]+")
_RERUN_SOURCE_STATUSES = ("Done", "In review")


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
    # The command this result answers; None for a correction that answers no
    # single command (a malformed block, a stray command line).
    command: Optional[WorkItemCommand] = None


def is_delegation_line(line: str) -> bool:
    """True if `line` is a bare `delegate_to_<agent>` command and nothing else."""
    return _DELEGATION_LINE_RE.match(line.strip()) is not None


_COMMAND_NAMES = (
    "open_work_item",
    "close_work_item",
    "list_work_items",
    "read_work_item",
)


def is_command_line(line: str) -> bool:
    """True if `line` starts with a work-item command name (whether or not
    the rest of the line follows the grammar)."""
    first = line.strip().split(None, 1)[0] if line.strip() else ""
    return first in _COMMAND_NAMES


@dataclass(frozen=True)
class CommandBlock:
    """The trailing command block of an assistant message.

    `lines`: the block's raw lines in message order (work-item command lines
    and bare `delegate_to_<agent>` lines). `commands`: the parsed work-item
    commands, in order, of every block line that is not a delegation.
    `malformed`: the first block line that starts with a command name but
    fails the grammar, else None; when set, `commands` holds only the lines
    before it and the block must not be applied (a half-applied block would
    hide the error).
    """

    lines: List[str]
    commands: List[WorkItemCommand]
    malformed: Optional[str]


def command_block(message: str) -> List[str]:
    """The lines of `message`'s trailing command block: the consecutive
    non-empty lines at the END of the message that are each either a
    work-item command line (start with a command name) or a bare
    `delegate_to_<agent>` line, in message order. Everything before the
    block is prose or code.

    Empty when there is no such line, or when the block would sit inside an
    unterminated ``` fence (the message ends inside a code block, so those
    lines are code, not commands).
    """
    if not message:
        return []
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    start = len(lines)
    while start > 0 and (
        is_command_line(lines[start - 1]) or is_delegation_line(lines[start - 1])
    ):
        start -= 1
    if start == len(lines):
        return []
    fence_open = False
    for line in lines[:start]:
        if line.startswith("```"):
            fence_open = not fence_open
    if fence_open:
        return []
    return lines[start:]


def command_line(message: str) -> Optional[str]:
    """The single work-item command line of `message`, if its trailing
    command block (see `command_block`) holds exactly one. Kept for callers
    that only ever send one command; raises ValueError when the block holds
    several, since dropping all but one would apply the wrong thing.
    """
    block = [line for line in command_block(message) if not is_delegation_line(line)]
    if not block:
        return None
    if len(block) > 1:
        raise ValueError(
            "the message's command block holds several work-item commands; "
            "use command_block/parse_work_item_commands"
        )
    return block[0]


def _parse_command_line(line: str) -> Optional[WorkItemCommand]:
    """One command line to a command, or None when it fails the grammar.
    Quoting follows shell-like rules solely for tokenization; nothing is
    ever passed through a shell."""
    try:
        tokens = shlex.split(line)
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


def parse_command_block(message: str) -> CommandBlock:
    """Parse the trailing command block of an assistant message (see
    `command_block`). Commands are single-line and come in the order they
    appear; a bare `delegate_to_<agent>` line may stand anywhere in the
    block and is not a work-item command.
    """
    lines = command_block(message)
    commands: List[WorkItemCommand] = []
    for line in lines:
        if is_delegation_line(line):
            continue
        command = _parse_command_line(line)
        if command is None:
            return CommandBlock(lines=lines, commands=commands, malformed=line)
        commands.append(command)
    return CommandBlock(lines=lines, commands=commands, malformed=None)


def parse_work_item_commands(message: str) -> List[WorkItemCommand]:
    """The work-item commands of `message`'s trailing command block, in
    order. Empty when the message has no command block, or when any line of
    the block fails the grammar (the whole block is then refused, so a
    caller never applies half of it; `parse_command_block` names the bad
    line). A command line anywhere but the block is ignored here, so a
    quoted or discussed command never fires accidentally.
    """
    block = parse_command_block(message)
    if block.malformed is not None:
        return []
    return block.commands


def parse_work_item_command(message: str) -> Optional[WorkItemCommand]:
    """One-command convenience over `parse_work_item_commands`: the single
    command of the message's command block, or None when there is none (or
    the block is malformed). Raises ValueError when the block holds more
    than one command, so a caller that can only handle one never silently
    drops the rest; such callers must migrate to the list form.
    """
    commands = parse_work_item_commands(message)
    if not commands:
        return None
    if len(commands) > 1:
        raise ValueError(
            f"the message's command block holds {len(commands)} work-item "
            "commands; use parse_work_item_commands"
        )
    return commands[0]


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

    def head_commit(self) -> Optional[str]:
        """The store's latest commit, or None when it has no commits yet."""
        with self._lock:
            if not self._has_head():
                return None
            return self._git("rev-parse", "HEAD").stdout.strip()

    def _resolve_commit(self, commit: str) -> str:
        """The full id of `commit`, which must be in this store's history.

        Raises WorkItemNotFound for an unknown commit or one that is not an
        ancestor of HEAD.
        """
        if not commit or commit.startswith("-"):
            raise WorkItemNotFound(f"work-item commit {commit!r} is not valid")
        if not self._has_head():
            raise WorkItemNotFound(
                f"work-item commit {commit!r} is unknown: the store has no commits"
            )
        resolved = self._git(
            "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}", check=False
        )
        if resolved.returncode != 0:
            raise WorkItemNotFound(f"work-item commit {commit!r} is unknown")
        full = resolved.stdout.strip()
        ancestor = self._git("merge-base", "--is-ancestor", full, "HEAD", check=False)
        if ancestor.returncode == 1:
            raise WorkItemNotFound(
                f"work-item commit {commit!r} is not in the store's history"
            )
        if ancestor.returncode != 0:
            raise WorkItemPersistenceError(
                f"work-item Git operation failed: {ancestor.stderr.strip()}"
            )
        return full

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

    def _fresh_index(self) -> Dict[str, Any]:
        """The committed index, re-read for a mutation.

        Mutations never trust the cache: another store instance on the same
        directory may have committed since it was filled, and allocating an id
        or rewriting the index from a stale copy would overwrite that commit.
        """
        self._index_cache = None
        return self._index()

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
                "reruns",
            )
        }

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            index = self._index()
            summaries = [dict(value) for value in index.get("items", [])]
            for summary in summaries:
                # Summaries written before the reruns link have no key.
                summary.setdefault("reruns", None)
            return summaries

    def read(self, item_id: int) -> Dict[str, Any]:
        with self._lock:
            item = dict(self._item(item_id))
            # v2 items predate anchors; they were never anchored.
            item.setdefault("anchor", None)
            # Items written before the reruns link were never linked.
            item.setdefault("reruns", None)
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

    @staticmethod
    def _validate_anchor(anchor: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if anchor is None:
            return None
        if not isinstance(anchor, dict) or not set(anchor) <= set(_ANCHOR_KEYS):
            raise WorkItemConflict(
                "work-item anchor must be an object whose keys are among "
                + ", ".join(_ANCHOR_KEYS)
            )
        # The keys are each optional (str or null); an absent key is null.
        anchor = {key: anchor.get(key) for key in _ANCHOR_KEYS}
        for key in _ANCHOR_KEYS:
            value = anchor[key]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise WorkItemConflict(
                    f"work-item anchor {key} must be a non-empty string or null"
                )
        if all(anchor[key] is None for key in _ANCHOR_KEYS):
            raise WorkItemConflict("work-item anchor needs at least one non-null value")
        return anchor

    @staticmethod
    def normalized_title(title: str) -> str:
        """`title` casefolded with punctuation and whitespace removed, the
        key two titles are compared on for the reruns link."""
        return _TITLE_NOISE_RE.sub("", title).casefold()

    @classmethod
    def _rerun_source(
        cls, summaries: List[Dict[str, Any]], *, owner: str, title: str
    ) -> Optional[int]:
        """The id of the latest Done or In review item `owner` already has
        under the same normalized title, or None: the item a new one with
        that title redoes."""
        key = cls.normalized_title(title)
        matches = [
            int(summary["id"])
            for summary in summaries
            if summary.get("owner") == owner
            and summary.get("status") in _RERUN_SOURCE_STATUSES
            and cls.normalized_title(str(summary.get("title", ""))) == key
        ]
        return max(matches) if matches else None

    def open(
        self,
        title: str,
        body: str,
        owner: str,
        turn: int,
        *,
        opened_by: Optional[str] = None,
        anchor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Open an item owned by `owner`.

        `opened_by` is the "opened" transition's actor (default: the owner);
        a person opening a ticket passes `HUMAN_REVIEWER`. `anchor` optionally
        ties the item to a block, code action and/or artifact. `reruns` is
        the id of the Done or In review item `owner` already has under the
        same normalized title (`normalized_title`), the one this item redoes,
        or null.
        """
        if not title.strip() or not body.strip():
            raise WorkItemConflict("work-item title and body must be non-empty")
        anchor = self._validate_anchor(anchor)
        actor = owner if opened_by is None else opened_by
        with self._lock:
            index = self._fresh_index()
            item_id = int(index.get("next_id", 0))
            reruns = self._rerun_source(
                index.get("items", []), owner=owner, title=title
            )
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
                        "actor": actor,
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
                "anchor": anchor,
                "reruns": reruns,
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
            index = self._fresh_index()
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
            index = self._fresh_index()
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
            index = self._fresh_index()
            item = self._item(item_id)
            timestamp = utc_now()
            item.setdefault("notes", []).append(
                {"actor": actor, "turn": turn, "timestamp": timestamp, "text": text}
            )
            self._update_index(index, item)
            self._commit(f"work-item {item_id}: note", index, item)
            return self.read(item_id)

    def record_brief_provenance(self, brief_json: str) -> None:
        """Commit a frozen session brief's JSON into this store's history
        (WS-5.3 step 2: "seed the work-item store's first commit with the
        brief"), so the brief that governed this session's work is part of
        its durable git history, not just a side file. A plain commit, not
        an item mutation — no `_index()`/`_item()` round trip needed.
        """
        with self._lock:
            self._restore_committed_tree()
            (self.root / "brief.json").write_text(brief_json, encoding="utf-8")
            self._git("add", "brief.json")
            self._git("commit", "--quiet", "-m", "brief: frozen")
            self._index_cache = None

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
        return self._copy(
            dst_dir,
            commit=None,
            child_session_id=child_session_id,
            forked_from_session_id=forked_from_session_id,
        )

    def copy_at(
        self,
        dst_dir: Path,
        *,
        commit: str,
        child_session_id: str,
        forked_from_session_id: str,
    ) -> "WorkItemStore":
        """`copy_to`, but the child's store is this store as of `commit`.

        The child keeps the history up to `commit`; later items, reviews and
        transitions are not in it. Raises WorkItemNotFound when `commit` is
        unknown or not in this store's history, before anything is copied.
        """
        return self._copy(
            dst_dir,
            commit=commit,
            child_session_id=child_session_id,
            forked_from_session_id=forked_from_session_id,
        )

    def _copy(
        self,
        dst_dir: Path,
        *,
        commit: Optional[str],
        child_session_id: str,
        forked_from_session_id: str,
    ) -> "WorkItemStore":
        dst_dir = Path(dst_dir)
        with self._lock:
            resolved = None if commit is None else self._resolve_commit(commit)
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
            if resolved is not None:
                # The child's own Git repo, never the project's: rewind its
                # branch and working tree to `commit`.
                child._git("reset", "--hard", "--quiet", resolved)
                # Git drops `items/` when no item existed at `commit`.
                child.items_dir.mkdir(parents=True, exist_ok=True)
                child._index_cache = None
            index = child._index()
            index["session_id"] = child_session_id
            index["forked_from_session_id"] = forked_from_session_id
            child._atomic_json(child.root / "index.json", index)
            item_paths = sorted(child.items_dir.glob("*.json"))
            for item_path in item_paths:
                item = json.loads(item_path.read_text(encoding="utf-8"))
                item["session_id"] = child_session_id
                child._atomic_json(item_path, item)
            # `git add items` fails on an empty directory (a store whose only
            # commit is its frozen brief), so name the item files instead.
            child._git(
                "add",
                "index.json",
                *(path.relative_to(child.root).as_posix() for path in item_paths),
            )
            child._git(
                "commit",
                "--quiet",
                "-m",
                f"fork: recorded from session {forked_from_session_id}"
                + ("" if resolved is None else f" at {resolved}"),
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
        evaluator: str = HUMAN_REVIEWER,
        turn: int,
        verdict: Optional[Literal["approve", "reject"]],
        assessment: str,
        provider_receipt: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record one review.

        Required QC: the review moves an In review item to Done (approve) or
        back to In progress (reject). Optional QC: the item stays Done, except
        that a human (`HUMAN_REVIEWER`) reject reopens it to In progress. A
        human reject needs a non-empty assessment.
        """
        human = evaluator == HUMAN_REVIEWER
        if human and verdict == "reject" and not assessment.strip():
            raise WorkItemConflict("a human reject needs a non-empty assessment")
        with self._lock:
            index = self._fresh_index()
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
            reopened = False
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
            elif (
                not error
                and human
                and verdict == "reject"
                and self.policy.qc_mode == "optional"
            ):
                reopened = True
                item["status"] = "In progress"
                item["completed_turn"] = None
                item["completed_at"] = None
                item["closed_turn"] = None
                item["closed_at"] = None
                item["transitions"].append(
                    {
                        "kind": "reopened",
                        "actor": HUMAN_REVIEWER,
                        "turn": turn,
                        "timestamp": timestamp,
                        "from_status": "Done",
                        "to_status": "In progress",
                        "from_owner": item["owner"],
                        "to_owner": item["owner"],
                    }
                )
            self._update_index(index, item)
            label = "failed" if error else str(verdict)
            if reopened:
                label += ", reopened"
            self._commit(f"work-item {item_id}: review {label}", index, item)
            return self.read(item_id)


def copy_work_items_at(
    src_dir: Path,
    dst_dir: Path,
    *,
    commit: str,
    child_session_id: str,
    forked_from_session_id: str,
) -> WorkItemStore:
    """Copy a session's work-item store as of `commit` for a branch.

    Uses the same lineage stamping as `work_item_runtime.copy_work_items`
    (`WorkItemStore.copy_to`). The policy is the one committed at `commit`.
    Raises WorkItemNotFound when `src_dir` holds no store or `commit` is not
    in its history.
    """
    src_dir = Path(src_dir)
    if not (src_dir / ".git").exists():
        raise WorkItemNotFound(f"no work-item store at {src_dir}")
    parent = WorkItemStore(
        src_dir, session_id=forked_from_session_id, policy=WorkItemPolicy()
    )
    with parent._lock:
        resolved = parent._resolve_commit(commit)
        shown = parent._git("show", f"{resolved}:index.json", check=False)
    if shown.returncode == 0:
        try:
            committed = json.loads(shown.stdout)
        except json.JSONDecodeError as exc:
            raise WorkItemPersistenceError(
                f"committed work-item index at {resolved} is invalid"
            ) from exc
        parent.policy = WorkItemPolicy.from_dict(
            {
                key: value
                for key, value in committed.items()
                if key in {"qc_mode", "stall_turns", "stall_halt_turns"}
            }
        )
    return parent.copy_at(
        dst_dir,
        commit=resolved,
        child_session_id=child_session_id,
        forked_from_session_id=forked_from_session_id,
    )


def render_work_item_prompt(policy: WorkItemPolicy) -> str:
    close_result = "Done" if policy.qc_mode == "optional" else "In review"
    return (
        "\n\nWork items are enforced in this run. Put your work-item commands in "
        "a COMMAND BLOCK at the END of the message: one command per line, "
        "several allowed, applied in the order written. A bare "
        "`delegate_to_<agent>` line belongs in the same block, in any "
        "position. Prose or code may come before the block, nothing after it. "
        "Commands anywhere else are not applied (you are told so). Do not wrap "
        "commands in backticks:\n"
        '- `open_work_item "<title>" "<body>"`\n'
        '- `close_work_item <id> "<completion summary>"`\n'
        "- `list_work_items`\n"
        "- `read_work_item <id>`\n"
        "Close before you delegate: a work item you ran code under must be "
        "closed by you (you did the work, so you write its completion summary) "
        "before a delegation is accepted; `close_work_item` and "
        "`delegate_to_<agent>` may stand in the same command block, the close "
        "first. A delegation that finds such an item still open is refused, and "
        "you cannot delegate to yourself. Items you opened but did no code work "
        "in (e.g. a plan) transfer to the agent you delegate to; you do not need "
        "to mention an item id. "
        "When you receive a delegation and own no open work item for that "
        "task, open one for your task first, with a short stage-style title "
        '(e.g. "QC filtering"); close it when the task is done. If you are '
        "redoing a stage you already completed, reuse the same title so the "
        "rerun is linked to the original. "
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
            return WorkItemCommandResult(feedback, True, item, command)
        if command.name == "close_work_item":
            if command.item_id is None:
                raise WorkItemConflict("close_work_item needs an item id")
            item = store.close(command.item_id, command.completion_summary, owner, turn)
            feedback = f"Work item {item['id']} moved to {item['status']}."
            return WorkItemCommandResult(feedback, True, item, command)
        if command.name == "list_work_items":
            return WorkItemCommandResult(
                "Work items: " + json.dumps(store.list(), sort_keys=True),
                True,
                command=command,
            )
        if command.item_id is None:
            raise WorkItemConflict("read_work_item needs an item id")
        return WorkItemCommandResult(
            "Work item: " + json.dumps(store.read(command.item_id), sort_keys=True),
            True,
            command=command,
        )
    except WorkItemError as exc:
        return WorkItemCommandResult(
            f"Work-item command refused: {exc}", False, command=command
        )
