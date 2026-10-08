"""Block attribution: which work item (and which attempt) each code action
implements.

A work item is the intent; a block is the implementation that goes with it.
One work item can have several blocks, one per attempt (an attempt ends when
a reviewer rejects it). Code that runs while its agent owns no open work item
goes to an *implicit* block (`work_item_id = None`).

The attribution rule lives here, once, and both engines (`runner.py` and
`server/streaming_runner.py`) drive a `BlockTracker` instead of inlining it.
`blocks.json` is the source of truth; it sits next to the `work-items/`
directory and is rewritten atomically on every change.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from caribou.execution.work_items import WorkItemStore

BLOCK_SCHEMA = "caribou.block.v1"
BLOCK_INDEX_SCHEMA = "caribou.block_index.v1"
BLOCKS_FILENAME = "blocks.json"

_OPEN = "running"
_DELEGATION_PREFIX = "delegate_to_"


class BlockError(RuntimeError):
    """blocks.json is malformed, or the tracker was driven inconsistently."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _block_id(index: int) -> str:
    return f"blk-{index:04d}"


def _rejections(item: Dict[str, Any]) -> int:
    """Reject verdicts on a full item. Errored reviews (verdict None) don't count."""
    return sum(1 for review in item["reviews"] if review.get("verdict") == "reject")


def blocks_path_for(work_item_store: WorkItemStore) -> Path:
    """`blocks.json` sits next to the store's `work-items/` directory."""
    return Path(work_item_store.root).parent / BLOCKS_FILENAME


def load_blocks(blocks_path: Path) -> Optional[Dict[str, Any]]:
    """The parsed block index, or None when `blocks_path` does not exist.

    Raises `BlockError` on malformed content (bad JSON, wrong schema version,
    missing keys, or block indices that are not exactly 1..n in order).
    """
    blocks_path = Path(blocks_path)
    if not blocks_path.exists():
        return None
    try:
        index = json.loads(blocks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BlockError(f"blocks file is not valid JSON: {blocks_path}") from exc
    if not isinstance(index, dict):
        raise BlockError(f"blocks file is not a JSON object: {blocks_path}")
    if index.get("schema_version") != BLOCK_INDEX_SCHEMA:
        raise BlockError(
            f"blocks file has schema_version {index.get('schema_version')!r}, "
            f"expected {BLOCK_INDEX_SCHEMA!r}: {blocks_path}"
        )
    if not isinstance(index.get("session_id"), str):
        raise BlockError(f"blocks file has no session_id: {blocks_path}")
    blocks = index.get("blocks")
    if not isinstance(blocks, list):
        raise BlockError(f"blocks file has no blocks list: {blocks_path}")
    for position, block in enumerate(blocks, start=1):
        if not isinstance(block, dict):
            raise BlockError(f"block {position} is not an object: {blocks_path}")
        if block.get("schema_version") != BLOCK_SCHEMA:
            raise BlockError(
                f"block {position} has schema_version "
                f"{block.get('schema_version')!r}: {blocks_path}"
            )
        if block.get("index") != position or block.get("block_id") != _block_id(
            position
        ):
            raise BlockError(
                f"block {position} is out of sequence "
                f"({block.get('block_id')!r}, index {block.get('index')!r}): "
                f"{blocks_path}"
            )
    return index


def _write_index(blocks_path: Path, index: Dict[str, Any]) -> None:
    temporary = blocks_path.with_suffix(blocks_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, blocks_path)


def init_blocks(blocks_path: Path, session_id: str) -> None:
    """Write an empty blocks.json for a new session; raises if one exists.

    Called when the session is created, before its runner (and tracker)
    starts, so the session reads as "recording blocks, none yet" from the
    first request rather than as one recorded before blocks existed.
    """
    if not session_id:
        raise ValueError("session_id must be non-empty")
    blocks_path = Path(blocks_path)
    if blocks_path.exists():
        raise BlockError(f"blocks file already exists: {blocks_path}")
    blocks_path.parent.mkdir(parents=True, exist_ok=True)
    _write_index(
        blocks_path,
        {"schema_version": BLOCK_INDEX_SCHEMA, "session_id": session_id, "blocks": []},
    )


def fork_blocks(src_path: Path, dst_path: Path, *, child_session_id: str) -> bool:
    """Copy a session's blocks.json for a forked child session.

    Rewrites `session_id` on the index and on every block, the same way
    `WorkItemStore.copy_to` re-stamps items, so the child's tracker loads it
    without a session mismatch. Returns False (and writes nothing) when the
    parent has no blocks.json; raises if `dst_path` already exists.
    """
    index = load_blocks(src_path)
    if index is None:
        return False
    dst_path = Path(dst_path)
    if dst_path.exists():
        raise BlockError(f"fork destination already exists: {dst_path}")
    index["session_id"] = child_session_id
    for block in index["blocks"]:
        block["session_id"] = child_session_id
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _write_index(dst_path, index)
    return True


class BlockTracker:
    """Attributes code actions to blocks and persists them to blocks.json.

    Single-threaded: drive it only from the session's runner loop. Work-item
    changes made elsewhere (reviews from `/evaluate` or the web evaluate and
    human-review routes) are picked up by `sync()`, which `begin_action` and
    `close_all` also run first, and which both runner loops call when they
    take a user message.
    """

    def __init__(
        self,
        blocks_path: Path,
        session_id: str,
        work_item_store: WorkItemStore,
        on_change: Callable[[Dict[str, Any]], None],
    ) -> None:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        self.blocks_path = Path(blocks_path)
        self.session_id = session_id
        self.work_items = work_item_store
        self._on_change = on_change
        self._blocks: List[Dict[str, Any]] = []
        index = load_blocks(self.blocks_path)
        if index is not None:
            # A run reusing an output dir (e.g. a CLI `--output-dir` with a new
            # timestamped run_id) continues under the session recorded in the
            # file, the same way WorkItemStore keeps its index's original
            # session_id. Forks re-stamp the file explicitly via fork_blocks.
            self.session_id = index["session_id"]
            self._blocks = index["blocks"]
        else:
            # Sessions the web server creates already have one (init_blocks);
            # CLI and control runs get theirs here.
            init_blocks(self.blocks_path, self.session_id)
        self._by_action: Dict[str, Dict[str, Any]] = {
            action_id: block
            for block in self._blocks
            for action_id in block["action_ids"]
        }
        # The block the previous code action went to, in memory only: after a
        # resume, the first implicit action starts a fresh implicit block.
        self._last_block: Optional[Dict[str, Any]] = None
        # Workbench focus (in memory only): while set, every action goes to
        # this block. `_focus_reopened` is True once a focused action reopened
        # a block that was closed when the focus was set.
        self._focus: Optional[Dict[str, Any]] = None
        self._focus_reopened = False
        # One-shot handoff titles for the next implicit block of an agent.
        self._pending_titles: Dict[str, str] = {}

    # -- persistence -------------------------------------------------------

    def _changed(self, block: Dict[str, Any]) -> None:
        block["updated_at"] = _utc_now()
        _write_index(
            self.blocks_path,
            {
                "schema_version": BLOCK_INDEX_SCHEMA,
                "session_id": self.session_id,
                "blocks": self._blocks,
            },
        )
        self._on_change(copy.deepcopy(block))

    # -- queries -----------------------------------------------------------

    def blocks(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._blocks)

    def block_for_action(self, action_id: str) -> Dict[str, Any]:
        block = self._by_action.get(action_id)
        if block is None:
            raise BlockError(f"action {action_id!r} is not attributed to any block")
        return copy.deepcopy(block)

    # -- lifecycle ---------------------------------------------------------

    def _new_block(
        self,
        *,
        owner: str,
        turn: int,
        work_item_id: Optional[int],
        attempt: int,
        title: str,
    ) -> Dict[str, Any]:
        index = len(self._blocks) + 1
        timestamp = _utc_now()
        block: Dict[str, Any] = {
            "schema_version": BLOCK_SCHEMA,
            "block_id": _block_id(index),
            "session_id": self.session_id,
            "index": index,
            "work_item_id": work_item_id,
            "attempt": attempt,
            "implicit": work_item_id is None,
            "title": title,
            "kind": None,
            "agents": [owner],
            "status": _OPEN,
            "turn_start": turn,
            "turn_end": turn,
            "action_ids": [],
            "failed_action_ids": [],
            "artifact_paths": [],
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        self._blocks.append(block)
        return block

    def _close(self, block: Dict[str, Any], *, rejected: bool) -> None:
        if block["status"] != _OPEN:
            return
        actions = block["action_ids"]
        failed = set(block["failed_action_ids"])
        if actions and actions[-1] in failed:
            block["status"] = "error"
        elif failed or rejected:
            block["status"] = "warn"
        else:
            block["status"] = "ok"
        self._changed(block)

    def _item_close_reason(self, item: Dict[str, Any], block: Dict[str, Any]) -> Optional[str]:
        if _rejections(item) >= block["attempt"]:
            return "rejected"
        if item["status"] == "Done":
            return "done"
        return None

    def _apply_item(self, item: Dict[str, Any]) -> None:
        for block in self._blocks:
            if block["work_item_id"] != item["id"]:
                continue
            if block is self._focus and block["status"] == _OPEN:
                # A focused block stays open until the focus clears;
                # `clear_focus` applies the close rules then.
                continue
            if block["status"] == "ok":
                # A reject that arrives after the attempt closed ok (e.g. the
                # item was Done, then reviewed) downgrades it to warn.
                if _rejections(item) >= block["attempt"]:
                    block["status"] = "warn"
                    self._changed(block)
                continue
            if block["status"] != _OPEN:
                continue
            reason = self._item_close_reason(item, block)
            if reason is not None:
                self._close(block, rejected=reason == "rejected")

    def on_work_item_changed(self, item: Dict[str, Any]) -> None:
        """Close blocks the change ends: the item is Done (a), or the block's
        attempt was rejected (b); and turn an `ok` block whose attempt was
        rejected afterwards into `warn`. `item` is a full item
        (`WorkItemStore.read`)."""
        self._apply_item(item)

    def sync(self) -> None:
        """Re-read the work item of every open or `ok` work-item block and
        apply it.

        Catches reviews, reopens and Done transitions made outside the runner
        loop. Idempotent.
        """
        item_ids = sorted(
            {
                block["work_item_id"]
                for block in self._blocks
                if block["status"] in (_OPEN, "ok") and not block["implicit"]
            }
        )
        for item_id in item_ids:
            self._apply_item(self.work_items.read(item_id))

    def _attribute(self, owner: str, turn: int) -> Dict[str, Any]:
        candidates = [
            self.work_items.read(int(summary["id"]))
            for summary in self.work_items.blocking_for_owner(owner)
        ]
        if candidates:
            item = max(
                candidates,
                key=lambda value: (
                    WorkItemStore.last_transition_turn(value),
                    int(value["id"]),
                ),
            )
            attempt = 1 + _rejections(item)
            for block in self._blocks:
                if block["work_item_id"] == item["id"] and block["attempt"] == attempt:
                    return block
            return self._new_block(
                owner=owner,
                turn=turn,
                work_item_id=int(item["id"]),
                attempt=attempt,
                title=str(item["title"]),
            )
        last = self._last_block
        if (
            last is not None
            and last["implicit"]
            and last["status"] == _OPEN
            and last["agents"][0] == owner
        ):
            return last
        return self._new_block(
            owner=owner,
            turn=turn,
            work_item_id=None,
            attempt=1,
            title=self._pending_titles.pop(owner, f"{owner} (no work item)"),
        )

    # -- workbench focus and handoffs --------------------------------------

    def set_focus(self, block_id: str) -> None:
        """Attribute every action to `block_id` until `clear_focus`.

        Overrides work-item and implicit attribution entirely. A closed block
        reopens on the first focused action, not here. Focusing another block
        clears the current focus first; refocusing the same block is a no-op.
        """
        block = next(
            (value for value in self._blocks if value["block_id"] == block_id), None
        )
        if block is None:
            raise BlockError(f"cannot focus unknown block {block_id!r}")
        if block is self._focus:
            return
        self.clear_focus()
        self._focus = block
        self._focus_reopened = False

    def clear_focus(self) -> None:
        """End the focus. A block the focus reopened closes again under the
        normal rules; one that was open before the focus stays open unless
        its work item's state now closes it. No-op with no focus set."""
        block = self._focus
        reopened = self._focus_reopened
        self._focus = None
        self._focus_reopened = False
        if block is None or block["status"] != _OPEN:
            return
        if block["implicit"]:
            if reopened:
                self._close(block, rejected=False)
            return
        item = self.work_items.read(int(block["work_item_id"]))
        if reopened:
            self._close(block, rejected=_rejections(item) >= block["attempt"])
        else:
            self._apply_item(item)

    def note_delegation(self, from_agent: str, to_agent: str, command: str) -> None:
        """Title the next implicit block created for `to_agent` after the
        handoff, e.g. "QC metrics (from input_agent)" for
        `delegate_to_QC_metrics`. One-shot: consumed by that block."""
        if not command.startswith(_DELEGATION_PREFIX):
            raise ValueError(
                f"delegation command must start with {_DELEGATION_PREFIX!r}: "
                f"{command!r}"
            )
        task = command[len(_DELEGATION_PREFIX):].replace("_", " ").strip()
        if not task:
            raise ValueError(f"delegation command names no task: {command!r}")
        self._pending_titles[to_agent] = f"{task} (from {from_agent})"

    def begin_action(self, owner: str, turn: int, action_id: str) -> str:
        """Attribute a code action that is about to execute; returns its block_id."""
        if action_id in self._by_action:
            raise BlockError(f"action {action_id!r} was already attributed")
        self.sync()
        if self._focus is not None:
            block = self._focus
            if block["status"] != _OPEN:
                self._focus_reopened = True
        else:
            block = self._attribute(owner, turn)
        last = self._last_block
        if last is not None and last is not block and last["implicit"]:
            self._close(last, rejected=False)
        # A work-item block closed only by session end (d) is matched again
        # after a resume: the same attempt continues, so it reopens.
        block["status"] = _OPEN
        if owner not in block["agents"]:
            block["agents"].append(owner)
        block["action_ids"].append(action_id)
        block["turn_end"] = turn
        self._by_action[action_id] = block
        self._last_block = block
        self._changed(block)
        return block["block_id"]

    def finish_action(self, action_id: str, success: bool) -> str:
        """Record an action's outcome; returns its block_id."""
        block = self._by_action.get(action_id)
        if block is None:
            raise BlockError(f"action {action_id!r} was never begun")
        if not success:
            block["failed_action_ids"].append(action_id)
            self._changed(block)
        return block["block_id"]

    def record_artifact(self, path: str, action_id: Optional[str]) -> Optional[str]:
        """Attribute an artifact path to its producing action's block.

        Returns that block_id, or None for an artifact with no action_id.
        """
        if action_id is None:
            return None
        block = self._by_action.get(action_id)
        if block is None:
            raise BlockError(f"artifact {path!r} names unknown action {action_id!r}")
        if path not in block["artifact_paths"]:
            block["artifact_paths"].append(path)
            self._changed(block)
        return block["block_id"]

    def close_all(self) -> None:
        """Session end (d): close every open block."""
        self.sync()
        self._focus = None
        self._focus_reopened = False
        for block in self._blocks:
            self._close(block, rejected=False)
        self._last_block = None
