"""Shared work-item mechanics used by both the CLI and web engines.

`runner.py` and `streaming_runner.py` call these functions instead of
inlining the same logic twice. This is the only way the two engines stay
consistent (see the implementation brief,
`notes/docs/session-work-items-and-brief-implementation.md`, WS-0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from caribou.execution.session_brief import SessionBrief
from caribou.execution.work_items import (
    WorkItemCommandResult,
    WorkItemPolicy,
    WorkItemStore,
    execute_work_item_command,
    parse_work_item_command,
)

MAX_STATE_CHARS = 1200
MAX_STATE_ITEMS = 8

_COMMAND_NAMES = (
    "open_work_item",
    "close_work_item",
    "list_work_items",
    "read_work_item",
)


def _looks_like_attempted_command(message: str) -> bool:
    """True if some line looks like a work-item command that
    `parse_work_item_command` nonetheless rejected — almost always because
    the agent added prose, a header, or extra lines around it, violating the
    one-command-per-message grammar. Used only to decide whether the silent
    `None` case below deserves an explicit correction instead.
    """
    for line in message.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(name) for name in _COMMAND_NAMES):
            return True
    return False


def apply_command(
    store: WorkItemStore, message: str, *, owner: str, turn: int
) -> Optional[WorkItemCommandResult]:
    """Parse and apply one work-item command from an assistant message.

    Returns `None` if the message is not a work-item command at all,
    distinct from a `WorkItemCommandResult(success=False, ...)`, which means
    it *was* a command but was refused.

    A message that merely *looks like* an attempted command (contains a
    command name on some line) but fails the strict one-command-per-message
    grammar — e.g. the agent narrates before issuing it — is not silently
    dropped as `None`: without feedback the agent has no signal that its
    message didn't register, and it will repeat the identical mistake
    forever (observed in practice — see the implementation brief's follow-up
    notes). It gets a `WorkItemCommandResult(success=False, ...)` explaining
    exactly what to fix instead.
    """
    command = parse_work_item_command(message)
    if command is None:
        if _looks_like_attempted_command(message):
            return WorkItemCommandResult(
                feedback=(
                    "Work-item command not recognized. A work-item command "
                    "must be the ENTIRE message, alone on its own line — no "
                    "narration, headers, or anything else before or after "
                    'it. Resend ONLY the command, e.g.: '
                    'open_work_item "<title>" "<body>"'
                ),
                success=False,
            )
        return None
    return execute_work_item_command(store, command, owner=owner, turn=turn)


def end_session_block(store: WorkItemStore, owner: str) -> List[Dict[str, Any]]:
    """Items that block `owner` from ending the session (non-Done items they own)."""
    return store.blocking_for_owner(owner)


def transfer_on_delegation(
    store: WorkItemStore,
    from_owner: str,
    to_owner: str,
    *,
    turn: int,
    evaluator_agent_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Hand off every active item `from_owner` owns to `to_owner`, with one
    exception: the evaluator agent never receives ownership this way.

    Required-QC review refuses a self-approve (`evaluator == owner`). If a
    bare `delegate_to_<evaluator>` transferred active items unconditionally,
    an item closed by the evaluator would land in `In review` owned by the
    evaluator, and `record_review` would then refuse it forever — a
    permanent deadlock. So a delegation that targets the evaluator agent
    leaves active items with `from_owner` instead of transferring them.
    """
    if evaluator_agent_name is not None and to_owner == evaluator_agent_name:
        return []
    return store.transfer_active(from_owner, to_owner, turn)


def render_work_item_state(
    store: WorkItemStore, owner: str, *, brief_goal: Optional[str] = None
) -> str:
    """The ambient, per-turn work-item state block.

    Derived, not persisted: recomputed from the store each turn and appended
    to the outgoing message list just before the LLM call. Persisting it
    would accumulate duplicates and pollute memory-strategy summaries.
    """
    items = [item for item in store.list() if item.get("status") != "Done"]
    lines: List[str] = []
    if brief_goal:
        lines.append(f"BRIEF GOAL: {brief_goal}")

    if not items:
        lines.append("WORK ITEMS: none open.")
        return "\n".join(lines)

    lines.append(f"WORK ITEMS (owner: {owner}):")
    shown = items[:MAX_STATE_ITEMS]
    for item in shown:
        since = store.last_transition_turn(item)
        marker = " (yours)" if item.get("owner") == owner else ""
        lines.append(
            f"- #{item['id']} [{item['status']}] {item['title']} — "
            f"owner {item['owner']}{marker}, opened turn {item.get('created_turn')}, "
            f"last change turn {since}"
        )
    remaining = len(items) - len(shown)
    if remaining > 0:
        lines.append(f"... +{remaining} more")

    blocking = [item for item in items if item.get("owner") == owner]
    if blocking:
        plural = "item" if len(blocking) == 1 else "items"
        lines.append(
            f"You own {len(blocking)} {plural} that {'is' if len(blocking) == 1 else 'are'} "
            "not Done; end_session is blocked until closed."
        )

    rendered = "\n".join(lines)
    if len(rendered) > MAX_STATE_CHARS:
        rendered = rendered[:MAX_STATE_CHARS] + "\n... (truncated)"
    return rendered


def stall_report(
    store: WorkItemStore, owner: str, *, turn: int, stall_turns: int
) -> Optional[Dict[str, Any]]:
    """The owner's active item stalled the longest, if any item has gone
    `stall_turns` or more turns without a transition. `None` if nothing is
    stalled.
    """
    stalled = []
    for item in store.list():
        if item.get("owner") != owner or item.get("status") == "Done":
            continue
        since = store.last_transition_turn(item)
        idle = turn - since
        if idle >= stall_turns:
            stalled.append((idle, item, since))
    if not stalled:
        return None
    stalled.sort(key=lambda entry: entry[0], reverse=True)
    idle, item, since = stalled[0]
    return {
        "item_id": item["id"],
        "title": item["title"],
        "owner": owner,
        "idle_turns": idle,
        "last_transition_turn": since,
    }


def copy_work_items(
    src_dir: Path, dst_dir: Path, *, child_session_id: str, forked_from_session_id: str
) -> Optional[WorkItemStore]:
    """Copy a session's work-item store for a fork. `None` if the source
    directory doesn't exist (e.g. work items were never opened).
    """
    src_dir = Path(src_dir)
    if not src_dir.exists():
        return None
    from caribou.execution.work_items import WorkItemStore as _Store

    parent_policy = _peek_policy(src_dir)
    parent = _Store(src_dir, session_id=forked_from_session_id, policy=parent_policy)
    return parent.copy_to(
        dst_dir,
        child_session_id=child_session_id,
        forked_from_session_id=forked_from_session_id,
    )


def _peek_policy(work_items_dir: Path) -> WorkItemPolicy:
    """Best-effort read of the committed `qc_mode` so a fork doesn't silently
    reset an existing store's policy before `WorkItemStore.__init__` runs its
    own reconciliation on first `_index()` call.
    """
    import json
    import subprocess

    try:
        result = subprocess.run(
            ["git", "show", "HEAD:index.json"],
            cwd=work_items_dir,
            check=True,
            text=True,
            capture_output=True,
        )
        raw = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return WorkItemPolicy()
    return WorkItemPolicy.from_dict(
        {k: v for k, v in raw.items() if k in {"qc_mode", "stall_turns", "stall_halt_turns"}}
    )


def freeze_brief(
    brief: SessionBrief,
    *,
    brief_path: Path,
    store: WorkItemStore,
    brief_mode: str,
    owner: str,
    turn: int,
) -> Optional[Dict[str, Any]]:
    """Freeze an accepted brief (WS-5.3, "Freeze (on accept)" steps 1-3):
    write `brief.json`, commit it into the work-item store's history as
    provenance, and — if `brief_mode == "seed_item"` — open the seed work
    item. Steps 4 (pin to history) and 5 (flip phase) are the caller's
    responsibility, since they touch engine-specific state (`history`/
    `_Session`) this module doesn't own.

    Returns the seed item dict, or `None` in `context` mode.
    """
    brief_json = brief.model_dump_json(indent=2)
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(brief_json + "\n", encoding="utf-8")
    store.record_brief_provenance(brief_json)
    if brief_mode == "seed_item":
        return seed_brief_item(store, brief, owner=owner, turn=turn)
    return None


def render_brief_pin(brief: SessionBrief) -> str:
    """The brief rendered as a pinned system message (WS-5.3 step 4, WS-5.4)."""
    lines = [
        "SESSION BRIEF (frozen — read-only for the rest of this session):",
        f"Deliverable: {brief.deliverable}",
        "In scope:",
        *(f"  - {item}" for item in brief.in_scope),
    ]
    if brief.out_of_scope:
        lines.append("Out of scope:")
        lines.extend(f"  - {item}" for item in brief.out_of_scope)
    lines.append("Done when:")
    lines.extend(f"  - {item}" for item in brief.done_when)
    if brief.risks:
        lines.append("Risks:")
        lines.extend(f"  - {item}" for item in brief.risks)
    return "\n".join(lines)


def seed_brief_item(
    store: WorkItemStore, brief: SessionBrief, *, owner: str, turn: int
) -> Dict[str, Any]:
    """Open the top-level work item #0 for a frozen, `seed_item`-mode brief,
    so `end_session` is blocked on the deliverable until it closes (WS-5.3
    step 3). The title is the deliverable; the body carries done_when so the
    owner sees the completion criteria without re-reading the brief.
    """
    body_lines = [brief.deliverable, "", "Done when:"]
    body_lines.extend(f"- {criterion}" for criterion in brief.done_when)
    return store.open(brief.deliverable, "\n".join(body_lines), owner, turn)
