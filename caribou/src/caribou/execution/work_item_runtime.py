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
    command_block,
    execute_work_item_command,
    is_command_line,
    parse_command_block,
)

MAX_STATE_CHARS = 1200
MAX_STATE_ITEMS = 8

NOT_APPLIED_AFTER_REFUSAL = (
    "not applied: an earlier command in this message was refused"
)

_BLOCK_RULE = (
    "Put work-item commands in a block at the END of the message, one per "
    "line (several allowed, applied in order), with every argument quoted, "
    'e.g.: open_work_item "<title>" "<body>" or '
    'close_work_item <id> "<completion summary>". A bare delegate_to_<agent> '
    "line may stand in the block; prose or code may come before the block, "
    "nothing after it."
)


def stray_command_lines(message: str) -> List[str]:
    """Lines of `message` that start with a work-item command name but are
    neither inside a code fence nor part of the trailing command block
    (`command_block`): a command followed by prose, for instance. Such a
    line is not applied; without feedback the agent has no signal that it
    didn't register and repeats the identical mistake forever (observed in
    practice — see the implementation brief's follow-up notes), so the
    caller reports them.
    """
    block = command_block(message)
    block_start = len(block)
    stray: List[str] = []
    in_fence = False
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    body = lines[: len(lines) - block_start] if block_start else lines
    for line in body:
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and is_command_line(line):
            stray.append(line)
    return stray


def _looks_like_attempted_command(message: str) -> bool:
    """True if `message` has a command-looking line that will not be applied:
    a malformed line in the trailing command block, or a command line
    outside the block (`stray_command_lines`). Used to decide whether the
    "no command" case deserves an explicit correction instead of the
    message being ignored without a word.
    """
    return parse_command_block(message).malformed is not None or bool(
        stray_command_lines(message)
    )


def apply_commands(
    store: WorkItemStore, message: str, *, owner: str, turn: int
) -> List[WorkItemCommandResult]:
    """Parse and apply the work-item commands of an assistant message's
    trailing command block, in order.

    Returns `[]` when the message holds no work-item command at all. The
    commands are applied in the order written and stop at the first
    refusal: every later command gets a `WorkItemCommandResult(success=
    False, feedback=NOT_APPLIED_AFTER_REFUSAL)`, so the agent sees which
    ones it must resend.

    A block with a malformed line is not applied at all (a half-applied
    block would hide the error): the result is one failed
    `WorkItemCommandResult` naming the bad line. A command-looking line
    outside the block (e.g. followed by prose) adds one failed result with
    the same correction, after the block's results — it is never silently
    dropped.
    """
    block = parse_command_block(message)
    results: List[WorkItemCommandResult] = []
    if block.malformed is not None:
        results.append(
            WorkItemCommandResult(
                feedback=(
                    "Work-item command not recognized: "
                    f"{block.malformed!r} does not follow the grammar, so none of "
                    f"this message's commands were applied. {_BLOCK_RULE}"
                ),
                success=False,
            )
        )
    else:
        refused = False
        for command in block.commands:
            if refused:
                results.append(
                    WorkItemCommandResult(
                        feedback=NOT_APPLIED_AFTER_REFUSAL, success=False, command=command
                    )
                )
                continue
            result = execute_work_item_command(store, command, owner=owner, turn=turn)
            results.append(result)
            if not result.success:
                refused = True
    stray = stray_command_lines(message)
    if stray:
        quoted = ", ".join(repr(line) for line in stray)
        results.append(
            WorkItemCommandResult(
                feedback=(
                    f"Work-item command not applied: {quoted} "
                    "(not in the command block at the end of the message). "
                    + _BLOCK_RULE
                ),
                success=False,
            )
        )
    return results


def apply_command(
    store: WorkItemStore, message: str, *, owner: str, turn: int
) -> Optional[WorkItemCommandResult]:
    """One-command convenience over `apply_commands`: the single result, or
    `None` when the message holds no command. Raises ValueError when the
    message yields several results (a block of several commands, or a block
    plus a stray command line), so a caller that handles one result never
    drops the rest; such callers must migrate to `apply_commands`.
    """
    results = apply_commands(store, message, owner=owner, turn=turn)
    if not results:
        return None
    if len(results) > 1:
        raise ValueError(
            f"the message yields {len(results)} work-item command results; "
            "use apply_commands"
        )
    return results[0]


def end_session_block(store: WorkItemStore, owner: str) -> List[Dict[str, Any]]:
    """Items that block `owner` from ending the session (non-Done items they own)."""
    return store.blocking_for_owner(owner)


def delegation_allowed(
    store: WorkItemStore,
    tracker_blocks: List[Dict[str, Any]],
    owner: str,
    target: str,
) -> Optional[str]:
    """Whether `owner` may delegate to `target` right now: a refusal reason,
    or None when the delegation may proceed.

    Refused when the target is the owner itself, and when the owner has an
    In-progress item it did code work under — a block of that item
    (`tracker_blocks`, the block tracker's current blocks) whose `agents`
    include the owner and whose `action_ids` are non-empty. The one who did
    the work writes the item's completion summary; a delegation must not
    push that onto the next agent (observed in practice: every stage was
    "closed" by the agent after it, see session ac59f7b8). Items the owner
    has NOT worked on (a plan opened for the next agent) are not a reason to
    refuse; `transfer_on_delegation` hands them over once this check passes.
    """
    if target == owner:
        return "Delegation refused: you cannot delegate to yourself."
    for summary in store.list():
        if summary.get("owner") != owner or summary.get("status") != "In progress":
            continue
        item_id = int(summary["id"])
        worked = any(
            block["work_item_id"] == item_id
            and owner in block["agents"]
            and bool(block["action_ids"])
            for block in tracker_blocks
        )
        if worked:
            return (
                f"Delegation refused: close work item #{item_id} first — you did "
                "the work, so you write its completion summary "
                f'(close_work_item {item_id} "<summary>"). Delegate in your '
                "next message."
            )
    return None


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

    Runs only after `delegation_allowed` returned None, so every item it
    moves is one the delegating agent did no code work under (a plan opened
    for the receiving agent, for instance).

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
