"""Evidence for a work-item review: the code that ran for the item, its
outputs, and the files it produced.

A work-item review used to send the evaluator only the item itself and the
git diff of `items/N.json`, so every review was rejected for "no evidence"
even when the code had run and produced artifacts. This module builds the
`evidence` section of the review payload from the session's block index
(`blocks.json`, see `execution/blocks.py`) and a `code_lookup` that resolves
an action id to the executed code and its outputs.

The lookup differs per engine: the web server keeps `session.code_events`
in memory, the CLI has only its `events.jsonl` log (written by
`cli/run_cli.py`). Both produce the same evidence shape:

    {
      "blocks": [
        {"block_id", "attempt", "status", "agents", "turn_start", "turn_end",
         "actions": [{"action_id", "agent", "success", "source", "stdout",
                      "stderr"} ...],
         "artifact_paths": [str]}
        ...
      ],
      "truncated": bool
    }

An action whose record the lookup cannot find is listed with `source: null`
and `missing_record: true`; it is never skipped. Source and outputs are
capped (head and tail kept, with a marker); `truncated` says whether any cap
applied. `drop_oldest_outputs` strips the outputs of the oldest attempt so a
payload that is still over the token budget can shrink further (the caller
records that in `evidence["note"]`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SOURCE_CAP = 4000
STDOUT_CAP = 2000
STDERR_CAP = 2000

# The CLI's runner event log; it sits in the session artifacts directory,
# next to `work-items/` and `blocks.json`.
CLI_EVENT_LOG_FILENAME = "events.jsonl"

CodeLookup = Callable[[str], Optional[Dict[str, Any]]]

_RECORD_KEYS = ("agent", "success", "source", "stdout", "stderr")


def cap_text(text: str, cap: int) -> Tuple[str, bool]:
    """Cap `text` at about `cap` characters, keeping its head and tail.

    Returns `(text, False)` when it fits, else the head and tail joined by a
    marker that states how many characters were cut, and True.
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    if len(text) <= cap:
        return text, False
    head = cap // 2
    tail = cap - head
    omitted = len(text) - cap
    marker = f"\n... [{omitted} characters truncated] ...\n"
    return text[:head] + marker + text[len(text) - tail :], True


def _action_evidence(
    action_id: str, record: Optional[Dict[str, Any]], failed: bool
) -> Tuple[Dict[str, Any], bool]:
    if record is None:
        return (
            {
                "action_id": action_id,
                "agent": None,
                # The block index knows the outcome even when the code and
                # its outputs are gone.
                "success": not failed,
                "source": None,
                "stdout": None,
                "stderr": None,
                "missing_record": True,
            },
            False,
        )
    missing = [key for key in _RECORD_KEYS if key not in record]
    if missing:
        raise KeyError(
            f"code record for action {action_id!r} lacks {missing}; "
            f"expected keys {list(_RECORD_KEYS)}"
        )
    source, source_cut = cap_text(str(record["source"]), SOURCE_CAP)
    stdout, stdout_cut = cap_text(str(record["stdout"]), STDOUT_CAP)
    stderr, stderr_cut = cap_text(str(record["stderr"]), STDERR_CAP)
    return (
        {
            "action_id": action_id,
            "agent": record["agent"],
            "success": bool(record["success"]),
            "source": source,
            "stdout": stdout,
            "stderr": stderr,
        },
        source_cut or stdout_cut or stderr_cut,
    )


def build_block_evidence(
    blocks_index: Dict[str, Any],
    item_id: int,
    code_lookup: CodeLookup,
) -> Dict[str, Any]:
    """Evidence for every block of work item `item_id`, in index order.

    `blocks_index` is a parsed `blocks.json` (`execution.blocks.load_blocks`).
    `code_lookup(action_id)` returns `{agent, success, source, stdout,
    stderr}` or None for an action it has no record of.
    """
    blocks: List[Dict[str, Any]] = []
    truncated = False
    for block in blocks_index["blocks"]:
        if block["work_item_id"] != item_id:
            continue
        failed = set(block["failed_action_ids"])
        actions: List[Dict[str, Any]] = []
        for action_id in block["action_ids"]:
            action, action_truncated = _action_evidence(
                action_id, code_lookup(action_id), action_id in failed
            )
            truncated = truncated or action_truncated
            actions.append(action)
        blocks.append(
            {
                "block_id": block["block_id"],
                "attempt": block["attempt"],
                "status": block["status"],
                "agents": list(block["agents"]),
                "turn_start": block["turn_start"],
                "turn_end": block["turn_end"],
                "actions": actions,
                "artifact_paths": list(block["artifact_paths"]),
            }
        )
    return {"blocks": blocks, "truncated": truncated}


def drop_oldest_outputs(evidence: Dict[str, Any]) -> Optional[str]:
    """Strip source/stdout/stderr from the oldest block that still has them.

    The block keeps its action ids, outcomes and artifact paths and gets
    `outputs_dropped: true`. Returns that block's id, or None when every
    block's outputs are already gone (nothing more can be dropped).
    """
    for block in evidence["blocks"]:
        if block.get("outputs_dropped"):
            continue
        if not any(
            action.get("source") is not None
            or action.get("stdout") is not None
            or action.get("stderr") is not None
            for action in block["actions"]
        ):
            continue
        for action in block["actions"]:
            action["source"] = None
            action["stdout"] = None
            action["stderr"] = None
        block["outputs_dropped"] = True
        return str(block["block_id"])
    return None


def code_lookup_from_records(records: List[Dict[str, Any]]) -> CodeLookup:
    """A lookup over records that already carry `action_id` (web sessions'
    `code_events`); records without one (written before action ids were
    recorded) cannot be matched and are left out."""
    by_action: Dict[str, Dict[str, Any]] = {}
    for record in records:
        action_id = record.get("action_id")
        if action_id is None:
            continue
        by_action[action_id] = {key: record[key] for key in _RECORD_KEYS}
    return by_action.get


def read_cli_event_log(events_path: Path) -> Dict[str, Dict[str, Any]]:
    """Join `code_submitted` and `code_result` lines of a CLI `events.jsonl`
    by action id. Raises FileNotFoundError when the log is absent and
    ValueError on a malformed line."""
    events_path = Path(events_path)
    if not events_path.exists():
        raise FileNotFoundError(f"no CLI event log at {events_path}")
    submitted: Dict[str, Dict[str, Any]] = {}
    records: Dict[str, Dict[str, Any]] = {}
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{events_path}:{line_number} is not valid JSON"
                ) from exc
            event_type = event.get("event_type")
            if event_type not in ("code_submitted", "code_result"):
                continue
            payload = event["payload"]
            action_id = payload["action_id"]
            if event_type == "code_submitted":
                submitted[action_id] = {
                    "agent": event["agent_name"],
                    "source": payload["source"],
                }
                continue
            if action_id not in submitted:
                raise ValueError(
                    f"{events_path}:{line_number} has a code_result for action "
                    f"{action_id!r} without a code_submitted"
                )
            start = submitted[action_id]
            records[action_id] = {
                "agent": start["agent"],
                "success": bool(payload["success"]),
                "source": start["source"],
                "stdout": payload["stdout"],
                "stderr": payload["stderr"],
            }
    return records


def code_lookup_from_cli_event_log(events_path: Path) -> CodeLookup:
    return read_cli_event_log(events_path).get
