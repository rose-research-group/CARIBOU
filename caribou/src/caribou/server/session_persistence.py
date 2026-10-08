"""
Disk persistence for CARIBOU sessions.

Every non-token event triggers a session.json write so a crash or restart
leaves us with an up-to-date snapshot to reload.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

from caribou.server.models import (
    ArtifactRecord,
    BranchRestoreMode,
    CodeEventRecord,
    MessageRecord,
    ResolvedModelInfo,
    RecoveryMode,
    RecoveryStatus,
    SessionCreateRequest,
    SessionStatus,
)
from caribou.core.python_environments import (
    PythonEnvironmentKind,
    ResolvedPythonEnvironment,
)
from caribou.server.session_state import (
    SEQ_RESERVATION_BLOCK,
    SESSIONS_DIR,
    _Session,
    append_session_event,
    backfill_event_seq,
)

_log = logging.getLogger(__name__)


def session_file(session_id: str, sessions_dir: Path = SESSIONS_DIR) -> Path:
    return sessions_dir / session_id / "session.json"


def session_dir(session_id: str, sessions_dir: Path = SESSIONS_DIR) -> Path:
    return sessions_dir / session_id


def save_session(
    session: _Session,
    is_deleted: Callable[[str], bool],
    sessions_dir: Path = SESSIONS_DIR,
) -> None:
    """
    Write session state to disk. Called after every non-token event.

    `is_deleted` is passed in (rather than checked once) so we can bail out
    if the session gets deleted between the check and the write.
    """
    if is_deleted(session.id):
        return
    try:
        path = session_file(session.id, sessions_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "caribou.web_session.v5",
            "phase": session.phase,
            "brief": session.brief,
            "id": session.id,
            "name": session.name,
            "config": session.config.model_dump(mode="json"),
            "resolved_model": (
                session.resolved_model.model_dump()
                if session.resolved_model is not None
                else None
            ),
            "resolved_evaluator_model": (
                session.resolved_evaluator_model.model_dump()
                if session.resolved_evaluator_model is not None
                else None
            ),
            "evaluator_model_revision": session.evaluator_model_revision,
            "python_environment": session.python_environment.model_dump(mode="json"),
            "status": session.status.value,
            "current_agent": session.current_agent,
            "current_turn": session.current_turn,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": [m.model_dump() for m in session.messages],
            "artifacts": [a.model_dump() for a in session.artifacts],
            "code_events": [c.model_dump() for c in session.code_events],
            "events": session.events,
            "event_seq_reserved": session.event_seq + SEQ_RESERVATION_BLOCK,
            "parent_session_id": session.parent_session_id,
            "forked_from_checkpoint_id": session.forked_from_checkpoint_id,
            "forked_from_block_id": session.forked_from_block_id,
            "branch_restore_mode": (
                session.branch_restore_mode.value
                if session.branch_restore_mode is not None
                else None
            ),
            "branch_instruction": session.branch_instruction,
            "attempt_number": session.attempt_number,
            "attempts": session.attempts,
            "recovery_mode": (
                session.recovery_mode.value
                if session.recovery_mode is not None
                else None
            ),
            "recovery_status": session.recovery_status.value,
            "recovery_detail": session.recovery_detail,
            "recovery_phase": session.recovery_phase,
            "recovery_step": session.recovery_step,
            "recovery_total_steps": session.recovery_total_steps,
            "recovery_substep": session.recovery_substep,
            "recovery_substep_total": session.recovery_substep_total,
            "checkpoint_id": session.checkpoint_id,
            "checkpoint_turn": session.checkpoint_turn,
            "checkpoint_healthy": session.checkpoint_healthy,
        }
        if is_deleted(session.id):
            return
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if is_deleted(session.id):
            temporary.unlink(missing_ok=True)
            return
        os.replace(temporary, path)
        session.event_seq_saved = data["event_seq_reserved"] - SEQ_RESERVATION_BLOCK
    except Exception as exc:
        # Persistence failure must never crash the server, but do log it.
        _log.warning("Failed to persist session %s: %s", session.id, exc)


def backfill_artifact_record(raw: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Give an artifact record from a pre-v5 session file its `path` and `mtime_ns`.

    The old artifact scanner only looked at the top level of output_dir, so
    an old record's path is its filename. Its mtime_ns is read from the file
    on disk; a record whose file no longer exists raises (the session is then
    skipped on load) rather than being given an invented mtime.
    """
    has_path, has_mtime = "path" in raw, "mtime_ns" in raw
    if has_path and has_mtime:
        return raw
    if has_path or has_mtime:
        raise ValueError(
            f"artifact record {raw.get('id')!r} has only one of path/mtime_ns"
        )
    filename = raw["filename"]
    if Path(raw.get("local_path") or filename).name != filename:
        raise ValueError(
            f"legacy artifact record {raw.get('id')!r} local_path "
            f"{raw.get('local_path')!r} does not end in its filename {filename!r}"
        )
    file_path = output_dir / filename
    if not file_path.is_file():
        raise FileNotFoundError(
            f"legacy artifact record {raw.get('id')!r}: {file_path} no longer exists, "
            "so its mtime_ns cannot be backfilled"
        )
    return {**raw, "path": filename, "mtime_ns": file_path.stat().st_mtime_ns}


def load_persisted_sessions(sessions_dir: Path = SESSIONS_DIR) -> Dict[str, _Session]:
    """On startup, reload all sessions saved to disk."""
    sessions: Dict[str, _Session] = {}
    if not sessions_dir.exists():
        return sessions
    for sess_dir in sorted(sessions_dir.iterdir()):
        f = sess_dir / "session.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
            config = SessionCreateRequest(**data["config"])
            resolved_model = (
                ResolvedModelInfo(**data["resolved_model"])
                if data.get("resolved_model")
                else None
            )
            resolved_evaluator_model = (
                ResolvedModelInfo(**data["resolved_evaluator_model"])
                if data.get("resolved_evaluator_model")
                else resolved_model
            )
            python_environment = ResolvedPythonEnvironment.model_validate(
                data.get("python_environment")
                or {
                    "mode": "bundled",
                    "python_executable": "/usr/local/envs/rapids/bin/python",
                    "kind": PythonEnvironmentKind.conda.value,
                }
            )
            raw_status = data.get("status", "stopped")
            # Sessions that were mid-run when the server died are now stopped
            interrupted_recovery = raw_status == "recovering"
            if raw_status in ("running", "initializing", "recovering"):
                raw_status = "stopped"
            status = SessionStatus(raw_status)
            # v4 and older files have no seq on events and no reservation;
            # number their events in log order so every event carries seq.
            # Otherwise resume from the reservation, past any seq handed out
            # to tokens that were never saved.
            events = data.get("events", [])
            event_seq = backfill_event_seq(events, data.get("event_seq_reserved"))

            session = _Session(
                id=data["id"],
                name=data.get("name") or data["id"][:8],
                config=config,
                status=status,
                current_agent=data.get("current_agent", ""),
                current_turn=data.get("current_turn", 0),
                messages=[MessageRecord(**m) for m in data.get("messages", [])],
                artifacts=[
                    ArtifactRecord(**backfill_artifact_record(a, sess_dir / "outputs"))
                    for a in data.get("artifacts", [])
                ],
                code_events=[CodeEventRecord(**c) for c in data.get("code_events", [])],
                output_dir=sess_dir / "outputs",
                events=events,
                event_seq=event_seq,
                # The file on disk reserves nothing beyond event_seq, so the
                # first new event forces a save (new reservation).
                event_seq_saved=event_seq - SEQ_RESERVATION_BLOCK,
                event_condition=asyncio.Condition(),
                stop_flag=threading.Event(),
                cancel_response_flag=threading.Event(),
                user_input_queue=queue.Queue(),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                model_name=(resolved_model.model if resolved_model is not None else ""),
                resolved_model=resolved_model,
                evaluator_model_name=(
                    resolved_evaluator_model.model
                    if resolved_evaluator_model is not None
                    else ""
                ),
                resolved_evaluator_model=resolved_evaluator_model,
                evaluator_model_revision=max(
                    1, int(data.get("evaluator_model_revision", 1) or 1)
                ),
                python_environment=python_environment,
                parent_session_id=data.get("parent_session_id"),
                forked_from_checkpoint_id=data.get("forked_from_checkpoint_id"),
                # Absent from files written before branching existed.
                forked_from_block_id=data.get("forked_from_block_id"),
                branch_restore_mode=(
                    BranchRestoreMode(data["branch_restore_mode"])
                    if data.get("branch_restore_mode")
                    else None
                ),
                branch_instruction=data.get("branch_instruction"),
                attempt_number=max(1, int(data.get("attempt_number", 1))),
                recovery_mode=(
                    RecoveryMode(data["recovery_mode"])
                    if data.get("recovery_mode")
                    else None
                ),
                recovery_status=(
                    RecoveryStatus.failed
                    if interrupted_recovery
                    else RecoveryStatus(data.get("recovery_status", "none"))
                ),
                recovery_detail=(
                    "Recovery was interrupted by a server restart. Retry either recovery mode."
                    if interrupted_recovery
                    else data.get("recovery_detail")
                ),
                recovery_phase=data.get("recovery_phase"),
                recovery_step=int(data.get("recovery_step", 0) or 0),
                recovery_total_steps=int(data.get("recovery_total_steps", 0) or 0),
                recovery_substep=data.get("recovery_substep"),
                recovery_substep_total=data.get("recovery_substep_total"),
                checkpoint_id=data.get("checkpoint_id"),
                checkpoint_turn=data.get("checkpoint_turn"),
                checkpoint_healthy=bool(data.get("checkpoint_healthy", False)),
                attempts=list(data.get("attempts", [])),
                # v3 files have neither key — a session saved before WS-5
                # loads as phase="execution", brief=None, which is correct:
                # it never had a briefing phase to begin with.
                # brief_policy is intentionally not persisted/restored here:
                # like work_item_policy, it's re-resolved from the
                # blueprint on the next resume (session.agent_system is
                # also None until then).
                phase=data.get("phase", "execution"),
                brief=data.get("brief"),
            )
            # If the session was interrupted, record that in the event log
            if raw_status != data.get("status"):
                append_session_event(
                    session,
                    {
                        "type": "status_change",
                        "session_id": session.id,
                        "turn": session.current_turn,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"status": "stopped", "reason": "server restarted"},
                    },
                )
            sessions[session.id] = session
        except Exception as exc:
            # Skip but log — silent skips have masked schema drift and
            # disk corruption in prior incidents.
            _log.warning("Skipping corrupt session file %s: %s", f, exc)
    return sessions
