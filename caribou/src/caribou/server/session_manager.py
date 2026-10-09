"""
SessionManager: owns all active CARIBOU sessions on the server.

Each session wraps an agent run: it holds the sandbox, LLM client, agent
system, event log, and the asyncio machinery to stream events to WebSocket
clients. Multiple WebSocket connections can observe the same session — they
all see full history on connect then live events from that point forward.

Persistence, session-record shape, and bootstrap helpers live in sibling
modules:
  - session_state.py       — the `_Session` dataclass and constants
  - session_persistence.py — save/load session state to/from disk
  - session_setup.py       — blueprint / LLM / sandbox construction
"""

from __future__ import annotations

import asyncio
import copy
import logging
import queue
import shutil
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv

from caribou.config import ENV_FILE
from caribou.core.python_environments import (
    assert_environment_unchanged,
    bundled_python_environment,
    resolved_host_environment,
    validate_python_environment_path,
)
from caribou.execution.evaluation import (
    build_evaluation_payload,
    evaluate_work_item,
    resolve_evaluator_agent,
    run_evaluation,
    evaluation_response_metadata,
)
from caribou.execution.session_brief import resolve_brief_policy
from caribou.execution.blocks import (
    BLOCKS_FILENAME,
    BlockError,
    fork_blocks,
    inherit_blocks,
    init_blocks,
    load_blocks,
)
from caribou.execution.review_evidence import (
    build_block_evidence,
    code_lookup_from_records,
)
from caribou.execution.user_input import UserTurn
from caribou.execution.work_item_runtime import copy_work_items
from caribou.execution.work_items import (
    HUMAN_REVIEWER,
    copy_work_items_at,
    WorkItemPolicy,
    WorkItemStore,
)
from caribou.execution.token_utils import estimate_tokens
from caribou.server.models import (
    ArtifactRecord,
    ArtifactType,
    BranchRequest,
    BranchRestoreMode,
    BranchSummary,
    CodeEventRecord,
    EvaluatorModelState,
    EvaluatorModelUpdateRequest,
    EvaluationResult,
    MemoryStrategy,
    RecoveryMode,
    RecoveryStatus,
    MessageRecord,
    QueuedMessage,
    SessionCreateRequest,
    SessionForkRequest,
    SessionMode,
    SessionResponse,
    SessionResumeRequest,
    SessionStatus,
)
from caribou.server.session_persistence import (
    load_persisted_sessions,
    save_session,
    session_dir,
)
from caribou.server.session_setup import (
    build_evaluator_client,
    build_llm_client,
    build_sandbox,
    find_blueprint,
    resolve_model_info,
    resolve_evaluator_model_info,
)
from caribou.server.session_state import (
    read_max_active_sessions,
    SANDBOX_DATA_PATH,
    SANDBOX_REF_DATA_PATH,
    SESSIONS_DIR,
    SKIP_PERSIST_TYPES,
    _Session,
    append_session_event,
    seq_reservation_exhausted,
)
from caribou.execution.session_recovery import (
    bootstrap_anndata,
    capture_checkpoint,
    LEDGER_BASE_ORIGINAL,
    checkpoint_dataset_path,
    copy_output_tree,
    copy_unchanged_artifacts,
    literal_replay,
    load_checkpoint,
    publish_checkpoint_pointer,
    smart_rebuild,
    verify_fingerprint,
)

# Backwards-compatible re-exports for callers that reach into this module.
_SESSIONS_DIR = SESSIONS_DIR
RECOVERY_TOTAL_STEPS = 8


def _create_session_logger(session_id: str, session_dir_path) -> logging.Logger:
    """Create a logger isolated to one session's stderr and session.log."""
    short = session_id[:8]
    logger = logging.getLogger(f"caribou.session.{short}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_file = session_dir_path / "session.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt=f"%(asctime)s.%(msecs)03d  [{short}]  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter(
            fmt=f"%(asctime)s  [session {short}]  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(stream_handler)
    return logger


def _close_session_logger(logger: logging.Logger) -> None:
    """Flush and close every handler owned by a per-session logger."""
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)


class UnknownWorkItemOwner(ValueError):
    """A human ticket named an owner that is not an agent in the blueprint."""


class SessionLimitReached(ValueError):
    """Starting another sandbox container would exceed the Settings limit."""


class BranchConflict(ValueError):
    """A branch request whose preconditions do not hold (HTTP 409)."""


# Sessions in these states hold (or are about to start) a sandbox container.
_ACTIVE_WITH_SANDBOX = {
    SessionStatus.initializing,
    SessionStatus.idle,
    SessionStatus.running,
    SessionStatus.recovering,
}
# Counted even before `sandbox_manager` exists: their container is starting,
# so two quick creates cannot both pass the limit (A6).
_ACTIVE_STARTING = {SessionStatus.initializing, SessionStatus.recovering}
_BRANCHABLE_STATUSES = {SessionStatus.idle, SessionStatus.stopped, SessionStatus.error}
# The RecoveryMode a branch child reports for each restore mode (resume and
# retry keep accepting only smart/literal_replay).
_BRANCH_RECOVERY_MODE = {
    BranchRestoreMode.replay: RecoveryMode.literal_replay,
    BranchRestoreMode.checkpoint: None,
    BranchRestoreMode.llm_regen: RecoveryMode.smart,
}
# Source events a fork or branch child keeps in its own log.
_RETAINED_EVENT_TYPES = {
    "message_complete",
    "system_message",
    "recovery_completed",
    "agent_switch",
    "code_submitted",
    "code_result",
    "artifact",
    "error",
}


class SessionManager:
    def __init__(self) -> None:
        self._deleted_session_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, _Session] = load_persisted_sessions()

    # ------------------------------------------------------------------
    # Persistence adapters (keep the manager as the single call site)
    # ------------------------------------------------------------------

    def _is_deleted(self, session_id: str) -> bool:
        return session_id in getattr(self, "_deleted_session_ids", set())

    def _save_session(self, session: _Session) -> None:
        save_session(session, self._is_deleted, _SESSIONS_DIR)

    def _session_dir(self, session_id: str):
        return session_dir(session_id, _SESSIONS_DIR)

    # ------------------------------------------------------------------
    # Active-session limit (Settings: CARIBOU_MAX_ACTIVE_SESSIONS)
    # ------------------------------------------------------------------

    def _active_session_count(self) -> int:
        return sum(
            1
            for session in self._sessions.values()
            if session.status in _ACTIVE_STARTING
            or (
                session.sandbox_manager is not None
                and session.status in _ACTIVE_WITH_SANDBOX
            )
        )

    def _check_session_limit(self) -> None:
        """Raise SessionLimitReached when no container slot is free.

        Callers hold `self._lock` and, without awaiting in between, mark the
        session that takes the slot as initializing/recovering (the
        reservation), so concurrent requests cannot both pass.
        """
        limit = read_max_active_sessions(ENV_FILE)
        if limit is None:
            return
        active = self._active_session_count()
        if active >= limit:
            raise SessionLimitReached(
                f"Active session limit reached ({active} of {limit}). Stop a "
                "session or raise the limit in Settings."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_session(self, config: SessionCreateRequest) -> SessionResponse:
        import queue
        import threading

        requested_python_environment = None
        if config.python_environment_path:
            requested_python_environment = resolved_host_environment(
                validate_python_environment_path(config.python_environment_path)
            )
            config = config.model_copy(
                update={"python_environment_path": requested_python_environment.path}
            )

        session_id = str(uuid4())
        output_dir = SESSIONS_DIR / session_id / "outputs"

        session = _Session(
            id=session_id,
            name=(config.name or "").strip() or session_id[:8],
            config=config,
            status=SessionStatus.initializing,
            current_agent="",
            current_turn=0,
            messages=[],
            artifacts=[],
            code_events=[],
            output_dir=output_dir,
            events=[],
            event_condition=asyncio.Condition(),
            stop_flag=threading.Event(),
            cancel_response_flag=threading.Event(),
            user_input_queue=queue.Queue(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resolved_model=resolve_model_info(config),
            resolved_evaluator_model=resolve_evaluator_model_info(
                config, worker_resolved=resolve_model_info(config)
            ),
            **(
                {"python_environment": requested_python_environment}
                if requested_python_environment is not None
                else {}
            ),
            attempts=[
                {
                    "attempt_number": 1,
                    "kind": "initial",
                    "started_at": datetime.utcnow().isoformat(),
                    "mode": config.mode.value,
                    "llm_backend": config.llm_backend,
                }
            ],
        )

        async with self._lock:
            # Check and reserve with no await in between: the slot is taken
            # by registering the initializing session.
            self._check_session_limit()
            output_dir.mkdir(parents=True, exist_ok=True)
            # Before the runner starts, so the page's first /blocks request
            # already sees a block-recording session.
            init_blocks(output_dir.parent / BLOCKS_FILENAME, session_id)
            self._deleted_session_ids.discard(session_id)
            self._sessions[session_id] = session

        session.logger = _create_session_logger(session_id, output_dir.parent)
        session.logger.info(
            "Session created | backend: %s | model: %s | mode: %s | sandbox: %s | log: %s",
            config.llm_backend,
            (
                session.resolved_model.model
                if session.resolved_model is not None
                else "unresolved"
            ),
            config.mode.value,
            config.sandbox_type.value,
            output_dir.parent / "session.log",
        )
        self._save_session(session)
        asyncio.create_task(self._initialize_session(session))
        return session.to_response()

    def get_session(self, session_id: str) -> Optional[_Session]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[SessionResponse]:
        return [s.to_response() for s in self._sessions.values()]

    async def resume_session(
        self, session_id: str, request: SessionResumeRequest
    ) -> SessionResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        if session.status != SessionStatus.stopped:
            raise ValueError("Only a stopped session can be resumed")
        self._validate_recovery_request(request)
        if session.recovery_task and not session.recovery_task.done():
            raise ValueError("Session recovery is already in progress")

        async with self._lock:
            self._check_session_limit()
            # Validated before the reservation so a bad request changes nothing.
            self._apply_target_mode(session, request)
            session.status = SessionStatus.recovering
        session.attempt_number += 1
        session.attempts.append(
            {
                "attempt_number": session.attempt_number,
                "kind": "resume",
                "started_at": datetime.utcnow().isoformat(),
                "source_checkpoint_id": session.checkpoint_id,
                "recovery_mode": request.recovery_mode.value,
            }
        )
        session.recovery_mode = request.recovery_mode
        session.recovery_status = RecoveryStatus.recovering
        session.recovery_phase = "checkpoint"
        session.recovery_step = 1
        session.recovery_total_steps = RECOVERY_TOTAL_STEPS
        session.recovery_substep = None
        session.recovery_substep_total = None
        session.recovery_detail = "Loading the latest safe checkpoint."
        session.status = SessionStatus.recovering
        self._save_session(session)
        session.recovery_task = asyncio.create_task(
            self._recover_session(session, request)
        )
        return session.to_response()

    async def fork_session(
        self, source_id: str, request: SessionForkRequest
    ) -> SessionResponse:
        source = self._sessions.get(source_id)
        if source is None:
            raise KeyError("Session not found")
        if not request.name.strip():
            raise ValueError("Fork name cannot be blank")
        self._validate_recovery_request(request)
        child_id = str(uuid4())
        config_updates: Dict[str, Any] = {
            "name": request.name.strip(),
            "llm_backend": request.llm_backend or source.config.llm_backend,
            "initial_prompt": None,
        }
        if request.llm_backend and request.llm_backend != source.config.llm_backend:
            # Provider-specific identifiers must not leak across a backend
            # switch when the new provider is using its configured default.
            config_updates["model_name"] = None
            config_updates["ollama_model"] = None
        if request.model_name and request.model_name.strip():
            config_updates["model_name"] = request.model_name.strip()
        if request.ollama_model and request.ollama_model.strip():
            config_updates["ollama_model"] = request.ollama_model.strip()
        if request.evaluator_model is not None:
            config_updates["evaluator_model"] = request.evaluator_model
        selected_python_environment = source.python_environment.model_copy(deep=True)
        if "python_environment_path" in request.model_fields_set:
            if request.python_environment_path is None:
                config_updates["python_environment_path"] = None
                selected_python_environment = bundled_python_environment()
            else:
                candidate = validate_python_environment_path(
                    request.python_environment_path
                )
                config_updates["python_environment_path"] = candidate.path
                selected_python_environment = resolved_host_environment(candidate)
        child_config = source.config.model_copy(update=config_updates)
        child_resolved_model = resolve_model_info(child_config)
        child_resolved_evaluator = resolve_evaluator_model_info(
            child_config, worker_resolved=child_resolved_model
        )
        child = _Session(
            id=child_id,
            name=request.name.strip(),
            config=child_config,
            status=SessionStatus.recovering,
            current_agent=source.current_agent,
            current_turn=source.current_turn,
            messages=[],
            artifacts=[],
            code_events=[],
            output_dir=SESSIONS_DIR / child_id / "outputs",
            events=[],
            event_condition=asyncio.Condition(),
            stop_flag=threading.Event(),
            cancel_response_flag=threading.Event(),
            user_input_queue=queue.Queue(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resolved_model=child_resolved_model,
            resolved_evaluator_model=child_resolved_evaluator,
            python_environment=selected_python_environment,
            parent_session_id=source.id,
            attempt_number=1,
            recovery_mode=request.recovery_mode,
            recovery_status=RecoveryStatus.awaiting_checkpoint,
            recovery_detail="Waiting for the source session's next safe turn boundary.",
            recovery_phase="awaiting_checkpoint",
            recovery_step=1,
            recovery_total_steps=RECOVERY_TOTAL_STEPS,
            attempts=[
                {
                    "attempt_number": 1,
                    "kind": "fork",
                    "started_at": datetime.utcnow().isoformat(),
                    "source_session_id": source.id,
                    "recovery_mode": request.recovery_mode.value,
                    "model_change_reason": request.model_change_reason,
                    "source_worker_model": (
                        source.resolved_model.model_dump()
                        if source.resolved_model is not None
                        else None
                    ),
                    "worker_model": (
                        child_resolved_model.model_dump()
                        if child_resolved_model is not None
                        else None
                    ),
                    "source_evaluator_model": (
                        source.resolved_evaluator_model.model_dump()
                        if source.resolved_evaluator_model is not None
                        else None
                    ),
                    "evaluator_model": (
                        child_resolved_evaluator.model_dump()
                        if child_resolved_evaluator is not None
                        else None
                    ),
                }
            ],
        )
        # Validate/resolve mode before publishing the child so invalid requests
        # cannot leave an orphaned recovering session in the registry.
        self._apply_target_mode(child, request)
        async with self._lock:
            self._check_session_limit()
            child.output_dir.mkdir(parents=True, exist_ok=True)
            # A placeholder until _fork_work_items copies the parent's blocks.
            init_blocks(child.output_dir.parent / BLOCKS_FILENAME, child_id)
            self._deleted_session_ids.discard(child_id)
            self._sessions[child_id] = child
        child.logger = _create_session_logger(child_id, child.output_dir.parent)
        self._save_session(child)
        child.recovery_task = asyncio.create_task(
            self._complete_fork(source, child, request)
        )
        return child.to_response()

    # ------------------------------------------------------------------
    # Branching from a block (contract §5)
    # ------------------------------------------------------------------

    async def branch_session(
        self, source_id: str, block_id: str, request: BranchRequest
    ) -> SessionResponse:
        """Create a branch child that starts at `block_id`'s entry checkpoint.

        Every precondition failure raises BranchConflict (409); an unknown
        session raises KeyError (404). The child's restore runs in the
        background and reports through the recovery_* fields and events.
        """
        source = self._sessions.get(source_id)
        if source is None:
            raise KeyError("Session not found")
        if source.status not in _BRANCHABLE_STATUSES:
            raise BranchConflict(
                f"Session {source_id} is {SessionStatus(source.status).value}; "
                "branch only from a session that is idle, stopped or in error."
            )
        index = load_blocks(source.output_dir.parent / BLOCKS_FILENAME)
        blocks = index["blocks"] if index is not None else []
        block = next((item for item in blocks if item["block_id"] == block_id), None)
        if block is None:
            raise BranchConflict(f"Block {block_id} does not exist in session {source_id}.")
        inherited = block.get("inherited_from")
        if inherited is not None:
            # Its entry checkpoint lives in the parent's .checkpoints.
            raise BranchConflict(
                f"Block {block_id} is inherited from {inherited['session_id']}; "
                "branch from it in that session."
            )
        entry = block.get("entry")
        if not entry or not entry.get("checkpoint_id"):
            raise BranchConflict(
                f"Block {block_id} was recorded before branching support; it has "
                "no entry checkpoint."
            )
        running = [
            item["block_id"]
            for item in blocks
            if item["index"] < block["index"] and item["status"] == "running"
        ]
        if running:
            raise BranchConflict(
                f"Blocks {running} before {block_id} are still running, so they "
                "cannot be inherited; stop the source session first."
            )
        checkpoint = load_checkpoint(source.output_dir, entry["checkpoint_id"])
        if checkpoint is None:
            raise BranchConflict(
                f"Entry checkpoint {entry['checkpoint_id']} of block {block_id} is "
                "missing from the session's checkpoints."
            )
        if checkpoint.get("pin_block_id") != block_id:
            raise BranchConflict(
                f"Checkpoint {entry['checkpoint_id']} is pinned to "
                f"{checkpoint.get('pin_block_id')!r}, not to block {block_id}."
            )
        self._check_branch_mode(
            checkpoint,
            block_id,
            request.restore_mode,
            acknowledge_unverified=request.acknowledge_unverified,
        )

        mode = request.restore_mode
        suffix = f" · branch from {block_id}"
        if request.name is not None:
            name = request.name.strip()
        else:
            # The persisted config caps names at 120 characters; shorten the
            # source name rather than the block reference.
            name = source.name[: max(0, 120 - len(suffix))].rstrip() + suffix
        child_id = str(uuid4())
        child_config = source.config.model_copy(
            update={"name": name, "initial_prompt": None}
        )
        child_resolved_model = resolve_model_info(child_config)
        now = datetime.utcnow()
        child = _Session(
            id=child_id,
            name=name,
            config=child_config,
            status=SessionStatus.recovering,
            current_agent=str(checkpoint.get("current_agent") or source.current_agent),
            current_turn=int(checkpoint["turn"]),
            messages=[],
            artifacts=[],
            code_events=[],
            output_dir=SESSIONS_DIR / child_id / "outputs",
            events=[],
            event_condition=asyncio.Condition(),
            stop_flag=threading.Event(),
            cancel_response_flag=threading.Event(),
            user_input_queue=queue.Queue(),
            created_at=now,
            updated_at=now,
            resolved_model=child_resolved_model,
            resolved_evaluator_model=resolve_evaluator_model_info(
                child_config, worker_resolved=child_resolved_model
            ),
            python_environment=source.python_environment.model_copy(deep=True),
            parent_session_id=source.id,
            forked_from_checkpoint_id=checkpoint["checkpoint_id"],
            forked_from_block_id=block_id,
            branch_restore_mode=mode,
            branch_instruction=request.instruction,
            attempt_number=1,
            recovery_mode=_BRANCH_RECOVERY_MODE[mode],
            recovery_status=RecoveryStatus.recovering,
            recovery_detail="Copying the block's entry checkpoint.",
            recovery_phase="copying_checkpoint",
            recovery_step=1,
            recovery_total_steps=RECOVERY_TOTAL_STEPS,
            attempts=[
                {
                    "attempt_number": 1,
                    "kind": "branch",
                    "started_at": now.isoformat(),
                    "source_session_id": source.id,
                    "source_block_id": block_id,
                    "source_checkpoint_id": checkpoint["checkpoint_id"],
                    "restore_mode": mode.value,
                    "acknowledge_unverified": request.acknowledge_unverified,
                    # Enough of the block to re-run the restore on a retry,
                    # even after the source session is deleted.
                    "source_block": {
                        "block_id": block_id,
                        "index": int(block["index"]),
                        "title": str(block.get("title") or ""),
                        "entry": dict(block["entry"]),
                    },
                }
            ],
        )
        async with self._lock:
            self._check_session_limit()
            child.output_dir.mkdir(parents=True, exist_ok=True)
            # A placeholder until inherit_blocks copies the parent's blocks.
            init_blocks(child.output_dir.parent / BLOCKS_FILENAME, child_id)
            self._deleted_session_ids.discard(child_id)
            self._sessions[child_id] = child
        child.logger = _create_session_logger(child_id, child.output_dir.parent)
        self._save_session(child)
        child.recovery_task = asyncio.create_task(
            self._complete_branch(source, child, checkpoint, block)
        )
        return child.to_response()

    @staticmethod
    def _check_branch_mode(
        checkpoint: Dict[str, Any],
        block_id: str,
        mode: BranchRestoreMode,
        *,
        acknowledge_unverified: bool,
    ) -> None:
        """The restore mode's requirements on the entry checkpoint (§4, A4)."""
        if mode == BranchRestoreMode.replay:
            base = checkpoint.get("ledger_base")
            if base != LEDGER_BASE_ORIGINAL:
                raise BranchConflict(
                    f"Replay needs an action ledger that runs from the original "
                    f"dataset; the entry checkpoint of block {block_id} records "
                    f"ledger_base={base!r}. Use checkpoint or llm_regen."
                )
            if checkpoint.get("fingerprint") is None and not acknowledge_unverified:
                raise BranchConflict(
                    f"The entry checkpoint of block {block_id} has no AnnData "
                    "fingerprint, so a replay cannot be verified. Set "
                    "acknowledge_unverified to replay anyway."
                )
            return
        if not checkpoint.get("complete"):
            raise BranchConflict(
                f"{mode.value} needs a complete entry checkpoint, but the "
                f"checkpoint of block {block_id} is incomplete"
                + (
                    f": {checkpoint['capture_error']}"
                    if checkpoint.get("capture_error")
                    else "."
                )
            )

    def list_branches(self, session_id: str) -> List[BranchSummary]:
        """Direct branch children of `session_id`, oldest first."""
        if session_id not in self._sessions:
            raise KeyError("Session not found")
        children = sorted(
            (
                item
                for item in self._sessions.values()
                if item.parent_session_id == session_id
                and item.forked_from_block_id is not None
            ),
            key=lambda item: item.created_at,
        )
        return [
            BranchSummary(
                session_id=item.id,
                name=item.name or item.id[:8],
                status=item.status,
                recovery_status=item.recovery_status,
                forked_from_block_id=item.forked_from_block_id,
                branch_restore_mode=item.branch_restore_mode,
                branch_instruction=item.branch_instruction,
                created_at=item.created_at,
            )
            for item in children
        ]

    def _copy_brief(self, source: _Session, child: _Session) -> None:
        """Carry the source's frozen brief (Goal) into a fork or branch child."""
        source_brief = source.output_dir.parent / "brief.json"
        if source_brief.is_file():
            shutil.copy2(source_brief, child.output_dir.parent / "brief.json")
        child.brief = copy.deepcopy(source.brief)
        child.phase = source.phase

    def _copy_branch_transcript(
        self, source: _Session, child: _Session, *, entry_turn: int
    ) -> None:
        """The child's visible transcript up to the block's entry.

        The entry checkpoint is taken in turn `entry_turn` before the
        assistant message that opened the block, so its history holds every
        earlier turn plus turn `entry_turn`'s user message. The transcript
        matches: earlier turns in full, and only the user message of
        `entry_turn`.
        """

        def keep(turn: int, role: str) -> bool:
            return turn < entry_turn or (turn == entry_turn and role == "user")

        child.messages = [
            MessageRecord(
                session_id=child.id,
                turn=item.turn,
                role=item.role,
                agent_name=item.agent_name,
                content=item.content,
                is_delegation=item.is_delegation,
            )
            for item in source.messages
            if keep(item.turn, item.role)
        ]
        child.code_events = [
            CodeEventRecord(
                session_id=child.id,
                turn=item.turn,
                agent_name=item.agent_name,
                source=item.source,
                stdout=item.stdout,
                stderr=item.stderr,
                success=item.success,
                duration_ms=item.duration_ms,
                action_id=item.action_id,
                block_id=item.block_id,
            )
            for item in source.code_events
            if item.turn < entry_turn
        ]
        for source_event in source.events:
            event_type = source_event.get("type")
            turn = int(source_event.get("turn", 0) or 0)
            if event_type not in _RETAINED_EVENT_TYPES:
                continue
            if event_type == "error" and bool(
                (source_event.get("data") or {}).get("fatal")
            ):
                continue
            role = (
                ((source_event.get("data") or {}).get("message") or {}).get("role")
                if event_type == "message_complete"
                else None
            )
            if not keep(turn, role or ""):
                continue
            child_event = copy.deepcopy(source_event)
            child_event["session_id"] = child.id
            # The parent's seq belongs to the parent's log.
            del child_event["seq"]
            append_session_event(child, child_event)

    def _branch_artifact_records(
        self, source: _Session, child: _Session, *, entry_turn: int
    ) -> List[ArtifactRecord]:
        """Records for the source artifacts that exist in the child's outputs
        after the restore, stamped with the child's own file mtimes."""
        records: List[ArtifactRecord] = []
        for item in source.artifacts:
            if item.turn >= entry_turn:
                continue
            path = child.output_dir / item.path
            if not path.is_file():
                continue
            records.append(
                ArtifactRecord(
                    session_id=child.id,
                    turn=item.turn,
                    type=item.type,
                    filename=item.filename,
                    mime_type=item.mime_type,
                    size_bytes=path.stat().st_size,
                    created_at=item.created_at,
                    local_path=str(path),
                    path=item.path,
                    mtime_ns=path.stat().st_mtime_ns,
                    action_id=item.action_id,
                )
            )
        return records

    async def _complete_branch(
        self,
        source: _Session,
        child: _Session,
        checkpoint: Dict[str, Any],
        block: Dict[str, Any],
    ) -> None:
        """Copy the child's starting state from the source, then restore it."""
        try:
            if self._is_deleted(child.id):
                return
            self._set_recovery_progress(
                child,
                phase="copying_checkpoint",
                detail="Copying the block's entry checkpoint, blocks, work items and brief.",
                step=1,
            )
            checkpoint_id = checkpoint["checkpoint_id"]
            child_root = child.output_dir.parent / ".checkpoints"
            child_root.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                shutil.copytree,
                source.output_dir.parent / ".checkpoints" / checkpoint_id,
                child_root / checkpoint_id,
            )
            # In the child, E is a clean starting point (its history ends
            # before the block's old attempt), so it is the child's latest
            # until the child's first turn boundary publishes its own.
            await asyncio.to_thread(
                publish_checkpoint_pointer, child.output_dir, checkpoint_id
            )
            child.checkpoint_id = checkpoint_id
            child.checkpoint_turn = int(checkpoint["turn"])
            child.checkpoint_healthy = bool(checkpoint.get("complete"))
            skipped: List[str] = []
            if child.branch_restore_mode != BranchRestoreMode.replay:
                # Replay regenerates the files, so its output dir starts empty.
                skipped = await asyncio.to_thread(
                    copy_unchanged_artifacts,
                    source.output_dir,
                    child.output_dir,
                    manifest=checkpoint.get("artifacts") or [],
                )
            if self._is_deleted(child.id):
                return
            await asyncio.to_thread(
                inherit_blocks,
                source.output_dir.parent / BLOCKS_FILENAME,
                child.output_dir.parent / BLOCKS_FILENAME,
                below_index=int(block["index"]),
                parent_session_id=source.id,
                child_session_id=child.id,
            )
            commit = block["entry"].get("work_items_commit")
            if commit is not None:
                child.work_item_store = await asyncio.to_thread(
                    copy_work_items_at,
                    source.output_dir.parent / "work-items",
                    child.output_dir.parent / "work-items",
                    commit=commit,
                    child_session_id=child.id,
                    forked_from_session_id=source.id,
                )
            # A null commit: the source had no work-item commits yet, and the
            # child's store is created empty on first use.
            await asyncio.to_thread(self._copy_brief, source, child)
            self._copy_branch_transcript(
                source, child, entry_turn=int(block["entry"]["turn"])
            )
            # A retry re-runs only the restore, so it needs this copy phase
            # to have completed.
            child.attempts[0]["prepared"] = True
            self._save_session(child)
            await self._restore_branch(child, checkpoint, block, skipped)
        except Exception as exc:
            await self._fail_branch(child, str(exc))

    async def _fail_branch(self, child: _Session, detail: str) -> None:
        """Mark a branch restore failed and release any sandbox it started."""
        if child.sandbox_manager is not None:
            sandbox, child.sandbox_manager = child.sandbox_manager, None
            try:
                await asyncio.to_thread(sandbox.stop_container)
            except Exception as stop_exc:
                detail = f"{detail} (stopping its sandbox also failed: {stop_exc})"
        child.status = SessionStatus.stopped
        child.recovery_status = RecoveryStatus.failed
        child.recovery_detail = detail
        child.recovery_phase = "failed"
        self._finish_latest_attempt(child, "recovery_failed")
        self._save_session(child)

    async def _restore_branch(
        self,
        child: _Session,
        checkpoint: Dict[str, Any],
        block: Dict[str, Any],
        skipped: List[str],
    ) -> None:
        """Run the child's restore mode (§4, A3) in its new sandbox, then
        launch its runner with the branch's first user turn."""
        mode = child.branch_restore_mode
        if mode == BranchRestoreMode.replay:
            dataset_source = Path(child.config.dataset_path)
        else:
            dataset_source = checkpoint_dataset_path(child.output_dir, checkpoint)
        runtime = await self._start_recovered_runtime(child, checkpoint, dataset_source)
        if runtime is None:
            return
        llm_client, model_name, driver = runtime
        emit = self._recovery_progress_emitter(child)
        self._set_recovery_progress(
            child,
            phase="restoring_dataset",
            detail=(
                "Loading the original dataset for replay."
                if mode == BranchRestoreMode.replay
                else "Loading the AnnData saved at the block's entry."
            ),
            step=6,
        )
        boot_ok, boot_detail = await asyncio.to_thread(
            bootstrap_anndata, child.sandbox_manager
        )
        if not boot_ok:
            raise RuntimeError(f"AnnData could not be loaded: {boot_detail}")
        if mode == BranchRestoreMode.replay:
            replayed, detail = await asyncio.to_thread(
                literal_replay,
                sandbox=child.sandbox_manager,
                checkpoint=checkpoint,
                emit=emit,
            )
            if not replayed:
                raise RuntimeError(f"Replay diverged from the recorded attempts: {detail}")
            expected = checkpoint.get("fingerprint")
            if expected is None:
                detail += " The AnnData fingerprint was not verified (no fingerprint; acknowledged)."
            else:
                self._set_recovery_progress(
                    child,
                    phase="verifying_fingerprint",
                    detail="Comparing the replayed AnnData with the checkpoint fingerprint.",
                    step=7,
                )
                matched, fingerprint_detail = await asyncio.to_thread(
                    verify_fingerprint, child.sandbox_manager, expected
                )
                if not matched:
                    raise RuntimeError(fingerprint_detail)
                detail += " " + fingerprint_detail
        elif mode == BranchRestoreMode.checkpoint:
            detail = "Restored the AnnData saved at the block's entry checkpoint."
        else:
            rebuilt, detail = await asyncio.to_thread(
                smart_rebuild,
                sandbox=child.sandbox_manager,
                llm_client=llm_client,
                model_name=model_name,
                current_agent_prompt=driver.get_full_prompt(None),
                checkpoint=checkpoint,
                emit=emit,
            )
            if not rebuilt:
                raise RuntimeError(f"LLM regeneration failed: {detail}")
        if skipped:
            detail += (
                " Output files not copied because they changed or are missing "
                f"since the checkpoint: {', '.join(skipped)}."
            )
        if self._is_deleted(child.id):
            await asyncio.to_thread(child.sandbox_manager.stop_container)
            return
        entry_turn = int(block["entry"]["turn"])
        source = self._sessions.get(child.parent_session_id or "")
        if source is not None:
            child.artifacts = self._branch_artifact_records(
                source, child, entry_turn=entry_turn
            )
        child.recovery_detail = detail
        self._set_recovery_progress(
            child,
            phase="finalizing",
            detail="Restore checks passed; starting the branch's runner.",
            step=8,
        )
        child.recovery_detail = detail
        child.recovery_status = RecoveryStatus.recovered
        child.recovery_phase = "completed"
        self._save_session(child)
        self._emit_recovery_completed(child)
        title = str(block.get("title") or "")
        await self._launch_recovered_runner(
            child,
            branch_turn=(
                f'[Branch from {block["block_id"]} "{title}" · {mode.value}]\n'
                f"{child.branch_instruction}"
            ),
        )

    async def retry_recovery(
        self, session_id: str, request: SessionResumeRequest
    ) -> SessionResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        if session.recovery_status not in {
            RecoveryStatus.partial,
            RecoveryStatus.failed,
        }:
            raise ValueError("Only partial or failed recovery can be retried")
        self._validate_recovery_request(request)
        if session.forked_from_block_id is not None:
            return await self._retry_branch_recovery(session, request)
        async with self._lock:
            # A retry starts a new container. The session's own partial
            # sandbox is not counted: its status is stopped.
            self._check_session_limit()
            self._apply_target_mode(session, request)
            session.status = SessionStatus.recovering
        if session.sandbox_manager is not None:
            try:
                await asyncio.to_thread(session.sandbox_manager.stop_container)
            except Exception:
                pass
            session.sandbox_manager = None
        self._finish_latest_attempt(session, "recovery_failed")
        session.attempt_number += 1
        session.attempts.append(
            {
                "attempt_number": session.attempt_number,
                "kind": "recovery_retry",
                "started_at": datetime.utcnow().isoformat(),
                "source_checkpoint_id": session.checkpoint_id,
                "recovery_mode": request.recovery_mode.value,
            }
        )
        session.recovery_mode = request.recovery_mode
        session.recovery_status = RecoveryStatus.recovering
        session.recovery_phase = "checkpoint"
        session.recovery_step = 1
        session.recovery_total_steps = RECOVERY_TOTAL_STEPS
        session.recovery_substep = None
        session.recovery_substep_total = None
        session.recovery_detail = (
            "Reloading the latest safe checkpoint for a clean retry."
        )
        self._save_session(session)
        session.recovery_task = asyncio.create_task(
            self._recover_session(session, request)
        )
        return session.to_response()

    async def _retry_branch_recovery(
        self, session: _Session, request: SessionResumeRequest
    ) -> SessionResponse:
        """Retry a branch child's restore against its block-entry checkpoint.

        The requested recovery mode maps to a branch mode (smart → llm_regen,
        literal_replay → replay); checkpoint mode is retried by branching
        again. The mode's requirements apply exactly as on the branch
        endpoint, and the branch instruction is queued again.
        """
        mode = {
            RecoveryMode.smart: BranchRestoreMode.llm_regen,
            RecoveryMode.literal_replay: BranchRestoreMode.replay,
        }[request.recovery_mode]
        branch_attempt = next(
            (item for item in session.attempts if item.get("kind") == "branch"), None
        )
        if branch_attempt is None or "source_block" not in branch_attempt:
            raise BranchConflict(
                f"Session {session.id} has no record of the block it branched "
                "from; branch from the parent again."
            )
        if not branch_attempt.get("prepared"):
            raise BranchConflict(
                f"Branch {session.id} never finished copying its blocks, work "
                "items and transcript from the parent; branch again instead."
            )
        block = branch_attempt["source_block"]
        block_id = session.forked_from_block_id
        checkpoint = load_checkpoint(session.output_dir, session.forked_from_checkpoint_id)
        if checkpoint is None:
            raise BranchConflict(
                f"Entry checkpoint {session.forked_from_checkpoint_id} of block "
                f"{block_id} is missing from the branch's checkpoints."
            )
        self._check_branch_mode(
            checkpoint,
            block_id,
            mode,
            acknowledge_unverified=bool(branch_attempt.get("acknowledge_unverified")),
        )
        if mode != BranchRestoreMode.replay and (
            self._sessions.get(session.parent_session_id or "") is None
        ):
            raise BranchConflict(
                f"{mode.value} copies output files from the parent session, "
                "which no longer exists."
            )
        async with self._lock:
            self._check_session_limit()
            self._apply_target_mode(session, request)
            session.status = SessionStatus.recovering
        if session.sandbox_manager is not None:
            sandbox, session.sandbox_manager = session.sandbox_manager, None
            await asyncio.to_thread(sandbox.stop_container)
        self._finish_latest_attempt(session, "recovery_failed")
        session.attempt_number += 1
        session.attempts.append(
            {
                "attempt_number": session.attempt_number,
                "kind": "branch_retry",
                "started_at": datetime.utcnow().isoformat(),
                "source_checkpoint_id": checkpoint["checkpoint_id"],
                "restore_mode": mode.value,
            }
        )
        session.branch_restore_mode = mode
        session.recovery_mode = _BRANCH_RECOVERY_MODE[mode]
        session.recovery_status = RecoveryStatus.recovering
        session.recovery_phase = "copying_checkpoint"
        session.recovery_step = 1
        session.recovery_total_steps = RECOVERY_TOTAL_STEPS
        session.recovery_substep = None
        session.recovery_substep_total = None
        session.recovery_detail = (
            f"Retrying the branch restore in {mode.value} mode from block "
            f"{block_id}'s entry checkpoint."
        )
        self._save_session(session)
        session.recovery_task = asyncio.create_task(
            self._retry_branch(session, checkpoint, block)
        )
        return session.to_response()

    async def _retry_branch(
        self, child: _Session, checkpoint: Dict[str, Any], block: Dict[str, Any]
    ) -> None:
        """Reset the child's outputs for `branch_restore_mode`, then restore."""
        try:
            # The runner never ran (the restore failed), so everything in the
            # output dir is a byproduct of the failed restore.
            if child.output_dir.exists():
                await asyncio.to_thread(shutil.rmtree, child.output_dir)
            child.output_dir.mkdir(parents=True)
            skipped: List[str] = []
            if child.branch_restore_mode != BranchRestoreMode.replay:
                source = self._sessions.get(child.parent_session_id or "")
                if source is None:
                    raise RuntimeError(
                        "the parent session no longer exists, so its output "
                        "files cannot be copied"
                    )
                skipped = await asyncio.to_thread(
                    copy_unchanged_artifacts,
                    source.output_dir,
                    child.output_dir,
                    manifest=checkpoint.get("artifacts") or [],
                )
            await self._restore_branch(child, checkpoint, block, skipped)
        except Exception as exc:
            await self._fail_branch(child, str(exc))

    async def accept_partial_recovery(self, session_id: str) -> SessionResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        if (
            session.recovery_status != RecoveryStatus.partial
            or session.sandbox_manager is None
        ):
            raise ValueError("Session has no usable partial recovery to accept")
        session.recovery_status = RecoveryStatus.accepted_partial
        session.recovery_detail = (
            "Partial recovery accepted. Transient variables may still be missing."
        )
        session.recovery_phase = "completed"
        session.recovery_step = RECOVERY_TOTAL_STEPS
        session.recovery_total_steps = RECOVERY_TOTAL_STEPS
        self._save_session(session)
        self._emit_recovery_completed(session, accepted_partial=True)
        await self._launch_recovered_runner(session)
        return session.to_response()

    @staticmethod
    def _validate_recovery_request(request: SessionResumeRequest) -> None:
        if (
            request.recovery_mode == RecoveryMode.literal_replay
            and not request.acknowledge_replay_risk
        ):
            raise ValueError(
                "Literal replay requires acknowledgement that external side effects may repeat"
            )

    def _set_recovery_progress(
        self,
        session: _Session,
        *,
        phase: str,
        detail: str,
        step: int,
        substep: Optional[int] = None,
        substep_total: Optional[int] = None,
    ) -> None:
        self._on_event(
            session,
            {
                "type": "recovery_progress",
                "session_id": session.id,
                "turn": session.current_turn,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "phase": phase,
                    "detail": detail,
                    "step": step,
                    "total_steps": RECOVERY_TOTAL_STEPS,
                    "substep": substep,
                    "substep_total": substep_total,
                    "mode": self._recovery_mode_label(session),
                    "attempt_number": session.attempt_number,
                },
            },
        )

    @staticmethod
    def _recovery_mode_label(session: _Session) -> Optional[str]:
        """A branch child reports its restore mode; other sessions their recovery mode."""
        if session.branch_restore_mode is not None:
            return session.branch_restore_mode.value
        return session.recovery_mode.value if session.recovery_mode else None

    def _emit_recovery_completed(
        self, session: _Session, *, accepted_partial: bool = False
    ) -> None:
        self._on_event(
            session,
            {
                "type": "recovery_completed",
                "session_id": session.id,
                "turn": session.current_turn,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "mode": self._recovery_mode_label(session) or "best_effort",
                    "attempt_number": session.attempt_number,
                    "checkpoint_id": session.checkpoint_id,
                    "checkpoint_turn": session.checkpoint_turn,
                    "detail": session.recovery_detail,
                    "accepted_partial": accepted_partial,
                },
            },
        )

    @staticmethod
    def _apply_target_mode(session: _Session, request: SessionResumeRequest) -> None:
        target = request.target_mode or session.config.mode
        updates: Dict[str, Any] = {"mode": target, "initial_prompt": None}
        if target == SessionMode.auto:
            if request.additional_turns is None:
                raise ValueError("Auto recovery requires an additional-turn budget")
            updates["max_turns"] = session.current_turn + request.additional_turns
        else:
            updates["max_turns"] = None
        session.config = session.config.model_copy(update=updates)

    async def _complete_fork(
        self, source: _Session, child: _Session, request: SessionForkRequest
    ) -> None:
        try:
            if self._is_deleted(child.id):
                return
            starting_checkpoint = source.checkpoint_id
            if source.status in {
                SessionStatus.running,
                SessionStatus.initializing,
                SessionStatus.recovering,
            }:
                async with source.event_condition:
                    await asyncio.wait_for(
                        source.event_condition.wait_for(
                            lambda: (
                                source.checkpoint_id != starting_checkpoint
                                or source.status
                                in {SessionStatus.stopped, SessionStatus.error}
                                or self._is_deleted(source.id)
                                or self._is_deleted(child.id)
                            )
                        ),
                        timeout=900,
                    )
            if self._is_deleted(child.id):
                return
            checkpoint = load_checkpoint(source.output_dir)
            if checkpoint is None:
                legacy_history = [
                    {"role": item.role, "content": item.content}
                    for item in source.messages
                    if item.role in {"user", "assistant", "system"}
                ]
                checkpoint = await asyncio.to_thread(
                    capture_checkpoint,
                    session=source,
                    history=legacy_history,
                    runner_state={
                        "schema_version": "caribou.web_runner_checkpoint_state.v1",
                        "current_agent_name": source.current_agent or "unknown",
                        "turns_completed": source.current_turn,
                        "next_turn": source.current_turn + 1,
                        "consecutive_exec_failures": 0,
                        "consecutive_no_action": 0,
                        "action_space_past_actions": [],
                    },
                )
                child.recovery_detail = "Legacy source had no durable checkpoint; using best-effort retained evidence."
            self._set_recovery_progress(
                child,
                phase="copying_checkpoint",
                detail="Copying the safe checkpoint, outputs, and recorded session history.",
                step=1,
            )
            await asyncio.to_thread(
                copy_output_tree, source.output_dir, child.output_dir
            )
            if self._is_deleted(child.id):
                return
            await asyncio.to_thread(self._fork_work_items, source, child)
            # The fork keeps the source's Goal (frozen brief).
            await asyncio.to_thread(self._copy_brief, source, child)
            if self._is_deleted(child.id):
                return
            source_checkpoint_dir = (
                source.output_dir.parent / ".checkpoints" / checkpoint["checkpoint_id"]
            )
            child_root = child.output_dir.parent / ".checkpoints"
            child_checkpoint_dir = child_root / checkpoint["checkpoint_id"]
            child_root.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                shutil.copytree,
                source_checkpoint_dir,
                child_checkpoint_dir,
                dirs_exist_ok=True,
            )
            await asyncio.to_thread(
                publish_checkpoint_pointer,
                child.output_dir,
                checkpoint["checkpoint_id"],
            )
            child.checkpoint_id = checkpoint["checkpoint_id"]
            child.forked_from_checkpoint_id = checkpoint["checkpoint_id"]
            child.checkpoint_turn = checkpoint["turn"]
            child.checkpoint_healthy = bool(checkpoint.get("complete"))
            child.current_turn = checkpoint["turn"]
            child.messages = [
                MessageRecord(
                    session_id=child.id,
                    turn=item.turn,
                    role=item.role,
                    agent_name=item.agent_name,
                    content=item.content,
                    is_delegation=item.is_delegation,
                )
                for item in source.messages
                if item.turn <= checkpoint["turn"]
            ]
            child.artifacts = []
            for item in source.artifacts:
                if item.turn > checkpoint["turn"]:
                    continue
                child.artifacts.append(
                    ArtifactRecord(
                        session_id=child.id,
                        turn=item.turn,
                        type=item.type,
                        filename=item.filename,
                        mime_type=item.mime_type,
                        size_bytes=item.size_bytes,
                        created_at=item.created_at,
                        local_path=str(child.output_dir / item.path),
                        path=item.path,
                        mtime_ns=item.mtime_ns,
                        action_id=item.action_id,
                    )
                )
            child.code_events = [
                CodeEventRecord(
                    session_id=child.id,
                    turn=item.turn,
                    agent_name=item.agent_name,
                    source=item.source,
                    stdout=item.stdout,
                    stderr=item.stderr,
                    success=item.success,
                    duration_ms=item.duration_ms,
                    action_id=item.action_id,
                    block_id=item.block_id,
                )
                for item in source.code_events
                if item.turn <= checkpoint["turn"]
            ]
            retained_event_types = _RETAINED_EVENT_TYPES
            # Recovery-progress events already emitted on the child stay in
            # its log; the retained history is appended after them under the
            # child's own seq counter.
            for source_event in source.events:
                if (
                    source_event.get("type") not in retained_event_types
                    or int(source_event.get("turn", 0) or 0) > checkpoint["turn"]
                    or (
                        source_event.get("type") == "error"
                        and bool((source_event.get("data") or {}).get("fatal"))
                    )
                ):
                    continue
                child_event = copy.deepcopy(source_event)
                child_event["session_id"] = child.id
                # The parent's seq belongs to the parent's log.
                del child_event["seq"]
                append_session_event(child, child_event)
            child.recovery_status = RecoveryStatus.recovering
            child.status = SessionStatus.recovering
            self._save_session(child)
            await self._recover_session(child, request)
        except Exception as exc:
            child.status = SessionStatus.stopped
            child.recovery_status = RecoveryStatus.failed
            child.recovery_detail = str(exc)
            self._finish_latest_attempt(child, "recovery_failed")
            self._save_session(child)

    async def _recover_session(
        self, session: _Session, request: SessionResumeRequest
    ) -> None:
        try:
            if self._is_deleted(session.id):
                return
            self._set_recovery_progress(
                session,
                phase="checkpoint",
                detail="Loading and validating the latest safe checkpoint.",
                step=1,
            )
            checkpoint = load_checkpoint(session.output_dir)
            if checkpoint is None:
                history = [
                    {"role": item.role, "content": item.content}
                    for item in session.messages
                    if item.role in {"user", "assistant", "system"}
                ]
                checkpoint = await asyncio.to_thread(
                    capture_checkpoint,
                    session=session,
                    history=history,
                    runner_state={
                        "schema_version": "caribou.web_runner_checkpoint_state.v1",
                        "current_agent_name": session.current_agent or "unknown",
                        "turns_completed": session.current_turn,
                        "next_turn": session.current_turn + 1,
                        "consecutive_exec_failures": 0,
                        "consecutive_no_action": 0,
                        "action_space_past_actions": [],
                    },
                )
                session.recovery_detail = "Legacy session: durable runtime evidence is incomplete; recovery is best-effort."
            self._set_recovery_progress(
                session,
                phase="retiring_runtime",
                detail="Closing the previous sandbox and clearing transient runtime state.",
                step=2,
            )
            if session.sandbox_manager is not None:
                try:
                    await asyncio.to_thread(session.sandbox_manager.stop_container)
                except Exception:
                    pass
            if self._is_deleted(session.id):
                return
            dataset_source = (
                Path(session.config.dataset_path)
                if request.recovery_mode == RecoveryMode.literal_replay
                else checkpoint_dataset_path(session.output_dir, checkpoint)
            )
            runtime = await self._start_recovered_runtime(
                session, checkpoint, dataset_source
            )
            if runtime is None:
                return
            llm_client, model_name, driver = runtime
            emit = self._recovery_progress_emitter(session)

            if request.recovery_mode == RecoveryMode.smart:
                self._set_recovery_progress(
                    session,
                    phase="restoring_dataset",
                    detail="Loading checkpointed AnnData and verifying the recovered dataset.",
                    step=6,
                )
                boot_ok, boot_detail = await asyncio.to_thread(
                    bootstrap_anndata, session.sandbox_manager
                )
                if not boot_ok:
                    raise RuntimeError(
                        f"Checkpointed AnnData could not be loaded: {boot_detail}"
                    )
                recovered, detail = await asyncio.to_thread(
                    smart_rebuild,
                    sandbox=session.sandbox_manager,
                    llm_client=llm_client,
                    model_name=model_name,
                    current_agent_prompt=driver.get_full_prompt(None),
                    checkpoint=checkpoint,
                    emit=emit,
                )
            else:
                self._set_recovery_progress(
                    session,
                    phase="preparing_replay",
                    detail=(
                        "Preparing the original dataset and the complete recorded code ledger "
                        "for literal replay."
                    ),
                    step=6,
                )
                if not checkpoint.get("actions"):
                    boot_ok, boot_detail = await asyncio.to_thread(
                        bootstrap_anndata, session.sandbox_manager
                    )
                    if not boot_ok:
                        raise RuntimeError(
                            f"Original AnnData could not be loaded: {boot_detail}"
                        )
                recovered, detail = await asyncio.to_thread(
                    literal_replay,
                    sandbox=session.sandbox_manager,
                    checkpoint=checkpoint,
                    emit=emit,
                )
            session.recovery_detail = detail
            if self._is_deleted(session.id):
                try:
                    await asyncio.to_thread(session.sandbox_manager.stop_container)
                except Exception:
                    pass
                return
            if not recovered:
                session.status = SessionStatus.stopped
                session.recovery_status = RecoveryStatus.partial
                session.recovery_phase = "partial"
                self._save_session(session)
                return
            self._set_recovery_progress(
                session,
                phase="finalizing",
                detail="Recovery checks passed; starting the recovered session runner.",
                step=8,
            )
            session.recovery_status = RecoveryStatus.recovered
            session.recovery_phase = "completed"
            self._save_session(session)
            self._emit_recovery_completed(session)
            await self._launch_recovered_runner(session)
        except Exception as exc:
            session.status = SessionStatus.stopped
            session.recovery_status = RecoveryStatus.failed
            session.recovery_detail = str(exc)
            session.recovery_phase = "failed"
            self._finish_latest_attempt(session, "recovery_failed")
            self._save_session(session)

    async def _start_recovered_runtime(
        self, session: _Session, checkpoint: Dict[str, Any], dataset_source: Path
    ) -> Optional[tuple]:
        """Steps 3-5 of a recovery: agent system, LLM clients, a fresh sandbox
        holding `dataset_source` as its primary dataset, and the restored
        transcript/runner position from `checkpoint`.

        Returns `(llm_client, model_name, driver)`, or None when the session
        was deleted meanwhile (its new sandbox is then stopped).
        """
        load_dotenv(dotenv_path=ENV_FILE, override=True)
        from caribou.agents.AgentSystem import AgentSystem

        self._set_recovery_progress(
            session,
            phase="loading_configuration",
            detail="Loading the agent system and selected LLM backend.",
            step=3,
        )
        agent_system = AgentSystem.load_from_json(
            str(find_blueprint(session.config.agent_system))
        )
        state = dict(checkpoint.get("runner_state") or {})
        current_name = state.get("current_agent_name") or next(
            iter(agent_system.agents)
        )
        driver = agent_system.get_agent(current_name) or agent_system.get_agent(
            next(iter(agent_system.agents))
        )
        session.agent_system = agent_system
        session.brief_policy = resolve_brief_policy(
            agent_system.brief_policy, session.config.brief_mode
        )
        session.driver_agent = driver
        session.current_agent = driver.name
        llm_client, model_name = build_llm_client(session.config)
        session.llm_client = llm_client
        session.model_name = model_name
        session.resolved_model = resolve_model_info(
            session.config, resolved_model_name=model_name
        )
        (
            session.evaluator_llm_client,
            session.evaluator_model_name,
            session.resolved_evaluator_model,
        ) = build_evaluator_client(
            session.config,
            worker_client=llm_client,
            worker_model_name=model_name,
        )
        runtime_config = session.config.model_copy(
            update={"dataset_path": str(dataset_source)}
        )
        self._set_recovery_progress(
            session,
            phase="starting_sandbox",
            detail=(
                f"Starting a fresh {session.config.sandbox_type.value} sandbox; "
                "container startup can take a little while."
            ),
            step=4,
        )
        sandbox_manager = await asyncio.to_thread(
            build_sandbox, runtime_config, session.output_dir
        )
        if self._is_deleted(session.id):
            try:
                await asyncio.to_thread(sandbox_manager.stop_container)
            except Exception:
                pass
            return None
        try:
            assert_environment_unchanged(
                session.python_environment,
                getattr(
                    sandbox_manager,
                    "python_environment",
                    session.python_environment,
                ),
            )
        except Exception:
            await asyncio.to_thread(sandbox_manager.stop_container)
            raise
        session.sandbox_manager = sandbox_manager
        session.python_environment = getattr(
            sandbox_manager, "python_environment", session.python_environment
        )
        self._save_session(session)
        session.analysis_context = textwrap.dedent(f"""\
            Primary dataset path: **{SANDBOX_DATA_PATH}**

            RECOVERED SESSION: durable AnnData, transcript, action history, and files were restored.
            Arbitrary Python globals, open handles, GPU objects, and external process state were not directly restored.
            Save all generated outputs to /workspace/outputs/.
        """).strip()
        self._set_recovery_progress(
            session,
            phase="restoring_history",
            detail="Restoring transcript, memory state, runner position, and copied files.",
            step=5,
        )
        history = [dict(item) for item in checkpoint.get("history", [])]
        if not history or history[0].get("role") != "system":
            history = [
                {
                    "role": "system",
                    "content": f"**GLOBAL POLICY**: {agent_system.global_policy}\n",
                },
                {
                    "role": "system",
                    "content": driver.get_full_prompt(None)
                    + "\n\n"
                    + session.analysis_context,
                },
                *history,
            ]
        session.initial_history = history[:2]
        session.resume_history = history
        session.resume_runner_state = state
        session.resume_memory_state = checkpoint.get("memory")

        return llm_client, model_name, driver

    def _recovery_progress_emitter(self, session: _Session):
        """A thread-safe `emit` for literal_replay / smart_rebuild progress."""
        loop = asyncio.get_running_loop()

        def emit(progress: Dict[str, Any]) -> None:
            phase = str(progress.get("phase") or "verifying_recovery")
            substep = int(progress.get("step", 0) or 0) or None
            substep_total = int(progress.get("total", 0) or 0) or None
            if phase == "literal_replay":
                detail = (
                    f"Replaying recorded code attempt {substep} of {substep_total}, "
                    "including attempts that originally failed."
                )
            else:
                detail = (
                    f"Agent-guided environment rebuild attempt {substep} of "
                    f"{substep_total}."
                )
            loop.call_soon_threadsafe(
                lambda: self._set_recovery_progress(
                    session,
                    phase=phase,
                    detail=detail,
                    step=7,
                    substep=substep,
                    substep_total=substep_total,
                )
            )

        return emit

    async def _launch_recovered_runner(
        self, session: _Session, *, branch_turn: Optional[str] = None
    ) -> None:
        """Start the runner of a recovered session.

        `branch_turn` (branch children only) is the branch's first user turn.
        Interactive: it is pre-loaded onto the fresh `user_input_queue`, which
        the runner (start_waiting) takes as its first input. Auto: the runner
        reads no queue, so it is appended to the restored history as the last
        user message, and the runner continues from it.
        """
        session.stop_flag.clear()
        session.cancel_response_flag.clear()
        session.user_input_queue = queue.Queue()
        history = [
            dict(item) for item in (session.resume_history or session.initial_history)
        ]
        recovery_notice = {
            "role": "system",
            "content": (
                f"RECOVERY NOTICE (attempt {session.attempt_number}): a fresh sandbox was "
                f"created using {self._recovery_mode_label(session) or 'best-effort'} recovery. "
                "The checkpointed AnnData, files, transcript, and recorded memory were restored. "
                "Arbitrary Python globals, open handles, GPU objects, and external process state "
                "were not directly restored; verify or rebuild transient state before relying on it. "
                f"Recovery result: {session.recovery_detail or 'ready'}."
            ),
        }
        history.append(recovery_notice)
        self._on_event(
            session,
            {
                "type": "system_message",
                "session_id": session.id,
                "turn": session.current_turn,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "content": recovery_notice["content"],
                    "category": "Recovery notice",
                },
            },
        )
        latest_attempt = session.attempts[-1] if session.attempts else {}
        if latest_attempt.get("kind") == "fork":
            source_worker = latest_attempt.get("source_worker_model")
            worker = latest_attempt.get("worker_model")
            source_evaluator = latest_attempt.get("source_evaluator_model")
            evaluator = latest_attempt.get("evaluator_model")
            if source_worker != worker or source_evaluator != evaluator:
                reason = latest_attempt.get("model_change_reason") or "not provided"
                model_notice = {
                    "role": "system",
                    "content": (
                        "[SYSTEM — CONFIGURATION CHANGE] This fork changed its model "
                        f"configuration. Worker: {source_worker} → {worker}. Evaluator: "
                        f"{source_evaluator} → {evaluator}. Reason: {reason}."
                    ),
                }
                history.append(model_notice)
                self._on_event(
                    session,
                    {
                        "type": "system_message",
                        "session_id": session.id,
                        "turn": session.current_turn,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "content": model_notice["content"],
                            "category": "Configuration change",
                        },
                    },
                )
        if branch_turn is not None:
            if session.config.mode == SessionMode.interactive:
                # After the queue reset above, before the runner thread first
                # reads it (A7); never via send_user_message, which refuses
                # messages while the session is still `recovering`.
                session.user_input_queue.put(UserTurn(content=branch_turn, block_id=None))
            else:
                history.append({"role": "user", "content": branch_turn})
                turn = session.current_turn + 1
                self._on_event(
                    session,
                    {
                        "type": "message_complete",
                        "session_id": session.id,
                        "turn": turn,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "message": {
                                "id": f"msg_{session.id}_branch_turn",
                                "turn": turn,
                                "role": "user",
                                "agent_name": "",
                                "content": branch_turn,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        },
                    },
                )
        await self._launch_runner(
            session,
            history,
            resume_state=session.resume_runner_state,
            start_waiting=session.config.mode == SessionMode.interactive,
        )

    async def stop_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.stop_flag.set()

    async def cancel_response(self, session_id: str) -> bool:
        """Cancel an in-flight interactive response without ending the session."""
        session = self._sessions.get(session_id)
        if (
            not session
            or session.config.mode != SessionMode.interactive
            or session.status != SessionStatus.running
            or not session.runner_task
            or session.runner_task.done()
        ):
            return False
        session.cancel_response_flag.set()
        return True

    def get_memory_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return a read-only snapshot of the session's memory state."""
        session = self._sessions.get(session_id)
        if not session or not session.memory_manager:
            return None
        return session.memory_manager.get_state()

    def get_context_breakdown(self, session_id: str) -> Dict[str, Any]:
        """Return the context breakdown for any session."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        if session.memory_manager is not None:
            return session.memory_manager.get_state()

        # Fallback: compute breakdown from the pinned system messages set up at
        # session init plus the raw per-turn message records.
        pinned_messages = session.initial_history
        pinned_tokens = sum(
            estimate_tokens(m.get("content", "")) for m in pinned_messages
        )

        user_count = assistant_count = system_count = 0
        user_tokens = assistant_tokens = system_tokens = 0
        for msg in session.messages:
            tokens = estimate_tokens(msg.content)
            if msg.role == "user":
                user_count += 1
                user_tokens += tokens
            elif msg.role == "assistant":
                assistant_count += 1
                assistant_tokens += tokens
            else:
                system_count += 1
                system_tokens += tokens

        working_tokens = user_tokens + assistant_tokens + system_tokens
        total_messages = len(pinned_messages) + len(session.messages)
        total_tokens = pinned_tokens + working_tokens

        return {
            "strategy": session.config.memory_strategy.value,
            "total_messages": total_messages,
            "context_breakdown": {
                "pinned_system": len(pinned_messages),
                "pivotal_code": 0,
                "summaries": 0,
                "working_user": user_count,
                "working_assistant": assistant_count,
                "working_system": system_count,
                "total": total_messages,
                "total_full_history": total_messages,
                "pinned_system_tokens": pinned_tokens,
                "pivotal_code_tokens": 0,
                "summaries_tokens": 0,
                "working_user_tokens": user_tokens,
                "working_assistant_tokens": assistant_tokens,
                "working_system_tokens": system_tokens,
                "total_tokens": total_tokens,
                "total_full_history_tokens": total_tokens,
            },
        }

    async def evaluate_session(self, session_id: str) -> EvaluationResult:
        """Send this session's full transcript to an evaluator agent and
        return its assessment. Shares its resolution/call logic with the
        CLI's /evaluate command (execution/user_commands.py) via
        caribou.execution.evaluation — callers must have already confirmed
        the session exists and has been started (llm_client/agent_system set).
        """
        session = self._sessions[session_id]

        evaluator_agent, source = resolve_evaluator_agent(session.agent_system)

        history = list(session.initial_history) + [
            {"role": m.role, "content": m.content} for m in session.messages
        ]
        payload = build_evaluation_payload(
            run_id=session.id,
            turn=session.current_turn,
            active_agent=session.current_agent,
            history=history,
        )

        # The LLM call blocks; run it off the event loop like every other
        # provider call in this module (see build_llm_client's callers).
        async with session.evaluator_model_lock:
            evaluator_client = session.evaluator_llm_client
            evaluator_model_name = session.evaluator_model_name
            resolved_evaluator_model = session.resolved_evaluator_model
            evaluator_model_revision = session.evaluator_model_revision
        if evaluator_client is None or not evaluator_model_name:
            raise ValueError("Evaluator model is not initialized")

        provider_receipt: Dict[str, object] = {}

        def capture_response(response: object) -> None:
            provider_receipt.update(evaluation_response_metadata(response))

        assessment = await asyncio.to_thread(
            run_evaluation,
            evaluator_agent=evaluator_agent,
            llm_client=evaluator_client,
            model_name=evaluator_model_name,
            payload=payload,
            response_callback=capture_response,
        )

        result = EvaluationResult(
            session_id=session.id,
            turn=session.current_turn,
            evaluator_agent=evaluator_agent.name,
            evaluator_source=source,
            model=evaluator_model_name,
            provider=(
                resolved_evaluator_model.provider
                if resolved_evaluator_model is not None
                else None
            ),
            evaluator_model=resolved_evaluator_model,
            evaluator_model_revision=evaluator_model_revision,
            provider_receipt=provider_receipt,
            assessment=assessment,
        )

        report_dir = session.output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = (
            report_dir
            / f"evaluation_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
        )
        report_path.write_text(result.model_dump_json(indent=2))

        return result

    def _fork_work_items(self, source: _Session, child: _Session) -> None:
        # Work items live in a sibling directory to `output_dir`, outside the
        # sandbox `copy_output_tree` walks (see WS-1) — copy it separately
        # and record lineage in the child's index. Synchronous: the caller
        # runs this via `asyncio.to_thread` since it shells out to `git`.
        copied_store = copy_work_items(
            source.output_dir.parent / "work-items",
            child.output_dir.parent / "work-items",
            child_session_id=child.id,
            forked_from_session_id=source.id,
        )
        if copied_store is not None:
            child.work_item_store = copied_store
        # blocks.json sits next to work-items/; copied (re-stamped with the
        # child's session id) before the child's runner can start a tracker.
        # The child was given an empty placeholder at fork time; replace it.
        child_blocks = child.output_dir.parent / BLOCKS_FILENAME
        placeholder = load_blocks(child_blocks)
        if placeholder is not None:
            if placeholder["blocks"] or placeholder["session_id"] != child.id:
                raise BlockError(
                    f"fork child already has blocks.json with content: {child_blocks}"
                )
            child_blocks.unlink()
        copied = fork_blocks(
            source.output_dir.parent / BLOCKS_FILENAME,
            child_blocks,
            child_session_id=child.id,
        )
        if not copied:
            init_blocks(child_blocks, child.id)

    def _work_item_store(self, session: _Session) -> WorkItemStore:
        # Cached on the session so this method, the turn loop
        # (streaming_runner receives the same instance via
        # run_session_async's work_item_store param), and every REST route
        # below all share one in-memory index cache (WS-0). Constructing a
        # fresh WorkItemStore per call — the previous behavior — silently
        # desynchronizes that cache the moment two call sites disagree.
        if session.work_item_store is None:
            policy = (
                session.agent_system.work_item_policy
                if session.agent_system is not None
                else WorkItemPolicy()
            )
            session.work_item_store = WorkItemStore(
                session.output_dir.parent / "work-items",
                session_id=session.id,
                policy=policy,
                origin_run_id=session.id,
            )
        return session.work_item_store

    def get_brief_state(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        return {"phase": session.phase, "brief": session.brief}

    def submit_brief_decision(
        self,
        session_id: str,
        *,
        decision: str,
        reason: Optional[str] = None,
        brief: Optional[Dict[str, Any]] = None,
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        if session.phase != "briefing":
            raise ValueError("session is not in the briefing phase")
        session.brief_decision_queue.put(
            {"decision": decision, "reason": reason, "brief": brief}
        )

    def list_work_items(self, session_id: str) -> List[Dict[str, Any]]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        return self._work_item_store(session).list()

    def read_blocks(self, session_id: str) -> Optional[Dict[str, Any]]:
        """The session's blocks.json index, or None if it never recorded blocks.

        BlockTracker's file is the source of truth; the manager keeps no copy.
        Raises KeyError for an unknown session and BlockError for a malformed
        file.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        return load_blocks(session.output_dir.parent / BLOCKS_FILENAME)

    def read_work_item(self, session_id: str, item_id: int) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        return self._work_item_store(session).read(item_id)

    async def review_work_item(
        self, session_id: str, item_id: int
    ) -> Dict[str, object]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        if session.agent_system is None or session.evaluator_llm_client is None:
            raise ValueError(
                "Session is not running — start or restart it before reviewing."
            )
        evaluator_agent, _source = resolve_evaluator_agent(session.agent_system)
        async with session.evaluator_model_lock:
            evaluator_client = session.evaluator_llm_client
            evaluator_model_name = session.evaluator_model_name
        store = self._work_item_store(session)
        evidence = self._review_evidence(session, item_id)
        try:
            result = await asyncio.to_thread(
                evaluate_work_item,
                store=store,
                item_id=item_id,
                run_id=session.id,
                turn=session.current_turn,
                evaluator_agent=evaluator_agent,
                llm_client=evaluator_client,
                model_name=evaluator_model_name,
                evidence=evidence,
            )
            changed_item = result["item"]
        except Exception:
            # Provider/contract failures are committed by evaluate_work_item;
            # publish that durable failed attempt before surfacing the error.
            changed_item = store.read(item_id)
            self._emit_work_item_change(session, changed_item)
            raise
        self._emit_work_item_change(session, changed_item)
        return result

    def _review_evidence(self, session: _Session, item_id: int) -> Dict[str, Any]:
        """What ran for `item_id`: its blocks from blocks.json joined to the
        session's code events by action id (`review_evidence`). A session
        that never recorded blocks yields no blocks and says so in `note`."""
        index = load_blocks(session.output_dir.parent / BLOCKS_FILENAME)
        if index is None:
            return {"blocks": [], "truncated": False, "note": "no blocks index"}
        code_lookup = code_lookup_from_records(
            [
                {
                    "action_id": record.action_id,
                    "agent": record.agent_name,
                    "success": record.success,
                    "source": record.source,
                    "stdout": record.stdout,
                    "stderr": record.stderr,
                }
                for record in session.code_events
            ]
        )
        return build_block_evidence(index, item_id, code_lookup)

    def _running_work_item_session(self, session_id: str) -> _Session:
        # Human work-item writes need the blueprint: its agents are the valid
        # owners and its work_item_policy decides what a review does.
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        if session.agent_system is None:
            raise ValueError(
                "Session is not running — start or restart it before changing "
                "work items."
            )
        return session

    async def open_human_work_item(
        self,
        session_id: str,
        *,
        title: str,
        body: str,
        owner: str,
        anchor: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Open a ticket a person filed for one of the blueprint's agents."""
        session = self._running_work_item_session(session_id)
        if owner not in session.agent_system.agents:
            raise UnknownWorkItemOwner(
                f"owner {owner!r} is not an agent in this session's blueprint "
                f"(agents: {sorted(session.agent_system.agents)})"
            )
        store = self._work_item_store(session)
        item = await asyncio.to_thread(
            store.open,
            title,
            body,
            owner,
            session.current_turn,
            opened_by=HUMAN_REVIEWER,
            anchor=anchor,
        )
        self._emit_work_item_change(session, item)
        return item

    async def record_human_review(
        self,
        session_id: str,
        item_id: int,
        *,
        verdict: str,
        assessment: str,
    ) -> Dict[str, Any]:
        """Record a person's approve/reject on a work item."""
        session = self._running_work_item_session(session_id)
        store = self._work_item_store(session)
        item = await asyncio.to_thread(
            store.record_review,
            item_id,
            evaluator=HUMAN_REVIEWER,
            turn=session.current_turn,
            verdict=verdict,
            assessment=assessment,
        )
        self._emit_work_item_change(session, item)
        return item

    def _emit_work_item_change(
        self, session: _Session, item: Dict[str, object]
    ) -> None:
        self._on_event(
            session,
            {
                "type": "work_item_changed",
                "session_id": session.id,
                "turn": session.current_turn,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"item": item},
            },
        )

    def get_evaluator_model(self, session_id: str) -> EvaluatorModelState:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        return EvaluatorModelState(
            selection=session.config.evaluator_model,
            resolved_model=session.resolved_evaluator_model,
            revision=session.evaluator_model_revision,
        )

    async def update_evaluator_model(
        self, session_id: str, request: EvaluatorModelUpdateRequest
    ) -> EvaluatorModelState:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session not found")
        async with session.evaluator_model_lock:
            if request.expected_revision != session.evaluator_model_revision:
                raise ValueError(
                    "Evaluator model configuration changed; refresh and retry "
                    f"with revision {session.evaluator_model_revision}"
                )
            old_selection = session.config.evaluator_model
            old_resolved = session.resolved_evaluator_model
            updated_config = session.config.model_copy(
                update={"evaluator_model": request.selection}
            )
            if session.llm_client is None or not session.model_name:
                evaluator_client = None
                evaluator_model_name = ""
                resolved = resolve_evaluator_model_info(
                    updated_config, worker_resolved=session.resolved_model
                )
            else:
                (
                    evaluator_client,
                    evaluator_model_name,
                    resolved,
                ) = await asyncio.to_thread(
                    build_evaluator_client,
                    updated_config,
                    worker_client=session.llm_client,
                    worker_model_name=session.model_name,
                )
            session.config = updated_config
            session.evaluator_llm_client = evaluator_client
            session.evaluator_model_name = evaluator_model_name
            session.resolved_evaluator_model = resolved
            session.evaluator_model_revision += 1
            revision = session.evaluator_model_revision

        now = datetime.utcnow().isoformat()
        self._on_event(
            session,
            {
                "type": "evaluator_model_changed",
                "session_id": session.id,
                "turn": session.current_turn,
                "timestamp": now,
                "data": {
                    "revision": revision,
                    "reason": request.reason,
                    "old_selection": old_selection.model_dump(mode="json"),
                    "new_selection": request.selection.model_dump(mode="json"),
                    "old_resolved_model": (
                        old_resolved.model_dump() if old_resolved is not None else None
                    ),
                    "resolved_model": resolved.model_dump()
                    if resolved is not None
                    else None,
                },
            },
        )
        old_label = old_resolved.model if old_resolved is not None else "unresolved"
        new_label = resolved.model if resolved is not None else "unresolved"
        reason_text = request.reason or "not provided"
        system_message = (
            "[SYSTEM — CONFIGURATION CHANGE] Evaluator model changed "
            f"from {old_label} to {new_label}. Configuration revision: "
            f"{revision - 1} → {revision}. Reason: {reason_text}. Worker model unchanged."
        )
        runner_active = (
            session.runner_task is not None and not session.runner_task.done()
        )
        if runner_active:
            session.control_message_queue.put(system_message)
        else:
            self._on_event(
                session,
                {
                    "type": "system_message",
                    "session_id": session.id,
                    "turn": session.current_turn,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "content": system_message,
                        "category": "Configuration change",
                    },
                },
            )
        return self.get_evaluator_model(session_id)

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self._deleted_session_ids.add(session_id)
            session = self._sessions.pop(session_id, None)
        if session:
            session.stop_flag.set()
            if (
                session.recovery_task
                and not session.recovery_task.done()
                and session.recovery_status == RecoveryStatus.awaiting_checkpoint
            ):
                session.recovery_task.cancel()
            if session.runner_task and not session.runner_task.done():
                session.runner_task.cancel()
            async with session.event_condition:
                session.event_condition.notify_all()
            if session.logger:
                session.logger.info("Session deleted — closing log")
                _close_session_logger(session.logger)
            if session.sandbox_manager:
                try:
                    await asyncio.to_thread(session.sandbox_manager.stop_container)
                except Exception:
                    pass
        try:
            await asyncio.to_thread(
                shutil.rmtree, self._session_dir(session_id), ignore_errors=True
            )
        except Exception:
            pass

    async def shutdown_all(self) -> List[str]:
        """
        Stop all active server-owned sessions and sandbox processes.

        This is used by the FastAPI lifespan shutdown hook. It is deliberately
        best-effort: one broken sandbox must not prevent the rest from being
        torn down.
        """
        errors: List[str] = []

        async with self._lock:
            sessions = list(self._sessions.values())

        for session in sessions:
            session.stop_flag.set()

        for session in sessions:
            if session.runner_task and not session.runner_task.done():
                session.runner_task.cancel()

        for session in sessions:
            if session.runner_task and not session.runner_task.done():
                try:
                    await asyncio.wait_for(session.runner_task, timeout=5)
                except asyncio.CancelledError:
                    pass
                except asyncio.TimeoutError:
                    errors.append(f"{session.id}: runner did not stop before timeout")
                except Exception as exc:
                    errors.append(f"{session.id}: runner shutdown failed: {exc}")

            if session.sandbox_manager:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(session.sandbox_manager.stop_container),
                        timeout=10,
                    )
                except asyncio.TimeoutError:
                    errors.append(f"{session.id}: sandbox did not stop before timeout")
                except Exception as exc:
                    errors.append(f"{session.id}: sandbox shutdown failed: {exc}")

            async with self._lock:
                if session.status in (
                    SessionStatus.initializing,
                    SessionStatus.running,
                    SessionStatus.idle,
                ):
                    session.status = SessionStatus.stopped
                    session.updated_at = datetime.utcnow()
                    append_session_event(
                        session,
                        {
                            "type": "status_change",
                            "session_id": session.id,
                            "turn": session.current_turn,
                            "timestamp": session.updated_at.isoformat(),
                            "data": {"status": "stopped", "reason": "server shutdown"},
                        },
                    )
                    self._save_session(session)

            if session.logger:
                session.logger.info(
                    "Session stopped during server shutdown — closing log"
                )
                _close_session_logger(session.logger)

        return errors

    def _reject_user_message(self, session: _Session, code: str, message: str) -> None:
        """Log a refused, dropped or failed message action as a non-fatal error event.

        The sender sees the refusal in the session log; a refused message used
        to vanish without a trace.
        """
        self._on_event(
            session,
            {
                "type": "error",
                "session_id": session.id,
                "turn": session.current_turn,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"code": code, "message": message, "fatal": False},
            },
        )

    @staticmethod
    def _runner_waiting(session: _Session) -> bool:
        """True when a live interactive runner is waiting for its next turn.

        The runner emits `idle` as it starts waiting on `user_input_queue`;
        a turn already on that queue (e.g. a branch child's first turn), or
        a requested stop, means it will not take another one.
        """
        return (
            session.config.mode == SessionMode.interactive
            and session.status == SessionStatus.idle
            and session.runner_task is not None
            and not session.runner_task.done()
            and not session.stop_flag.is_set()
            and session.user_input_queue.empty()
        )

    @staticmethod
    def _can_queue(session: _Session) -> bool:
        """True when a message may wait in the queue for the runner's next idle.

        Only interactive sessions whose runner is live or starting: a message
        queued on a stopped, errored or auto session would never be delivered.
        """
        if session.config.mode != SessionMode.interactive:
            return False
        if session.status in (SessionStatus.initializing, SessionStatus.recovering):
            return True
        return (
            session.status in (SessionStatus.running, SessionStatus.idle)
            and session.runner_task is not None
            and not session.runner_task.done()
        )

    @staticmethod
    def _block_refusal(
        session: _Session, block_id: Optional[str]
    ) -> Optional[tuple[str, str]]:
        """Why `block_id` cannot focus a turn, as (code, reason), or None if it can.

        `block_id` (from the workbench) must name a block in the session's
        blocks.json that is not inherited. Checked when a message is sent or
        queued, and again when a queued message is delivered.
        """
        if block_id is None:
            return None
        if not isinstance(block_id, str) or not block_id:
            return (
                "INVALID_BLOCK_ID",
                f"block_id must be a non-empty string, got {block_id!r}.",
            )
        index = load_blocks(session.output_dir.parent / BLOCKS_FILENAME)
        block = (
            next((b for b in index["blocks"] if b["block_id"] == block_id), None)
            if index is not None
            else None
        )
        if block is None:
            return (
                "UNKNOWN_BLOCK",
                f"block {block_id} does not exist in this session.",
            )
        if block.get("inherited_from") is not None:
            # Inherited blocks are immutable in a branch (A5); focusing
            # one would fail the runner.
            return (
                "INHERITED_BLOCK",
                f"block {block_id} is inherited from session "
                f"{block['inherited_from']['session_id']} and cannot be "
                "focused in this branch.",
            )
        return None

    def _deliver_user_turn(self, session: _Session, turn: UserTurn) -> None:
        """Hand one turn to the waiting runner (caller checked _runner_waiting)."""
        self._on_event(
            session,
            {
                "type": "status_change",
                "session_id": session.id,
                "turn": session.current_turn,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"status": "running", "reason": "user_message_queued"},
            },
        )
        session.user_input_queue.put(turn)

    def _emit_message_queue_changed(self, session: _Session) -> None:
        """Log the full current queue; the latest such event is the truth."""
        self._on_event(
            session,
            {
                "type": "message_queue_changed",
                "session_id": session.id,
                "turn": session.current_turn,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "queue": [
                        item.model_dump(mode="json") for item in session.message_queue
                    ]
                },
            },
        )

    def _deliver_next_queued(self, session: _Session) -> None:
        """Deliver the queue head if the runner is waiting (one per idle).

        Runs on the event loop, after the runner's `idle` status_change has
        been processed. A head whose block_id is no longer valid is dropped
        with a visible error (never sent with the wrong focus) and the next
        item is tried, so a drop does not strand the rest of the queue.
        """
        while session.message_queue and self._runner_waiting(session):
            item = session.message_queue.pop(0)
            self._emit_message_queue_changed(session)
            refusal = self._block_refusal(session, item.block_id)
            if refusal is not None:
                code, reason = refusal
                self._reject_user_message(
                    session,
                    code,
                    f"Queued message {item.id} dropped, not delivered: {reason}",
                )
                continue
            self._deliver_user_turn(
                session, UserTurn(content=item.content, block_id=item.block_id)
            )
            return

    async def send_user_message(
        self, session_id: str, content: str, block_id: Optional[str] = None
    ) -> bool:
        """Deliver one message now, or queue it while the agent is busy.

        Delivered at once when a live interactive runner is waiting and
        nothing is queued ahead of it; queued (delivered on a later idle)
        while an interactive session's runner is busy or starting. Anything
        else, and a bad `block_id`, is refused with a non-fatal error event.
        Returns True when the message was delivered or queued.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        if not self._runner_waiting(session) and not self._can_queue(session):
            self._reject_user_message(
                session,
                "MESSAGE_NOT_ACCEPTED",
                "Message not delivered: the session accepts messages only in "
                "interactive mode while its runner is live or starting "
                f"(status: {SessionStatus(session.status).value}, mode: "
                f"{SessionMode(session.config.mode).value}).",
            )
            return False
        refusal = self._block_refusal(session, block_id)
        if refusal is not None:
            code, reason = refusal
            self._reject_user_message(
                session, code, f"Message not delivered: {reason}"
            )
            return False
        if self._runner_waiting(session) and not session.message_queue:
            self._deliver_user_turn(
                session, UserTurn(content=content, block_id=block_id)
            )
            return True
        session.message_queue.append(
            QueuedMessage(
                id=uuid4().hex,
                content=content,
                block_id=block_id,
                created_at=datetime.utcnow(),
            )
        )
        self._emit_message_queue_changed(session)
        self._deliver_next_queued(session)
        return True

    async def unqueue_message(self, session_id: str, message_id: str) -> bool:
        """Remove one queued message before delivery; it is then never sent.

        Runs on the event loop like delivery, so the two cannot race. An id
        that is no longer queued (delivered, dropped or already removed) is
        logged as a non-fatal QUEUED_MESSAGE_GONE error event.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        index = next(
            (
                position
                for position, item in enumerate(session.message_queue)
                if item.id == message_id
            ),
            None,
        )
        if index is None:
            self._reject_user_message(
                session,
                "QUEUED_MESSAGE_GONE",
                f"Queued message {message_id!r} is no longer queued: it was "
                "already delivered or removed.",
            )
            return False
        del session.message_queue[index]
        self._emit_message_queue_changed(session)
        return True

    async def start_run(self, session_id: str, initial_prompt: str) -> bool:
        """
        Called when the WebSocket client sends a 'run' message.
        Launches the runner task from the beginning.
        """
        session = self._sessions.get(session_id)
        if not session or session.status not in (SessionStatus.idle,):
            return False
        if session.runner_task and not session.runner_task.done():
            return False

        history = list(session.initial_history)
        history.append({"role": "user", "content": initial_prompt})
        session.status = SessionStatus.running

        self._on_event(
            session,
            {
                "type": "message_complete",
                "session_id": session.id,
                "turn": 1,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "message": {
                        "id": f"msg_{session.id}_user_0",
                        "turn": 1,
                        "role": "user",
                        "agent_name": "",
                        "content": initial_prompt,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                },
            },
        )

        return await self._launch_runner(session, history)

    async def _launch_runner(
        self,
        session: _Session,
        history: List[Dict],
        *,
        resume_state: Optional[Dict[str, Any]] = None,
        start_waiting: bool = False,
    ) -> bool:
        """Launch a session runner."""
        from caribou.server.streaming_runner import run_session_async

        is_auto = session.config.mode == SessionMode.auto
        max_turns = session.config.max_turns or 20

        # Determine effective memory strategy: new fields take precedence,
        # legacy booleans are the fallback.
        memory_strategy = session.config.memory_strategy
        if memory_strategy == MemoryStrategy.full:
            if session.config.agent_report_memory:
                memory_strategy = MemoryStrategy.agent_report
            elif session.config.compress_memory:
                memory_strategy = MemoryStrategy.episodic
        # Persist the effective strategy so to_response()/get_context_breakdown()
        # report what's actually running, even when it was derived from the
        # legacy compress_memory/agent_report_memory flags.
        session.config.memory_strategy = memory_strategy

        memory_manager = None
        report_memory = None

        if memory_strategy == MemoryStrategy.episodic:
            from caribou.execution.MemoryManager import MemoryManager

            whs = session.config.memory_working_history_size or 4
            st = session.config.memory_summarization_threshold or 20
            cs = session.config.memory_chunk_size or 10
            memory_manager = MemoryManager(
                llm_client=session.llm_client,
                model_name=session.model_name,
                initial_history=history,
                working_history_size=whs,
                summarization_threshold=st,
                chunk_size_to_summarize=cs,
            )
            if session.resume_memory_state and session.resume_memory_state.get(
                "restorable"
            ):
                memory_manager.restore_checkpoint(session.resume_memory_state)
                if resume_state and history:
                    memory_manager.add_message(
                        history[-1].get("role", "system"),
                        history[-1].get("content", ""),
                    )
            session.memory_manager = memory_manager
            if session.logger:
                session.logger.info(
                    "Episodic memory enabled | working_history: %s | threshold: %s | chunk: %s",
                    whs,
                    st,
                    cs,
                )
            # agent_report and episodic are mutually exclusive
            if session.config.agent_report_memory:
                session.config.agent_report_memory = False

        elif memory_strategy == MemoryStrategy.agent_report:
            from caribou.execution.report_generation import AgentReportMemory

            base_globals = [history[0]] if history else []
            agent_prompt_content = history[1]["content"] if len(history) > 1 else ""
            report_memory = AgentReportMemory(base_globals, agent_prompt_content)
            if session.resume_memory_state and session.resume_memory_state.get(
                "restorable"
            ):
                report_memory.restore_checkpoint(session.resume_memory_state)
            session.memory_manager = report_memory
            if session.logger:
                session.logger.info("Agent report memory enabled")

        if session.logger:
            session.logger.info(
                "Runner launching | mode: %s | max_turns: %s | history_messages: %s | memory: %s",
                session.config.mode.value,
                max_turns,
                len(history),
                memory_strategy.value,
            )

        loop = asyncio.get_running_loop()

        def _checkpoint_callback(
            checkpoint_history: List[Dict], checkpoint_state: Dict[str, Any]
        ) -> None:
            try:
                checkpoint = capture_checkpoint(
                    session=session,
                    history=checkpoint_history,
                    runner_state=checkpoint_state,
                )
            except Exception as exc:
                loop.call_soon_threadsafe(self._checkpoint_failed, session, str(exc))
                return
            loop.call_soon_threadsafe(self._checkpoint_published, session, checkpoint)

        def _block_entry_checkpoint(
            entry_history: List[Dict], entry_state: Dict[str, Any], block_id: str
        ) -> Dict[str, Any]:
            # Called synchronously in the runner thread before a new block's
            # first action. Never published (A1): it is taken mid-turn, so
            # resume and fork must not load it as latest. Whatever it raises
            # propagates to the runner as a runner error.
            return capture_checkpoint(
                session=session,
                history=entry_history,
                runner_state=entry_state,
                pin_block_id=block_id,
                publish=False,
            )

        async def _guarded_runner() -> None:
            # Ensures the sandbox is torn down and the stop flag reset even if the
            # runner task is cancelled mid-turn (e.g., session deleted, server
            # shutdown). Without this, cancellations leak sandbox containers.
            try:
                await run_session_async(
                    session_id=session.id,
                    agent_system=session.agent_system,
                    driver_agent=session.driver_agent,
                    analysis_context=session.analysis_context,
                    llm_client=session.llm_client,
                    sandbox_manager=session.sandbox_manager,
                    history=history,
                    is_auto=is_auto,
                    max_turns=max_turns,
                    model_name=session.model_name,
                    output_dir=session.output_dir,
                    event_callback=lambda ev: self._on_event(session, ev),
                    stop_flag=session.stop_flag,
                    cancel_response_flag=session.cancel_response_flag,
                    user_input_queue=session.user_input_queue if not is_auto else None,
                    control_message_queue=session.control_message_queue,
                    logger=session.logger,
                    memory_manager=memory_manager,
                    report_memory=report_memory,
                    checkpoint_callback=_checkpoint_callback,
                    block_entry_checkpoint=_block_entry_checkpoint,
                    resume_state=resume_state,
                    start_waiting=start_waiting,
                    work_item_store=self._work_item_store(session),
                    phase=session.phase,
                    brief_policy=session.brief_policy,
                    brief_decision_queue=(
                        session.brief_decision_queue if not is_auto else None
                    ),
                    brief=session.brief,
                    # Files already recorded (e.g. before a resume) are not
                    # re-emitted unless they change.
                    known_artifacts={
                        record.path: record.mtime_ns for record in session.artifacts
                    },
                )
            except asyncio.CancelledError:
                # Propagate after cleanup so shutdown_all/delete_session can await it.
                raise
            finally:
                # Best-effort sandbox shutdown for cancellation paths that don't
                # go through delete_session (e.g., server SIGTERM race).
                if self._is_deleted(session.id) and session.sandbox_manager is not None:
                    try:
                        await asyncio.to_thread(session.sandbox_manager.stop_container)
                    except Exception:
                        pass

        session.runner_task = asyncio.create_task(_guarded_runner())
        return True

    def _checkpoint_published(
        self, session: _Session, checkpoint: Dict[str, Any]
    ) -> None:
        self._on_event(
            session,
            {
                "type": "checkpoint_created",
                "session_id": session.id,
                "turn": checkpoint["turn"],
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "healthy": checkpoint["complete"],
                    "capture_error": checkpoint.get("capture_error"),
                },
            },
        )

    def _checkpoint_failed(self, session: _Session, detail: str) -> None:
        session.checkpoint_healthy = False
        self._on_event(
            session,
            {
                "type": "checkpoint_failed",
                "session_id": session.id,
                "turn": session.current_turn,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"detail": detail},
            },
        )

    def append_event(self, session: _Session, event: Dict[str, Any]) -> None:
        """Synchronously append event and update derived state."""
        # Derived state first: if the event is inconsistent (e.g. a
        # code_result with no code_submitted) it raises before being logged.
        self._process_event(session, event)
        append_session_event(session, event)
        session.updated_at = datetime.utcnow()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_event(self, session: _Session, event: Dict[str, Any]) -> None:
        """Called from run_session_async (main thread via call_soon_threadsafe)."""
        if self._is_deleted(session.id):
            return
        # Derived state first: if the event is inconsistent (e.g. a
        # code_result with no code_submitted) it raises before being logged.
        self._process_event(session, event)
        append_session_event(session, event)
        session.updated_at = datetime.utcnow()
        if event.get("type") not in SKIP_PERSIST_TYPES or seq_reservation_exhausted(
            session
        ):
            self._save_session(session)
        asyncio.ensure_future(self._notify_condition(session))
        if event.get("type") == "status_change" and (event.get("data") or {}).get(
            "status"
        ) == SessionStatus.idle.value:
            # The runner emits `idle` as it starts waiting for input: hand it
            # the oldest queued message, if any (after `idle` is in the log).
            self._deliver_next_queued(session)

    async def _notify_condition(self, session: _Session) -> None:
        async with session.event_condition:
            session.event_condition.notify_all()

    def _process_event(self, session: _Session, event: Dict[str, Any]) -> None:
        """Update session state from incoming events."""
        t = event.get("type")
        data = event.get("data", {})

        if t == "status_change":
            status_str = data.get("status", "")
            try:
                session.status = SessionStatus(status_str)
            except ValueError:
                pass
            if session.status in {SessionStatus.stopped, SessionStatus.error}:
                self._finish_latest_attempt(session, session.status.value)

        elif t == "recovery_progress":
            session.recovery_phase = data.get("phase")
            session.recovery_detail = data.get("detail", session.recovery_detail)
            session.recovery_step = int(data.get("step", session.recovery_step) or 0)
            session.recovery_total_steps = int(
                data.get("total_steps", session.recovery_total_steps) or 0
            )
            session.recovery_substep = data.get("substep")
            session.recovery_substep_total = data.get("substep_total")

        elif t == "system_message":
            message_id = data.setdefault("id", str(uuid4()))
            session.messages.append(
                MessageRecord(
                    id=message_id,
                    session_id=session.id,
                    turn=event.get("turn", session.current_turn),
                    role="system",
                    agent_name=data.get("category", "System"),
                    content=data.get("content", ""),
                    timestamp=datetime.fromisoformat(event["timestamp"]),
                )
            )

        elif t == "message_complete":
            msg_data = data.get("message", {})
            session.current_turn = msg_data.get("turn", session.current_turn)
            session.current_agent = msg_data.get("agent_name", session.current_agent)
            session.messages.append(
                MessageRecord(
                    id=msg_data.get("id", str(uuid4())),
                    session_id=session.id,
                    turn=msg_data.get("turn", 0),
                    role=msg_data.get("role", "assistant"),
                    agent_name=msg_data.get("agent_name", ""),
                    content=msg_data.get("content", ""),
                    timestamp=datetime.utcnow(),
                )
            )

        elif t == "agent_switch":
            session.current_agent = data.get("to_agent", session.current_agent)

        elif t == "code_submitted":
            action_id = data["action_id"]
            if action_id in session.pending_code_sources:
                raise RuntimeError(
                    f"code_submitted for action {action_id!r} while an earlier "
                    "submission of it has no code_result yet"
                )
            session.pending_code_sources[action_id] = data["source"]

        elif t == "code_result":
            action_id = data["action_id"]
            if action_id not in session.pending_code_sources:
                raise RuntimeError(
                    f"code_result for action {action_id!r} has no matching "
                    "code_submitted"
                )
            source = session.pending_code_sources.pop(action_id)
            session.code_events.append(
                CodeEventRecord(
                    session_id=session.id,
                    turn=event.get("turn", 0),
                    agent_name=data.get("agent_name", ""),
                    source=source,
                    stdout=data.get("stdout", ""),
                    stderr=data.get("stderr", ""),
                    success=data.get("success", True),
                    duration_ms=data.get("duration_ms", 0),
                    action_id=action_id,
                    block_id=data.get("block_id"),
                )
            )

        elif t == "artifact":
            art = data.get("artifact", {})
            art_type_str = art.get("type", "data")
            try:
                art_type = ArtifactType(art_type_str)
            except ValueError:
                art_type = ArtifactType.data
            record = ArtifactRecord(
                session_id=session.id,
                turn=art.get("turn", event.get("turn", 0)),
                type=art_type,
                filename=art.get("filename", ""),
                mime_type=art.get("mime_type", "application/octet-stream"),
                size_bytes=art.get("size_bytes", 0),
                local_path=art.get("local_path", ""),
                path=art["path"],
                mtime_ns=art["mtime_ns"],
                action_id=art["action_id"],
            )
            # `path` is the artifact's identity: an overwrite replaces the
            # earlier record in place (keeping its id, so download URLs stay
            # valid) instead of adding a duplicate.
            existing = next(
                (
                    index
                    for index, item in enumerate(session.artifacts)
                    if item.path == record.path
                ),
                None,
            )
            if existing is None:
                session.artifacts.append(record)
            else:
                record.id = session.artifacts[existing].id
                session.artifacts[existing] = record

        elif t == "block_changed":
            # blocks.json (written by BlockTracker) is the source of truth and
            # the event log already holds this event for live updates and
            # replay, so the manager keeps no second copy of block state.
            pass

        elif t == "phase_change":
            session.phase = data.get("phase", session.phase)

        elif t == "brief_accepted":
            session.brief = data.get("brief")

    @staticmethod
    def _finish_latest_attempt(session: _Session, outcome: str) -> None:
        """Close the current provenance record once; completed attempts stay immutable."""

        if not session.attempts:
            return
        latest = session.attempts[-1]
        if latest.get("attempt_number") != session.attempt_number or latest.get(
            "ended_at"
        ):
            return
        latest["ended_at"] = datetime.utcnow().isoformat()
        latest["outcome"] = outcome
        latest["final_turn"] = session.current_turn

    async def _initialize_session(self, session: _Session) -> None:
        """
        Runs in background after session creation.
        Sets up sandbox, LLM client, agent system, and emits status events.
        """
        load_dotenv(dotenv_path=ENV_FILE, override=True)
        log = session.logger

        def _emit_init(event_type: str, data: Dict) -> None:
            if self._is_deleted(session.id):
                return
            self._on_event(
                session,
                {
                    "type": event_type,
                    "session_id": session.id,
                    "turn": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data,
                },
            )

        try:
            if self._is_deleted(session.id):
                return

            # --- Agent system ---
            from caribou.agents.AgentSystem import AgentSystem

            blueprint_path = find_blueprint(session.config.agent_system)
            if log:
                log.info("Loading blueprint: %s", blueprint_path)
            agent_sys = AgentSystem.load_from_json(str(blueprint_path))
            if self._is_deleted(session.id):
                return
            session.agent_system = agent_sys
            session.brief_policy = resolve_brief_policy(
                agent_sys.brief_policy, session.config.brief_mode
            )

            # Pick driver agent: first agent in the system
            driver_name = next(iter(agent_sys.agents))
            session.driver_agent = agent_sys.get_agent(driver_name)
            session.current_agent = driver_name
            if log:
                log.info(
                    "Blueprint loaded | agents: %s | driver: %s",
                    list(agent_sys.agents),
                    driver_name,
                )

            # --- LLM client ---
            if log:
                log.info(
                    "Building LLM client | backend: %s", session.config.llm_backend
                )
            llm_client, model_name = build_llm_client(session.config)
            if self._is_deleted(session.id):
                return
            session.llm_client = llm_client
            session.model_name = model_name
            session.resolved_model = resolve_model_info(
                session.config,
                resolved_model_name=model_name,
            )
            (
                session.evaluator_llm_client,
                session.evaluator_model_name,
                session.resolved_evaluator_model,
            ) = build_evaluator_client(
                session.config,
                worker_client=llm_client,
                worker_model_name=model_name,
            )
            self._save_session(session)
            if log:
                log.info(
                    "LLM client ready | backend: %s | model: %s",
                    session.config.llm_backend,
                    model_name,
                )

            # --- Analysis context + initial history ---
            analysis_context = textwrap.dedent(f"""\
                Primary dataset path: **{SANDBOX_DATA_PATH}**
                {"Reference dataset path: **" + SANDBOX_REF_DATA_PATH + "**" if session.config.reference_dataset_path else ""}

                **IMPORTANT**: Please save all generated output files (plots, .h5ad, .csv) to the /workspace/outputs/ directory.
            """).strip()
            session.analysis_context = analysis_context

            driver = session.driver_agent

            # --- Session brief (WS-5) ---
            from caribou.execution.session_brief import SessionBrief

            frozen_brief: Optional[SessionBrief] = None
            if session.config.brief is not None:
                raw = dict(session.config.brief)
                raw.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
                raw.setdefault("created_by", "human")
                # created_at is a string here (JSON has no datetime type);
                # SessionBrief's strict=True config rejects a str where a
                # real datetime is expected unless told not to be strict.
                frozen_brief = SessionBrief.model_validate(raw, strict=False)
            should_run_briefing = (
                frozen_brief is None
                and session.brief_policy is not None
                and session.brief_policy.enabled
                and session.config.mode == SessionMode.interactive
            )
            session.phase = "briefing" if should_run_briefing else "execution"

            # Work items are active in both interactive and auto sessions
            # (WS-1) — the grammar appendix must not be suppressed for auto.
            # During briefing, the work-item grammar is suppressed in favor
            # of the briefing instructions (mirrors runner.py's `elif`).
            if should_run_briefing:
                from caribou.execution.session_brief import render_briefing_prompt

                appendix = render_briefing_prompt()
            else:
                appendix = ""
            system_prompt = (
                driver.get_full_prompt(
                    None, None if should_run_briefing else agent_sys.work_item_policy
                )
                + appendix
                + "\n\n"
                + analysis_context
            )
            session.initial_history = [
                {
                    "role": "system",
                    "content": f"**GLOBAL POLICY**: {agent_sys.global_policy}\n",
                },
                {"role": "system", "content": system_prompt},
            ]

            if frozen_brief is not None:
                from caribou.execution.work_item_runtime import (
                    freeze_brief,
                    render_brief_pin,
                )

                seed_item = freeze_brief(
                    frozen_brief,
                    brief_path=session.output_dir.parent / "brief.json",
                    store=self._work_item_store(session),
                    brief_mode=session.brief_policy.mode
                    if session.brief_policy is not None
                    else "context",
                    owner=driver.name,
                    turn=0,
                )
                session.brief = frozen_brief.model_dump(mode="json")
                session.initial_history.append(
                    {"role": "system", "content": render_brief_pin(frozen_brief)}
                )
                if seed_item is not None and log:
                    log.info(
                        "Seeded work item %s from frozen brief: %s",
                        seed_item["id"],
                        seed_item["title"],
                    )
            _emit_init(
                "system_message",
                {
                    "content": session.initial_history[0]["content"],
                    "category": "Global policy",
                },
            )
            _emit_init(
                "system_message",
                {
                    "content": session.initial_history[1]["content"],
                    "category": "Agent prompt",
                },
            )

            # --- Sandbox ---
            _emit_init(
                "status_change",
                {"status": "initializing", "reason": "starting_sandbox"},
            )
            if log:
                log.info(
                    "Starting sandbox | type: %s", session.config.sandbox_type.value
                )
            sandbox_started = time.monotonic()
            sandbox_manager = await asyncio.to_thread(
                build_sandbox, session.config, session.output_dir
            )
            if self._is_deleted(session.id):
                try:
                    await asyncio.to_thread(sandbox_manager.stop_container)
                except Exception:
                    pass
                return
            try:
                assert_environment_unchanged(
                    session.python_environment,
                    getattr(
                        sandbox_manager,
                        "python_environment",
                        session.python_environment,
                    ),
                )
            except Exception:
                await asyncio.to_thread(sandbox_manager.stop_container)
                raise
            session.sandbox_manager = sandbox_manager
            session.python_environment = getattr(
                sandbox_manager, "python_environment", session.python_environment
            )
            self._save_session(session)
            if log:
                log.info(
                    "Sandbox ready | elapsed: %sms",
                    int((time.monotonic() - sandbox_started) * 1000),
                )

            try:
                await asyncio.to_thread(
                    capture_checkpoint,
                    session=session,
                    history=session.initial_history,
                    runner_state={
                        "schema_version": "caribou.web_runner_checkpoint_state.v1",
                        "current_agent_name": session.current_agent,
                        "turns_completed": 0,
                        "next_turn": 1,
                        "consecutive_exec_failures": 0,
                        "consecutive_no_action": 0,
                        "action_space_past_actions": [],
                    },
                )
            except Exception as checkpoint_exc:
                session.checkpoint_healthy = False
                if log:
                    log.warning("Baseline checkpoint failed: %s", checkpoint_exc)

            session.status = SessionStatus.idle
            _emit_init("status_change", {"status": "idle", "reason": "ready"})
            if log:
                log.info("Session ready (idle)")

            # Auto sessions with a prompt start immediately — no WebSocket run message needed
            if (
                session.config.mode == SessionMode.auto
                and session.config.initial_prompt
            ):
                if log:
                    log.info(
                        "Auto-mode run starting | prompt: %r",
                        session.config.initial_prompt[:80],
                    )
                await self.start_run(session.id, session.config.initial_prompt)

        except BaseException as exc:  # noqa: BLE001 — includes SystemExit/KeyboardInterrupt
            # SystemExit from legacy sandbox helpers must NOT propagate out of
            # this task, or asyncio treats it as a fatal shutdown and takes the
            # server lifespan down with it. Log the failure and forward a clean
            # error event to the UI.
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                raise
            if log:
                log.error("Session init failed: %s", exc, exc_info=True)
            session.status = SessionStatus.error
            _emit_init(
                "error",
                {
                    "code": getattr(exc, "code", "INIT_ERROR"),
                    "message": str(exc) or exc.__class__.__name__,
                    "fatal": True,
                    "suggested_fix": getattr(exc, "suggested_fix", None),
                },
            )
            _emit_init(
                "status_change",
                {"status": "error", "reason": str(exc) or exc.__class__.__name__},
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

session_manager = SessionManager()
