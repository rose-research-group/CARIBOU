"""
Session data types and shared constants for the CARIBOU server.

Kept separate from `session_manager` so persistence, setup, and routing
code can import the record type without pulling in the whole manager.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from caribou.config import CARIBOU_HOME
from caribou.core.python_environments import (
    PythonEnvironmentKind,
    ResolvedPythonEnvironment,
)
from caribou.server.models import (
    ArtifactRecord,
    CodeEventRecord,
    EvaluatorModelState,
    MemoryConfigResponse,
    MessageRecord,
    ResolvedModelInfo,
    RecoveryMode,
    RecoveryStatus,
    SessionCreateRequest,
    SessionResponse,
    SessionStatus,
)

# Filesystem layout
UPLOADS_DIR = CARIBOU_HOME / "server_uploads"
SESSIONS_DIR = CARIBOU_HOME / "server_sessions"

# Sandbox paths mounted inside every session's container
SANDBOX_DATA_PATH = "/workspace/dataset.h5ad"
SANDBOX_REF_DATA_PATH = "/workspace/reference.h5ad"

# Events that don't need to be persisted mid-stream (too frequent)
SKIP_PERSIST_TYPES = {"token", "pong"}

# Streaming token events are only kept until their message completes. The
# runner streams one message at a time per session, and every
# `message_complete` (user or assistant) ends whatever was streaming — the
# frontend flushes its streaming buffer on any message_complete too. So when a
# message_complete arrives, every token event before it is dropped from the
# log: a client reconnecting mid-stream still replays the partial message,
# while completed messages are represented by their message_complete alone.
# Structural (non-token) events are never dropped and the log has no cap:
# without tokens it grows by a few events per turn, and every one of those is
# needed to rebuild the session view on reconnect.
TOKEN_EVENT_TYPE = "token"

# Token events don't trigger a session.json save, so after a crash the seqs
# of unsaved tokens would be handed out again on restart — and a client that
# saw them would drop the new events as already applied. To make seq never
# reused, session.json stores a *reservation* `event_seq_reserved` =
# (seq at save time + this block). Restart resumes the counter from the
# reservation, and the manager forces a save before any seq exceeds it.
SEQ_RESERVATION_BLOCK = 1024


def append_session_event(session: "_Session", event: Dict[str, Any]) -> Dict[str, Any]:
    """Append `event` to the session's event log — the only way events enter it.

    Assigns the event's `seq` (strictly increasing per session, starting at
    1, persisted as `session.event_seq`) and compacts the token events of the
    message that `event` completes. Mutates and returns `event`.
    """
    if "seq" in event:
        raise ValueError(
            f"event already carries seq={event['seq']!r}; seq is assigned only "
            "when the event is appended to a session's log"
        )
    session.event_seq += 1
    event["seq"] = session.event_seq
    if event["type"] == "message_complete":
        session.events[:] = [
            e for e in session.events if e["type"] != TOKEN_EVENT_TYPE
        ]
    session.events.append(event)
    return event


def seq_reservation_exhausted(session: "_Session") -> bool:
    """True when the last assigned seq has reached the persisted reservation."""
    return session.event_seq >= session.event_seq_saved + SEQ_RESERVATION_BLOCK


def backfill_event_seq(
    events: List[Dict[str, Any]], stored_event_seq: Optional[int]
) -> int:
    """Validate (or, for pre-seq session files, assign) event seqs in place.

    `stored_event_seq` is the file's `event_seq_reserved`. Returns the
    session's seq counter. Files written before seq existed have no `seq` on
    any event and no stored reservation; their events are numbered 1..n in
    log order. Anything in between is inconsistent and raises.
    """
    with_seq = sum(1 for e in events if "seq" in e)
    if with_seq == 0:
        if events and stored_event_seq is not None:
            raise ValueError(
                "session file stores event_seq but its events carry no seq"
            )
        for index, event in enumerate(events, start=1):
            event["seq"] = index
        return len(events) if events else (stored_event_seq or 0)
    if with_seq != len(events):
        raise ValueError(
            f"only {with_seq} of {len(events)} persisted events carry seq"
        )
    if stored_event_seq is None:
        raise ValueError("persisted events carry seq but event_seq is missing")
    previous = 0
    for event in events:
        if not isinstance(event["seq"], int) or event["seq"] <= previous:
            raise ValueError(
                f"persisted event seq {event['seq']!r} does not follow {previous}"
            )
        previous = event["seq"]
    if stored_event_seq < previous:
        raise ValueError(
            f"stored event_seq {stored_event_seq} is below the last event seq {previous}"
        )
    return stored_event_seq


@dataclass
class _Session:
    id: str
    config: SessionCreateRequest
    status: SessionStatus
    current_agent: str
    current_turn: int
    messages: List[MessageRecord]
    artifacts: List[ArtifactRecord]
    code_events: List[CodeEventRecord]
    output_dir: Path
    # Event log: every event emitted is appended here, only via
    # append_session_event (which assigns `seq` and compacts tokens)
    events: List[Dict[str, Any]]
    # Notified whenever a new event is appended
    event_condition: asyncio.Condition
    stop_flag: threading.Event
    cancel_response_flag: threading.Event
    user_input_queue: queue.Queue
    created_at: datetime
    updated_at: datetime
    control_message_queue: queue.Queue = field(default_factory=queue.Queue)
    name: str = ""
    # Set once sandbox + runner are wired up
    sandbox_manager: Any = None
    llm_client: Any = None
    agent_system: Any = None
    driver_agent: Any = None
    initial_history: List[Dict] = field(default_factory=list)
    model_name: str = ""
    resolved_model: Optional[ResolvedModelInfo] = None
    evaluator_llm_client: Any = None
    evaluator_model_name: str = ""
    resolved_evaluator_model: Optional[ResolvedModelInfo] = None
    evaluator_model_revision: int = 1
    evaluator_model_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    python_environment: ResolvedPythonEnvironment = field(
        default_factory=lambda: ResolvedPythonEnvironment(
            mode="bundled",
            python_executable="/usr/local/envs/rapids/bin/python",
            kind=PythonEnvironmentKind.conda,
        )
    )
    analysis_context: str = ""
    runner_task: Optional[asyncio.Task] = None
    logger: Any = None
    memory_manager: Any = None
    parent_session_id: Optional[str] = None
    forked_from_checkpoint_id: Optional[str] = None
    attempt_number: int = 1
    recovery_mode: Optional[RecoveryMode] = None
    recovery_status: RecoveryStatus = RecoveryStatus.none
    recovery_detail: Optional[str] = None
    recovery_phase: Optional[str] = None
    recovery_step: int = 0
    recovery_total_steps: int = 0
    recovery_substep: Optional[int] = None
    recovery_substep_total: Optional[int] = None
    checkpoint_id: Optional[str] = None
    checkpoint_turn: Optional[int] = None
    checkpoint_healthy: bool = False
    recovery_task: Optional[asyncio.Task] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    resume_history: Optional[List[Dict[str, str]]] = None
    resume_runner_state: Optional[Dict[str, Any]] = None
    resume_memory_state: Optional[Dict[str, Any]] = None
    # Lazily built, then cached here for the life of the session so every
    # call site (turn loop, REST review route) shares one instance and one
    # coherent in-memory index cache. See WS-0 of
    # notes/docs/session-work-items-and-brief-implementation.md.
    work_item_store: Any = None
    # The blueprint's brief_policy, resolved against this session's
    # `config.brief_mode` override (see session_brief.resolve_brief_policy).
    # Set once in _initialize_session/_recover_session, right after
    # session.agent_system is set. Not yet consumed by the runner — the
    # briefing conversation phase itself (WS-5.3-5.5) isn't implemented, so
    # this currently only records the effective policy for display/future use.
    brief_policy: Any = None
    # "briefing" while the interview conversation is active, "execution"
    # once a frozen brief is pinned (or briefing never applied). See WS-5.
    phase: str = "execution"
    # The frozen brief, as a plain dict (SessionBrief.model_dump), once
    # accepted — None until then. Draft proposals live only in the WS event
    # stream (brief_draft), not here; only the frozen brief is durable state.
    brief: Optional[Dict[str, Any]] = None
    brief_decision_queue: queue.Queue = field(default_factory=queue.Queue)
    # Last `seq` assigned to an event of this session (0 = none yet).
    # Persisted (as a reservation, see SEQ_RESERVATION_BLOCK) separately from
    # `events`, because token compaction removes the events carrying the
    # highest seqs; never recomputed from the log.
    event_seq: int = 0
    # `event_seq` as of the last successful session.json write.
    event_seq_saved: int = 0
    # Source of each code_submitted whose code_result hasn't arrived yet,
    # keyed by the opaque `action_id`, so the CodeEventRecord built on
    # code_result carries the real source.
    pending_code_sources: Dict[str, str] = field(default_factory=dict)

    def to_response(self) -> SessionResponse:
        memory = None
        if (
            self.config.memory_strategy.value != "full"
            or self.memory_manager is not None
        ):
            # Only MemoryManager (episodic strategy) carries a `.config` dict;
            # AgentReportMemory (agent_report strategy) has no equivalent settings.
            mgr_config = getattr(self.memory_manager, "config", {}) or {}
            memory = MemoryConfigResponse(
                strategy=self.config.memory_strategy.value,
                working_history_size=(
                    self.config.memory_working_history_size
                    if self.config.memory_working_history_size is not None
                    else mgr_config.get("working_history_size")
                ),
                summarization_threshold=(
                    self.config.memory_summarization_threshold
                    if self.config.memory_summarization_threshold is not None
                    else mgr_config.get("summarization_threshold")
                ),
                chunk_size=(
                    self.config.memory_chunk_size
                    if self.config.memory_chunk_size is not None
                    else mgr_config.get("chunk_size_to_summarize")
                ),
            )
        return SessionResponse(
            id=self.id,
            name=self.name or self.id[:8],
            status=self.status,
            mode=self.config.mode,
            run_mode=self.config.run_mode,
            agent_system=self.config.agent_system,
            llm_backend=self.config.llm_backend,
            resolved_model=self.resolved_model,
            evaluator_model=EvaluatorModelState(
                selection=self.config.evaluator_model,
                resolved_model=self.resolved_evaluator_model,
                revision=self.evaluator_model_revision,
            ),
            sandbox_type=self.config.sandbox_type,
            python_environment=self.python_environment,
            dataset_path=self.config.dataset_path,
            max_turns=self.config.max_turns,
            current_turn=self.current_turn,
            current_agent=self.current_agent,
            created_at=self.created_at,
            updated_at=self.updated_at,
            artifact_count=len(self.artifacts),
            message_count=len(self.messages),
            memory=memory,
            can_evaluate=(
                self.evaluator_llm_client is not None and self.agent_system is not None
            ),
            parent_session_id=self.parent_session_id,
            forked_from_checkpoint_id=self.forked_from_checkpoint_id,
            attempt_number=self.attempt_number,
            recovery_mode=self.recovery_mode,
            recovery_status=self.recovery_status,
            recovery_detail=self.recovery_detail,
            recovery_phase=self.recovery_phase,
            recovery_step=self.recovery_step,
            recovery_total_steps=self.recovery_total_steps,
            recovery_substep=self.recovery_substep,
            recovery_substep_total=self.recovery_substep_total,
            checkpoint_turn=self.checkpoint_turn,
            checkpoint_healthy=self.checkpoint_healthy,
            brief_mode=(
                (self.brief_policy.mode if self.brief_policy.enabled else "off")
                if self.brief_policy is not None
                else None
            ),
            phase=self.phase,
            brief=self.brief,
        )
