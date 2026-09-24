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

# Bound the in-memory event log per session so long-running sessions don't
# leak memory as tokens/messages accumulate. When exceeded, oldest events
# are dropped; the persisted session.json still reflects the latest state.
MAX_EVENTS_PER_SESSION = 5000
# When trimming kicks in, keep this many recent events.
EVENTS_TRIM_TARGET = 4000


def trim_events(events: List[Dict[str, Any]]) -> None:
    """Trim the events list in place if it exceeds the max."""
    if len(events) > MAX_EVENTS_PER_SESSION:
        drop = len(events) - EVENTS_TRIM_TARGET
        del events[:drop]


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
    # Event log: every event emitted is appended here
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
        )
