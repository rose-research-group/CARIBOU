from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from caribou.core.python_environments import ResolvedPythonEnvironment


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SessionStatus(str, Enum):
    initializing = "initializing"
    idle = "idle"
    running = "running"
    stopped = "stopped"
    error = "error"
    recovering = "recovering"


class SessionMode(str, Enum):
    interactive = "interactive"
    auto = "auto"


class RunMode(str, Enum):
    full_system = "full_system"
    single_agent = "single_agent"
    one_shot = "one_shot"


class SandboxType(str, Enum):
    singularity = "singularity"
    docker = "docker"
    offline = "offline"


class ArtifactType(str, Enum):
    plot = "plot"
    data = "data"
    code = "code"
    report = "report"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class MemoryStrategy(str, Enum):
    full = "full"
    episodic = "episodic"
    agent_report = "agent_report"
    none = "none"


class RecoveryMode(str, Enum):
    smart = "smart"
    literal_replay = "literal_replay"


class BranchRestoreMode(str, Enum):
    """How a branch child rebuilds the state at its block's entry checkpoint."""

    replay = "replay"
    checkpoint = "checkpoint"
    llm_regen = "llm_regen"


class RecoveryStatus(str, Enum):
    none = "none"
    awaiting_checkpoint = "awaiting_checkpoint"
    recovering = "recovering"
    recovered = "recovered"
    partial = "partial"
    failed = "failed"
    accepted_partial = "accepted_partial"


class MemoryConfigResponse(BaseModel):
    strategy: str = "full"
    working_history_size: Optional[int] = None
    summarization_threshold: Optional[int] = None
    chunk_size: Optional[int] = None


class EvaluatorModelConfig(BaseModel):
    """Requested evaluator model binding.

    ``inherit_worker`` preserves the historical behaviour while keeping the
    evaluator role explicit in persisted state and API responses.
    """

    mode: Literal["inherit_worker", "explicit"] = "inherit_worker"
    llm_backend: Optional[str] = None
    model_name: Optional[str] = None
    ollama_model: Optional[str] = None

    @model_validator(mode="after")
    def validate_explicit_binding(self) -> "EvaluatorModelConfig":
        if self.mode == "explicit" and not (self.llm_backend or "").strip():
            raise ValueError("explicit evaluator model requires llm_backend")
        if self.mode == "inherit_worker" and any(
            value is not None
            for value in (self.llm_backend, self.model_name, self.ollama_model)
        ):
            raise ValueError(
                "inherit_worker evaluator model cannot declare provider-specific fields"
            )
        return self


def _normalize_optional_reason(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class SessionCreateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    mode: SessionMode
    run_mode: RunMode = RunMode.full_system
    agent_system: str
    llm_backend: str
    model_name: Optional[str] = None
    ollama_model: Optional[str] = None
    sandbox_type: SandboxType = SandboxType.singularity
    python_environment_path: Optional[str] = None
    dataset_path: str
    reference_dataset_path: Optional[str] = None
    max_turns: Optional[int] = None
    initial_prompt: Optional[str] = None
    memory_strategy: MemoryStrategy = MemoryStrategy.full
    memory_working_history_size: Optional[int] = None
    memory_summarization_threshold: Optional[int] = None
    memory_chunk_size: Optional[int] = None
    compress_memory: bool = False
    agent_report_memory: bool = False
    evaluator_model: EvaluatorModelConfig = Field(default_factory=EvaluatorModelConfig)
    # None = use the blueprint's own brief_policy (its default). An explicit
    # value overrides the blueprint for this session only — e.g. turning
    # briefing on for a blueprint that doesn't declare brief_policy at all.
    brief_mode: Optional[Literal["off", "context", "seed_item"]] = None
    # A pre-authored, already-frozen brief (raw fields, not a SessionBrief
    # instance — created_at/created_by are harness-set regardless of what's
    # supplied here). Mirrors the CLI's `--brief <path>`: skips the briefing
    # conversation entirely and pins this brief immediately. Auto sessions
    # never run a briefing conversation, so this is their only way to have
    # a brief at all.
    brief: Optional[Dict[str, Any]] = None


class ResolvedModelInfo(BaseModel):
    """Exact model identity and effective request controls for provenance."""

    provider: str
    model: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EvaluatorModelState(BaseModel):
    selection: EvaluatorModelConfig
    resolved_model: Optional[ResolvedModelInfo] = None
    revision: int = Field(default=1, ge=1)


class EvaluatorModelUpdateRequest(BaseModel):
    selection: EvaluatorModelConfig
    expected_revision: int = Field(ge=1)
    reason: Optional[str] = Field(default=None, max_length=1000)

    _normalize_reason = field_validator("reason", mode="before")(
        _normalize_optional_reason
    )


class MessageRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    turn: int
    role: str
    agent_name: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_delegation: bool = False


class ArtifactRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    turn: int
    type: ArtifactType
    filename: str
    mime_type: str
    size_bytes: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    local_path: str = ""
    # Posix path relative to the session's output_dir — the artifact's
    # identity. An overwrite of the same path replaces the record.
    path: str
    # st_mtime_ns of the file version this record describes.
    mtime_ns: int
    # The code block that produced it (execution.event_ids.make_action_id),
    # or None when produced outside a code block.
    action_id: Optional[str] = None

    @property
    def download_url(self) -> str:
        return f"/api/sessions/{self.session_id}/artifacts/{self.id}/download"


class CodeEventRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    turn: int
    agent_name: str
    source: str
    stdout: str = ""
    stderr: str = ""
    success: bool = True
    duration_ms: int = 0


class EvaluationResult(BaseModel):
    session_id: str
    turn: int
    evaluator_agent: str
    evaluator_source: str
    model: str
    provider: Optional[str] = None
    evaluator_model: Optional[ResolvedModelInfo] = None
    evaluator_model_revision: int = 1
    provider_receipt: Dict[str, Any] = Field(default_factory=dict)
    assessment: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkItemSummary(BaseModel):
    id: int
    title: str
    status: str
    owner: str
    created_turn: int
    created_at: str
    completed_turn: Optional[int] = None
    completed_at: Optional[str] = None


class WorkItemAnchor(BaseModel):
    """What a human ticket points at; the store requires at least one non-null."""

    model_config = ConfigDict(extra="forbid")

    block_id: Optional[str] = None
    action_id: Optional[str] = None
    artifact_path: Optional[str] = None


class WorkItemDetail(WorkItemSummary):
    schema_version: str
    # The store writes session_id/origin_run_id; it has never written run_id
    # since work items became session-owned.
    session_id: str
    origin_run_id: str
    body: str
    # Null for agent-opened items and for v2 items written before anchors.
    anchor: Optional[WorkItemAnchor] = None
    completion_summary: Optional[str] = None
    closed_turn: Optional[int] = None
    closed_at: Optional[str] = None
    transitions: List[Dict[str, Any]] = Field(default_factory=list)
    reviews: List[Dict[str, Any]] = Field(default_factory=list)
    opening_commit: Optional[str] = None
    latest_commit: Optional[str] = None


class BlockEntry(BaseModel):
    """What the web loop captured when a block was created: its entry checkpoint.

    `checkpoint_id` is null when no checkpoint exists (the CLI loop records
    entries without one); such a block cannot be branched from.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    turn: int
    checkpoint_id: Optional[str]
    checkpoint_complete: bool
    fingerprint: Optional[Dict[str, Any]]
    work_items_commit: Optional[str]


class BlockInheritedFrom(BaseModel):
    """The parent block a branch child's block was copied from."""

    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str
    block_id: str


class BlockRecord(BaseModel):
    """One workbench block, as BlockTracker writes it to blocks.json.

    `caribou.block.v2` adds `entry` and `inherited_from`; both are required
    keys in v2 (possibly null). A `caribou.block.v1` record has neither: it
    is legitimate legacy and reads as entry=null, inherited_from=null.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["caribou.block.v1", "caribou.block.v2"]
    block_id: str = Field(pattern=r"^blk-\d{4,}$")
    session_id: str
    index: int = Field(ge=1)
    work_item_id: Optional[int]
    attempt: int = Field(ge=1)
    implicit: bool
    title: str
    kind: None
    agents: List[str]
    status: Literal["running", "ok", "warn", "error"]
    turn_start: int
    turn_end: int
    action_ids: List[str]
    failed_action_ids: List[str]
    artifact_paths: List[str]
    created_at: str
    updated_at: str
    entry: Optional[BlockEntry] = None
    inherited_from: Optional[BlockInheritedFrom] = None

    @model_validator(mode="before")
    @classmethod
    def _fields_match_schema_version(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        version = value.get("schema_version")
        v2_fields = ("entry", "inherited_from")
        if version == "caribou.block.v1":
            present = [name for name in v2_fields if name in value]
            if present:
                raise ValueError(
                    f"a caribou.block.v1 record cannot carry {present}"
                )
        elif version == "caribou.block.v2":
            missing = [name for name in v2_fields if name not in value]
            if missing:
                raise ValueError(
                    f"a caribou.block.v2 record must carry {missing} (null allowed)"
                )
        return value


class BlocksResponse(BaseModel):
    """`GET /api/sessions/{id}/blocks`; `recorded` is false when the session has no blocks.json."""

    model_config = ConfigDict(extra="forbid", strict=True)

    recorded: bool
    blocks: List[BlockRecord]


class WorkItemCreateRequest(BaseModel):
    """`POST /api/sessions/{id}/work-items`: a ticket a person opens for an agent."""

    model_config = ConfigDict(extra="forbid")

    title: str
    body: str
    owner: str
    anchor: Optional[WorkItemAnchor] = None


class WorkItemHumanReviewRequest(BaseModel):
    """`POST /api/sessions/{id}/work-items/{n}/human-review`."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["approve", "reject"]
    assessment: str


class WorkItemReviewResult(BaseModel):
    item: WorkItemDetail
    verdict: str
    assessment: str
    provider_receipt: Dict[str, Any] = Field(default_factory=dict)


class BriefDecisionRequest(BaseModel):
    decision: Literal["accept", "reject", "edit"]
    reason: Optional[str] = None
    # Required when decision == "edit": the corrected brief fields, in the
    # same shape as the agent's ```brief block (deliverable, in_scope,
    # out_of_scope, done_when, and the optional fields).
    brief: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _edit_requires_brief(self) -> "BriefDecisionRequest":
        if self.decision == "edit" and self.brief is None:
            raise ValueError("decision 'edit' requires a 'brief' payload")
        return self


class QueuedMessage(BaseModel):
    """A user message waiting in a session's message queue.

    Queued while the agent is busy and delivered (as a UserTurn keeping its
    block_id) the next time the runner waits for input. Wire shape of every
    item in `message_queue_changed` events and `SessionResponse.message_queue`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    block_id: Optional[str] = None
    created_at: datetime

    @field_validator("id", "content")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("block_id")
    @classmethod
    def _non_empty_block_id(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value:
            raise ValueError("block_id must be a non-empty string or null")
        return value


class SessionResponse(BaseModel):
    id: str
    name: str
    status: SessionStatus
    mode: SessionMode
    run_mode: RunMode
    agent_system: str
    llm_backend: str
    resolved_model: Optional[ResolvedModelInfo] = None
    evaluator_model: EvaluatorModelState = Field(
        default_factory=lambda: EvaluatorModelState(selection=EvaluatorModelConfig())
    )
    sandbox_type: SandboxType
    python_environment: ResolvedPythonEnvironment
    dataset_path: str
    max_turns: Optional[int]
    current_turn: int
    current_agent: str
    created_at: datetime
    updated_at: datetime
    artifact_count: int
    message_count: int
    memory: Optional[MemoryConfigResponse] = None
    # False after a server restart until the runner is relaunched — restored
    # sessions don't carry a live llm_client/agent_system (see
    # session_persistence.py), so /evaluate would 400 even though current_turn > 0.
    can_evaluate: bool = False
    parent_session_id: Optional[str] = None
    forked_from_checkpoint_id: Optional[str] = None
    # Set only on a branch child (POST /blocks/{block_id}/branch).
    forked_from_block_id: Optional[str] = None
    branch_restore_mode: Optional[BranchRestoreMode] = None
    branch_instruction: Optional[str] = None
    attempt_number: int = 1
    recovery_mode: Optional[RecoveryMode] = None
    recovery_status: RecoveryStatus = RecoveryStatus.none
    recovery_detail: Optional[str] = None
    recovery_phase: Optional[str] = None
    recovery_step: int = 0
    recovery_total_steps: int = 0
    recovery_substep: Optional[int] = None
    recovery_substep_total: Optional[int] = None
    checkpoint_turn: Optional[int] = None
    checkpoint_healthy: bool = False
    # The effective brief mode for this session (session override, if any,
    # else the blueprint's own brief_policy) — "off" if briefing is
    # disabled. None until the agent system has loaded (still initializing).
    brief_mode: Optional[Literal["off", "context", "seed_item"]] = None
    # "briefing" while the interview conversation is active, "execution"
    # once a frozen brief is pinned (or briefing never applied/ran).
    phase: Literal["briefing", "execution"] = "execution"
    # The frozen brief, once accepted — None until then. Draft proposals
    # arrive only via the brief_draft WS event, not this field.
    brief: Optional[Dict[str, Any]] = None
    # Messages queued while the agent was busy, oldest first; the latest
    # `message_queue_changed` event supersedes this snapshot.
    message_queue: List[QueuedMessage] = Field(default_factory=list)


class SessionResumeRequest(BaseModel):
    recovery_mode: RecoveryMode = RecoveryMode.smart
    target_mode: Optional[SessionMode] = None
    additional_turns: Optional[int] = Field(default=None, ge=1, le=10_000)
    acknowledge_replay_risk: bool = False


class SessionForkRequest(SessionResumeRequest):
    name: str = Field(min_length=1, max_length=120)
    llm_backend: Optional[str] = None
    model_name: Optional[str] = None
    ollama_model: Optional[str] = None
    evaluator_model: Optional[EvaluatorModelConfig] = None
    model_change_reason: Optional[str] = Field(default=None, max_length=1000)
    # Omitted means inherit; explicit null selects the bundled environment.
    python_environment_path: Optional[str] = None

    _normalize_reason = field_validator("model_change_reason", mode="before")(
        _normalize_optional_reason
    )


class BranchRequest(BaseModel):
    """`POST /api/sessions/{id}/blocks/{block_id}/branch`."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1, max_length=4000)
    restore_mode: BranchRestoreMode
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    acknowledge_unverified: bool = False

    @field_validator("instruction")
    @classmethod
    def _instruction_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction cannot be blank")
        return value

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("name cannot be blank")
        return value


class BranchSummary(BaseModel):
    """One entry of `GET /api/sessions/{id}/branches`."""

    session_id: str
    name: str
    status: SessionStatus
    recovery_status: RecoveryStatus
    forked_from_block_id: str
    branch_restore_mode: Optional[BranchRestoreMode] = None
    branch_instruction: Optional[str] = None
    created_at: datetime


class PythonEnvironmentPathRequest(BaseModel):
    path: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class LLMBackend(BaseModel):
    id: str
    provider: str
    display_name: str
    available: bool
    model_name: Optional[str] = None
    thinking: Optional[bool] = None
    status: Optional[str] = None
    message: Optional[str] = None
    suggested_fix: Optional[str] = None


class AgentBlueprint(BaseModel):
    name: str
    description: str
    agents: List[str]
    has_rag: bool
    path: str
    is_package_default: bool = False


# ---------------------------------------------------------------------------
# Blueprint editor
# ---------------------------------------------------------------------------


class CommandConfig(BaseModel):
    target_agent: str
    description: str


class AgentConfig(BaseModel):
    prompt: str
    rag_enabled: bool = False
    neighbors: Dict[str, CommandConfig] = {}
    code_samples: List[str] = []


class WorkItemPolicyConfig(BaseModel):
    qc_mode: Literal["optional", "required"] = "optional"


class BriefPolicyConfig(BaseModel):
    """A blueprint's `brief_policy` block, in the shape BriefPolicy.from_dict reads."""

    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool = False
    mode: Literal["context", "seed_item"] = "context"
    require_confirmation: StrictBool = True


class BlueprintContent(BaseModel):
    name: str
    global_policy: str
    agents: Dict[str, AgentConfig]
    is_package_default: bool
    evaluator_agent: Optional[str] = None
    work_item_policy: WorkItemPolicyConfig = Field(default_factory=WorkItemPolicyConfig)
    # None when the blueprint file has no brief_policy block.
    brief_policy: Optional[BriefPolicyConfig] = None


class SaveBlueprintRequest(BaseModel):
    name: str
    global_policy: str
    agents: Dict[str, AgentConfig]
    evaluator_agent: Optional[str] = None
    work_item_policy: WorkItemPolicyConfig = Field(default_factory=WorkItemPolicyConfig)
    # None when the blueprint file has no brief_policy block.
    brief_policy: Optional[BriefPolicyConfig] = None


class ServerStatus(BaseModel):
    version: str = "0.1.0"
    sandbox_type: str
    active_sessions: int


class OllamaModelsResponse(BaseModel):
    host: str
    running: bool
    models: List[str]
    default_model: str
    status: str
    message: str
    suggested_fix: Optional[str] = None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DatasetRecord(BaseModel):
    filename: str
    path: str
    size_bytes: int
    uploaded_at: datetime


class DatasetPathValidationRequest(BaseModel):
    path: str


# ---------------------------------------------------------------------------
# WebSocket messages (client → server)
# ---------------------------------------------------------------------------


class WSRunMessage(BaseModel):
    type: str = "run"
    content: str


class WSUserMessage(BaseModel):
    type: str = "user_message"
    content: str


class WSStopMessage(BaseModel):
    type: str = "stop"


# ---------------------------------------------------------------------------
# WebSocket events (server → client)  — raw dicts emitted by streaming_runner
# ---------------------------------------------------------------------------
# Shape: { type: str, session_id: str, turn: int, timestamp: str, data: dict,
#          seq: int }  — seq is assigned when the session manager appends the
#          event to the session's log (session_state.append_session_event).
# Types: token | message_complete | agent_switch | code_submitted |
#        code_result | artifact | block_changed | status_change |
#        metrics_result | error | pong


def make_event(
    event_type: str,
    data: Dict[str, Any],
    session_id: str = "",
    turn: int = 0,
) -> Dict[str, Any]:
    return {
        "type": event_type,
        "session_id": session_id,
        "turn": turn,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }
