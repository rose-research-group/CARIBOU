"""Strict shared domain models for CARIBOU experiments and run attempts."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from caribou.config import get_caribou_slurm_partition

from .enums import (
    AggregateStatus,
    ArtifactType,
    BudgetStatus,
    CheckpointComponent,
    CheckpointStatus,
    EventType,
    ExecutorKind,
    ExperimentState,
    FailureCategory,
    FailureDisposition,
    InterfaceOrigin,
    MemoryStrategy,
    MetricRole,
    MetricStatus,
    RetentionPolicy,
    RunOutcome,
    RunState,
    SandboxKind,
    StudyClass,
    TopologyKind,
    UncertaintyRole,
)
from .ids import new_id


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must use UTC offset +00:00")
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]
NonEmptyStr = Annotated[StrictStr, StringConstraints(min_length=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0, allow_inf_nan=False)]
NonNegativeNumber = Union[NonNegativeInt, NonNegativeFloat]
ContentHash = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
GitCommit = Annotated[StrictStr, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
SpecId = Annotated[StrictStr, StringConstraints(pattern=r"^spec_[0-9a-f]{32}$")]
ExperimentId = Annotated[StrictStr, StringConstraints(pattern=r"^exp_[0-9a-f]{32}$")]
RunId = Annotated[StrictStr, StringConstraints(pattern=r"^run_[0-9a-f]{32}$")]
EventId = Annotated[StrictStr, StringConstraints(pattern=r"^evt_[0-9a-f]{32}$")]
ArtifactId = Annotated[StrictStr, StringConstraints(pattern=r"^art_[0-9a-f]{32}$")]
FailureId = Annotated[StrictStr, StringConstraints(pattern=r"^fail_[0-9a-f]{32}$")]
MetricId = Annotated[StrictStr, StringConstraints(pattern=r"^metric_[0-9a-f]{32}$")]
CheckpointId = Annotated[StrictStr, StringConstraints(pattern=r"^chk_[0-9a-f]{32}$")]
BudgetId = Annotated[StrictStr, StringConstraints(pattern=r"^budget_[0-9a-f]{32}$")]
AggregateId = Annotated[StrictStr, StringConstraints(pattern=r"^agg_[0-9a-f]{32}$")]


class FrozenDict(dict):
    """Serialization-friendly immutable mapping used after validation."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("domain mappings are immutable")

    __setitem__ = _immutable  # type: ignore[assignment]
    __delitem__ = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable  # type: ignore[assignment]
    update = _immutable  # type: ignore[assignment]
    __ior__ = _immutable  # type: ignore[assignment]


class FrozenList(list):
    """Serialization-friendly immutable sequence used after validation."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("domain sequences are immutable")

    __setitem__ = _immutable  # type: ignore[assignment]
    __delitem__ = _immutable  # type: ignore[assignment]
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]
    append = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    extend = _immutable  # type: ignore[assignment]
    insert = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    remove = _immutable  # type: ignore[assignment]
    reverse = _immutable  # type: ignore[assignment]
    sort = _immutable  # type: ignore[assignment]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, FrozenList):
        return value
    if isinstance(value, (list, tuple)):
        return FrozenList(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class DomainModel(BaseModel):
    """Base contract: immutable, strict, recursively closed, and finite."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def make_nested_collections_immutable(self) -> "DomainModel":
        for field_name in type(self).model_fields:
            object.__setattr__(
                self, field_name, _deep_freeze(getattr(self, field_name))
            )
        return self


class ContentReference(DomainModel):
    uri: NonEmptyStr
    content_hash: ContentHash
    size_bytes: Optional[NonNegativeInt] = None
    media_type: Optional[NonEmptyStr] = None
    accession: Optional[NonEmptyStr] = None
    version: Optional[NonEmptyStr] = None
    license: Optional[NonEmptyStr] = None


class CodeIdentity(DomainModel):
    repository: NonEmptyStr
    branch: NonEmptyStr
    commit: GitCommit
    dirty: StrictBool = False


class ModelSpec(DomainModel):
    provider: NonEmptyStr
    model: NonEmptyStr
    artifact: Optional[ContentReference] = None
    quantization: Optional[NonEmptyStr] = None
    context_length: Optional[PositiveInt] = None
    parameters: Dict[StrictStr, JsonValue] = Field(default_factory=dict)


class ToolSpec(DomainModel):
    name: NonEmptyStr
    version: NonEmptyStr
    content_hash: Optional[ContentHash] = None
    permissions: List[NonEmptyStr] = Field(default_factory=list)


class BlueprintSpec(DomainModel):
    source: ContentReference
    topology: TopologyKind
    driver_agent: NonEmptyStr
    global_policy_hash: ContentHash
    topology_hash: ContentHash
    prompt_hashes: Dict[NonEmptyStr, ContentHash]
    code_sample_hashes: Dict[NonEmptyStr, ContentHash] = Field(default_factory=dict)
    rag_corpus: Optional[ContentReference] = None
    tools: List[ToolSpec] = Field(default_factory=list)


class MemorySpec(DomainModel):
    strategy: MemoryStrategy = MemoryStrategy.full
    working_history_size: Optional[PositiveInt] = None
    summarization_threshold: Optional[PositiveInt] = None
    chunk_size: Optional[PositiveInt] = None
    summarizer_model: Optional[ModelSpec] = None

    @model_validator(mode="after")
    def validate_strategy_parameters(self) -> "MemorySpec":
        tuning = (
            self.working_history_size,
            self.summarization_threshold,
            self.chunk_size,
            self.summarizer_model,
        )
        if self.strategy not in (
            MemoryStrategy.episodic,
            MemoryStrategy.agent_report,
        ) and any(item is not None for item in tuning):
            raise ValueError(
                "memory tuning is only valid for episodic or agent_report strategies"
            )
        return self


class ResourceRequest(DomainModel):
    cpu_cores: PositiveInt = 1
    gpu_count: NonNegativeInt = 0
    memory_bytes: PositiveInt
    wall_seconds: PositiveInt
    scratch_bytes: NonNegativeInt = 0
    storage_bytes: NonNegativeInt = 0


class ContainerSpec(DomainModel):
    sandbox: SandboxKind
    image: ContentReference
    runtime_version: Optional[NonEmptyStr] = None
    gpu_enabled: StrictBool = False
    network_enabled: StrictBool = False
    force_refresh: StrictBool = False
    bind_mounts: Dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)


class ExecutionSpec(DomainModel):
    executor: ExecutorKind
    resources: ResourceRequest
    container: ContainerSpec
    partition: Optional[NonEmptyStr] = None
    account: Optional[NonEmptyStr] = None
    qos: Optional[NonEmptyStr] = None
    output_root: NonEmptyStr

    @model_validator(mode="after")
    def enforce_executor_contract(self) -> "ExecutionSpec":
        if self.executor == ExecutorKind.slurm and self.partition != get_caribou_slurm_partition():
            raise ValueError(
                f"CARIBOU Slurm execution requires partition '{get_caribou_slurm_partition()}'"
            )
        if self.executor == ExecutorKind.local and any(
            value is not None for value in (self.partition, self.account, self.qos)
        ):
            raise ValueError(
                "local execution cannot declare Slurm partition/account/qos"
            )
        return self


class RetryPolicy(DomainModel):
    maximum_attempts: PositiveInt = 1
    retryable_categories: List[FailureCategory] = Field(default_factory=list)
    base_delay_seconds: NonNegativeFloat = 0.0
    maximum_delay_seconds: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def validate_delay(self) -> "RetryPolicy":
        if self.maximum_delay_seconds < self.base_delay_seconds:
            raise ValueError("maximum retry delay cannot be less than base delay")
        return self


class StopRules(DomainModel):
    maximum_turns: PositiveInt
    timeout_seconds: PositiveInt
    maximum_consecutive_execution_failures: PositiveInt
    maximum_consecutive_no_action: PositiveInt
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class MetricDefinition(DomainModel):
    metric_key: NonEmptyStr
    name: NonEmptyStr
    role: MetricRole
    evaluator: ContentReference
    unit: Optional[NonEmptyStr] = None
    direction: Optional[Literal["minimize", "maximize", "target"]] = None
    denominator_definition: Optional[NonEmptyStr] = None
    acceptance_rule: Optional[NonEmptyStr] = None


class ConditionSpec(DomainModel):
    condition_id: NonEmptyStr
    label: NonEmptyStr
    blueprint: BlueprintSpec
    model: ModelSpec
    memory: MemorySpec = Field(default_factory=MemorySpec)
    prompt: ContentReference
    parameters: Dict[StrictStr, JsonValue] = Field(default_factory=dict)


class BudgetCounter(DomainModel):
    unit: NonEmptyStr
    limit: Optional[NonNegativeNumber] = None
    consumed: NonNegativeNumber = 0
    reserved: NonNegativeNumber = 0

    def observed(self) -> Decimal:
        return Decimal(str(self.consumed)) + Decimal(str(self.reserved))

    def is_over_limit(self) -> bool:
        return self.limit is not None and self.observed() > Decimal(str(self.limit))


class BudgetAllocation(DomainModel):
    api_calls: BudgetCounter
    input_tokens: BudgetCounter
    output_tokens: BudgetCounter
    cached_tokens: BudgetCounter
    cost: BudgetCounter
    cpu_seconds: BudgetCounter
    gpu_seconds: BudgetCounter
    memory_byte_seconds: BudgetCounter
    storage_bytes: BudgetCounter
    wall_seconds: BudgetCounter
    concurrency: BudgetCounter

    @model_validator(mode="after")
    def validate_units(self) -> "BudgetAllocation":
        expected = {
            "api_calls": "calls",
            "input_tokens": "tokens",
            "output_tokens": "tokens",
            "cached_tokens": "tokens",
            "cost": "usd",
            "cpu_seconds": "cpu_seconds",
            "gpu_seconds": "gpu_seconds",
            "memory_byte_seconds": "byte_seconds",
            "storage_bytes": "bytes",
            "wall_seconds": "seconds",
            "concurrency": "runs",
        }
        for field_name, unit in expected.items():
            if getattr(self, field_name).unit != unit:
                raise ValueError(f"{field_name} budget unit must be {unit!r}")
        return self

    def over_limit_counters(self) -> Dict[str, BudgetCounter]:
        return {
            field_name: getattr(self, field_name)
            for field_name in type(self).model_fields
            if getattr(self, field_name).is_over_limit()
        }


BudgetCounterName = Literal[
    "api_calls",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cost",
    "cpu_seconds",
    "gpu_seconds",
    "memory_byte_seconds",
    "storage_bytes",
    "wall_seconds",
    "concurrency",
]


class BudgetBreach(DomainModel):
    counter: BudgetCounterName
    limit: NonNegativeNumber
    observed: NonNegativeNumber
    detail: NonEmptyStr

    @model_validator(mode="after")
    def validate_overage(self) -> "BudgetBreach":
        if Decimal(str(self.observed)) <= Decimal(str(self.limit)):
            raise ValueError("budget breach observed value must exceed its limit")
        return self


class ExperimentSpec(DomainModel):
    schema_version: Literal["caribou.experiment_spec.v1"] = "caribou.experiment_spec.v1"
    spec_id: SpecId = Field(default_factory=lambda: new_id("spec"))
    spec_version: PositiveInt = 1
    title: NonEmptyStr
    study_class: StudyClass
    question: NonEmptyStr
    hypothesis: NonEmptyStr
    negative_interpretation: NonEmptyStr
    owner: NonEmptyStr
    reviewers: List[NonEmptyStr] = Field(min_length=1)
    code: CodeIdentity
    inputs: List[ContentReference]
    conditions: List[ConditionSpec]
    repetitions: PositiveInt
    execution: ExecutionSpec
    budget: BudgetAllocation
    metrics: List[MetricDefinition]
    stop_rules: StopRules
    randomization: NonEmptyStr
    exclusions: List[NonEmptyStr] = Field(default_factory=list)
    created_at: UtcDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> "ExperimentSpec":
        if not self.inputs:
            raise ValueError("experiment spec requires at least one input")
        if not self.conditions:
            raise ValueError("experiment spec requires at least one condition")
        if not self.metrics:
            raise ValueError("experiment spec requires at least one metric")
        if self.budget.over_limit_counters():
            raise ValueError(
                "frozen experiment budget is already over its declared limit"
            )
        condition_ids = [item.condition_id for item in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("condition IDs must be unique")
        metric_keys = [item.metric_key for item in self.metrics]
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("metric keys must be unique")
        return self


class Experiment(DomainModel):
    schema_version: Literal["caribou.experiment.v1"] = "caribou.experiment.v1"
    experiment_id: ExperimentId = Field(default_factory=lambda: new_id("exp"))
    spec_id: SpecId
    spec_version: PositiveInt
    spec_hash: ContentHash
    owner: NonEmptyStr
    state: ExperimentState = ExperimentState.draft
    transition_sequence: NonNegativeInt = 0
    run_ids: List[RunId] = Field(default_factory=list)
    aggregate_ids: List[AggregateId] = Field(default_factory=list)
    created_at: UtcDatetime = Field(default_factory=utc_now)
    updated_at: UtcDatetime = Field(default_factory=utc_now)
    completed_at: Optional[UtcDatetime] = None

    @model_validator(mode="after")
    def validate_timestamps_and_terminal_state(self) -> "Experiment":
        if len(self.run_ids) != len(set(self.run_ids)):
            raise ValueError("experiment run IDs must be unique")
        if len(self.aggregate_ids) != len(set(self.aggregate_ids)):
            raise ValueError("experiment aggregate IDs must be unique")
        if self.updated_at < self.created_at:
            raise ValueError("experiment updated_at cannot precede created_at")
        terminal = {
            ExperimentState.completed,
            ExperimentState.cancelled,
            ExperimentState.failed,
            ExperimentState.rejected,
        }
        if self.state in terminal and self.completed_at is None:
            raise ValueError("terminal experiment requires completed_at")
        if self.state not in terminal and self.completed_at is not None:
            raise ValueError("nonterminal experiment cannot have completed_at")
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("experiment completed_at cannot precede created_at")
        return self


class ExperimentTransitionRecord(DomainModel):
    schema_version: Literal["caribou.experiment_transition.v1"] = (
        "caribou.experiment_transition.v1"
    )
    event_id: EventId = Field(default_factory=lambda: new_id("evt"))
    experiment_id: ExperimentId
    sequence: PositiveInt
    occurred_at: UtcDatetime = Field(default_factory=utc_now)
    from_state: ExperimentState
    to_state: ExperimentState
    reason: NonEmptyStr
    actor: NonEmptyStr


class Run(DomainModel):
    schema_version: Literal["caribou.run.v1"] = "caribou.run.v1"
    run_id: RunId = Field(default_factory=lambda: new_id("run"))
    experiment_id: ExperimentId
    spec_hash: ContentHash
    condition_id: NonEmptyStr
    replicate_index: NonNegativeInt
    attempt_index: PositiveInt = 1
    idempotency_key: NonEmptyStr
    interface: InterfaceOrigin
    owner: NonEmptyStr
    initial_state: RunState = RunState.draft
    state: RunState = RunState.draft
    executor: ExecutorKind
    code: CodeIdentity
    resolved_model: ModelSpec
    resolved_blueprint: BlueprintSpec
    resolved_prompt: ContentReference
    resolved_memory: MemorySpec
    resolved_inputs: List[ContentReference]
    resolved_stop_rules: StopRules
    resolved_budget: BudgetAllocation
    container: ContainerSpec
    resources: ResourceRequest
    scheduler_job_id: Optional[NonEmptyStr] = None
    partition: Optional[NonEmptyStr] = None
    resumed_from_run_id: Optional[RunId] = None
    resume_checkpoint_id: Optional[CheckpointId] = None
    current_turn: NonNegativeInt = 0
    current_agent: Optional[NonEmptyStr] = None
    event_sequence: NonNegativeInt = 0
    artifact_ids: List[ArtifactId] = Field(default_factory=list)
    metric_record_ids: List[MetricId] = Field(default_factory=list)
    failure_ids: List[FailureId] = Field(default_factory=list)
    checkpoint_ids: List[CheckpointId] = Field(default_factory=list)
    budget_record_ids: List[BudgetId] = Field(default_factory=list)
    created_at: UtcDatetime = Field(default_factory=utc_now)
    updated_at: UtcDatetime = Field(default_factory=utc_now)
    queued_at: Optional[UtcDatetime] = None
    started_at: Optional[UtcDatetime] = None
    ended_at: Optional[UtcDatetime] = None
    terminal_outcome: Optional[RunOutcome] = None
    end_reason: Optional[NonEmptyStr] = None
    exit_code: Optional[StrictInt] = None
    resume_eligible: StrictBool = False

    @model_validator(mode="after")
    def validate_run_contract(self) -> "Run":
        if not self.resolved_inputs:
            raise ValueError("run requires at least one resolved input")
        if self.updated_at < self.created_at:
            raise ValueError("run updated_at cannot precede created_at")
        for label, identifiers in (
            ("artifact", self.artifact_ids),
            ("metric", self.metric_record_ids),
            ("failure", self.failure_ids),
            ("checkpoint", self.checkpoint_ids),
            ("budget", self.budget_record_ids),
        ):
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"run {label} IDs must be unique")
        ordered = [self.created_at, self.queued_at, self.started_at, self.ended_at]
        previous: datetime = self.created_at
        for timestamp in ordered[1:]:
            if timestamp is not None:
                if timestamp < previous:
                    raise ValueError("run lifecycle timestamps cannot move backward")
                previous = timestamp
        if any(
            timestamp is not None and timestamp > self.updated_at
            for timestamp in (self.queued_at, self.started_at, self.ended_at)
        ):
            raise ValueError("run updated_at cannot precede lifecycle timestamps")
        if self.state == RunState.queued and self.queued_at is None:
            raise ValueError("queued run requires queued_at")
        if self.state in {RunState.running, RunState.checkpointed, RunState.succeeded}:
            if self.started_at is None:
                raise ValueError(f"{self.state.value} run requires started_at")
        if self.executor == ExecutorKind.slurm:
            if self.partition != get_caribou_slurm_partition():
                raise ValueError(
                    f"CARIBOU Slurm run must resolve to partition '{get_caribou_slurm_partition()}'"
                )
        elif self.partition is not None or self.scheduler_job_id is not None:
            raise ValueError("local run cannot carry Slurm partition or job ID")
        if (self.resumed_from_run_id is None) != (self.resume_checkpoint_id is None):
            raise ValueError(
                "resumed_from_run_id and resume_checkpoint_id must be supplied together"
            )

        outcomes = {
            RunState.succeeded: RunOutcome.succeeded,
            RunState.failed: RunOutcome.failed,
            RunState.cancelled: RunOutcome.cancelled,
            RunState.rejected: RunOutcome.rejected,
            RunState.resumable: RunOutcome.interrupted_resumable,
        }
        terminal = set(outcomes)
        if self.state in terminal:
            if (
                self.ended_at is None
                or self.terminal_outcome != outcomes[self.state]
                or self.end_reason is None
            ):
                raise ValueError(
                    "terminal run requires matching ended_at and terminal_outcome"
                )
        elif any(
            value is not None
            for value in (
                self.ended_at,
                self.terminal_outcome,
                self.end_reason,
                self.exit_code,
            )
        ):
            raise ValueError("nonterminal run cannot have terminal result fields")
        if self.state == RunState.checkpointed and not self.checkpoint_ids:
            raise ValueError("checkpointed run requires a checkpoint")
        if self.state == RunState.resumable:
            if not self.resume_eligible or not self.checkpoint_ids:
                raise ValueError(
                    "resumable run requires a checkpoint and resume_eligible=true"
                )
        elif self.resume_eligible:
            raise ValueError("only a resumable terminal attempt may be resume eligible")
        return self


class StateTransitionPayload(DomainModel):
    from_state: RunState
    to_state: RunState
    reason: NonEmptyStr


class MessagePayload(DomainModel):
    role: NonEmptyStr
    agent_name: Optional[NonEmptyStr] = None
    content: NonEmptyStr
    is_delegation: StrictBool = False


class TokenPayload(DomainModel):
    agent_name: NonEmptyStr
    token: StrictStr


class AgentSwitchPayload(DomainModel):
    from_agent: NonEmptyStr
    to_agent: NonEmptyStr
    command: NonEmptyStr
    reason: Optional[NonEmptyStr] = None


class RagPayload(DomainModel):
    query: NonEmptyStr
    result_artifact_id: Optional[ArtifactId] = None
    success: StrictBool


class CodeSubmittedPayload(DomainModel):
    action_id: NonEmptyStr
    source_artifact_id: ArtifactId
    agent_name: NonEmptyStr
    block_index: PositiveInt
    total_blocks: PositiveInt


class CodeResultPayload(DomainModel):
    action_id: NonEmptyStr
    success: StrictBool
    duration_ms: NonNegativeInt
    stdout_artifact_id: Optional[ArtifactId] = None
    stderr_artifact_id: Optional[ArtifactId] = None


class ArtifactCreatedPayload(DomainModel):
    artifact_id: ArtifactId


class MetricRecordedPayload(DomainModel):
    metric_record_id: MetricId


class CheckpointCreatedPayload(DomainModel):
    checkpoint_id: CheckpointId


class BudgetRecordedPayload(DomainModel):
    budget_record_id: BudgetId


class FailureRecordedPayload(DomainModel):
    failure_id: FailureId


class HeartbeatPayload(DomainModel):
    message: Optional[NonEmptyStr] = None


EventPayload = Union[
    StateTransitionPayload,
    MessagePayload,
    TokenPayload,
    AgentSwitchPayload,
    RagPayload,
    CodeSubmittedPayload,
    CodeResultPayload,
    ArtifactCreatedPayload,
    MetricRecordedPayload,
    CheckpointCreatedPayload,
    BudgetRecordedPayload,
    FailureRecordedPayload,
    HeartbeatPayload,
]


class Event(DomainModel):
    schema_version: Literal["caribou.event.v1"] = "caribou.event.v1"
    event_id: EventId = Field(default_factory=lambda: new_id("evt"))
    experiment_id: ExperimentId
    run_id: RunId
    sequence: PositiveInt
    occurred_at: UtcDatetime = Field(default_factory=utc_now)
    event_type: EventType
    turn: NonNegativeInt = 0
    stage: Optional[NonEmptyStr] = None
    actor: NonEmptyStr
    correlation_id: Optional[NonEmptyStr] = None
    causation_event_id: Optional[EventId] = None
    durable: StrictBool = True
    payload: EventPayload

    @model_validator(mode="after")
    def validate_payload_type(self) -> "Event":
        expected = {
            EventType.state_transition: StateTransitionPayload,
            EventType.message: MessagePayload,
            EventType.token: TokenPayload,
            EventType.agent_switch: AgentSwitchPayload,
            EventType.rag: RagPayload,
            EventType.code_submitted: CodeSubmittedPayload,
            EventType.code_result: CodeResultPayload,
            EventType.artifact_created: ArtifactCreatedPayload,
            EventType.metric_recorded: MetricRecordedPayload,
            EventType.checkpoint_created: CheckpointCreatedPayload,
            EventType.budget_recorded: BudgetRecordedPayload,
            EventType.failure_recorded: FailureRecordedPayload,
            EventType.heartbeat: HeartbeatPayload,
        }
        if not isinstance(self.payload, expected[self.event_type]):
            raise ValueError(
                f"payload does not match event type {self.event_type.value}"
            )
        if self.event_type == EventType.token and self.durable:
            raise ValueError("token events must be explicitly ephemeral")
        return self


class Artifact(DomainModel):
    schema_version: Literal["caribou.artifact.v1"] = "caribou.artifact.v1"
    artifact_id: ArtifactId = Field(default_factory=lambda: new_id("art"))
    experiment_id: ExperimentId
    run_id: RunId
    producer_event_id: EventId
    producer: NonEmptyStr
    artifact_type: ArtifactType
    role: NonEmptyStr
    filename: NonEmptyStr
    storage_uri: NonEmptyStr
    content_hash: ContentHash
    media_type: NonEmptyStr
    schema_type: Optional[NonEmptyStr] = None
    schema_version_name: Optional[NonEmptyStr] = None
    size_bytes: NonNegativeInt
    created_at: UtcDatetime = Field(default_factory=utc_now)
    parent_artifact_ids: List[ArtifactId] = Field(default_factory=list)
    retention: RetentionPolicy = RetentionPolicy.experiment
    owner: NonEmptyStr
    sensitivity: NonEmptyStr = "internal"

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if value in (".", "..") or "/" in value or "\\" in value:
            raise ValueError("filename must be one path-safe component")
        return value

    @field_validator("storage_uri")
    @classmethod
    def validate_storage_uri(cls, value: str) -> str:
        if "://" not in value:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "relative storage path cannot be absolute or traverse parents"
                )
        return value

    @model_validator(mode="after")
    def validate_parent_ids(self) -> "Artifact":
        if self.artifact_id in self.parent_artifact_ids:
            raise ValueError("artifact cannot be its own parent")
        if len(self.parent_artifact_ids) != len(set(self.parent_artifact_ids)):
            raise ValueError("artifact parent IDs must be unique")
        return self


class FailureRecord(DomainModel):
    schema_version: Literal["caribou.failure.v1"] = "caribou.failure.v1"
    failure_id: FailureId = Field(default_factory=lambda: new_id("fail"))
    experiment_id: ExperimentId
    run_id: RunId
    event_id: EventId
    occurred_at: UtcDatetime = Field(default_factory=utc_now)
    category: FailureCategory
    stage: NonEmptyStr
    code: NonEmptyStr
    message: NonEmptyStr
    detail: Dict[StrictStr, JsonValue] = Field(default_factory=dict)
    traceback_artifact_id: Optional[ArtifactId] = None
    fatal: StrictBool
    retryable: StrictBool
    attempt: PositiveInt
    detected_by: NonEmptyStr
    correction_attempted: StrictBool = False
    correction_status: Optional[NonEmptyStr] = None
    downstream_effect: NonEmptyStr
    disposition: FailureDisposition
    caused_by_failure_id: Optional[FailureId] = None

    @model_validator(mode="after")
    def validate_failure_semantics(self) -> "FailureRecord":
        if self.caused_by_failure_id == self.failure_id:
            raise ValueError("failure cannot cause itself")
        if self.fatal and self.retryable:
            raise ValueError("a fatal failure cannot be retryable")
        if self.disposition == FailureDisposition.retry and not self.retryable:
            raise ValueError("retry disposition requires retryable=true")
        if self.fatal and self.disposition in {
            FailureDisposition.retry,
            FailureDisposition.resume,
            FailureDisposition.corrected,
        }:
            raise ValueError("fatal failure has a nonterminal disposition")
        if self.correction_attempted != (self.correction_status is not None):
            raise ValueError("correction_attempted and correction_status must agree")
        if self.disposition == FailureDisposition.corrected:
            if not self.correction_attempted or self.correction_status != "succeeded":
                raise ValueError(
                    "corrected disposition requires a succeeded correction"
                )
        return self


class MetricRecord(DomainModel):
    schema_version: Literal["caribou.metric.v1"] = "caribou.metric.v1"
    metric_record_id: MetricId = Field(default_factory=lambda: new_id("metric"))
    experiment_id: ExperimentId
    run_id: RunId
    event_id: EventId
    condition_id: NonEmptyStr
    replicate_index: NonNegativeInt
    metric_key: NonEmptyStr
    metric_name: NonEmptyStr
    evaluator: ContentReference
    recorded_at: UtcDatetime = Field(default_factory=utc_now)
    status: MetricStatus
    value: Optional[JsonValue] = None
    unit: Optional[NonEmptyStr] = None
    direction: Optional[Literal["minimize", "maximize", "target"]] = None
    denominator: Optional[NonNegativeInt] = None
    sample_count: Optional[NonNegativeInt] = None
    role: MetricRole
    uncertainty_role: UncertaintyRole
    input_artifact_ids: List[ArtifactId]
    parameters: Dict[StrictStr, JsonValue] = Field(default_factory=dict)
    failure_id: Optional[FailureId] = None
    exclusion_reason: Optional[NonEmptyStr] = None

    @model_validator(mode="after")
    def validate_measurement(self) -> "MetricRecord":
        if self.status == MetricStatus.measured:
            if (
                self.value is None
                or self.failure_id is not None
                or self.exclusion_reason is not None
            ):
                raise ValueError("measured metric requires a value and no failure")
        elif self.status in (MetricStatus.missing, MetricStatus.failed):
            if self.value is not None or self.exclusion_reason is not None:
                raise ValueError("missing or failed metric cannot contain a value")
            if self.status == MetricStatus.failed and self.failure_id is None:
                raise ValueError("failed metric requires a failure record")
        elif self.status == MetricStatus.excluded:
            if (
                self.value is not None
                or self.failure_id is not None
                or self.exclusion_reason is None
            ):
                raise ValueError("excluded metric requires only an exclusion reason")
        return self


class Checkpoint(DomainModel):
    schema_version: Literal["caribou.checkpoint.v1"] = "caribou.checkpoint.v1"
    checkpoint_id: CheckpointId = Field(default_factory=lambda: new_id("chk"))
    experiment_id: ExperimentId
    run_id: RunId
    event_id: EventId
    event_sequence: NonNegativeInt
    stage: NonEmptyStr
    turn: NonNegativeInt
    created_at: UtcDatetime = Field(default_factory=utc_now)
    parent_checkpoint_id: Optional[CheckpointId] = None
    components: List[CheckpointComponent]
    dataset_artifact_id: Optional[ArtifactId] = None
    message_history_artifact_id: Optional[ArtifactId] = None
    agent_state_artifact_id: Optional[ArtifactId] = None
    executed_actions_artifact_id: Optional[ArtifactId] = None
    artifact_manifest_id: ArtifactId
    random_state_artifact_id: Optional[ArtifactId] = None
    spec_hash: ContentHash
    code_commit: GitCommit
    container_digest: ContentHash
    model_identity: NonEmptyStr
    integrity_hash: ContentHash
    status: CheckpointStatus = CheckpointStatus.complete
    resume_requirements: List[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_components(self) -> "Checkpoint":
        if self.parent_checkpoint_id == self.checkpoint_id:
            raise ValueError("checkpoint cannot be its own parent")
        if CheckpointComponent.artifact_manifest not in self.components:
            raise ValueError("checkpoint must include its artifact manifest component")
        if len(self.components) != len(set(self.components)):
            raise ValueError("checkpoint components must be unique")
        required = {
            CheckpointComponent.dataset_state: self.dataset_artifact_id,
            CheckpointComponent.message_history: self.message_history_artifact_id,
            CheckpointComponent.agent_state: self.agent_state_artifact_id,
            CheckpointComponent.executed_actions: self.executed_actions_artifact_id,
            CheckpointComponent.random_state: self.random_state_artifact_id,
        }
        for component, artifact_id in required.items():
            if component in self.components and artifact_id is None:
                raise ValueError(
                    f"checkpoint component {component.value} requires its artifact"
                )
        return self


def checkpoint_integrity_hash(checkpoint: Checkpoint) -> str:
    """Hash the immutable checkpoint envelope excluding its hash field."""

    payload = checkpoint.model_dump(mode="json")
    payload.pop("integrity_hash", None)
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class BudgetRecord(DomainModel):
    schema_version: Literal["caribou.budget.v1"] = "caribou.budget.v1"
    budget_record_id: BudgetId = Field(default_factory=lambda: new_id("budget"))
    experiment_id: ExperimentId
    run_id: Optional[RunId] = None
    event_id: Optional[EventId] = None
    recorded_at: UtcDatetime = Field(default_factory=utc_now)
    allocation: BudgetAllocation
    status: BudgetStatus
    source: Literal["estimate", "measured", "reconciled"]
    breaches: List[BudgetBreach] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> "BudgetRecord":
        if (self.run_id is None) != (self.event_id is None):
            raise ValueError(
                "run-scoped budget and event IDs must be supplied together"
            )
        overages = set(self.allocation.over_limit_counters())
        declared = {breach.counter for breach in self.breaches}
        if self.status == BudgetStatus.within_limit:
            if self.breaches or overages:
                raise ValueError("within-limit budget cannot contain an overage")
        else:
            if not overages or declared != overages:
                raise ValueError(
                    "exceeded or rejected budget must identify every over-limit counter"
                )
            for breach in self.breaches:
                counter = getattr(self.allocation, breach.counter)
                if (
                    Decimal(str(breach.limit)) != Decimal(str(counter.limit))
                    or Decimal(str(breach.observed)) != counter.observed()
                ):
                    raise ValueError(
                        "budget breach values do not match the recorded allocation"
                    )
        return self


class Aggregate(DomainModel):
    schema_version: Literal["caribou.aggregate.v1"] = "caribou.aggregate.v1"
    aggregate_id: AggregateId = Field(default_factory=lambda: new_id("agg"))
    experiment_id: ExperimentId
    spec_hash: ContentHash
    status: AggregateStatus
    included_run_ids: List[RunId]
    excluded_run_ids: List[RunId] = Field(default_factory=list)
    exclusion_reasons: Dict[RunId, NonEmptyStr] = Field(default_factory=dict)
    method: ContentReference
    created_at: UtcDatetime = Field(default_factory=utc_now)
    metric_record_ids: List[MetricId] = Field(default_factory=list)
    artifact_ids: List[ArtifactId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_run_sets(self) -> "Aggregate":
        included = set(self.included_run_ids)
        excluded = set(self.excluded_run_ids)
        if not included:
            raise ValueError("aggregate requires at least one included run")
        if len(self.included_run_ids) != len(included) or len(
            self.excluded_run_ids
        ) != len(excluded):
            raise ValueError("aggregate run IDs must be unique")
        if included & excluded:
            raise ValueError("a run cannot be both included and excluded")
        if set(self.exclusion_reasons) != excluded:
            raise ValueError("every excluded run requires exactly one exclusion reason")
        return self
