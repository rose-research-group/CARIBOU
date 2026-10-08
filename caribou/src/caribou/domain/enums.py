"""Enumerations shared by CARIBOU CLI, web, and benchmark adapters."""

from enum import Enum


class StringEnum(str, Enum):
    """Python 3.10-compatible string enum."""


class StudyClass(StringEnum):
    exploratory = "exploratory"
    pilot = "pilot"
    confirmatory = "confirmatory"


class ExperimentState(StringEnum):
    draft = "draft"
    validated = "validated"
    planned = "planned"
    active = "active"
    aggregating = "aggregating"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"
    rejected = "rejected"


class RunState(StringEnum):
    draft = "draft"
    validated = "validated"
    planned = "planned"
    queued = "queued"
    starting = "starting"
    running = "running"
    checkpointed = "checkpointed"
    cancelling = "cancelling"
    cancelled = "cancelled"
    failed = "failed"
    resumable = "resumable"
    succeeded = "succeeded"
    rejected = "rejected"


class RunOutcome(StringEnum):
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    rejected = "rejected"
    interrupted_resumable = "interrupted_resumable"


class InterfaceOrigin(StringEnum):
    cli = "cli"
    web = "web"
    benchmark = "benchmark"
    migration = "migration"


class ExecutorKind(StringEnum):
    local = "local"
    slurm = "slurm"


class SandboxKind(StringEnum):
    docker = "docker"
    apptainer = "apptainer"
    singularity = "singularity"
    offline = "offline"


class TopologyKind(StringEnum):
    one_shot = "one_shot"
    single_agent = "single_agent"
    multi_agent = "multi_agent"


class MemoryStrategy(StringEnum):
    full = "full"
    episodic = "episodic"
    agent_report = "agent_report"
    none = "none"


class EventType(StringEnum):
    state_transition = "state_transition"
    message = "message"
    token = "token"
    agent_switch = "agent_switch"
    rag = "rag"
    code_submitted = "code_submitted"
    code_result = "code_result"
    artifact_created = "artifact_created"
    metric_recorded = "metric_recorded"
    checkpoint_created = "checkpoint_created"
    budget_recorded = "budget_recorded"
    failure_recorded = "failure_recorded"
    heartbeat = "heartbeat"
    work_item_changed = "work_item_changed"
    block_changed = "block_changed"


class ArtifactType(StringEnum):
    dataset = "dataset"
    plot = "plot"
    code = "code"
    report = "report"
    note = "note"
    todo = "todo"
    notebook = "notebook"
    log = "log"
    checkpoint = "checkpoint"
    manifest = "manifest"
    metric_table = "metric_table"
    message_history = "message_history"
    other = "other"


class RetentionPolicy(StringEnum):
    temporary = "temporary"
    experiment = "experiment"
    evidence = "evidence"
    release = "release"


class FailureCategory(StringEnum):
    validation = "validation"
    permission = "permission"
    budget = "budget"
    provider = "provider"
    scheduler = "scheduler"
    container = "container"
    execution = "execution"
    metric = "metric"
    persistence = "persistence"
    cancellation = "cancellation"
    scientific = "scientific"
    timeout = "timeout"
    internal = "internal"


class FailureDisposition(StringEnum):
    retry = "retry"
    resume = "resume"
    terminate = "terminate"
    corrected = "corrected"
    not_claimed = "not_claimed"
    investigate = "investigate"


class MetricRole(StringEnum):
    primary = "primary"
    secondary = "secondary"
    exploratory = "exploratory"
    diagnostic = "diagnostic"


class MetricStatus(StringEnum):
    measured = "measured"
    missing = "missing"
    failed = "failed"
    excluded = "excluded"


class UncertaintyRole(StringEnum):
    not_applicable = "not_applicable"
    point_estimate = "point_estimate"
    replicate = "replicate"
    lower_bound = "lower_bound"
    upper_bound = "upper_bound"
    standard_error = "standard_error"
    standard_deviation = "standard_deviation"
    confidence_interval = "confidence_interval"
    credible_interval = "credible_interval"
    distribution = "distribution"


class BudgetStatus(StringEnum):
    within_limit = "within_limit"
    exceeded = "exceeded"
    rejected = "rejected"


class CheckpointStatus(StringEnum):
    complete = "complete"
    invalid = "invalid"
    incompatible = "incompatible"


class CheckpointComponent(StringEnum):
    dataset_state = "dataset_state"
    message_history = "message_history"
    agent_state = "agent_state"
    executed_actions = "executed_actions"
    artifact_manifest = "artifact_manifest"
    random_state = "random_state"


class AggregateStatus(StringEnum):
    complete = "complete"
    partial = "partial"
    invalid = "invalid"
