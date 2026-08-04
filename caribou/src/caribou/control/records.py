"""Versioned operational records kept outside the scientific domain objects."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from caribou.config import get_caribou_slurm_partition
from caribou.domain.models import (
    Artifact,
    ContentHash,
    ExperimentId,
    NonEmptyStr,
    RunId,
    UtcDatetime,
    utc_now,
)


class ControlRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
    )


class IdempotencyClaim(ControlRecord):
    spec_hash: ContentHash
    experiment_id: ExperimentId
    run_ids: Tuple[RunId, ...]
    created_at: UtcDatetime = Field(default_factory=utc_now)


class StoreIndex(ControlRecord):
    schema_version: Literal["caribou.control_index.v1"] = "caribou.control_index.v1"
    experiments: Dict[ExperimentId, Tuple[RunId, ...]] = Field(default_factory=dict)
    runs: Dict[RunId, ExperimentId] = Field(default_factory=dict)
    idempotency: Dict[ContentHash, IdempotencyClaim] = Field(default_factory=dict)
    updated_at: UtcDatetime = Field(default_factory=utc_now)


class ExecutionHandle(ControlRecord):
    schema_version: Literal["caribou.execution_handle.v1"] = (
        "caribou.execution_handle.v1"
    )
    run_id: RunId
    pid: StrictInt = Field(gt=0)
    hostname: NonEmptyStr
    process_start_identity: NonEmptyStr
    launch_nonce: NonEmptyStr
    worker_module: NonEmptyStr
    log_path: NonEmptyStr
    launched_at: UtcDatetime = Field(default_factory=utc_now)


class SlurmExecutionHandle(ControlRecord):
    schema_version: Literal["caribou.slurm_execution_handle.v1"] = (
        "caribou.slurm_execution_handle.v1"
    )
    run_id: RunId
    job_id: NonEmptyStr
    partition: NonEmptyStr = Field(default_factory=get_caribou_slurm_partition)
    account: Optional[NonEmptyStr] = None
    qos: Optional[NonEmptyStr] = None
    script_path: NonEmptyStr
    script_hash: ContentHash
    stdout_path: NonEmptyStr
    submitted_at: UtcDatetime = Field(default_factory=utc_now)
    released_at: Optional[UtcDatetime] = None

    @model_validator(mode="after")
    def validate_root_job_id(self) -> "SlurmExecutionHandle":
        if not self.job_id.isascii() or not self.job_id.isdigit():
            raise ValueError("Slurm job_id must be one numeric root job ID")
        return self

    @model_validator(mode="after")
    def validate_partition(self) -> "SlurmExecutionHandle":
        if self.partition != get_caribou_slurm_partition():
            raise ValueError(
                f"Slurm execution handle must bind partition '{get_caribou_slurm_partition()}'"
            )
        return self


class SlurmSubmissionLedger(ControlRecord):
    schema_version: Literal["caribou.slurm_submission_ledger.v1"] = (
        "caribou.slurm_submission_ledger.v1"
    )
    run_id: RunId
    job_name: NonEmptyStr
    script_hash: ContentHash
    attempts: Tuple[UtcDatetime, ...] = ()

    @model_validator(mode="after")
    def validate_attempt_order(self) -> "SlurmSubmissionLedger":
        if tuple(sorted(self.attempts)) != self.attempts:
            raise ValueError("Slurm submission attempts must be time ordered")
        return self


class SlurmAccounting(ControlRecord):
    schema_version: Literal["caribou.slurm_accounting.v1"] = (
        "caribou.slurm_accounting.v1"
    )
    run_id: RunId
    job_id: NonEmptyStr
    partition: NonEmptyStr = Field(default_factory=get_caribou_slurm_partition)
    state: NonEmptyStr
    terminal: StrictBool
    exit_code: Optional[NonEmptyStr] = None
    elapsed_seconds: StrictInt = Field(ge=0)
    allocated_cpus: StrictInt = Field(ge=0)
    requested_memory: Optional[NonEmptyStr] = None
    max_rss_kib: Optional[StrictInt] = Field(default=None, ge=0)
    node_list: Optional[NonEmptyStr] = None
    started_at_raw: Optional[NonEmptyStr] = None
    ended_at_raw: Optional[NonEmptyStr] = None
    raw_output_path: NonEmptyStr
    raw_output_hash: ContentHash
    consistent_with_run: Optional[StrictBool] = None
    recorded_at: UtcDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_root_job_id(self) -> "SlurmAccounting":
        if not self.job_id.isascii() or not self.job_id.isdigit():
            raise ValueError("Slurm job_id must be one numeric root job ID")
        return self

    @model_validator(mode="after")
    def validate_partition(self) -> "SlurmAccounting":
        if self.partition != get_caribou_slurm_partition():
            raise ValueError(
                f"Slurm accounting must record partition '{get_caribou_slurm_partition()}'"
            )
        return self


class SlurmCancellationAttempt(ControlRecord):
    attempted_at: UtcDatetime = Field(default_factory=utc_now)
    succeeded: StrictBool
    error_code: Optional[NonEmptyStr] = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "SlurmCancellationAttempt":
        if self.succeeded == (self.error_code is not None):
            raise ValueError(
                "successful cancellation has no error; failure requires one"
            )
        return self


class SlurmCancellationLedger(ControlRecord):
    schema_version: Literal["caribou.slurm_cancellation_ledger.v1"] = (
        "caribou.slurm_cancellation_ledger.v1"
    )
    run_id: RunId
    job_id: NonEmptyStr
    attempts: Tuple[SlurmCancellationAttempt, ...] = ()

    @model_validator(mode="after")
    def validate_identity(self) -> "SlurmCancellationLedger":
        if not self.job_id.isascii() or not self.job_id.isdigit():
            raise ValueError("Slurm job_id must be one numeric root job ID")
        if sum(attempt.succeeded for attempt in self.attempts) > 1:
            raise ValueError("Slurm cancellation can be durably successful only once")
        return self


class CancelRequest(ControlRecord):
    schema_version: Literal["caribou.cancel_request.v1"] = "caribou.cancel_request.v1"
    run_id: RunId
    requested_at: UtcDatetime = Field(default_factory=utc_now)
    actor: NonEmptyStr
    reason: NonEmptyStr


class CheckpointRequest(ControlRecord):
    """One idempotent request to stop a run at its next safe turn boundary."""

    schema_version: Literal["caribou.checkpoint_request.v1"] = (
        "caribou.checkpoint_request.v1"
    )
    run_id: RunId
    idempotency_key_hash: ContentHash
    requested_at: UtcDatetime = Field(default_factory=utc_now)
    actor: NonEmptyStr
    reason: NonEmptyStr


class ProviderCallUsage(ControlRecord):
    """Whitelisted token counts returned by one provider call attempt."""

    prompt_tokens: Optional[StrictInt] = Field(default=None, ge=0)
    completion_tokens: Optional[StrictInt] = Field(default=None, ge=0)
    total_tokens: Optional[StrictInt] = Field(default=None, ge=0)
    cached_tokens: Optional[StrictInt] = Field(default=None, ge=0)
    cache_miss_tokens: Optional[StrictInt] = Field(default=None, ge=0)
    reasoning_tokens: Optional[StrictInt] = Field(default=None, ge=0)


class ProviderCallReceipt(ControlRecord):
    """Redacted, immutable observation of one actual provider SDK attempt."""

    schema_version: Literal["caribou.provider_call_receipt.v1"] = (
        "caribou.provider_call_receipt.v1"
    )
    call_id: NonEmptyStr
    run_id: RunId
    turn: StrictInt = Field(ge=1)
    agent_name: NonEmptyStr
    attempt: StrictInt = Field(ge=1)
    maximum_attempts: StrictInt = Field(ge=1)
    provider: NonEmptyStr
    requested_model: NonEmptyStr
    outcome: Literal["succeeded", "failed"]
    started_at: UtcDatetime
    ended_at: UtcDatetime
    duration_ms: StrictInt = Field(ge=0)
    response_id: Optional[NonEmptyStr] = None
    request_id: Optional[NonEmptyStr] = None
    response_model: Optional[NonEmptyStr] = None
    system_fingerprint: Optional[NonEmptyStr] = None
    finish_reason: Optional[NonEmptyStr] = None
    usage: ProviderCallUsage = Field(default_factory=ProviderCallUsage)
    failure_type: Optional[NonEmptyStr] = None
    http_status_code: Optional[StrictInt] = Field(default=None, gt=0)
    cost_usd: None = None
    cost_basis: Literal["unavailable"] = "unavailable"
    sdk_retries: Literal[0] = 0

    @model_validator(mode="after")
    def validate_attempt(self) -> "ProviderCallReceipt":
        expected_call_id = f"{self.run_id}:turn:{self.turn}:attempt:{self.attempt}"
        if self.call_id != expected_call_id:
            raise ValueError("provider call_id does not match its run and attempt")
        if self.attempt > self.maximum_attempts:
            raise ValueError("provider attempt exceeds maximum_attempts")
        if self.ended_at < self.started_at:
            raise ValueError("provider attempt ended before it started")
        usage_values = tuple(
            getattr(self.usage, name) for name in type(self.usage).model_fields
        )
        if self.outcome == "succeeded":
            if self.failure_type is not None or self.http_status_code is not None:
                raise ValueError(
                    "successful provider call cannot contain failure fields"
                )
        else:
            if self.failure_type is None:
                raise ValueError("failed provider call requires failure_type")
            if any(
                value is not None
                for value in (
                    self.response_id,
                    self.response_model,
                    self.system_fingerprint,
                    self.finish_reason,
                    *usage_values,
                )
            ):
                raise ValueError(
                    "failed provider call cannot contain response or usage fields"
                )
        return self


class ProviderCallReceiptV2(ProviderCallReceipt):
    """Provider receipt with provider-reported routing and cost accounting."""

    schema_version: Literal["caribou.provider_call_receipt.v2"] = (
        "caribou.provider_call_receipt.v2"
    )
    upstream_provider: Optional[NonEmptyStr] = None
    cost_usd: Optional[StrictFloat] = Field(default=None, ge=0, allow_inf_nan=False)
    upstream_cost_usd: Optional[StrictFloat] = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    cost_basis: Literal["provider_reported"] = "provider_reported"

    @model_validator(mode="after")
    def validate_v2_cost(self) -> "ProviderCallReceiptV2":
        if self.outcome == "failed" and (
            self.cost_usd is not None or self.upstream_cost_usd is not None
        ):
            raise ValueError("failed provider call cannot contain cost fields")
        return self


class ArtifactManifest(ControlRecord):
    schema_version: Literal["caribou.artifact_manifest.v1"] = (
        "caribou.artifact_manifest.v1"
    )
    run_id: RunId
    artifacts: Tuple[Artifact, ...] = ()
    updated_at: UtcDatetime = Field(default_factory=utc_now)

    def artifact(self, artifact_id: str) -> Optional[Artifact]:
        return next(
            (item for item in self.artifacts if item.artifact_id == artifact_id),
            None,
        )

    @model_validator(mode="after")
    def validate_unique_artifacts(self) -> "ArtifactManifest":
        identifiers = [artifact.artifact_id for artifact in self.artifacts]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("artifact manifest IDs must be unique")
        return self
