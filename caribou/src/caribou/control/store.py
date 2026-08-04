"""Filesystem-backed authoritative store for durable experiment lifecycles."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel

from caribou.config import CARIBOU_HOME, get_caribou_slurm_partition
from caribou.domain.enums import (
    ArtifactType,
    CheckpointComponent,
    EventType,
    ExecutorKind,
    ExperimentState,
    InterfaceOrigin,
    MemoryStrategy,
    RetentionPolicy,
    RunState,
)
from caribou.domain.ids import new_id
from caribou.domain.lifecycle import (
    create_resume_attempt,
    transition_experiment,
    transition_run,
)
from caribou.domain.models import (
    Artifact,
    ArtifactCreatedPayload,
    Checkpoint,
    CheckpointCreatedPayload,
    Event,
    EventPayload,
    Experiment,
    ExperimentSpec,
    HeartbeatPayload,
    Run,
    checkpoint_integrity_hash,
    utc_now,
)
from caribou.domain.serialization import (
    IntegrityError,
    canonical_json_bytes,
    commit_experiment_run_link,
    commit_experiment_transition,
    commit_run_checkpoint,
    commit_run_event,
    commit_run_scheduler_binding,
    commit_run_transition,
    file_hash,
    initialize_experiment_journal,
    initialize_run_journal,
    model_hash,
    read_experiment_journal,
    read_model,
    read_run_journal,
    sha256_bytes,
    verify_artifact,
    write_model,
)

from .api import ControlError, ExitCode
from .records import (
    ArtifactManifest,
    CancelRequest,
    CheckpointRequest,
    ExecutionHandle,
    IdempotencyClaim,
    SlurmAccounting,
    SlurmCancellationAttempt,
    SlurmCancellationLedger,
    SlurmExecutionHandle,
    SlurmSubmissionLedger,
    StoreIndex,
)
from .specs import (
    ADAPTER_PARAMETER,
    AGENT_PATH_SMOKE_ADAPTER,
    CARIBOU_AGENT_ADAPTER,
    build_local_plan,
    validate_control_spec,
)


TERMINAL_RUN_STATES = frozenset(
    {
        RunState.succeeded,
        RunState.failed,
        RunState.cancelled,
        RunState.rejected,
        RunState.resumable,
    }
)

SUPPORTED_RESUME_REQUIREMENTS = frozenset(
    {
        "dataset_binding=checkpoint",
        "memory_strategy=full",
        "runner_state_schema=caribou.agent_session_checkpoint_state.v1",
    }
)


def _validate_path_identifier(value: str, kind: str) -> str:
    if re.fullmatch(rf"{kind}_[0-9a-f]{{32}}", value) is None:
        raise ControlError(
            f"{kind.upper()}_ID_INVALID",
            f"{kind} identifier is not a canonical CARIBOU ID",
            exit_code=ExitCode.validation,
            details={"identifier": value},
        )
    return value


@dataclass(frozen=True)
class Submission:
    experiment: Experiment
    runs: tuple[Run, ...]
    plan: dict
    idempotent_replay: bool


@dataclass(frozen=True)
class ResumeSubmission:
    source: Run
    checkpoint: Checkpoint
    child: Run
    idempotent_replay: bool


def default_store_root() -> Path:
    configured = os.environ.get("CARIBOU_EXPERIMENT_HOME")
    return (
        Path(configured).expanduser()
        if configured
        else CARIBOU_HOME / "experiment_store" / "v1"
    )


class ExperimentStore:
    """One low-inode, crash-safe local control-plane store."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or default_store_root()).expanduser().resolve()

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def experiment_dir(self, experiment_id: str) -> Path:
        return (
            self.root / "experiments" / _validate_path_identifier(experiment_id, "exp")
        )

    def experiment_journal_path(self, experiment_id: str) -> Path:
        return self.experiment_dir(experiment_id) / "experiment-journal.json"

    def experiment_spec_path(self, experiment_id: str) -> Path:
        return self.experiment_dir(experiment_id) / "spec.json"

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / _validate_path_identifier(run_id, "run")

    def run_journal_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run-journal.json"

    def artifact_manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "artifacts.json"

    def execution_handle_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "executor.json"

    def cancel_request_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "control.json"

    def checkpoint_request_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "checkpoint-request.json"

    def scheduler_handle_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "scheduler.json"

    def scheduler_accounting_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "scheduler-accounting.json"

    def scheduler_accounting_raw_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "scheduler-accounting.raw"

    def scheduler_script_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "slurm-job.sh"

    def scheduler_cancellation_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "scheduler-cancellation.json"

    def scheduler_submission_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "scheduler-submission.json"

    @contextmanager
    def mutation_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.root / ".caribou-control.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read_index_unlocked(self) -> StoreIndex:
        if not self.index_path.exists():
            return StoreIndex()
        return self._read(self.index_path, StoreIndex)

    def index(self) -> StoreIndex:
        if not self.index_path.is_file():
            raise ControlError(
                "STORE_NOT_FOUND",
                "the experiment store has not been initialized",
                exit_code=ExitCode.not_found,
                details={"root": str(self.root)},
            )
        return self._read(self.index_path, StoreIndex)

    def _read(self, path: Path, model_type: type[BaseModel]):
        if not path.is_file():
            raise ControlError(
                "RECORD_NOT_FOUND",
                f"durable record does not exist: {path.name}",
                exit_code=ExitCode.not_found,
                details={"path": str(path)},
            )
        try:
            return read_model(path, model_type)
        except IntegrityError as exc:
            raise ControlError(
                "STORE_INTEGRITY_ERROR",
                "a durable control-plane record failed validation",
                exit_code=ExitCode.integrity,
                details={"path": str(path), "reason": str(exc)},
            ) from exc

    @staticmethod
    def _idempotency_hash(key: str) -> str:
        return sha256_bytes(key.encode("utf-8"))

    @staticmethod
    def _validated_run(run: Run, updates: dict) -> Run:
        candidate = run.model_copy(update=updates)
        return Run.model_validate_json(candidate.model_dump_json())

    def submit(
        self,
        spec: ExperimentSpec,
        idempotency_key: str,
        *,
        interface: InterfaceOrigin = InterfaceOrigin.cli,
    ) -> Submission:
        """Atomically claim an idempotency key and create the complete run matrix."""

        if not idempotency_key.strip():
            raise ControlError(
                "IDEMPOTENCY_KEY_REQUIRED",
                "submission requires a non-empty idempotency key",
                exit_code=ExitCode.validation,
            )
        if idempotency_key != idempotency_key.strip() or len(idempotency_key) > 256:
            raise ControlError(
                "IDEMPOTENCY_KEY_INVALID",
                "idempotency key must be trimmed and at most 256 characters",
                exit_code=ExitCode.validation,
            )
        validate_control_spec(spec, require_submit_adapter=True)
        plan = build_local_plan(spec)
        spec_hash = model_hash(spec)
        key_hash = self._idempotency_hash(idempotency_key)
        with self.mutation_lock():
            index = self._read_index_unlocked()
            existing = index.idempotency.get(key_hash)
            if existing is not None:
                if existing.spec_hash != spec_hash:
                    raise ControlError(
                        "IDEMPOTENCY_CONFLICT",
                        "the idempotency key is already bound to another specification",
                        exit_code=ExitCode.conflict,
                        details={
                            "experiment_id": existing.experiment_id,
                            "existing_spec_hash": existing.spec_hash,
                            "submitted_spec_hash": spec_hash,
                        },
                    )
                experiment = self.experiment(existing.experiment_id)
                existing_runs = tuple(self.run(run_id) for run_id in existing.run_ids)
                return Submission(experiment, existing_runs, plan, True)

            now = utc_now()
            experiment_id = new_id("exp")
            new_runs: list[Run] = []
            for condition in spec.conditions:
                for replicate_index in range(spec.repetitions):
                    run = Run(
                        experiment_id=experiment_id,
                        spec_hash=spec_hash,
                        condition_id=condition.condition_id,
                        replicate_index=replicate_index,
                        idempotency_key=(
                            f"{idempotency_key}:{condition.condition_id}:"
                            f"{replicate_index}:1"
                        ),
                        interface=interface,
                        owner=spec.owner,
                        initial_state=RunState.planned,
                        state=RunState.planned,
                        executor=spec.execution.executor,
                        code=spec.code,
                        resolved_model=condition.model,
                        resolved_blueprint=condition.blueprint,
                        resolved_prompt=condition.prompt,
                        resolved_memory=condition.memory,
                        resolved_inputs=list(spec.inputs),
                        resolved_stop_rules=spec.stop_rules,
                        resolved_budget=spec.budget,
                        container=spec.execution.container,
                        resources=spec.execution.resources,
                        partition=spec.execution.partition,
                        created_at=now,
                        updated_at=now,
                    )
                    new_runs.append(run)
            experiment = Experiment(
                experiment_id=experiment_id,
                spec_id=spec.spec_id,
                spec_version=spec.spec_version,
                spec_hash=spec_hash,
                owner=spec.owner,
                run_ids=[run.run_id for run in new_runs],
                created_at=now,
                updated_at=now,
            )

            experiment_directory = self.experiment_dir(experiment_id)
            experiment_directory.mkdir(parents=True, mode=0o700)
            write_model(self.experiment_spec_path(experiment_id), spec)
            experiment_hash = initialize_experiment_journal(
                self.experiment_journal_path(experiment_id), experiment
            )
            executor_label = spec.execution.executor.value
            for target, reason in (
                (ExperimentState.validated, "specification validated"),
                (ExperimentState.planned, "run matrix planned"),
                (ExperimentState.active, f"{executor_label} workers authorized"),
            ):
                experiment_transition = transition_experiment(
                    experiment, target, reason=reason, actor="control-plane"
                )
                experiment_hash = commit_experiment_transition(
                    self.experiment_journal_path(experiment_id),
                    experiment_transition,
                    expected_hash=experiment_hash,
                )
                experiment = experiment_transition.experiment

            queued_runs: list[Run] = []
            for run in new_runs:
                run_directory = self.run_dir(run.run_id)
                run_directory.mkdir(parents=True, mode=0o700)
                run_hash = initialize_run_journal(
                    self.run_journal_path(run.run_id), run
                )
                write_model(
                    self.artifact_manifest_path(run.run_id),
                    ArtifactManifest(run_id=run.run_id),
                )
                run_transition = transition_run(
                    run,
                    RunState.queued,
                    reason=f"accepted by {executor_label} executor",
                    actor="control-plane",
                )
                commit_run_transition(
                    self.run_journal_path(run.run_id),
                    run_transition,
                    expected_hash=run_hash,
                )
                queued_runs.append(run_transition.run)

            experiments = dict(index.experiments)
            experiments[experiment_id] = tuple(run.run_id for run in queued_runs)
            run_index = dict(index.runs)
            run_index.update({run.run_id: experiment_id for run in queued_runs})
            claims = dict(index.idempotency)
            claims[key_hash] = IdempotencyClaim(
                spec_hash=spec_hash,
                experiment_id=experiment_id,
                run_ids=tuple(run.run_id for run in queued_runs),
                created_at=now,
            )
            updated_index = StoreIndex(
                experiments=experiments,
                runs=run_index,
                idempotency=claims,
                updated_at=utc_now(),
            )
            write_model(
                self.index_path,
                updated_index,
                expected_hash=file_hash(self.index_path)
                if self.index_path.exists()
                else None,
            )
            return Submission(experiment, tuple(queued_runs), plan, False)

    def experiment(self, experiment_id: str) -> Experiment:
        try:
            return read_experiment_journal(
                self.experiment_journal_path(experiment_id)
            ).experiment
        except IntegrityError as exc:
            if not self.experiment_journal_path(experiment_id).exists():
                raise ControlError(
                    "EXPERIMENT_NOT_FOUND",
                    f"unknown experiment: {experiment_id}",
                    exit_code=ExitCode.not_found,
                ) from exc
            raise ControlError(
                "STORE_INTEGRITY_ERROR",
                "experiment journal failed validation",
                exit_code=ExitCode.integrity,
                details={"experiment_id": experiment_id, "reason": str(exc)},
            ) from exc

    def spec(self, experiment_id: str) -> ExperimentSpec:
        return self._read(self.experiment_spec_path(experiment_id), ExperimentSpec)

    def run(self, run_id: str) -> Run:
        try:
            return read_run_journal(self.run_journal_path(run_id)).run
        except IntegrityError as exc:
            if not self.run_journal_path(run_id).exists():
                raise ControlError(
                    "RUN_NOT_FOUND",
                    f"unknown run: {run_id}",
                    exit_code=ExitCode.not_found,
                ) from exc
            raise ControlError(
                "STORE_INTEGRITY_ERROR",
                "run journal failed validation",
                exit_code=ExitCode.integrity,
                details={"run_id": run_id, "reason": str(exc)},
            ) from exc

    def experiment_id_for_run(self, run_id: str) -> str:
        experiment_id = self.index().runs.get(run_id)
        if experiment_id is None:
            raise ControlError(
                "RUN_NOT_FOUND",
                f"unknown run: {run_id}",
                exit_code=ExitCode.not_found,
            )
        return experiment_id

    def events(
        self, run_id: str, *, after: int = 0, limit: int = 1000
    ) -> tuple[Event, ...]:
        if after < 0 or limit < 1 or limit > 10000:
            raise ControlError(
                "CURSOR_INVALID",
                "after must be nonnegative and limit must be between 1 and 10000",
                exit_code=ExitCode.validation,
            )
        self.run(run_id)
        journal = read_run_journal(self.run_journal_path(run_id))
        return tuple(event for event in journal.events if event.sequence > after)[
            :limit
        ]

    def append_run_event(
        self,
        run_id: str,
        *,
        event_type: EventType,
        payload: EventPayload,
        actor: str,
        turn: int,
        current_agent: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Event:
        """Append one typed durable runner event to the authoritative journal."""

        if event_type in {EventType.state_transition, EventType.token}:
            raise ControlError(
                "EVENT_TYPE_UNSUPPORTED",
                "runner events cannot append transitions or ephemeral tokens",
                exit_code=ExitCode.validation,
                details={"event_type": event_type.value},
            )
        with self.mutation_lock():
            path = self.run_journal_path(run_id)
            journal = read_run_journal(path)
            if journal.run.state not in {
                RunState.running,
                RunState.checkpointed,
                RunState.cancelling,
            }:
                raise ControlError(
                    "RUN_NOT_EVENT_WRITABLE",
                    "runner events require an active attempt",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": journal.run.state.value},
                )
            if turn < journal.run.current_turn:
                raise ControlError(
                    "EVENT_TURN_REGRESSION",
                    "runner event turn cannot move backward",
                    exit_code=ExitCode.integrity,
                    details={
                        "run_id": run_id,
                        "current_turn": journal.run.current_turn,
                        "event_turn": turn,
                    },
                )
            timestamp = utc_now()
            event = Event(
                experiment_id=journal.run.experiment_id,
                run_id=run_id,
                sequence=journal.run.event_sequence + 1,
                occurred_at=timestamp,
                event_type=event_type,
                turn=turn,
                stage=stage,
                actor=actor,
                payload=payload,
            )
            updates: dict = {
                "event_sequence": event.sequence,
                "current_turn": turn,
                "updated_at": timestamp,
            }
            if current_agent is not None:
                updates["current_agent"] = current_agent
            updated_run = self._validated_run(journal.run, updates)
            commit_run_event(
                path,
                updated_run,
                event,
                expected_hash=file_hash(path),
            )
            return event

    def _transition_run_unlocked(
        self,
        run_id: str,
        target: RunState,
        *,
        reason: str,
        actor: str,
        checkpoint: Optional[Checkpoint] = None,
        exit_code: Optional[int] = None,
    ) -> tuple[Run, bool]:
        path = self.run_journal_path(run_id)
        journal = read_run_journal(path)
        transition = transition_run(
            journal.run,
            target,
            reason=reason,
            actor=actor,
            checkpoint=checkpoint,
            exit_code=exit_code,
        )
        if not transition.applied:
            return transition.run, False
        commit_run_transition(path, transition, expected_hash=file_hash(path))
        return transition.run, True

    def transition_run(
        self,
        run_id: str,
        target: RunState,
        *,
        reason: str,
        actor: str,
        checkpoint: Optional[Checkpoint] = None,
        exit_code: Optional[int] = None,
    ) -> tuple[Run, bool]:
        with self.mutation_lock():
            return self._transition_run_unlocked(
                run_id,
                target,
                reason=reason,
                actor=actor,
                checkpoint=checkpoint,
                exit_code=exit_code,
            )

    def request_cancel(
        self, run_id: str, *, actor: str, reason: str
    ) -> tuple[Run, bool]:
        with self.mutation_lock():
            run = self.run(run_id)
            if run.state in TERMINAL_RUN_STATES:
                return run, False
            request_path = self.cancel_request_path(run_id)
            if not request_path.exists():
                write_model(
                    request_path,
                    CancelRequest(run_id=run_id, actor=actor, reason=reason),
                )
            if run.state == RunState.cancelling:
                return run, False
            target = (
                RunState.cancelled
                if run.state in {RunState.draft, RunState.validated, RunState.planned}
                else RunState.cancelling
            )
            return self._transition_run_unlocked(
                run_id, target, reason=reason, actor=actor
            )

    def cancel_requested(self, run_id: str) -> bool:
        return self.cancel_request_path(run_id).is_file()

    def checkpoint_request(self, run_id: str) -> Optional[CheckpointRequest]:
        path = self.checkpoint_request_path(run_id)
        return self._read(path, CheckpointRequest) if path.is_file() else None

    @staticmethod
    def _require_supported_checkpoint_memory(run: Run) -> None:
        if run.resolved_memory.strategy != MemoryStrategy.full:
            raise ControlError(
                "CHECKPOINT_MEMORY_UNSUPPORTED",
                "checkpoint recovery currently supports only full-history memory",
                exit_code=ExitCode.conflict,
                details={"memory_strategy": run.resolved_memory.strategy.value},
            )

    def _require_supported_checkpoint_adapter(self, run: Run) -> None:
        condition = next(
            item
            for item in self.spec(run.experiment_id).conditions
            if item.condition_id == run.condition_id
        )
        adapter = condition.parameters.get(ADAPTER_PARAMETER)
        if adapter not in {AGENT_PATH_SMOKE_ADAPTER, CARIBOU_AGENT_ADAPTER}:
            raise ControlError(
                "CHECKPOINT_ADAPTER_UNSUPPORTED",
                "checkpoint recovery is available only for agent workloads",
                exit_code=ExitCode.conflict,
                details={"run_id": run.run_id, "adapter": adapter},
            )

    def request_checkpoint(
        self,
        run_id: str,
        *,
        idempotency_key: str,
        actor: str,
        reason: str,
    ) -> tuple[Run, CheckpointRequest, bool]:
        """Request one cooperative stop at the next completed-turn boundary."""

        if (
            not idempotency_key.strip()
            or idempotency_key != idempotency_key.strip()
            or len(idempotency_key) > 256
        ):
            raise ControlError(
                "IDEMPOTENCY_KEY_INVALID",
                "checkpoint idempotency key must be trimmed and at most 256 characters",
                exit_code=ExitCode.validation,
            )
        key_hash = self._idempotency_hash(f"checkpoint:v1:{idempotency_key}")
        with self.mutation_lock():
            run = self.run(run_id)
            self._require_supported_checkpoint_memory(run)
            self._require_supported_checkpoint_adapter(run)
            existing = self.checkpoint_request(run_id)
            if existing is not None:
                if existing.idempotency_key_hash != key_hash:
                    raise ControlError(
                        "CHECKPOINT_REQUEST_CONFLICT",
                        "the run already has another checkpoint request",
                        exit_code=ExitCode.conflict,
                        details={"run_id": run_id},
                    )
                return run, existing, False
            if run.state in TERMINAL_RUN_STATES:
                raise ControlError(
                    "RUN_TERMINAL",
                    "a terminal run cannot accept a new checkpoint request",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": run.state.value},
                )
            request = CheckpointRequest(
                run_id=run_id,
                idempotency_key_hash=key_hash,
                actor=actor,
                reason=reason,
            )
            write_model(self.checkpoint_request_path(run_id), request)
            return run, request, True

    def checkpoint_requested(self, run_id: str) -> bool:
        return self.checkpoint_request_path(run_id).is_file()

    def checkpoints(self, run_id: str) -> tuple[Checkpoint, ...]:
        self.run(run_id)
        return tuple(read_run_journal(self.run_journal_path(run_id)).checkpoints)

    def checkpoint(self, run_id: str, checkpoint_id: str) -> Checkpoint:
        checkpoint = next(
            (
                item
                for item in self.checkpoints(run_id)
                if item.checkpoint_id == checkpoint_id
            ),
            None,
        )
        if checkpoint is None:
            raise ControlError(
                "CHECKPOINT_NOT_FOUND",
                f"checkpoint {checkpoint_id} is not linked to run {run_id}",
                exit_code=ExitCode.not_found,
            )
        return checkpoint

    @staticmethod
    def _checkpoint_id(run_id: str, request: CheckpointRequest) -> str:
        digest = sha256_bytes(
            f"checkpoint:v1:{run_id}:{request.idempotency_key_hash}".encode("utf-8")
        )
        return f"chk_{digest.removeprefix('sha256:')[:32]}"

    def record_checkpoint(
        self,
        run_id: str,
        *,
        stage: str,
        turn: int,
        current_agent: str,
        dataset_artifact_id: str,
        message_history_artifact_id: str,
        agent_state_artifact_id: str,
        executed_actions_artifact_id: str,
        artifact_manifest_id: str,
        resume_requirements: list[str],
        actor: str,
    ) -> Checkpoint:
        """Atomically publish one complete checkpoint envelope and event."""

        if (
            len(resume_requirements) != len(set(resume_requirements))
            or frozenset(resume_requirements) != SUPPORTED_RESUME_REQUIREMENTS
        ):
            raise ControlError(
                "CHECKPOINT_REQUIREMENTS_UNSUPPORTED",
                "checkpoint resume requirements do not match the supported slice",
                exit_code=ExitCode.validation,
                details={"requirements": resume_requirements},
            )
        with self.mutation_lock():
            request = self.checkpoint_request(run_id)
            if request is None:
                raise ControlError(
                    "CHECKPOINT_NOT_REQUESTED",
                    "a checkpoint cannot be published without a durable request",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id},
                )
            checkpoint_id = self._checkpoint_id(run_id, request)
            journal_path = self.run_journal_path(run_id)
            journal = read_run_journal(journal_path)
            self._require_supported_checkpoint_memory(journal.run)
            self._require_supported_checkpoint_adapter(journal.run)
            existing = next(
                (
                    item
                    for item in journal.checkpoints
                    if item.checkpoint_id == checkpoint_id
                ),
                None,
            )
            if existing is not None:
                expected_ids = (
                    existing.dataset_artifact_id,
                    existing.message_history_artifact_id,
                    existing.agent_state_artifact_id,
                    existing.executed_actions_artifact_id,
                    existing.artifact_manifest_id,
                )
                supplied_ids = (
                    dataset_artifact_id,
                    message_history_artifact_id,
                    agent_state_artifact_id,
                    executed_actions_artifact_id,
                    artifact_manifest_id,
                )
                if expected_ids != supplied_ids:
                    raise ControlError(
                        "CHECKPOINT_REPLAY_CONFLICT",
                        "checkpoint replay supplied different component artifacts",
                        exit_code=ExitCode.integrity,
                        details={"checkpoint_id": checkpoint_id},
                    )
                if journal.run.state == RunState.running:
                    self._transition_run_unlocked(
                        run_id,
                        RunState.checkpointed,
                        reason=request.reason,
                        actor=actor,
                        checkpoint=existing,
                    )
                return existing
            if journal.checkpoints:
                raise ControlError(
                    "CHECKPOINT_ALREADY_EXISTS",
                    "the first checkpoint slice supports one checkpoint per attempt",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id},
                )
            if journal.run.state != RunState.running:
                raise ControlError(
                    "RUN_NOT_CHECKPOINTABLE",
                    "checkpoint finalization requires a running attempt",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": journal.run.state.value},
                )
            if (
                turn != journal.run.current_turn
                or current_agent != journal.run.current_agent
            ):
                raise ControlError(
                    "CHECKPOINT_CURSOR_MISMATCH",
                    "checkpoint state differs from the durable completed-turn cursor",
                    exit_code=ExitCode.integrity,
                    details={
                        "run_id": run_id,
                        "checkpoint_turn": turn,
                        "run_turn": journal.run.current_turn,
                    },
                )
            component_ids = (
                dataset_artifact_id,
                message_history_artifact_id,
                agent_state_artifact_id,
                executed_actions_artifact_id,
                artifact_manifest_id,
            )
            if len(set(component_ids)) != len(component_ids):
                raise ControlError(
                    "CHECKPOINT_COMPONENT_DUPLICATED",
                    "checkpoint components must reference distinct artifacts",
                    exit_code=ExitCode.integrity,
                )
            manifest = self.artifact_manifest(run_id)
            artifacts = {
                artifact.artifact_id: artifact for artifact in manifest.artifacts
            }
            if any(identifier not in artifacts for identifier in component_ids):
                raise ControlError(
                    "CHECKPOINT_COMPONENT_MISSING",
                    "checkpoint references an artifact absent from the run manifest",
                    exit_code=ExitCode.integrity,
                    details={"run_id": run_id},
                )
            expected_roles = (
                "checkpoint_dataset_state",
                "checkpoint_message_history",
                "checkpoint_agent_state",
                "checkpoint_executed_actions",
                "checkpoint_artifact_manifest",
            )
            observed_roles = tuple(
                artifacts[identifier].role for identifier in component_ids
            )
            if observed_roles != expected_roles:
                raise ControlError(
                    "CHECKPOINT_COMPONENT_ROLE_INVALID",
                    "checkpoint components do not match their declared semantic roles",
                    exit_code=ExitCode.integrity,
                    details={
                        "expected_roles": list(expected_roles),
                        "observed_roles": list(observed_roles),
                    },
                )
            self.verify_artifacts(run_id)

            timestamp = utc_now()
            event = Event(
                experiment_id=journal.run.experiment_id,
                run_id=run_id,
                sequence=journal.run.event_sequence + 1,
                occurred_at=timestamp,
                event_type=EventType.checkpoint_created,
                turn=turn,
                actor=actor,
                payload=CheckpointCreatedPayload(checkpoint_id=checkpoint_id),
            )
            provisional = Checkpoint(
                checkpoint_id=checkpoint_id,
                experiment_id=journal.run.experiment_id,
                run_id=run_id,
                event_id=event.event_id,
                event_sequence=event.sequence,
                stage=stage,
                turn=turn,
                created_at=timestamp,
                components=[
                    CheckpointComponent.dataset_state,
                    CheckpointComponent.message_history,
                    CheckpointComponent.agent_state,
                    CheckpointComponent.executed_actions,
                    CheckpointComponent.artifact_manifest,
                ],
                dataset_artifact_id=dataset_artifact_id,
                message_history_artifact_id=message_history_artifact_id,
                agent_state_artifact_id=agent_state_artifact_id,
                executed_actions_artifact_id=executed_actions_artifact_id,
                artifact_manifest_id=artifact_manifest_id,
                spec_hash=journal.run.spec_hash,
                code_commit=journal.run.code.commit,
                container_digest=journal.run.container.image.content_hash,
                model_identity=(
                    f"{journal.run.resolved_model.provider}:"
                    f"{journal.run.resolved_model.model}"
                ),
                integrity_hash="sha256:" + "0" * 64,
                resume_requirements=resume_requirements,
            )
            checkpoint = Checkpoint.model_validate_json(
                provisional.model_copy(
                    update={"integrity_hash": checkpoint_integrity_hash(provisional)}
                ).model_dump_json()
            )
            updated_run = self._validated_run(
                journal.run,
                {
                    "checkpoint_ids": [*journal.run.checkpoint_ids, checkpoint_id],
                    "event_sequence": event.sequence,
                    "current_turn": turn,
                    "current_agent": current_agent,
                    "updated_at": timestamp,
                },
            )
            commit_run_checkpoint(
                journal_path,
                updated_run,
                checkpoint,
                event,
                expected_hash=file_hash(journal_path),
            )
            self._transition_run_unlocked(
                run_id,
                RunState.checkpointed,
                reason=request.reason,
                actor=actor,
                checkpoint=checkpoint,
            )
            return checkpoint

    @staticmethod
    def _resume_run_id(idempotency_key: str) -> str:
        digest = sha256_bytes(
            f"resume:v1:{idempotency_key}".encode("utf-8")
        ).removeprefix("sha256:")
        return f"run_{digest[:32]}"

    @staticmethod
    def _validate_resume_child(
        source: Run,
        checkpoint: Checkpoint,
        child: Run,
        *,
        idempotency_key: str,
        interface: InterfaceOrigin,
    ) -> None:
        expected = create_resume_attempt(
            source,
            checkpoint=checkpoint,
            idempotency_key=idempotency_key,
            run_id=child.run_id,
            interface=interface,
            at=child.created_at,
        )
        mutable_fields = {
            "state",
            "scheduler_job_id",
            "current_turn",
            "current_agent",
            "event_sequence",
            "artifact_ids",
            "metric_record_ids",
            "failure_ids",
            "checkpoint_ids",
            "budget_record_ids",
            "updated_at",
            "queued_at",
            "started_at",
            "ended_at",
            "terminal_outcome",
            "end_reason",
            "exit_code",
            "resume_eligible",
        }
        observed_values = child.model_dump(mode="python")
        expected_values = expected.model_dump(mode="python")
        for field in mutable_fields:
            observed_values.pop(field, None)
            expected_values.pop(field, None)
        if observed_values != expected_values:
            raise ControlError(
                "RESUME_CHILD_INCOMPATIBLE",
                "the durable child differs from its frozen source checkpoint",
                exit_code=ExitCode.integrity,
                details={"child_run_id": child.run_id},
            )

    def resume(
        self,
        source_run_id: str,
        *,
        checkpoint_id: str,
        idempotency_key: str,
        interface: InterfaceOrigin = InterfaceOrigin.cli,
    ) -> ResumeSubmission:
        """Create or repair one idempotent child attempt from a checkpoint."""

        if (
            not idempotency_key.strip()
            or idempotency_key != idempotency_key.strip()
            or len(idempotency_key) > 256
        ):
            raise ControlError(
                "IDEMPOTENCY_KEY_INVALID",
                "resume idempotency key must be trimmed and at most 256 characters",
                exit_code=ExitCode.validation,
            )
        child_id = self._resume_run_id(idempotency_key)
        with self.mutation_lock():
            source = self.run(source_run_id)
            self._require_supported_checkpoint_memory(source)
            self._require_supported_checkpoint_adapter(source)
            checkpoint = self.checkpoint(source_run_id, checkpoint_id)
            if source.state != RunState.resumable or not source.resume_eligible:
                raise ControlError(
                    "RUN_NOT_RESUMABLE",
                    "only a terminal resumable attempt can create a child",
                    exit_code=ExitCode.conflict,
                    details={"run_id": source_run_id, "state": source.state.value},
                )
            if checkpoint_id not in source.checkpoint_ids:
                raise ControlError(
                    "CHECKPOINT_NOT_ATTACHED",
                    "the selected checkpoint is not attached to the source attempt",
                    exit_code=ExitCode.integrity,
                )
            if (
                frozenset(checkpoint.resume_requirements)
                != SUPPORTED_RESUME_REQUIREMENTS
            ):
                raise ControlError(
                    "CHECKPOINT_REQUIREMENTS_UNSUPPORTED",
                    "the checkpoint requires an unsupported restore capability",
                    exit_code=ExitCode.conflict,
                    details={"requirements": checkpoint.resume_requirements},
                )
            self.verify_artifacts(source_run_id)

            experiment_path = self.experiment_journal_path(source.experiment_id)
            experiment_journal = read_experiment_journal(experiment_path)
            existing_children = []
            for run_id in experiment_journal.experiment.run_ids:
                candidate = self.run(run_id)
                if (
                    candidate.resumed_from_run_id == source_run_id
                    and candidate.resume_checkpoint_id == checkpoint_id
                ):
                    existing_children.append(candidate)
            if len(existing_children) > 1:
                raise ControlError(
                    "CHECKPOINT_RESUME_AMBIGUOUS",
                    "multiple children consume one non-branching checkpoint",
                    exit_code=ExitCode.integrity,
                    details={"checkpoint_id": checkpoint_id},
                )
            if existing_children:
                existing_child = existing_children[0]
                if (
                    existing_child.run_id != child_id
                    or existing_child.idempotency_key != idempotency_key
                ):
                    raise ControlError(
                        "CHECKPOINT_ALREADY_RESUMED",
                        "the checkpoint is already bound to another resume request",
                        exit_code=ExitCode.conflict,
                        details={"child_run_id": existing_child.run_id},
                    )
                child = existing_child
                replay = True
            elif self.run_journal_path(child_id).exists():
                child = self.run(child_id)
                if (
                    child.resumed_from_run_id != source_run_id
                    or child.resume_checkpoint_id != checkpoint_id
                    or child.idempotency_key != idempotency_key
                ):
                    raise ControlError(
                        "RESUME_IDEMPOTENCY_CONFLICT",
                        "the resume key is already bound to different lineage",
                        exit_code=ExitCode.conflict,
                        details={"child_run_id": child_id},
                    )
                replay = True
            else:
                child = create_resume_attempt(
                    source,
                    checkpoint=checkpoint,
                    idempotency_key=idempotency_key,
                    run_id=child_id,
                    interface=interface,
                )
                child_directory = self.run_dir(child.run_id)
                child_directory.mkdir(parents=True, mode=0o700)
                initialize_run_journal(self.run_journal_path(child.run_id), child)
                write_model(
                    self.artifact_manifest_path(child.run_id),
                    ArtifactManifest(run_id=child.run_id),
                )
                replay = False

            self._validate_resume_child(
                source,
                checkpoint,
                child,
                idempotency_key=idempotency_key,
                # An idempotent replay reports the immutable first request.  A
                # later client may reach it through another adapter, but that
                # must not rewrite or invalidate the persisted origin.
                interface=child.interface if replay else interface,
            )

            experiment_journal = read_experiment_journal(experiment_path)
            if child.run_id not in experiment_journal.experiment.run_ids:
                linked_experiment = Experiment.model_validate_json(
                    experiment_journal.experiment.model_copy(
                        update={
                            "run_ids": [
                                *experiment_journal.experiment.run_ids,
                                child.run_id,
                            ],
                            "updated_at": utc_now(),
                        }
                    ).model_dump_json()
                )
                commit_experiment_run_link(
                    experiment_path,
                    linked_experiment,
                    child,
                    expected_hash=file_hash(experiment_path),
                )

            index = self._read_index_unlocked()
            indexed_experiment = index.runs.get(child.run_id)
            if indexed_experiment not in {None, source.experiment_id}:
                raise ControlError(
                    "RESUME_INDEX_CONFLICT",
                    "the child run ID is indexed to another experiment",
                    exit_code=ExitCode.integrity,
                )
            experiment_runs = list(index.experiments.get(source.experiment_id, ()))
            if child.run_id not in experiment_runs:
                experiment_runs.append(child.run_id)
            experiments = dict(index.experiments)
            experiments[source.experiment_id] = tuple(experiment_runs)
            runs = dict(index.runs)
            runs[child.run_id] = source.experiment_id
            if (
                index.experiments.get(source.experiment_id) != tuple(experiment_runs)
                or indexed_experiment is None
            ):
                write_model(
                    self.index_path,
                    StoreIndex(
                        experiments=experiments,
                        runs=runs,
                        idempotency=index.idempotency,
                        updated_at=utc_now(),
                    ),
                    expected_hash=file_hash(self.index_path),
                )

            child = self.run(child.run_id)
            if child.state == RunState.planned:
                child, _ = self._transition_run_unlocked(
                    child.run_id,
                    RunState.queued,
                    reason=f"resume accepted by {child.executor.value} executor",
                    actor="control-plane",
                )
            return ResumeSubmission(
                source=source,
                checkpoint=checkpoint,
                child=child,
                idempotent_replay=replay,
            )

    def artifact_manifest(self, run_id: str) -> ArtifactManifest:
        return self._read(self.artifact_manifest_path(run_id), ArtifactManifest)

    @staticmethod
    def _atomic_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def record_json_artifact(
        self,
        run_id: str,
        *,
        filename: str,
        role: str,
        value: dict,
        producer: str,
        artifact_type: ArtifactType = ArtifactType.manifest,
        media_type: str = "application/json",
        schema_type: Optional[str] = "caribou.lifecycle_smoke_result",
        schema_version_name: Optional[str] = "v1",
        turn: Optional[int] = None,
        current_agent: Optional[str] = None,
    ) -> Artifact:
        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise ControlError(
                "ARTIFACT_FILENAME_INVALID",
                "artifact filename must be one path-safe component",
                exit_code=ExitCode.validation,
                details={"filename": filename},
            )
        with self.mutation_lock():
            journal_path = self.run_journal_path(run_id)
            journal = read_run_journal(journal_path)
            if journal.run.state not in {
                RunState.running,
                RunState.checkpointed,
                RunState.cancelling,
            }:
                raise ControlError(
                    "RUN_NOT_WRITABLE",
                    "artifacts can be registered only while an attempt is running",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": journal.run.state.value},
                )
            data = (
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            return self._record_artifact_bytes_unlocked(
                journal_path=journal_path,
                journal_run=journal.run,
                filename=filename,
                role=role,
                data=data,
                producer=producer,
                artifact_type=artifact_type,
                media_type=media_type,
                schema_type=schema_type,
                schema_version_name=schema_version_name,
                turn=turn,
                current_agent=current_agent,
            )

    def record_idempotent_json_artifact(
        self,
        run_id: str,
        *,
        filename: str,
        role: str,
        value: dict,
        producer: str,
        artifact_type: ArtifactType = ArtifactType.manifest,
        media_type: str = "application/json",
        schema_type: Optional[str] = None,
        schema_version_name: Optional[str] = None,
        turn: Optional[int] = None,
        current_agent: Optional[str] = None,
    ) -> Artifact:
        """Record one named JSON artifact exactly once within a live worker.

        The ``(role, filename)`` pair is the idempotency identity. A replay must
        provide byte-identical content and descriptor metadata. If the manifest
        write succeeded but the journal update raised an ordinary exception,
        this method repairs that boundary before returning. A hard process death
        still requires a later recovery facility.
        """

        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise ControlError(
                "ARTIFACT_FILENAME_INVALID",
                "artifact filename must be one path-safe component",
                exit_code=ExitCode.validation,
                details={"filename": filename},
            )
        data = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with self.mutation_lock():
            journal_path = self.run_journal_path(run_id)
            journal = read_run_journal(journal_path)
            if journal.run.state not in {
                RunState.running,
                RunState.checkpointed,
                RunState.cancelling,
            }:
                raise ControlError(
                    "RUN_NOT_WRITABLE",
                    "artifacts can be registered only while an attempt is running",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": journal.run.state.value},
                )
            existing = self._repair_named_artifact_unlocked(
                journal_path=journal_path,
                journal_run=journal.run,
                filename=filename,
                role=role,
                data=data,
                producer=producer,
                artifact_type=artifact_type,
                media_type=media_type,
                schema_type=schema_type,
                schema_version_name=schema_version_name,
                turn=turn,
                current_agent=current_agent,
            )
            if existing is not None:
                return existing
            try:
                return self._record_artifact_bytes_unlocked(
                    journal_path=journal_path,
                    journal_run=journal.run,
                    filename=filename,
                    role=role,
                    data=data,
                    producer=producer,
                    artifact_type=artifact_type,
                    media_type=media_type,
                    schema_type=schema_type,
                    schema_version_name=schema_version_name,
                    turn=turn,
                    current_agent=current_agent,
                )
            except Exception:
                # The shared artifact path commits the manifest before the run
                # journal. Repair that one recoverable boundary immediately so
                # the paid call is neither duplicated nor left half-linked.
                current = read_run_journal(journal_path)
                recovered = self._repair_named_artifact_unlocked(
                    journal_path=journal_path,
                    journal_run=current.run,
                    filename=filename,
                    role=role,
                    data=data,
                    producer=producer,
                    artifact_type=artifact_type,
                    media_type=media_type,
                    schema_type=schema_type,
                    schema_version_name=schema_version_name,
                    turn=turn,
                    current_agent=current_agent,
                )
                if recovered is None:
                    raise
                return recovered

    def record_text_artifact(
        self,
        run_id: str,
        *,
        filename: str,
        role: str,
        text: str,
        producer: str,
        artifact_type: ArtifactType,
        media_type: str = "text/plain",
        turn: Optional[int] = None,
        current_agent: Optional[str] = None,
    ) -> Artifact:
        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise ControlError(
                "ARTIFACT_FILENAME_INVALID",
                "artifact filename must be one path-safe component",
                exit_code=ExitCode.validation,
                details={"filename": filename},
            )
        with self.mutation_lock():
            journal_path = self.run_journal_path(run_id)
            journal = read_run_journal(journal_path)
            if journal.run.state not in {
                RunState.running,
                RunState.checkpointed,
                RunState.cancelling,
            }:
                raise ControlError(
                    "RUN_NOT_WRITABLE",
                    "artifacts can be registered only while an attempt is running",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": journal.run.state.value},
                )
            return self._record_artifact_bytes_unlocked(
                journal_path=journal_path,
                journal_run=journal.run,
                filename=filename,
                role=role,
                data=text.encode("utf-8"),
                producer=producer,
                artifact_type=artifact_type,
                media_type=media_type,
                schema_type=None,
                schema_version_name=None,
                turn=turn,
                current_agent=current_agent,
            )

    def record_file_artifact(
        self,
        run_id: str,
        *,
        source: Path,
        filename: str,
        role: str,
        producer: str,
        artifact_type: ArtifactType,
        media_type: str,
        turn: Optional[int] = None,
        current_agent: Optional[str] = None,
    ) -> Artifact:
        """Copy one regular generated file into immutable run artifact storage."""

        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise ControlError(
                "ARTIFACT_FILENAME_INVALID",
                "artifact filename must be one path-safe component",
                exit_code=ExitCode.validation,
                details={"filename": filename},
            )
        candidate = source.expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise ControlError(
                "ARTIFACT_SOURCE_INVALID",
                "artifact source must be a regular non-symlink file",
                exit_code=ExitCode.integrity,
                details={"source": str(candidate)},
            )
        with self.mutation_lock():
            journal_path = self.run_journal_path(run_id)
            journal = read_run_journal(journal_path)
            if journal.run.state not in {
                RunState.running,
                RunState.checkpointed,
                RunState.cancelling,
            }:
                raise ControlError(
                    "RUN_NOT_WRITABLE",
                    "artifacts can be registered only while an attempt is running",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": journal.run.state.value},
                )
            artifact_id = new_id("art")
            storage_uri = f"artifacts/{artifact_id}-{filename}"
            destination = self.run_dir(run_id) / storage_uri
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            try:
                with (
                    os.fdopen(descriptor, "wb") as output,
                    candidate.open("rb") as input_file,
                ):
                    shutil.copyfileobj(input_file, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, destination)
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            return self._commit_artifact_unlocked(
                journal_path=journal_path,
                journal_run=journal.run,
                artifact_id=artifact_id,
                filename=filename,
                storage_uri=storage_uri,
                role=role,
                producer=producer,
                artifact_type=artifact_type,
                media_type=media_type,
                schema_type=None,
                schema_version_name=None,
                content_hash=file_hash(destination),
                size_bytes=destination.stat().st_size,
                turn=turn,
                current_agent=current_agent,
            )

    def record_idempotent_file_artifact(
        self,
        run_id: str,
        *,
        source: Path,
        filename: str,
        role: str,
        producer: str,
        artifact_type: ArtifactType,
        media_type: str,
        schema_type: Optional[str] = None,
        schema_version_name: Optional[str] = None,
        turn: Optional[int] = None,
        current_agent: Optional[str] = None,
    ) -> Artifact:
        """Copy or repair one immutable named file artifact exactly once."""

        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise ControlError(
                "ARTIFACT_FILENAME_INVALID",
                "artifact filename must be one path-safe component",
                exit_code=ExitCode.validation,
                details={"filename": filename},
            )
        candidate = source.expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise ControlError(
                "ARTIFACT_SOURCE_INVALID",
                "artifact source must be a regular non-symlink file",
                exit_code=ExitCode.integrity,
                details={"source": str(candidate)},
            )
        expected_hash = file_hash(candidate)
        expected_size = candidate.stat().st_size
        with self.mutation_lock():
            journal_path = self.run_journal_path(run_id)
            journal = read_run_journal(journal_path)
            if journal.run.state not in {
                RunState.running,
                RunState.checkpointed,
                RunState.cancelling,
            }:
                raise ControlError(
                    "RUN_NOT_WRITABLE",
                    "artifacts can be registered only while an attempt is running",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": journal.run.state.value},
                )
            existing = self._repair_named_artifact_unlocked(
                journal_path=journal_path,
                journal_run=journal.run,
                filename=filename,
                role=role,
                content_hash=expected_hash,
                size_bytes=expected_size,
                producer=producer,
                artifact_type=artifact_type,
                media_type=media_type,
                schema_type=schema_type,
                schema_version_name=schema_version_name,
                turn=turn,
                current_agent=current_agent,
            )
            if existing is not None:
                return existing
            artifact_id = new_id("art")
            storage_uri = f"artifacts/{artifact_id}-{filename}"
            destination = self.run_dir(run_id) / storage_uri
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            try:
                with (
                    os.fdopen(descriptor, "wb") as output,
                    candidate.open("rb") as input_file,
                ):
                    shutil.copyfileobj(input_file, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if (
                    file_hash(temporary) != expected_hash
                    or temporary.stat().st_size != expected_size
                ):
                    raise ControlError(
                        "ARTIFACT_SOURCE_CHANGED",
                        "artifact source changed while it was copied",
                        exit_code=ExitCode.integrity,
                        details={"source": str(candidate)},
                    )
                os.replace(temporary, destination)
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            try:
                return self._commit_artifact_unlocked(
                    journal_path=journal_path,
                    journal_run=journal.run,
                    artifact_id=artifact_id,
                    filename=filename,
                    storage_uri=storage_uri,
                    role=role,
                    producer=producer,
                    artifact_type=artifact_type,
                    media_type=media_type,
                    schema_type=schema_type,
                    schema_version_name=schema_version_name,
                    content_hash=expected_hash,
                    size_bytes=expected_size,
                    turn=turn,
                    current_agent=current_agent,
                )
            except Exception:
                current = read_run_journal(journal_path)
                recovered = self._repair_named_artifact_unlocked(
                    journal_path=journal_path,
                    journal_run=current.run,
                    filename=filename,
                    role=role,
                    content_hash=expected_hash,
                    size_bytes=expected_size,
                    producer=producer,
                    artifact_type=artifact_type,
                    media_type=media_type,
                    schema_type=schema_type,
                    schema_version_name=schema_version_name,
                    turn=turn,
                    current_agent=current_agent,
                )
                if recovered is None:
                    raise
                return recovered

    def _record_artifact_bytes_unlocked(
        self,
        *,
        journal_path: Path,
        journal_run: Run,
        filename: str,
        role: str,
        data: bytes,
        producer: str,
        artifact_type: ArtifactType,
        media_type: str,
        schema_type: Optional[str],
        schema_version_name: Optional[str],
        turn: Optional[int],
        current_agent: Optional[str],
    ) -> Artifact:
        artifact_id = new_id("art")
        storage_uri = f"artifacts/{artifact_id}-{filename}"
        self._atomic_bytes(self.run_dir(journal_run.run_id) / storage_uri, data)
        return self._commit_artifact_unlocked(
            journal_path=journal_path,
            journal_run=journal_run,
            artifact_id=artifact_id,
            filename=filename,
            storage_uri=storage_uri,
            role=role,
            producer=producer,
            artifact_type=artifact_type,
            media_type=media_type,
            schema_type=schema_type,
            schema_version_name=schema_version_name,
            content_hash=sha256_bytes(data),
            size_bytes=len(data),
            turn=turn,
            current_agent=current_agent,
        )

    def _repair_named_artifact_unlocked(
        self,
        *,
        journal_path: Path,
        journal_run: Run,
        filename: str,
        role: str,
        data: Optional[bytes] = None,
        content_hash: Optional[str] = None,
        size_bytes: Optional[int] = None,
        producer: str,
        artifact_type: ArtifactType,
        media_type: str,
        schema_type: Optional[str],
        schema_version_name: Optional[str],
        turn: Optional[int],
        current_agent: Optional[str],
    ) -> Optional[Artifact]:
        if data is not None:
            expected_hash = sha256_bytes(data)
            expected_size = len(data)
            if content_hash is not None and content_hash != expected_hash:
                raise ValueError("artifact data and supplied hash disagree")
            if size_bytes is not None and size_bytes != expected_size:
                raise ValueError("artifact data and supplied size disagree")
            content_hash = expected_hash
            size_bytes = expected_size
        if content_hash is None or size_bytes is None:
            raise ValueError("idempotent artifact repair requires hash and size")
        matches = tuple(
            artifact
            for artifact in self.artifact_manifest(journal_run.run_id).artifacts
            if artifact.role == role and artifact.filename == filename
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise ControlError(
                "IDEMPOTENT_ARTIFACT_DUPLICATED",
                "an idempotent artifact identity has multiple manifest entries",
                exit_code=ExitCode.integrity,
                details={
                    "run_id": journal_run.run_id,
                    "role": role,
                    "filename": filename,
                },
            )
        existing = matches[0]
        if (
            existing.producer != producer
            or existing.artifact_type != artifact_type
            or existing.media_type != media_type
            or existing.schema_type != schema_type
            or existing.schema_version_name != schema_version_name
            or existing.content_hash != content_hash
            or existing.size_bytes != size_bytes
        ):
            raise ControlError(
                "IDEMPOTENT_ARTIFACT_CONFLICT",
                "an idempotent artifact replay differs from its immutable record",
                exit_code=ExitCode.integrity,
                details={
                    "run_id": journal_run.run_id,
                    "role": role,
                    "filename": filename,
                },
            )
        try:
            verify_artifact(
                self.artifact_path(existing),
                existing.content_hash,
                existing.size_bytes,
                root=self.run_dir(journal_run.run_id),
            )
        except IntegrityError as exc:
            raise ControlError(
                "IDEMPOTENT_ARTIFACT_TAMPERED",
                "an idempotent artifact failed content verification",
                exit_code=ExitCode.integrity,
                details={
                    "run_id": journal_run.run_id,
                    "role": role,
                    "filename": filename,
                },
            ) from exc
        if existing.artifact_id in journal_run.artifact_ids:
            return existing

        event_turn = journal_run.current_turn if turn is None else turn
        if event_turn < journal_run.current_turn:
            raise ControlError(
                "IDEMPOTENT_ARTIFACT_REPAIR_AMBIGUOUS",
                "artifact repair cannot insert an event before the durable cursor",
                exit_code=ExitCode.integrity,
                details={
                    "run_id": journal_run.run_id,
                    "current_turn": journal_run.current_turn,
                    "artifact_turn": event_turn,
                },
            )
        event = Event(
            event_id=existing.producer_event_id,
            experiment_id=journal_run.experiment_id,
            run_id=journal_run.run_id,
            sequence=journal_run.event_sequence + 1,
            occurred_at=existing.created_at,
            event_type=EventType.artifact_created,
            turn=event_turn,
            actor=producer,
            payload=ArtifactCreatedPayload(artifact_id=existing.artifact_id),
        )
        updates: dict = {
            "artifact_ids": [*journal_run.artifact_ids, existing.artifact_id],
            "event_sequence": event.sequence,
            "current_turn": event_turn,
            "updated_at": event.occurred_at,
        }
        if current_agent is not None:
            updates["current_agent"] = current_agent
        updated_run = self._validated_run(journal_run, updates)
        commit_run_event(
            journal_path,
            updated_run,
            event,
            expected_hash=file_hash(journal_path),
        )
        return existing

    def _commit_artifact_unlocked(
        self,
        *,
        journal_path: Path,
        journal_run: Run,
        artifact_id: str,
        filename: str,
        storage_uri: str,
        role: str,
        producer: str,
        artifact_type: ArtifactType,
        media_type: str,
        schema_type: Optional[str],
        schema_version_name: Optional[str],
        content_hash: str,
        size_bytes: int,
        turn: Optional[int],
        current_agent: Optional[str],
    ) -> Artifact:
        timestamp = utc_now()
        event_id = new_id("evt")
        event_turn = journal_run.current_turn if turn is None else turn
        if event_turn < journal_run.current_turn:
            raise ControlError(
                "EVENT_TURN_REGRESSION",
                "artifact event turn cannot move backward",
                exit_code=ExitCode.integrity,
                details={
                    "run_id": journal_run.run_id,
                    "current_turn": journal_run.current_turn,
                    "event_turn": event_turn,
                },
            )
        event = Event(
            event_id=event_id,
            experiment_id=journal_run.experiment_id,
            run_id=journal_run.run_id,
            sequence=journal_run.event_sequence + 1,
            occurred_at=timestamp,
            event_type=EventType.artifact_created,
            turn=event_turn,
            actor=producer,
            payload=ArtifactCreatedPayload(artifact_id=artifact_id),
        )
        artifact = Artifact(
            artifact_id=artifact_id,
            experiment_id=journal_run.experiment_id,
            run_id=journal_run.run_id,
            producer_event_id=event_id,
            producer=producer,
            artifact_type=artifact_type,
            role=role,
            filename=filename,
            storage_uri=storage_uri,
            content_hash=content_hash,
            media_type=media_type,
            schema_type=schema_type,
            schema_version_name=schema_version_name,
            size_bytes=size_bytes,
            created_at=timestamp,
            retention=RetentionPolicy.experiment,
            owner=journal_run.owner,
        )
        manifest_path = self.artifact_manifest_path(journal_run.run_id)
        manifest = self.artifact_manifest(journal_run.run_id)
        updated_manifest = ArtifactManifest(
            run_id=journal_run.run_id,
            artifacts=(*manifest.artifacts, artifact),
            updated_at=timestamp,
        )
        write_model(
            manifest_path,
            updated_manifest,
            expected_hash=file_hash(manifest_path),
        )
        updates: dict = {
            "artifact_ids": [*journal_run.artifact_ids, artifact_id],
            "event_sequence": event.sequence,
            "current_turn": event_turn,
            "updated_at": timestamp,
        }
        if current_agent is not None:
            updates["current_agent"] = current_agent
        updated_run = self._validated_run(journal_run, updates)
        commit_run_event(
            journal_path,
            updated_run,
            event,
            expected_hash=file_hash(journal_path),
        )
        return artifact

    def artifact_path(self, artifact: Artifact) -> Path:
        path = self.run_dir(artifact.run_id) / artifact.storage_uri
        try:
            if path.is_symlink():
                raise ValueError("artifact is a symlink")
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.run_dir(artifact.run_id).resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ControlError(
                "ARTIFACT_PATH_INVALID",
                "artifact storage path escapes its run root or is missing",
                exit_code=ExitCode.integrity,
                details={"artifact_id": artifact.artifact_id},
            ) from exc
        return path

    def verify_artifacts(self, run_id: str) -> tuple[Artifact, ...]:
        manifest = self.artifact_manifest(run_id)
        for artifact in manifest.artifacts:
            try:
                verify_artifact(
                    self.artifact_path(artifact),
                    artifact.content_hash,
                    artifact.size_bytes,
                    root=self.run_dir(run_id),
                )
            except IntegrityError as exc:
                raise ControlError(
                    "ARTIFACT_INTEGRITY_ERROR",
                    "artifact content failed hash, size, or path validation",
                    exit_code=ExitCode.integrity,
                    details={
                        "artifact_id": artifact.artifact_id,
                        "reason": str(exc),
                    },
                ) from exc
        return manifest.artifacts

    def execution_handle(self, run_id: str) -> Optional[ExecutionHandle]:
        path = self.execution_handle_path(run_id)
        return self._read(path, ExecutionHandle) if path.exists() else None

    def write_execution_handle(self, handle: ExecutionHandle) -> None:
        path = self.execution_handle_path(handle.run_id)
        if path.exists():
            raise ControlError(
                "WORKER_ALREADY_LAUNCHED",
                "an execution handle already exists for this run",
                exit_code=ExitCode.conflict,
                details={"run_id": handle.run_id},
            )
        write_model(path, handle)

    def scheduler_handle(self, run_id: str) -> Optional[SlurmExecutionHandle]:
        path = self.scheduler_handle_path(run_id)
        return self._read(path, SlurmExecutionHandle) if path.exists() else None

    def scheduler_cancellation(self, run_id: str) -> Optional[SlurmCancellationLedger]:
        path = self.scheduler_cancellation_path(run_id)
        return self._read(path, SlurmCancellationLedger) if path.exists() else None

    def scheduler_submission(self, run_id: str) -> Optional[SlurmSubmissionLedger]:
        path = self.scheduler_submission_path(run_id)
        return self._read(path, SlurmSubmissionLedger) if path.exists() else None

    def _record_scheduler_submission_attempt_unlocked(
        self,
        *,
        run_id: str,
        job_name: str,
        script_hash: str,
    ) -> SlurmSubmissionLedger:
        path = self.scheduler_submission_path(run_id)
        existing = (
            self._read(path, SlurmSubmissionLedger)
            if path.exists()
            else SlurmSubmissionLedger(
                run_id=run_id,
                job_name=job_name,
                script_hash=script_hash,
            )
        )
        if existing.job_name != job_name or existing.script_hash != script_hash:
            raise ControlError(
                "SCHEDULER_SUBMISSION_CONFLICT",
                "submission ledger differs from the frozen Slurm job identity",
                exit_code=ExitCode.integrity,
                details={"run_id": run_id},
            )
        updated = SlurmSubmissionLedger.model_validate_json(
            existing.model_copy(
                update={"attempts": (*existing.attempts, utc_now())}
            ).model_dump_json()
        )
        write_model(
            path,
            updated,
            expected_hash=file_hash(path) if path.exists() else None,
        )
        return updated

    def _record_scheduler_cancellation_unlocked(
        self,
        *,
        run_id: str,
        job_id: str,
        succeeded: bool,
        error_code: Optional[str] = None,
    ) -> SlurmCancellationLedger:
        path = self.scheduler_cancellation_path(run_id)
        existing = (
            self._read(path, SlurmCancellationLedger)
            if path.exists()
            else SlurmCancellationLedger(run_id=run_id, job_id=job_id)
        )
        if existing.job_id != job_id:
            raise ControlError(
                "SCHEDULER_JOB_MISMATCH",
                "cancellation ledger job ID differs from the bound run",
                exit_code=ExitCode.integrity,
                details={"run_id": run_id},
            )
        if any(attempt.succeeded for attempt in existing.attempts):
            return existing
        updated = SlurmCancellationLedger.model_validate_json(
            existing.model_copy(
                update={
                    "attempts": (
                        *existing.attempts,
                        SlurmCancellationAttempt(
                            succeeded=succeeded,
                            error_code=error_code,
                        ),
                    )
                }
            ).model_dump_json()
        )
        write_model(
            path,
            updated,
            expected_hash=file_hash(path) if path.exists() else None,
        )
        return updated

    def _bind_scheduler_job_unlocked(
        self, handle: SlurmExecutionHandle
    ) -> tuple[Run, bool]:
        path = self.run_journal_path(handle.run_id)
        journal = read_run_journal(path)
        run = journal.run
        if run.executor != ExecutorKind.slurm or run.partition != get_caribou_slurm_partition():
            raise ControlError(
                "RUN_NOT_SLURM",
                f"scheduler identity can be bound only to a {get_caribou_slurm_partition()} Slurm run",
                exit_code=ExitCode.conflict,
                details={"run_id": handle.run_id, "executor": run.executor.value},
            )
        if run.state not in {RunState.queued, RunState.cancelling}:
            raise ControlError(
                "RUN_NOT_QUEUED",
                "a Slurm job can be bound only while the run is queued or cancelling",
                exit_code=ExitCode.conflict,
                details={"run_id": handle.run_id, "state": run.state.value},
            )
        handle_path = self.scheduler_handle_path(run.run_id)
        if handle_path.exists():
            existing = self._read(handle_path, SlurmExecutionHandle)
            if existing != handle:
                raise ControlError(
                    "SCHEDULER_HANDLE_CONFLICT",
                    "a different scheduler handle already exists for the run",
                    exit_code=ExitCode.integrity,
                    details={"run_id": run.run_id},
                )
        else:
            # Persist scheduler identity before modifying the run journal. If the
            # process stops between these writes, launch can finish the binding
            # from this held-job handle without submitting a duplicate job.
            write_model(handle_path, handle)
        if run.scheduler_job_id is not None:
            if run.scheduler_job_id == handle.job_id:
                return run, False
            raise ControlError(
                "SCHEDULER_JOB_CONFLICT",
                "the run is already bound to a different Slurm job",
                exit_code=ExitCode.conflict,
                details={
                    "run_id": handle.run_id,
                    "existing_job_id": run.scheduler_job_id,
                    "submitted_job_id": handle.job_id,
                },
            )
        timestamp = utc_now()
        event = Event(
            experiment_id=run.experiment_id,
            run_id=run.run_id,
            sequence=run.event_sequence + 1,
            occurred_at=timestamp,
            event_type=EventType.heartbeat,
            turn=run.current_turn,
            stage="scheduler_submission",
            actor="slurm-executor",
            payload=HeartbeatPayload(
                message=(
                    f"Slurm job {handle.job_id} bound on partition "
                    f"{get_caribou_slurm_partition()} while held"
                )
            ),
        )
        updated = self._validated_run(
            run,
            {
                "scheduler_job_id": handle.job_id,
                "event_sequence": event.sequence,
                "updated_at": timestamp,
            },
        )
        commit_run_scheduler_binding(
            path,
            updated,
            event,
            expected_hash=file_hash(path),
        )
        return updated, True

    def bind_scheduler_job(self, handle: SlurmExecutionHandle) -> tuple[Run, bool]:
        with self.mutation_lock():
            return self._bind_scheduler_job_unlocked(handle)

    def write_scheduler_script(self, run_id: str, script: str) -> tuple[Path, str]:
        run = self.run(run_id)
        if run.executor != ExecutorKind.slurm:
            raise ControlError(
                "RUN_NOT_SLURM",
                "scheduler scripts belong only to Slurm runs",
                exit_code=ExitCode.conflict,
                details={"run_id": run_id},
            )
        data = script.encode("utf-8")
        content_hash = sha256_bytes(data)
        path = self.scheduler_script_path(run_id)
        with self.mutation_lock():
            if path.exists():
                if file_hash(path) != content_hash:
                    raise ControlError(
                        "SCHEDULER_SCRIPT_CONFLICT",
                        "the frozen Slurm script changed after it was written",
                        exit_code=ExitCode.integrity,
                        details={"run_id": run_id},
                    )
            else:
                self._atomic_bytes(path, data)
                path.chmod(0o700)
        return path, content_hash

    def _mark_scheduler_released_unlocked(self, run_id: str) -> SlurmExecutionHandle:
        path = self.scheduler_handle_path(run_id)
        handle = self._read(path, SlurmExecutionHandle)
        if handle.released_at is not None:
            return handle
        updated = SlurmExecutionHandle.model_validate_json(
            handle.model_copy(update={"released_at": utc_now()}).model_dump_json()
        )
        write_model(path, updated, expected_hash=file_hash(path))
        return updated

    def mark_scheduler_released(self, run_id: str) -> SlurmExecutionHandle:
        with self.mutation_lock():
            return self._mark_scheduler_released_unlocked(run_id)

    def scheduler_accounting(self, run_id: str) -> Optional[SlurmAccounting]:
        path = self.scheduler_accounting_path(run_id)
        if not path.exists():
            return None
        accounting = self._read(path, SlurmAccounting)
        raw_path = self.scheduler_accounting_raw_path(run_id)
        if (
            accounting.raw_output_path != raw_path.name
            or not raw_path.is_file()
            or raw_path.is_symlink()
            or file_hash(raw_path) != accounting.raw_output_hash
        ):
            raise ControlError(
                "SCHEDULER_ACCOUNTING_TAMPERED",
                "durable Slurm accounting raw output is missing or does not match its hash",
                exit_code=ExitCode.integrity,
                details={"run_id": run_id},
            )
        return accounting

    def _record_system_artifact_unlocked(
        self,
        *,
        run_id: str,
        filename: str,
        role: str,
        data: bytes,
        artifact_type: ArtifactType,
        media_type: str,
        schema_type: Optional[str] = None,
        schema_version_name: Optional[str] = None,
    ) -> tuple[Artifact, bool]:
        manifest = self.artifact_manifest(run_id)
        existing = next(
            (artifact for artifact in manifest.artifacts if artifact.role == role),
            None,
        )
        content_hash = sha256_bytes(data)
        if existing is not None:
            if (
                existing.filename != filename
                or existing.content_hash != content_hash
                or existing.size_bytes != len(data)
            ):
                raise ControlError(
                    "SYSTEM_ARTIFACT_CONFLICT",
                    "a scheduler artifact role already refers to different content",
                    exit_code=ExitCode.integrity,
                    details={"run_id": run_id, "role": role},
                )
            try:
                verify_artifact(
                    self.artifact_path(existing),
                    existing.content_hash,
                    existing.size_bytes,
                    root=self.run_dir(run_id),
                )
            except IntegrityError as exc:
                raise ControlError(
                    "SYSTEM_ARTIFACT_TAMPERED",
                    "a durable scheduler artifact failed content verification",
                    exit_code=ExitCode.integrity,
                    details={"run_id": run_id, "role": role},
                ) from exc
            journal_path = self.run_journal_path(run_id)
            journal = read_run_journal(journal_path)
            if existing.artifact_id not in journal.run.artifact_ids:
                # The manifest is written before the journal event by the shared
                # artifact path. Recover a stop at that boundary deterministically
                # from the immutable artifact descriptor rather than duplicating it.
                timestamp = utc_now()
                event = Event(
                    event_id=existing.producer_event_id,
                    experiment_id=journal.run.experiment_id,
                    run_id=run_id,
                    sequence=journal.run.event_sequence + 1,
                    occurred_at=timestamp,
                    event_type=EventType.artifact_created,
                    turn=journal.run.current_turn,
                    actor=existing.producer,
                    payload=ArtifactCreatedPayload(artifact_id=existing.artifact_id),
                )
                updated = self._validated_run(
                    journal.run,
                    {
                        "artifact_ids": [
                            *journal.run.artifact_ids,
                            existing.artifact_id,
                        ],
                        "event_sequence": event.sequence,
                        "updated_at": timestamp,
                    },
                )
                commit_run_event(
                    journal_path,
                    updated,
                    event,
                    expected_hash=file_hash(journal_path),
                )
            return existing, False
        journal_path = self.run_journal_path(run_id)
        journal = read_run_journal(journal_path)
        artifact = self._record_artifact_bytes_unlocked(
            journal_path=journal_path,
            journal_run=journal.run,
            filename=filename,
            role=role,
            data=data,
            producer="slurm-reconciler",
            artifact_type=artifact_type,
            media_type=media_type,
            schema_type=schema_type,
            schema_version_name=schema_version_name,
            turn=None,
            current_agent=None,
        )
        return artifact, True

    def ensure_scheduler_artifacts(self, run_id: str) -> tuple[Artifact, ...]:
        """Expose immutable terminal Slurm evidence through the shared artifact API."""

        with self.mutation_lock():
            accounting = self.scheduler_accounting(run_id)
            if accounting is None:
                raise ControlError(
                    "SCHEDULER_ACCOUNTING_UNAVAILABLE",
                    "terminal Slurm accounting has not been persisted",
                    exit_code=ExitCode.not_found,
                    details={"run_id": run_id},
                )
            raw_path = self.scheduler_accounting_raw_path(run_id)
            artifacts = [
                self._record_system_artifact_unlocked(
                    run_id=run_id,
                    filename="scheduler-accounting.json",
                    role="slurm_accounting",
                    data=canonical_json_bytes(accounting),
                    artifact_type=ArtifactType.manifest,
                    media_type="application/json",
                    schema_type="caribou.slurm_accounting",
                    schema_version_name="v1",
                )[0],
                self._record_system_artifact_unlocked(
                    run_id=run_id,
                    filename="scheduler-accounting.raw",
                    role="slurm_accounting_raw",
                    data=raw_path.read_bytes(),
                    artifact_type=ArtifactType.log,
                    media_type="text/plain",
                )[0],
            ]
            handle = self.scheduler_handle(run_id)
            if handle is not None:
                script_path = self.run_dir(run_id) / handle.script_path
                if (
                    Path(handle.script_path).name != handle.script_path
                    or not script_path.is_file()
                    or script_path.is_symlink()
                    or file_hash(script_path) != handle.script_hash
                ):
                    raise ControlError(
                        "SCHEDULER_SCRIPT_TAMPERED",
                        "the submitted Slurm script is missing or differs from its durable hash",
                        exit_code=ExitCode.integrity,
                        details={"run_id": run_id},
                    )
                artifacts.append(
                    self._record_system_artifact_unlocked(
                        run_id=run_id,
                        filename="slurm-job.sh",
                        role="slurm_job_script",
                        data=script_path.read_bytes(),
                        artifact_type=ArtifactType.code,
                        media_type="text/x-shellscript",
                    )[0]
                )
                stdout_name = handle.stdout_path.replace("%j", handle.job_id)
                if Path(stdout_name).name != stdout_name:
                    raise ControlError(
                        "SCHEDULER_STDOUT_PATH_INVALID",
                        "the durable Slurm stdout path is not a safe run-local filename",
                        exit_code=ExitCode.integrity,
                        details={"run_id": run_id},
                    )
                stdout_path = self.run_dir(run_id) / stdout_name
                if stdout_path.is_file() and not stdout_path.is_symlink():
                    artifacts.append(
                        self._record_system_artifact_unlocked(
                            run_id=run_id,
                            filename="slurm-stdout.log",
                            role="slurm_stdout",
                            data=stdout_path.read_bytes(),
                            artifact_type=ArtifactType.log,
                            media_type="text/plain",
                        )[0]
                    )
            return tuple(artifacts)

    def write_scheduler_accounting(
        self,
        accounting: SlurmAccounting,
        *,
        raw_output: str,
    ) -> tuple[SlurmAccounting, bool]:
        with self.mutation_lock():
            run = self.run(accounting.run_id)
            if run.executor != ExecutorKind.slurm:
                raise ControlError(
                    "RUN_NOT_SLURM",
                    "scheduler accounting belongs only to Slurm runs",
                    exit_code=ExitCode.conflict,
                    details={"run_id": accounting.run_id},
                )
            if run.scheduler_job_id != accounting.job_id:
                raise ControlError(
                    "SCHEDULER_JOB_MISMATCH",
                    "scheduler accounting job ID differs from the durable run",
                    exit_code=ExitCode.integrity,
                    details={
                        "run_id": accounting.run_id,
                        "run_job_id": run.scheduler_job_id,
                        "accounting_job_id": accounting.job_id,
                    },
                )
            raw = raw_output.encode("utf-8")
            if sha256_bytes(raw) != accounting.raw_output_hash:
                raise ControlError(
                    "SCHEDULER_ACCOUNTING_HASH_MISMATCH",
                    "raw scheduler accounting does not match its declared hash",
                    exit_code=ExitCode.integrity,
                )
            path = self.scheduler_accounting_path(accounting.run_id)
            if path.exists():
                existing = self._read(path, SlurmAccounting)
                if existing.raw_output_hash == accounting.raw_output_hash:
                    return existing, False
                raise ControlError(
                    "SCHEDULER_ACCOUNTING_CONFLICT",
                    "terminal scheduler accounting changed after collection",
                    exit_code=ExitCode.integrity,
                    details={"run_id": accounting.run_id},
                )
            raw_path = self.scheduler_accounting_raw_path(accounting.run_id)
            self._atomic_bytes(raw_path, raw)
            write_model(path, accounting)
            return accounting, True

    def reconcile_experiment(self, experiment_id: str) -> Experiment:
        with self.mutation_lock():
            index = self._read_index_unlocked()
            run_ids = index.experiments.get(experiment_id)
            if run_ids is None:
                raise ControlError(
                    "EXPERIMENT_NOT_FOUND",
                    f"unknown experiment: {experiment_id}",
                    exit_code=ExitCode.not_found,
                )
            runs = [self.run(run_id) for run_id in run_ids]
            experiment_path = self.experiment_journal_path(experiment_id)
            journal = read_experiment_journal(experiment_path)
            if journal.experiment.state != ExperimentState.active:
                return journal.experiment
            attempts: dict[tuple[str, int], list[Run]] = {}
            for run in runs:
                attempts.setdefault((run.condition_id, run.replicate_index), []).append(
                    run
                )
            leaves: list[Run] = []
            for key, lineage in attempts.items():
                indices = [run.attempt_index for run in lineage]
                if len(indices) != len(set(indices)):
                    raise ControlError(
                        "ATTEMPT_LINEAGE_AMBIGUOUS",
                        "a condition replicate has duplicate attempt indices",
                        exit_code=ExitCode.integrity,
                        details={
                            "experiment_id": experiment_id,
                            "condition_id": key[0],
                            "replicate_index": key[1],
                        },
                    )
                leaves.append(max(lineage, key=lambda item: item.attempt_index))
            if any(run.state == RunState.resumable for run in leaves):
                # A resumable leaf is deliberately terminal at the attempt level,
                # but the logical experiment remains active while awaiting a child.
                return journal.experiment
            if not all(run.state in TERMINAL_RUN_STATES for run in leaves):
                return journal.experiment
            if all(run.state == RunState.succeeded for run in leaves):
                first = transition_experiment(
                    journal.experiment,
                    ExperimentState.aggregating,
                    reason="all attempts reached a successful terminal state",
                    actor="control-plane",
                )
                first_hash = commit_experiment_transition(
                    experiment_path, first, expected_hash=file_hash(experiment_path)
                )
                second = transition_experiment(
                    first.experiment,
                    ExperimentState.completed,
                    reason="no aggregate was required for the lifecycle probe",
                    actor="control-plane",
                )
                commit_experiment_transition(
                    experiment_path, second, expected_hash=first_hash
                )
                return second.experiment
            target = (
                ExperimentState.failed
                if any(
                    run.state in {RunState.failed, RunState.rejected} for run in leaves
                )
                else ExperimentState.cancelled
            )
            transition = transition_experiment(
                journal.experiment,
                target,
                reason="one or more attempts ended without success",
                actor="control-plane",
            )
            commit_experiment_transition(
                experiment_path,
                transition,
                expected_hash=file_hash(experiment_path),
            )
            return transition.experiment
