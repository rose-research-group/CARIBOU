"""Narrow Slurm transport for the shared durable CARIBOU worker."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Optional

from caribou.config import CARIBOU_HOME, get_caribou_slurm_partition
from caribou.domain.enums import ExecutorKind, RunState
from caribou.domain.models import Run, utc_now
from caribou.domain.serialization import file_hash, sha256_bytes

from .api import ControlError, ExitCode
from .executor import LaunchResult
from .records import SlurmAccounting, SlurmExecutionHandle
from .specs import ADAPTER_PARAMETER, CARIBOU_AGENT_ADAPTER
from .store import ExperimentStore, TERMINAL_RUN_STATES


_JOB_ID = re.compile(r"^[0-9]+$")
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 15.0
_DEFAULT_COMMAND_TIMEOUTS = {
    "squeue": 75.0,
    "sacct": 75.0,
}
_SUBMISSION_VISIBILITY_GRACE_SECONDS = 30.0
_SQUEUE_TERMINAL_ABSENCE_MARKERS = ("invalid job id specified",)
_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)


@dataclass(frozen=True)
class SchedulerObservation:
    job_id: str
    partition: str
    state: str
    terminal: bool
    source: str
    exit_code: Optional[str] = None
    elapsed_seconds: int = 0
    allocated_cpus: int = 0
    requested_memory: Optional[str] = None
    max_rss_kib: Optional[int] = None
    node_list: Optional[str] = None
    reason: Optional[str] = None
    started_at_raw: Optional[str] = None
    ended_at_raw: Optional[str] = None
    raw_output: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "partition": self.partition,
            "state": self.state,
            "terminal": self.terminal,
            "source": self.source,
            "exit_code": self.exit_code,
            "elapsed_seconds": self.elapsed_seconds,
            "allocated_cpus": self.allocated_cpus,
            "requested_memory": self.requested_memory,
            "max_rss_kib": self.max_rss_kib,
            "node_list": self.node_list,
            "reason": self.reason,
            "started_at_raw": self.started_at_raw,
            "ended_at_raw": self.ended_at_raw,
        }


@dataclass(frozen=True)
class ReconciliationResult:
    run: Run
    observation: SchedulerObservation
    accounting: Optional[SlurmAccounting]
    accounting_created: bool
    run_transition_applied: bool


def _normalized_state(value: str) -> str:
    return value.strip().split(maxsplit=1)[0].rstrip("+").upper()


def _optional(value: str) -> Optional[str]:
    stripped = value.strip()
    return None if stripped in {"", "(null)", "Unknown", "None"} else stripped


def _integer(value: str, field: str) -> int:
    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise ControlError(
            "SLURM_ACCOUNTING_INVALID",
            f"Slurm returned a non-integer {field}",
            exit_code=ExitCode.integrity,
            details={"field": field, "value": value[:100]},
        ) from exc
    if parsed < 0:
        raise ControlError(
            "SLURM_ACCOUNTING_INVALID",
            f"Slurm returned a negative {field}",
            exit_code=ExitCode.integrity,
            details={"field": field, "value": value[:100]},
        )
    return parsed


def _memory_kib(value: str) -> Optional[int]:
    stripped = value.strip()
    if not stripped:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)", stripped.upper())
    if match is None:
        raise ControlError(
            "SLURM_ACCOUNTING_INVALID",
            "Slurm returned an invalid MaxRSS value",
            exit_code=ExitCode.integrity,
            details={"field": "MaxRSS", "value": value[:100]},
        )
    factors = {
        "": Decimal(1),
        "K": Decimal(1),
        "M": Decimal(1024),
        "G": Decimal(1024) ** 2,
        "T": Decimal(1024) ** 3,
        "P": Decimal(1024) ** 4,
        "E": Decimal(1024) ** 5,
    }
    try:
        return int(Decimal(match.group(1)) * factors[match.group(2)])
    except InvalidOperation as exc:
        raise ControlError(
            "SLURM_ACCOUNTING_INVALID",
            "Slurm returned an invalid MaxRSS value",
            exit_code=ExitCode.integrity,
            details={"field": "MaxRSS", "value": value[:100]},
        ) from exc


def _slurm_time(seconds: int) -> str:
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{clock}" if days else clock


def _job_name(run_id: str) -> str:
    return f"caribou_{run_id}"


def _render_script(store: ExperimentStore, run: Run) -> str:
    source_root = Path(__file__).resolve().parents[2]
    repository_root = Path(__file__).resolve().parents[4]
    python = Path(sys.executable).resolve()
    home = Path.home().resolve()
    run_dir = store.run_dir(run.run_id)
    for field, value in (
        ("repository_root", str(repository_root)),
        ("run_dir", str(run_dir)),
    ):
        if any(character.isspace() or ord(character) == 127 for character in value):
            raise ControlError(
                "SLURM_PATH_INVALID",
                f"{field} contains whitespace unsafe for #SBATCH directives",
                exit_code=ExitCode.validation,
                details={"field": field},
            )
    memory_mib = (run.resources.memory_bytes + 1024**2 - 1) // 1024**2
    directives = [
        f"#SBATCH --job-name={_job_name(run.run_id)}",
        f"#SBATCH --partition={get_caribou_slurm_partition()}",
        f"#SBATCH --cpus-per-task={run.resources.cpu_cores}",
        f"#SBATCH --mem={memory_mib}M",
        f"#SBATCH --time={_slurm_time(run.resources.wall_seconds)}",
        "#SBATCH --export=NIL",
        f"#SBATCH --chdir={repository_root}",
        f"#SBATCH --output={run_dir / 'slurm-%j.out'}",
    ]
    if run.resources.gpu_count:
        directives.append(f"#SBATCH --gpus={run.resources.gpu_count}")
    spec = store.spec(run.experiment_id)
    condition = next(
        item for item in spec.conditions if item.condition_id == run.condition_id
    )
    adapter = condition.parameters.get(ADAPTER_PARAMETER)
    if adapter == CARIBOU_AGENT_ADAPTER and run.resolved_model.provider in {
        "openai",
        "deepseek",
        "openrouter",
    }:
        credential_file = CARIBOU_HOME / ".env"
        try:
            credential_stat = credential_file.stat()
        except OSError as exc:
            raise ControlError(
                "PROVIDER_CREDENTIAL_FILE_UNAVAILABLE",
                "provider-backed Slurm execution requires the per-user CARIBOU .env file",
                exit_code=ExitCode.permission,
                details={"path": str(credential_file)},
            ) from exc
        mode = credential_stat.st_mode
        if (
            credential_file.is_symlink()
            or not credential_file.is_file()
            or credential_stat.st_uid != os.geteuid()
            or mode & 0o077
            or mode & 0o600 != 0o600
        ):
            raise ControlError(
                "PROVIDER_CREDENTIAL_FILE_UNSAFE",
                "the per-user CARIBOU .env file must be owner-controlled, regular, non-symlinked, and mode 0600",
                exit_code=ExitCode.permission,
                details={"path": str(credential_file)},
            )
    if spec.execution.account is not None:
        directives.append(f"#SBATCH --account={spec.execution.account}")
    if spec.execution.qos is not None:
        directives.append(f"#SBATCH --qos={spec.execution.qos}")
    body = [
        "#!/bin/bash",
        *directives,
        "",
        "set -euo pipefail",
        "",
        "if (( $# != 0 )); then",
        '  echo "ERROR: this generated Slurm worker accepts no arguments" >&2',
        "  exit 64",
        "fi",
        "",
        f"export HOME={shlex.quote(str(home))}",
        'export PATH="/usr/bin:/bin"',
        'export PYTHONNOUSERSITE="1"',
        f"export PYTHONPATH={shlex.quote(str(source_root))}",
        f"export CARIBOU_HOME={shlex.quote(str(CARIBOU_HOME.resolve()))}",
        f"export CARIBOU_EXPERIMENT_HOME={shlex.quote(str(store.root))}",
        f"export TMPDIR={shlex.quote(str(run_dir / 'tmp'))}/$SLURM_JOB_ID",
        '/usr/bin/mkdir -p "$TMPDIR"',
        f"cd {shlex.quote(str(repository_root))}",
        (
            f"exec {shlex.quote(str(python))} -m caribou.control.worker "
            f"--store-root {shlex.quote(str(store.root))} "
            f"--run-id {shlex.quote(run.run_id)}"
        ),
        "",
    ]
    return "\n".join(body)


class SlurmExecutor:
    """Submit exactly one held job on the configured Slurm partition and bind it before release."""

    def __init__(
        self,
        *,
        sbatch: str = "sbatch",
        scontrol: str = "scontrol",
        scancel: str = "scancel",
        squeue: str = "squeue",
        sacct: str = "sacct",
        command_timeouts: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.sbatch = sbatch
        self.scontrol = scontrol
        self.scancel = scancel
        self.squeue = squeue
        self.sacct = sacct
        self.command_timeouts = dict(_DEFAULT_COMMAND_TIMEOUTS)
        if command_timeouts is not None:
            self.command_timeouts.update(command_timeouts)

    def _run(
        self, command: list[str], *, timeout: Optional[float] = None
    ) -> subprocess.CompletedProcess[str]:
        command_role = next(
            (
                role
                for role in ("sbatch", "scontrol", "scancel", "squeue", "sacct")
                if command[0] == getattr(self, role)
            ),
            Path(command[0]).name,
        )
        effective_timeout = (
            timeout
            if timeout is not None
            else self.command_timeouts.get(
                command[0],
                self.command_timeouts.get(
                    command_role,
                    self.command_timeouts.get(
                        Path(command[0]).name,
                        _DEFAULT_COMMAND_TIMEOUT_SECONDS,
                    ),
                ),
            )
        )
        environment = None
        if Path(command[0]).name == "sbatch":
            # SBATCH_* variables override script directives. Remove them so the
            # frozen script and explicit submit flags remain authoritative.
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("SBATCH_")
            }
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise ControlError(
                "SLURM_COMMAND_NOT_FOUND",
                f"required Slurm command is unavailable: {command[0]}",
                exit_code=ExitCode.not_found,
                details={"command": command[0]},
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ControlError(
                "SLURM_COMMAND_TIMEOUT",
                f"Slurm command timed out: {command[0]}",
                exit_code=ExitCode.transient,
                retryable=True,
                details={
                    "command": command[0],
                    "timeout_seconds": effective_timeout,
                },
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1:] or [""]
            raise ControlError(
                "SLURM_COMMAND_FAILED",
                f"Slurm command failed: {command[0]}",
                exit_code=ExitCode.transient,
                retryable=True,
                details={
                    "command": command[0],
                    "returncode": result.returncode,
                    "stderr": detail[0][:500],
                },
            )
        return result

    def _release(self, job_id: str) -> None:
        self._run([self.scontrol, "release", job_id])

    def _cleanup_launch_job(
        self,
        *,
        store: ExperimentStore,
        run_id: str,
        handle: SlurmExecutionHandle,
        cause: ControlError,
    ) -> ControlError:
        try:
            self._run([self.scancel, handle.job_id])
        except ControlError as cleanup_error:
            store._record_scheduler_cancellation_unlocked(
                run_id=run_id,
                job_id=handle.job_id,
                succeeded=False,
                error_code=cleanup_error.code,
            )
            return ControlError(
                "SLURM_LAUNCH_CLEANUP_FAILED",
                "Slurm launch failed and its held job could not be cancelled",
                exit_code=ExitCode.transient,
                retryable=True,
                details={
                    "run_id": run_id,
                    "job_id": handle.job_id,
                    "launch_error": cause.code,
                    "cancel_error": cleanup_error.code,
                },
            )
        store._record_scheduler_cancellation_unlocked(
            run_id=run_id,
            job_id=handle.job_id,
            succeeded=True,
        )
        return cause

    def _release_bound_handle(
        self,
        *,
        store: ExperimentStore,
        run_id: str,
        handle: SlurmExecutionHandle,
    ) -> SlurmExecutionHandle:
        """Release a held job, resolving a response/local-write ambiguity safely."""

        try:
            self._release(handle.job_id)
            return store._mark_scheduler_released_unlocked(run_id)
        except ControlError as release_error:
            try:
                observation = self._observe_squeue(handle)
            except ControlError as observation_error:
                raise ControlError(
                    "SLURM_RELEASE_STATE_UNKNOWN",
                    "Slurm release failed and held-job state could not be observed",
                    exit_code=ExitCode.transient,
                    retryable=True,
                    details={
                        "run_id": run_id,
                        "job_id": handle.job_id,
                        "release_error": release_error.code,
                        "observation_error": observation_error.code,
                    },
                ) from release_error
            if observation is None:
                try:
                    accounting_observation = self._observe_sacct(handle)
                except ControlError as accounting_error:
                    raise ControlError(
                        "SLURM_RELEASE_STATE_UNKNOWN",
                        "released job is absent from squeue and unavailable in sacct",
                        exit_code=ExitCode.transient,
                        retryable=True,
                        details={
                            "run_id": run_id,
                            "job_id": handle.job_id,
                            "release_error": release_error.code,
                            "accounting_error": accounting_error.code,
                        },
                    ) from release_error
                if not accounting_observation.terminal:
                    raise ControlError(
                        "SLURM_RELEASE_STATE_UNKNOWN",
                        "released job has no terminal or queue state",
                        exit_code=ExitCode.transient,
                        retryable=True,
                        details={"run_id": run_id, "job_id": handle.job_id},
                    ) from release_error
                return store._mark_scheduler_released_unlocked(run_id)
            released_or_finished = not (
                observation.state == "PENDING"
                and (observation.reason or "").startswith("JobHeld")
            )
            if released_or_finished:
                return store._mark_scheduler_released_unlocked(run_id)
            raise release_error

    def _handle(
        self,
        *,
        store: ExperimentStore,
        run: Run,
        job_id: str,
        script_path: Path,
        script_hash: str,
    ) -> SlurmExecutionHandle:
        spec = store.spec(run.experiment_id)
        return SlurmExecutionHandle(
            run_id=run.run_id,
            job_id=job_id,
            account=spec.execution.account,
            qos=spec.execution.qos,
            script_path=str(script_path.relative_to(store.run_dir(run.run_id))),
            script_hash=script_hash,
            stdout_path="slurm-%j.out",
        )

    def _recover_held_submission(
        self,
        *,
        store: ExperimentStore,
        run: Run,
        script_path: Path,
        script_hash: str,
    ) -> Optional[SlurmExecutionHandle]:
        name = _job_name(run.run_id)
        result = self._run(
            [
                self.squeue,
                "--local",
                "--noheader",
                "--user",
                str(os.geteuid()),
                "--name",
                name,
                "--states=PENDING",
                "--format=%i|%j|%P|%r|%U",
            ]
        )
        candidates: list[str] = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            fields = line.split("|")
            if len(fields) != 5:
                raise ControlError(
                    "SLURM_RECOVERY_OUTPUT_INVALID",
                    "squeue returned an invalid held-job recovery record",
                    exit_code=ExitCode.integrity,
                    details={"run_id": run.run_id},
                )
            job_id, observed_name, partition, reason, user_id = (
                field.strip() for field in fields
            )
            if observed_name != name:
                continue
            if (
                partition != get_caribou_slurm_partition()
                or not reason.startswith("JobHeld")
                or user_id != str(os.geteuid())
            ):
                raise ControlError(
                    "SLURM_RECOVERY_MISMATCH",
                    f"a same-name recovery candidate violates the held {get_caribou_slurm_partition()} contract",
                    exit_code=ExitCode.integrity,
                    details={"run_id": run.run_id, "job_id": job_id},
                )
            if _JOB_ID.fullmatch(job_id) is None:
                raise ControlError(
                    "SLURM_JOB_ID_INVALID",
                    "held-job recovery returned a non-numeric job ID",
                    exit_code=ExitCode.integrity,
                )
            candidates.append(job_id)
        if len(candidates) > 1:
            raise ControlError(
                "SLURM_SUBMISSION_AMBIGUOUS",
                "multiple held jobs match the durable CARIBOU run identity",
                exit_code=ExitCode.integrity,
                details={"run_id": run.run_id, "job_ids": candidates},
            )
        if not candidates:
            return None
        return self._handle(
            store=store,
            run=run,
            job_id=candidates[0],
            script_path=script_path,
            script_hash=script_hash,
        )

    @staticmethod
    def _submission_grace_elapsed(
        store: ExperimentStore,
        run_id: str,
    ) -> bool:
        ledger = store.scheduler_submission(run_id)
        if ledger is None or not ledger.attempts:
            return True
        elapsed = (utc_now() - ledger.attempts[-1]).total_seconds()
        return elapsed >= _SUBMISSION_VISIBILITY_GRACE_SECONDS

    def launch(self, store: ExperimentStore, run_id: str) -> LaunchResult:
        run = store.run(run_id)
        if run.executor != ExecutorKind.slurm or run.partition != get_caribou_slurm_partition():
            raise ControlError(
                "RUN_NOT_SLURM",
                f"the Slurm executor requires a {get_caribou_slurm_partition()} Slurm run",
                exit_code=ExitCode.conflict,
                details={"run_id": run_id},
            )
        script_path, script_hash = store.write_scheduler_script(
            run_id, _render_script(store, run)
        )
        launch_error: Optional[ControlError] = None
        handle: Optional[SlurmExecutionHandle] = None
        launched = False
        with store.mutation_lock():
            existing = store.scheduler_handle(run_id)
            current = store.run(run_id)
            if existing is not None:
                if existing.script_hash != script_hash:
                    raise ControlError(
                        "SCHEDULER_SCRIPT_CONFLICT",
                        "the scheduler handle refers to another script",
                        exit_code=ExitCode.integrity,
                        details={"run_id": run_id},
                    )
                if (
                    current.scheduler_job_id is None
                    and current.state == RunState.queued
                ):
                    current, _ = store._bind_scheduler_job_unlocked(existing)
                elif current.scheduler_job_id != existing.job_id:
                    raise ControlError(
                        "SCHEDULER_HANDLE_MISMATCH",
                        "the scheduler handle differs from the durable run",
                        exit_code=ExitCode.integrity,
                        details={"run_id": run_id},
                    )
                handle = existing
                if existing.released_at is None:
                    try:
                        handle = self._release_bound_handle(
                            store=store,
                            run_id=run_id,
                            handle=existing,
                        )
                    except ControlError as exc:
                        launch_error = self._cleanup_launch_job(
                            store=store,
                            run_id=run_id,
                            handle=existing,
                            cause=exc,
                        )
            elif current.scheduler_job_id is not None:
                launch_error = ControlError(
                    "SCHEDULER_HANDLE_MISSING",
                    "the run has a Slurm job ID but no scheduler handle",
                    exit_code=ExitCode.integrity,
                    details={"run_id": run_id},
                )
            elif current.state in TERMINAL_RUN_STATES:
                launch_error = ControlError(
                    "RUN_TERMINAL",
                    "a terminal run cannot submit a Slurm worker",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": current.state.value},
                )
            elif current.state != RunState.queued:
                launch_error = ControlError(
                    "RUN_NOT_QUEUED",
                    "the Slurm executor submits only a durably queued run",
                    exit_code=ExitCode.conflict,
                    details={"run_id": run_id, "state": current.state.value},
                )
            else:
                try:
                    handle = self._recover_held_submission(
                        store=store,
                        run=current,
                        script_path=script_path,
                        script_hash=script_hash,
                    )
                    if handle is None:
                        if not self._submission_grace_elapsed(store, run_id):
                            raise ControlError(
                                "SLURM_SUBMISSION_UNCERTAIN",
                                "a recent sbatch attempt has no visible held job yet; retry after the visibility grace period",
                                exit_code=ExitCode.transient,
                                retryable=True,
                                details={
                                    "run_id": run_id,
                                    "job_name": _job_name(run_id),
                                    "grace_seconds": _SUBMISSION_VISIBILITY_GRACE_SECONDS,
                                },
                            )
                        store._record_scheduler_submission_attempt_unlocked(
                            run_id=run_id,
                            job_name=_job_name(run_id),
                            script_hash=script_hash,
                        )
                        try:
                            submitted = self._run(
                                [
                                    self.sbatch,
                                    "--parsable",
                                    "--hold",
                                    f"--partition={get_caribou_slurm_partition()}",
                                    "--export=NIL",
                                    str(script_path),
                                ]
                            )
                        except ControlError as submit_error:
                            if submit_error.code != "SLURM_COMMAND_TIMEOUT":
                                raise
                            handle = self._recover_held_submission(
                                store=store,
                                run=current,
                                script_path=script_path,
                                script_hash=script_hash,
                            )
                            if handle is None:
                                raise ControlError(
                                    "SLURM_SUBMISSION_UNCERTAIN",
                                    "sbatch timed out and no held job is visible yet; retry with the same idempotency key",
                                    exit_code=ExitCode.transient,
                                    retryable=True,
                                    details={
                                        "run_id": run_id,
                                        "job_name": _job_name(run_id),
                                    },
                                ) from submit_error
                        if handle is None:
                            job_id = submitted.stdout.strip().split(";", 1)[0]
                            if _JOB_ID.fullmatch(job_id) is None:
                                raise ControlError(
                                    "SLURM_SUBMISSION_UNCERTAIN",
                                    "sbatch returned an invalid job ID; retry after held-job recovery",
                                    exit_code=ExitCode.transient,
                                    retryable=True,
                                    details={
                                        "run_id": run_id,
                                        "job_name": _job_name(run_id),
                                    },
                                )
                            handle = self._handle(
                                store=store,
                                run=current,
                                job_id=job_id,
                                script_path=script_path,
                                script_hash=script_hash,
                            )
                    store._bind_scheduler_job_unlocked(handle)
                    handle = self._release_bound_handle(
                        store=store,
                        run_id=run_id,
                        handle=handle,
                    )
                    launched = True
                except ControlError as exc:
                    launch_error = exc
                    if handle is not None:
                        launch_error = self._cleanup_launch_job(
                            store=store,
                            run_id=run_id,
                            handle=handle,
                            cause=exc,
                        )
            if launch_error is not None:
                current = store.run(run_id)
                if (
                    current.state == RunState.cancelling
                    and store.cancel_requested(run_id)
                    and launch_error.code != "SLURM_LAUNCH_CLEANUP_FAILED"
                ):
                    try:
                        store._transition_run_unlocked(
                            run_id,
                            RunState.cancelled,
                            reason="cancelled before the Slurm worker started",
                            actor="slurm-executor",
                            exit_code=int(ExitCode.cancelled),
                        )
                    except Exception:
                        pass
                elif (
                    current.state == RunState.queued
                    and (handle is not None or not launch_error.retryable)
                    and launch_error.code != "SLURM_LAUNCH_CLEANUP_FAILED"
                ):
                    try:
                        store._transition_run_unlocked(
                            run_id,
                            RunState.failed,
                            reason=f"Slurm submission failure: {launch_error.code}",
                            actor="slurm-executor",
                            exit_code=int(ExitCode.execution),
                        )
                    except Exception:
                        pass
        if launch_error is not None:
            if launch_error.code == "SLURM_LAUNCH_CLEANUP_FAILED":
                store.request_cancel(
                    run_id,
                    actor="slurm-executor",
                    reason="Slurm launch cleanup requires an idempotent retry",
                )
            try:
                store.reconcile_experiment(run.experiment_id)
            except Exception:
                pass
            raise launch_error
        if handle is None:  # pragma: no cover - guarded by the branches above
            raise ControlError(
                "SLURM_LAUNCH_INCOMPLETE",
                "Slurm launch finished without a scheduler handle",
                exit_code=ExitCode.internal,
            )
        return LaunchResult(handle=handle, launched=launched)

    def _observe_squeue(
        self, handle: SlurmExecutionHandle
    ) -> Optional[SchedulerObservation]:
        try:
            result = self._run(
                [
                    self.squeue,
                    "--local",
                    "--noheader",
                    "--jobs",
                    handle.job_id,
                    "--format=%i|%T|%P|%N|%M|%C|%m|%r",
                ]
            )
        except ControlError as exc:
            stderr = str(exc.details.get("stderr", "")).casefold()
            if exc.code == "SLURM_COMMAND_FAILED" and any(
                marker in stderr for marker in _SQUEUE_TERMINAL_ABSENCE_MARKERS
            ):
                # Some Slurm versions return exit 1 once a terminal job has
                # left squeue instead of returning an empty result. sacct is
                # authoritative for terminal identity, state, and resources.
                return None
            raise
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        fields = lines[0].split("|")
        if len(fields) != 8 or fields[0].strip() != handle.job_id:
            raise ControlError(
                "SLURM_OUTPUT_INVALID",
                "squeue returned an unexpected record",
                exit_code=ExitCode.integrity,
                details={"job_id": handle.job_id},
            )
        state = _normalized_state(fields[1])
        return SchedulerObservation(
            job_id=handle.job_id,
            partition=fields[2].strip(),
            state=state,
            terminal=state in _TERMINAL_STATES,
            source="squeue",
            allocated_cpus=_integer(fields[5], "allocated CPUs"),
            requested_memory=_optional(fields[6]),
            node_list=_optional(fields[3]),
            reason=_optional(fields[7]),
            raw_output=result.stdout,
        )

    def _observe_sacct(self, handle: SlurmExecutionHandle) -> SchedulerObservation:
        result = self._run(
            [
                self.sacct,
                "--local",
                "--jobs",
                handle.job_id,
                "--noheader",
                "--parsable2",
                "--units=K",
                "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,NodeList,Start,End,Partition",
            ]
        )
        records = [
            line.split("|") for line in result.stdout.splitlines() if line.strip()
        ]
        root = next(
            (
                fields
                for fields in records
                if fields and fields[0].strip() == handle.job_id
            ),
            None,
        )
        if root is None or len(root) < 11:
            raise ControlError(
                "SLURM_ACCOUNTING_UNAVAILABLE",
                "sacct has no root record for the submitted job",
                exit_code=ExitCode.transient,
                retryable=True,
                details={"job_id": handle.job_id},
            )
        batch = next(
            (
                fields
                for fields in records
                if fields and fields[0].strip() == f"{handle.job_id}.batch"
            ),
            None,
        )
        state = _normalized_state(root[1])
        max_rss = batch[6] if batch is not None and len(batch) >= 7 else root[6]
        partition = root[10].strip()
        if partition != handle.partition:
            raise ControlError(
                "SLURM_PARTITION_MISMATCH",
                "Slurm accounting partition differs from the durable binding",
                exit_code=ExitCode.integrity,
                details={
                    "job_id": handle.job_id,
                    "expected": handle.partition,
                    "observed": partition,
                },
            )
        return SchedulerObservation(
            job_id=handle.job_id,
            partition=partition,
            state=state,
            terminal=state in _TERMINAL_STATES,
            source="sacct",
            exit_code=_optional(root[2]),
            elapsed_seconds=_integer(root[3], "elapsed seconds"),
            allocated_cpus=_integer(root[4], "allocated CPUs"),
            requested_memory=_optional(root[5]),
            max_rss_kib=_memory_kib(max_rss),
            node_list=_optional(root[7]),
            started_at_raw=_optional(root[8]),
            ended_at_raw=_optional(root[9]),
            raw_output=result.stdout,
        )

    def inspect(self, store: ExperimentStore, run_id: str) -> SchedulerObservation:
        run = store.run(run_id)
        handle = store.scheduler_handle(run_id)
        if run.executor != ExecutorKind.slurm or handle is None:
            raise ControlError(
                "SCHEDULER_HANDLE_NOT_FOUND",
                "the run has no durable Slurm submission handle",
                exit_code=ExitCode.not_found,
                details={"run_id": run_id},
            )
        if run.scheduler_job_id != handle.job_id:
            raise ControlError(
                "SCHEDULER_HANDLE_MISMATCH",
                "the scheduler handle differs from the durable run",
                exit_code=ExitCode.integrity,
                details={"run_id": run_id},
            )
        accounting = store.scheduler_accounting(run_id)
        if accounting is not None:
            return SchedulerObservation(
                job_id=accounting.job_id,
                partition=accounting.partition,
                state=accounting.state,
                terminal=accounting.terminal,
                source="durable-accounting",
                exit_code=accounting.exit_code,
                elapsed_seconds=accounting.elapsed_seconds,
                allocated_cpus=accounting.allocated_cpus,
                requested_memory=accounting.requested_memory,
                max_rss_kib=accounting.max_rss_kib,
                node_list=accounting.node_list,
                started_at_raw=accounting.started_at_raw,
                ended_at_raw=accounting.ended_at_raw,
            )
        queued = self._observe_squeue(handle)
        if queued is not None:
            if queued.partition != handle.partition:
                raise ControlError(
                    "SLURM_PARTITION_MISMATCH",
                    "Slurm queue partition differs from the durable binding",
                    exit_code=ExitCode.integrity,
                    details={
                        "job_id": handle.job_id,
                        "expected": handle.partition,
                        "observed": queued.partition,
                    },
                )
            if not queued.terminal:
                return queued
        # Only sacct carries terminal exit/resource evidence. Never freeze a
        # terminal squeue row as accounting.
        return self._observe_sacct(handle)

    @staticmethod
    def _consistent(run: Run, observation: SchedulerObservation) -> Optional[bool]:
        if run.state == RunState.succeeded:
            return observation.state == "COMPLETED" and observation.exit_code in {
                None,
                "0:0",
            }
        if run.state == RunState.cancelled:
            return observation.state == "CANCELLED"
        if run.state == RunState.failed:
            return observation.state != "COMPLETED" or observation.exit_code not in {
                None,
                "0:0",
            }
        return None

    def reconcile(self, store: ExperimentStore, run_id: str) -> ReconciliationResult:
        observation = self.inspect(store, run_id)
        run = store.run(run_id)
        transition_applied = False
        if observation.terminal and run.state not in TERMINAL_RUN_STATES:
            cancelled = observation.state == "CANCELLED" and (
                run.state == RunState.cancelling or store.cancel_requested(run_id)
            )
            target = RunState.cancelled if cancelled else RunState.failed
            reason = (
                f"Slurm job {observation.job_id} reached {observation.state} "
                "before the worker recorded a terminal outcome"
            )
            run, transition_applied = store.transition_run(
                run_id,
                target,
                reason=reason,
                actor="slurm-reconciler",
                exit_code=(
                    int(ExitCode.cancelled)
                    if target == RunState.cancelled
                    else int(ExitCode.execution)
                ),
            )
            store.reconcile_experiment(run.experiment_id)
        accounting: Optional[SlurmAccounting] = None
        accounting_created = False
        if observation.terminal:
            if observation.source == "durable-accounting":
                accounting = store.scheduler_accounting(run_id)
                store.ensure_scheduler_artifacts(run_id)
            else:
                raw = observation.raw_output
                accounting = SlurmAccounting(
                    run_id=run_id,
                    job_id=observation.job_id,
                    partition=get_caribou_slurm_partition(),
                    state=observation.state,
                    terminal=True,
                    exit_code=observation.exit_code,
                    elapsed_seconds=observation.elapsed_seconds,
                    allocated_cpus=observation.allocated_cpus,
                    requested_memory=observation.requested_memory,
                    max_rss_kib=observation.max_rss_kib,
                    node_list=observation.node_list,
                    started_at_raw=observation.started_at_raw,
                    ended_at_raw=observation.ended_at_raw,
                    raw_output_path="scheduler-accounting.raw",
                    raw_output_hash=sha256_bytes(raw.encode("utf-8")),
                    consistent_with_run=self._consistent(run, observation),
                    recorded_at=utc_now(),
                )
                accounting, accounting_created = store.write_scheduler_accounting(
                    accounting, raw_output=raw
                )
                store.ensure_scheduler_artifacts(run_id)
        return ReconciliationResult(
            run=store.run(run_id),
            observation=observation,
            accounting=accounting,
            accounting_created=accounting_created,
            run_transition_applied=transition_applied,
        )

    def cancel(self, store: ExperimentStore, run_id: str) -> bool:
        with store.mutation_lock():
            run = store.run(run_id)
            handle = store.scheduler_handle(run_id)
            if run.executor != ExecutorKind.slurm or run.state in TERMINAL_RUN_STATES:
                return False
            if handle is None and run.state == RunState.cancelling:
                script_path = store.scheduler_script_path(run_id)
                if script_path.is_file() and not script_path.is_symlink():
                    handle = self._recover_held_submission(
                        store=store,
                        run=run,
                        script_path=script_path,
                        script_hash=file_hash(script_path),
                    )
            if handle is None:
                # A timed-out sbatch may not be visible immediately. Preserve the
                # cancelling state so a later idempotent cancel can recover it.
                if not self._submission_grace_elapsed(store, run_id):
                    return False
                store._transition_run_unlocked(
                    run_id,
                    RunState.cancelled,
                    reason="no Slurm job appeared during the submission visibility grace period",
                    actor="slurm-executor",
                    exit_code=int(ExitCode.cancelled),
                )
                return False
            partial_binding = run.scheduler_job_id is None
            if partial_binding:
                run, _ = store._bind_scheduler_job_unlocked(handle)
            elif run.scheduler_job_id != handle.job_id:
                raise ControlError(
                    "SCHEDULER_HANDLE_MISMATCH",
                    "the scheduler handle differs from the durable run",
                    exit_code=ExitCode.integrity,
                    details={"run_id": run_id},
                )
            ledger = store.scheduler_cancellation(run_id)
            if ledger is not None and any(
                attempt.succeeded for attempt in ledger.attempts
            ):
                return False
            try:
                self._run([self.scancel, handle.job_id])
            except ControlError as exc:
                store._record_scheduler_cancellation_unlocked(
                    run_id=run_id,
                    job_id=handle.job_id,
                    succeeded=False,
                    error_code=exc.code,
                )
                raise
            store._record_scheduler_cancellation_unlocked(
                run_id=run_id,
                job_id=handle.job_id,
                succeeded=True,
            )
            if partial_binding and run.state == RunState.cancelling:
                store._transition_run_unlocked(
                    run_id,
                    RunState.cancelled,
                    reason="cancelled durable held job before scheduler binding completed",
                    actor="slurm-executor",
                    exit_code=int(ExitCode.cancelled),
                )
            return True
