"""Focused Slurm lifecycle tests for the shared durable control plane."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

import pytest

import caribou.control.slurm as control_slurm_module
import caribou.control.store as control_store_module
from caribou.control.api import ControlError, ExitCode
from caribou.control.records import SlurmExecutionHandle
from caribou.control.service import ExperimentService
from caribou.control.slurm import SlurmExecutor, _render_script
from caribou.control.specs import ADAPTER_PARAMETER, LOCAL_LIFECYCLE_ADAPTER
from caribou.control.store import ExperimentStore
from caribou.control.worker import execute
from caribou.domain.enums import ExecutorKind, RunState
from caribou.domain.models import ExperimentSpec
from caribou.domain.serialization import sha256_bytes, write_model

from .test_domain_models import make_spec


def _completed(
    command: list[str], *, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=stderr)


def _slurm_store(
    tmp_path: Path,
    *,
    idempotency_key: str = "slurm-unit",
    partition: str = "peerd",
) -> tuple[ExperimentStore, str]:
    base = make_spec()
    condition = base.conditions[0].model_copy(
        update={
            "parameters": {
                ADAPTER_PARAMETER: LOCAL_LIFECYCLE_ADAPTER,
                "caribou.lifecycle_smoke_seconds": 0.0,
            }
        }
    )
    resources = base.execution.resources.model_copy(
        update={
            "cpu_cores": 3,
            "memory_bytes": 3 * 1024**3,
            "wall_seconds": 3661,
        }
    )
    execution = base.execution.model_copy(
        update={
            "executor": ExecutorKind.slurm,
            "partition": partition,
            "resources": resources,
        }
    )
    spec = ExperimentSpec.model_validate_json(
        base.model_copy(
            update={
                "conditions": [condition],
                "execution": execution,
                "repetitions": 1,
            }
        ).model_dump_json()
    )
    store = ExperimentStore(tmp_path / "store")
    run_id = store.submit(spec, idempotency_key).runs[0].run_id
    return store, run_id


def _bind_job(
    store: ExperimentStore,
    run_id: str,
    *,
    job_id: str = "742",
    released: bool = True,
) -> SlurmExecutionHandle:
    script_path, script_hash = store.write_scheduler_script(
        run_id, "#!/bin/bash\nexit 0\n"
    )
    handle = SlurmExecutionHandle(
        run_id=run_id,
        job_id=job_id,
        partition=store.run(run_id).partition,
        script_path=str(script_path.relative_to(store.run_dir(run_id))),
        script_hash=script_hash,
        stdout_path="slurm-%j.out",
    )
    store.bind_scheduler_job(handle)
    return store.mark_scheduler_released(run_id) if released else handle


def _write_partial_handle(
    store: ExperimentStore, run_id: str, *, job_id: str = "742"
) -> SlurmExecutionHandle:
    run = store.run(run_id)
    script_path, script_hash = store.write_scheduler_script(
        run_id, _render_script(store, run)
    )
    partial = SlurmExecutionHandle(
        run_id=run_id,
        job_id=job_id,
        partition=run.partition,
        script_path=str(script_path.relative_to(store.run_dir(run_id))),
        script_hash=script_hash,
        stdout_path="slurm-%j.out",
    )
    write_model(store.scheduler_handle_path(run_id), partial)
    return partial


def _terminal_sacct(
    job_id: str, state: str, exit_code: str, *, partition: str = "peerd"
) -> str:
    root = (
        f"{job_id}|{state}|{exit_code}|12|3|3072M||node-a|"
        f"2026-07-14T01:00:00|2026-07-14T01:00:12|{partition}"
    )
    batch = (
        f"{job_id}.batch|{state}|{exit_code}|12|3|3072M|2048K|node-a|"
        f"2026-07-14T01:00:00|2026-07-14T01:00:12|{partition}"
    )
    return f"{root}\n{batch}\n"


def _terminal_runner(raw: str) -> tuple[list[list[str]], Callable[[list[str]], subprocess.CompletedProcess[str]]]:
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            return _completed(command)
        if command[0] == "sacct":
            return _completed(command, stdout=raw)
        if command[0] == "scancel":
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    return commands, run


def test_command_timeout_map_is_copied_and_resolved_per_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = {"squeue": 91.0, "sbatch": 22.0}
    executor = SlurmExecutor(command_timeouts=configured)
    configured["squeue"] = 1.0
    observed: list[tuple[list[str], float]] = []

    def run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
        env: object,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert capture_output is True
        assert text is True
        observed.append((command, timeout))
        return _completed(command)

    monkeypatch.setattr(control_slurm_module.subprocess, "run", run)

    executor._run(["squeue"])
    executor._run(["/usr/bin/sacct"])
    executor._run(["sbatch"])
    executor._run(["scontrol"])
    executor._run(["custom-slurm-command"], timeout=2.0)

    custom_executor = SlurmExecutor(
        squeue="site-queue-client",
        sacct="/site/bin/accounting-client",
    )
    custom_executor._run(["site-queue-client"])
    custom_executor._run(["/site/bin/accounting-client"])

    assert observed == [
        (["squeue"], 91.0),
        (["/usr/bin/sacct"], 75.0),
        (["sbatch"], 22.0),
        (["scontrol"], 15.0),
        (["custom-slurm-command"], 2.0),
        (["site-queue-client"], 75.0),
        (["/site/bin/accounting-client"], 75.0),
    ]


def test_command_timeout_error_reports_effective_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = SlurmExecutor(command_timeouts={"squeue": 91.0})

    def timeout(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 91.0)

    monkeypatch.setattr(control_slurm_module.subprocess, "run", timeout)

    with pytest.raises(ControlError) as exc_info:
        executor._run(["squeue", "--local"])

    assert exc_info.value.code == "SLURM_COMMAND_TIMEOUT"
    assert exc_info.value.retryable is True
    assert exc_info.value.details == {
        "command": "squeue",
        "timeout_seconds": 91.0,
    }


def test_pre_submission_squeue_timeout_leaves_run_retryable_without_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="query-timeout")
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command[0:2] == ["squeue", "--local"]
        raise ControlError(
            "SLURM_COMMAND_TIMEOUT",
            "squeue timed out",
            exit_code=ExitCode.transient,
            retryable=True,
            details={"command": "squeue", "timeout_seconds": 75.0},
        )

    monkeypatch.setattr(executor, "_run", run)

    with pytest.raises(ControlError) as exc_info:
        executor.launch(store, run_id)

    assert exc_info.value.code == "SLURM_COMMAND_TIMEOUT"
    assert [command[0] for command in commands] == ["squeue"]
    assert store.run(run_id).state == RunState.queued
    assert store.run(run_id).scheduler_job_id is None
    assert store.scheduler_submission(run_id) is None
    assert store.scheduler_handle(run_id) is None


def test_held_submit_is_bound_before_release_and_script_is_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "never-copy-openai-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "never-copy-deepseek-secret")
    store, run_id = _slurm_store(tmp_path)
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            assert command == [
                "squeue",
                "--local",
                "--noheader",
                "--user",
                str(os.geteuid()),
                "--name",
                f"caribou_{run_id}",
                "--states=PENDING",
                "--format=%i|%j|%P|%r|%U",
            ]
            return _completed(command)
        if command[0] == "sbatch":
            assert command[1:5] == [
                "--parsable",
                "--hold",
                "--partition=peerd",
                "--export=NIL",
            ]
            assert len(command) == 6
            assert store.scheduler_handle(run_id) is None
            assert store.run(run_id).scheduler_job_id is None
            return _completed(command, stdout="742;cluster\n")
        if command == ["scontrol", "release", "742"]:
            durable = store.scheduler_handle(run_id)
            assert durable is not None
            assert durable.job_id == "742"
            assert durable.released_at is None
            assert store.run(run_id).scheduler_job_id == "742"
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run)
    launched = executor.launch(store, run_id)

    assert launched.launched is True
    assert [command[0] for command in commands] == [
        "squeue",
        "sbatch",
        "scontrol",
    ]
    assert launched.handle.job_id == "742"
    assert launched.handle.released_at is not None

    script = store.scheduler_script_path(run_id).read_text(encoding="utf-8")
    assert "#SBATCH --partition=peerd" in script
    assert "#SBATCH --cpus-per-task=3" in script
    assert "#SBATCH --mem=3072M" in script
    assert "#SBATCH --time=01:01:01" in script
    assert "#SBATCH --export=NIL" in script
    assert "--export=NONE" not in script
    assert 'export PATH="/usr/bin:/bin"' in script
    assert '/usr/bin/mkdir -p "$TMPDIR"' in script
    assert "if (( $# != 0 )); then" in script
    assert "this generated Slurm worker accepts no arguments" in script
    assert "-m caribou.control.worker" in script
    assert f"--store-root {store.root}" in script
    assert f"--run-id {run_id}" in script
    submitted_text = "\n".join(" ".join(command) for command in commands)
    for forbidden in (
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "never-copy-openai-secret",
        "never-copy-deepseek-secret",
    ):
        assert forbidden not in script
        assert forbidden not in submitted_text
    assert "--export=NONE" not in submitted_text
    assert "--export=NIL" in submitted_text


def test_first_submit_release_timeout_accepts_observed_nonheld_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="release-ambiguity")
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue" and "--name" in command:
            return _completed(command)
        if command[0] == "sbatch":
            return _completed(command, stdout="742;cluster\n")
        if command == ["scontrol", "release", "742"]:
            assert store.run(run_id).scheduler_job_id == "742"
            raise ControlError(
                "SLURM_COMMAND_TIMEOUT",
                "release response was lost",
                exit_code=ExitCode.transient,
                retryable=True,
            )
        if command[0] == "squeue" and "--jobs" in command:
            return _completed(
                command,
                stdout="742|PENDING|peerd|(null)|00:00|3|3072M|Priority\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run)
    result = executor.launch(store, run_id)

    assert result.launched is True
    assert result.handle.job_id == "742"
    assert result.handle.released_at is not None
    assert store.run(run_id).state == RunState.queued
    assert store.run(run_id).scheduler_job_id == "742"
    assert [command[0] for command in commands] == [
        "squeue",
        "sbatch",
        "scontrol",
        "squeue",
    ]
    assert all(command[0] != "scancel" for command in commands)


def test_first_submit_release_failure_cleans_up_still_held_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="release-held")
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue" and "--name" in command:
            return _completed(command)
        if command[0] == "sbatch":
            return _completed(command, stdout="742;cluster\n")
        if command == ["scontrol", "release", "742"]:
            raise ControlError(
                "SLURM_COMMAND_TIMEOUT",
                "release response was lost",
                exit_code=ExitCode.transient,
                retryable=True,
            )
        if command[0] == "squeue" and "--jobs" in command:
            return _completed(
                command,
                stdout=(
                    "742|PENDING|peerd|(null)|00:00|3|3072M|JobHeldUser\n"
                ),
            )
        if command == ["scancel", "742"]:
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run)
    with pytest.raises(ControlError) as exc_info:
        executor.launch(store, run_id)

    assert exc_info.value.code == "SLURM_COMMAND_TIMEOUT"
    assert store.run(run_id).state == RunState.failed
    assert store.run(run_id).scheduler_job_id == "742"
    handle = store.scheduler_handle(run_id)
    assert handle is not None
    assert handle.released_at is None
    assert [command[0] for command in commands] == [
        "squeue",
        "sbatch",
        "scontrol",
        "squeue",
        "scancel",
    ]


def test_failed_held_cleanup_is_durable_and_cancel_retries_exact_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="cleanup-retry")
    spec = store.spec(store.run(run_id).experiment_id)
    executor = SlurmExecutor()
    service = ExperimentService(store=store, slurm_executor=executor)
    commands: list[list[str]] = []
    scancel_attempts = 0

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal scancel_attempts
        commands.append(command)
        if command[0] == "squeue" and "--name" in command:
            return _completed(command)
        if command[0] == "sbatch":
            return _completed(command, stdout="742;cluster\n")
        if command == ["scontrol", "release", "742"]:
            raise ControlError(
                "SLURM_COMMAND_TIMEOUT",
                "release response was lost",
                exit_code=ExitCode.transient,
                retryable=True,
            )
        if command[0] == "squeue" and "--jobs" in command:
            return _completed(
                command,
                stdout=(
                    "742|PENDING|peerd|(null)|00:00|3|3072M|JobHeldUser\n"
                ),
            )
        if command == ["scancel", "742"]:
            scancel_attempts += 1
            if scancel_attempts == 1:
                raise ControlError(
                    "SLURM_COMMAND_FAILED",
                    "first cleanup scancel failed",
                    exit_code=ExitCode.transient,
                    retryable=True,
                )
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run)
    with pytest.raises(ControlError) as exc_info:
        service.submit(spec, idempotency_key="cleanup-retry")

    assert exc_info.value.code == "SLURM_LAUNCH_CLEANUP_FAILED"
    assert exc_info.value.retryable is True
    after_launch = store.run(run_id)
    assert after_launch.state == RunState.cancelling
    assert after_launch.scheduler_job_id == "742"
    first_ledger = store.scheduler_cancellation(run_id)
    assert first_ledger is not None
    assert len(first_ledger.attempts) == 1
    assert first_ledger.attempts[0].succeeded is False
    assert first_ledger.attempts[0].error_code == "SLURM_COMMAND_FAILED"
    assert commands[-1] == ["scancel", "742"]

    retried = service.submit(
        spec,
        idempotency_key="cleanup-retry",
    )

    assert retried.submission.idempotent_replay is True
    assert retried.launches == ()
    retried_run = retried.submission.runs[0]
    assert retried_run.run_id == run_id
    assert retried_run.state == RunState.cancelling
    assert retried_run.scheduler_job_id == "742"
    assert commands[-1] == ["scancel", "742"]
    assert scancel_attempts == 2
    assert sum(command[0] == "sbatch" for command in commands) == 1
    ledger = store.scheduler_cancellation(run_id)
    assert ledger is not None
    assert [attempt.succeeded for attempt in ledger.attempts] == [False, True]
    assert ledger.attempts[0].error_code == "SLURM_COMMAND_FAILED"
    assert ledger.attempts[1].error_code is None


def test_partial_handle_before_journal_binding_recovers_without_resubmit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="partial-binding")
    _write_partial_handle(store, run_id)
    assert store.run(run_id).scheduler_job_id is None

    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command == ["scontrol", "release", "742"]
        assert store.run(run_id).scheduler_job_id == "742"
        return _completed(command)

    monkeypatch.setattr(executor, "_run", run_command)
    result = executor.launch(store, run_id)

    assert result.launched is False
    assert result.handle.job_id == "742"
    assert result.handle.released_at is not None
    assert store.run(run_id).scheduler_job_id == "742"
    assert commands == [["scontrol", "release", "742"]]


def test_partial_handle_cancel_binds_job_then_closes_and_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="partial-cancel")
    _write_partial_handle(store, run_id, job_id="742")
    assert store.run(run_id).scheduler_job_id is None
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command == ["scancel", "742"]
        return _completed(command)

    monkeypatch.setattr(executor, "_run", run)
    service = ExperimentService(store=store, slurm_executor=executor)
    result = service.cancel(
        run_id, reason="cancel partial scheduler submission"
    )

    assert result.applied is True
    assert result.scheduler_signalled is True
    assert result.run.state == RunState.cancelled
    assert result.run.scheduler_job_id == "742"
    assert commands == [["scancel", "742"]]
    ledger = store.scheduler_cancellation(run_id)
    assert ledger is not None
    assert len(ledger.attempts) == 1
    assert ledger.attempts[0].succeeded is True

    raw = _terminal_sacct("742", "CANCELLED", "0:15")
    scheduler_commands, terminal_run = _terminal_runner(raw)
    monkeypatch.setattr(executor, "_run", terminal_run)
    reconciled = service.reconcile_scheduler(run_id)
    assert reconciled.run.state == RunState.cancelled
    assert reconciled.run_transition_applied is False
    assert reconciled.accounting_created is True
    assert reconciled.accounting is not None
    assert reconciled.accounting.consistent_with_run is True
    assert [command[0] for command in scheduler_commands] == ["squeue", "sacct"]


def test_cancel_without_handle_retries_named_lookup_and_then_cancels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="lookup-cancel")
    run = store.run(run_id)
    _, script_hash = store.write_scheduler_script(run_id, _render_script(store, run))
    with store.mutation_lock():
        store._record_scheduler_submission_attempt_unlocked(
            run_id=run_id,
            job_name=f"caribou_{run_id}",
            script_hash=script_hash,
        )
    monkeypatch.setattr(
        control_slurm_module, "_SUBMISSION_VISIBILITY_GRACE_SECONDS", 30.0
    )
    executor = SlurmExecutor()
    commands: list[list[str]] = []
    lookups = 0
    recovery_command = [
        "squeue",
        "--local",
        "--noheader",
        "--user",
        str(os.geteuid()),
        "--name",
        f"caribou_{run_id}",
        "--states=PENDING",
        "--format=%i|%j|%P|%r|%U",
    ]

    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal lookups
        commands.append(command)
        if command[0] == "squeue":
            assert command == recovery_command
            lookups += 1
            if lookups == 1:
                return _completed(command)
            return _completed(
                command,
                stdout=(
                    f"742|caribou_{run_id}|peerd|JobHeldUser|{os.geteuid()}\n"
                ),
            )
        if command == ["scancel", "742"]:
            assert store.run(run_id).scheduler_job_id == "742"
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run_command)
    service = ExperimentService(store=store, slurm_executor=executor)

    first = service.cancel(run_id, reason="cancel while sbatch visibility is delayed")
    assert first.applied is True
    assert first.scheduler_signalled is False
    assert first.run.state == RunState.cancelling
    assert first.run.scheduler_job_id is None
    assert store.scheduler_handle(run_id) is None

    retried = service.cancel(run_id, reason="retry delayed scheduler lookup")
    assert retried.applied is False
    assert retried.scheduler_signalled is True
    assert retried.run.state == RunState.cancelled
    assert retried.run.scheduler_job_id == "742"
    assert [command[0] for command in commands] == [
        "squeue",
        "squeue",
        "scancel",
    ]
    ledger = store.scheduler_cancellation(run_id)
    assert ledger is not None
    assert len(ledger.attempts) == 1
    assert ledger.attempts[0].succeeded is True


def test_submit_replay_never_duplicates_sbatch_during_visibility_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="visibility-grace")
    spec = store.spec(store.run(run_id).experiment_id)
    executor = SlurmExecutor()
    service = ExperimentService(store=store, slurm_executor=executor)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        control_slurm_module, "_SUBMISSION_VISIBILITY_GRACE_SECONDS", 30.0
    )

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            return _completed(command)
        if command[0] == "sbatch":
            raise ControlError(
                "SLURM_COMMAND_TIMEOUT",
                "sbatch response was lost",
                exit_code=ExitCode.transient,
                retryable=True,
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run)
    for _ in range(2):
        with pytest.raises(ControlError) as exc_info:
            service.submit(spec, idempotency_key="visibility-grace")
        assert exc_info.value.code == "SLURM_SUBMISSION_UNCERTAIN"
        assert exc_info.value.retryable is True

    assert sum(command[0] == "sbatch" for command in commands) == 1
    assert [command[0] for command in commands] == [
        "squeue",
        "sbatch",
        "squeue",
        "squeue",
    ]
    ledger = store.scheduler_submission(run_id)
    assert ledger is not None
    assert ledger.job_name == f"caribou_{run_id}"
    assert len(ledger.attempts) == 1
    assert store.scheduler_handle(run_id) is None
    assert store.run(run_id).state == RunState.queued


def test_cancel_without_visible_job_closes_after_zero_visibility_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="expired-grace")
    run = store.run(run_id)
    _, script_hash = store.write_scheduler_script(
        run_id, _render_script(store, run)
    )
    with store.mutation_lock():
        store._record_scheduler_submission_attempt_unlocked(
            run_id=run_id,
            job_name=f"caribou_{run_id}",
            script_hash=script_hash,
        )
    monkeypatch.setattr(
        control_slurm_module, "_SUBMISSION_VISIBILITY_GRACE_SECONDS", 0.0
    )
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command[0] == "squeue"
        return _completed(command)

    monkeypatch.setattr(executor, "_run", run_command)
    result = ExperimentService(store=store, slurm_executor=executor).cancel(
        run_id, reason="cancel after visibility grace"
    )

    assert result.applied is True
    assert result.scheduler_signalled is False
    assert result.run.state == RunState.cancelled
    assert result.run.scheduler_job_id is None
    assert store.scheduler_handle(run_id) is None
    assert [command[0] for command in commands] == ["squeue"]


def test_launch_recovers_one_named_held_job_without_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="named-recovery")
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            assert command == [
                "squeue",
                "--local",
                "--noheader",
                "--user",
                str(os.geteuid()),
                "--name",
                f"caribou_{run_id}",
                "--states=PENDING",
                "--format=%i|%j|%P|%r|%U",
            ]
            return _completed(
                command,
                stdout=(
                    f"742|caribou_{run_id}|peerd|JobHeldUser|{os.geteuid()}\n"
                ),
            )
        if command == ["scontrol", "release", "742"]:
            assert store.run(run_id).scheduler_job_id == "742"
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run)
    result = executor.launch(store, run_id)

    assert result.launched is True
    assert result.handle.job_id == "742"
    assert result.handle.released_at is not None
    assert store.run(run_id).scheduler_job_id == "742"
    assert [command[0] for command in commands] == ["squeue", "scontrol"]


@pytest.mark.parametrize(
    ("candidate_rows", "expected_code"),
    [
        (
            [
                "742|{name}|peerd|JobHeldUser|{uid}",
                "743|{name}|peerd|JobHeldUser|{uid}",
            ],
            "SLURM_SUBMISSION_AMBIGUOUS",
        ),
        (
            ["742|{name}|another-partition|JobHeldUser|{uid}"],
            "SLURM_RECOVERY_MISMATCH",
        ),
        (
            ["742|{name}|peerd|Priority|{uid}"],
            "SLURM_RECOVERY_MISMATCH",
        ),
        (
            ["742|{name}|peerd|JobHeldUser|{wrong_uid}"],
            "SLURM_RECOVERY_MISMATCH",
        ),
    ],
)
def test_named_recovery_multiple_or_mismatched_candidates_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_rows: list[str],
    expected_code: str,
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="invalid-recovery")
    executor = SlurmExecutor()
    commands: list[list[str]] = []
    job_name = f"caribou_{run_id}"

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command[0] == "squeue"
        return _completed(
            command,
            stdout="\n".join(
                row.format(
                    name=job_name,
                    uid=os.geteuid(),
                    wrong_uid=os.geteuid() + 1,
                )
                for row in candidate_rows
            )
            + "\n",
        )

    monkeypatch.setattr(executor, "_run", run)
    with pytest.raises(ControlError) as exc_info:
        executor.launch(store, run_id)

    assert exc_info.value.code == expected_code
    assert [command[0] for command in commands] == ["squeue"]
    assert store.scheduler_handle(run_id) is None
    assert store.run(run_id).scheduler_job_id is None
    assert store.run(run_id).state == RunState.failed


def test_worker_accepts_exact_slurm_context_and_rejects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted_store, accepted_run_id = _slurm_store(
        tmp_path / "accepted", idempotency_key="accepted"
    )
    _bind_job(accepted_store, accepted_run_id, job_id="742")
    monkeypatch.setenv("SLURM_JOB_ID", "742")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "peerd")

    assert execute(accepted_store, accepted_run_id) == 0
    assert accepted_store.run(accepted_run_id).state == RunState.succeeded

    rejected_store, rejected_run_id = _slurm_store(
        tmp_path / "rejected", idempotency_key="rejected"
    )
    _bind_job(rejected_store, rejected_run_id, job_id="743")
    monkeypatch.setenv("SLURM_JOB_ID", "wrong")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "peerd")

    assert execute(rejected_store, rejected_run_id) == int(ExitCode.execution)
    rejected = rejected_store.run(rejected_run_id)
    assert rejected.state == RunState.failed
    assert rejected.end_reason == "worker failure: SLURM_CONTEXT_MISMATCH"

    partition_store, partition_run_id = _slurm_store(
        tmp_path / "wrong-partition", idempotency_key="wrong-partition"
    )
    _bind_job(partition_store, partition_run_id, job_id="744")
    monkeypatch.setenv("SLURM_JOB_ID", "744")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "another-partition")

    assert execute(partition_store, partition_run_id) == int(ExitCode.execution)
    partition_rejected = partition_store.run(partition_run_id)
    assert partition_rejected.state == RunState.failed
    assert partition_rejected.end_reason == (
        "worker failure: SLURM_CONTEXT_MISMATCH"
    )


def test_completed_accounting_is_hashed_consistent_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path)
    _bind_job(store, run_id, job_id="742")
    monkeypatch.setenv("SLURM_JOB_ID", "742")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "peerd")
    assert execute(store, run_id) == 0

    raw = _terminal_sacct("742", "COMPLETED", "0:0")
    commands, run = _terminal_runner(raw)
    executor = SlurmExecutor()
    monkeypatch.setattr(executor, "_run", run)

    first = executor.reconcile(store, run_id)
    assert first.run.state == RunState.succeeded
    assert first.run_transition_applied is False
    assert first.accounting_created is True
    assert first.accounting is not None
    assert first.accounting.consistent_with_run is True
    assert first.accounting.raw_output_hash == sha256_bytes(raw.encode("utf-8"))
    assert store.scheduler_accounting_raw_path(run_id).read_text() == raw
    assert [command[0] for command in commands] == ["squeue", "sacct"]

    second = executor.reconcile(store, run_id)
    assert second.accounting_created is False
    assert second.run_transition_applied is False
    assert second.observation.source == "durable-accounting"
    assert second.accounting == first.accounting
    assert [command[0] for command in commands] == ["squeue", "sacct"]

    service = ExperimentService(store=store)
    artifacts = service.artifacts(run_id)
    scheduler_artifacts = {
        artifact.role: artifact
        for artifact in artifacts
        if artifact.role
        in {"slurm_accounting", "slurm_accounting_raw", "slurm_job_script"}
    }
    assert set(scheduler_artifacts) == {
        "slurm_accounting",
        "slurm_accounting_raw",
        "slurm_job_script",
    }
    for role in scheduler_artifacts:
        assert [artifact.role for artifact in artifacts].count(role) == 1
    assert scheduler_artifacts["slurm_accounting_raw"].content_hash == (
        first.accounting.raw_output_hash
    )
    handle = store.scheduler_handle(run_id)
    assert handle is not None
    assert scheduler_artifacts["slurm_job_script"].content_hash == handle.script_hash
    verified_roles = {artifact.role for artifact in service.verify_artifacts(run_id)}
    assert set(scheduler_artifacts) <= verified_roles
    for role, artifact in scheduler_artifacts.items():
        _, output = service.fetch_artifact(
            run_id,
            artifact.artifact_id,
            tmp_path / f"fetched-{role}",
        )
        assert sha256_bytes(output.read_bytes()) == artifact.content_hash

    store.scheduler_accounting_raw_path(run_id).write_text(
        "tampered durable accounting\n", encoding="utf-8"
    )
    with pytest.raises(ControlError) as exc_info:
        store.scheduler_accounting(run_id)
    assert exc_info.value.code == "SCHEDULER_ACCOUNTING_TAMPERED"

    raw_artifact = scheduler_artifacts["slurm_accounting_raw"]
    store.artifact_path(raw_artifact).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ControlError) as exc_info:
        service.verify_artifacts(run_id)
    assert exc_info.value.code == "ARTIFACT_INTEGRITY_ERROR"


def test_terminal_squeue_row_is_followed_by_authoritative_sacct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path)
    _bind_job(store, run_id, job_id="742")
    raw = _terminal_sacct("742", "COMPLETED", "0:0")
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            return _completed(
                command,
                stdout="742|COMPLETED|peerd|node-a|00:12|3|3072M|None\n",
            )
        if command[0] == "sacct":
            return _completed(command, stdout=raw)
        raise AssertionError(f"unexpected command: {command}")

    executor = SlurmExecutor()
    monkeypatch.setattr(executor, "_run", run)

    observed = executor.inspect(store, run_id)

    assert observed.source == "sacct"
    assert observed.state == "COMPLETED"
    assert observed.exit_code == "0:0"
    assert [command[0] for command in commands] == ["squeue", "sacct"]


def test_squeue_invalid_job_id_falls_back_to_sacct_during_inspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path)
    _bind_job(store, run_id, job_id="742")
    raw = _terminal_sacct("742", "COMPLETED", "0:0")
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="slurm_load_jobs error: Invalid job id specified\n",
            )
        if command[0] == "sacct":
            return _completed(command, stdout=raw)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(control_slurm_module.subprocess, "run", run)

    observed = SlurmExecutor().inspect(store, run_id)

    assert observed.source == "sacct"
    assert observed.state == "COMPLETED"
    assert observed.exit_code == "0:0"
    assert [command[0] for command in commands] == ["squeue", "sacct"]


def test_squeue_invalid_job_id_falls_back_to_sacct_during_reconcile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path)
    _bind_job(store, run_id, job_id="742")
    raw = _terminal_sacct("742", "FAILED", "1:0")
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="slurm_load_jobs error: Invalid job id specified\n",
            )
        if command[0] == "sacct":
            return _completed(command, stdout=raw)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(control_slurm_module.subprocess, "run", run)

    result = SlurmExecutor().reconcile(store, run_id)

    assert result.observation.source == "sacct"
    assert result.run_transition_applied is True
    assert result.run.state == RunState.failed
    assert result.accounting_created is True
    assert [command[0] for command in commands] == ["squeue", "sacct"]


@pytest.mark.parametrize("operation", ["inspect", "reconcile"])
def test_unrelated_squeue_failure_does_not_fall_back_to_sacct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    store, run_id = _slurm_store(tmp_path)
    _bind_job(store, run_id, job_id="742")
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command[0] == "squeue"
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="slurm_load_jobs error: Unable to contact slurm controller\n",
        )

    monkeypatch.setattr(control_slurm_module.subprocess, "run", run)
    executor = SlurmExecutor()

    with pytest.raises(ControlError) as exc_info:
        getattr(executor, operation)(store, run_id)

    assert exc_info.value.code == "SLURM_COMMAND_FAILED"
    assert exc_info.value.retryable is True
    assert [command[0] for command in commands] == ["squeue"]


def test_scheduler_artifact_retry_repairs_manifest_journal_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path)
    _bind_job(store, run_id, job_id="742")
    monkeypatch.setenv("SLURM_JOB_ID", "742")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "peerd")
    assert execute(store, run_id) == 0

    raw = _terminal_sacct("742", "COMPLETED", "0:0")
    commands, run = _terminal_runner(raw)
    executor = SlurmExecutor()
    monkeypatch.setattr(executor, "_run", run)
    original_commit = control_store_module.commit_run_event
    injected = False

    def fail_after_manifest(*args: object, **kwargs: object) -> str:
        nonlocal injected
        if not injected:
            injected = True
            raise RuntimeError("injected stop after artifact manifest write")
        return original_commit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        control_store_module, "commit_run_event", fail_after_manifest
    )
    with pytest.raises(
        RuntimeError, match="injected stop after artifact manifest write"
    ):
        executor.reconcile(store, run_id)

    partial_artifacts = store.artifact_manifest(run_id).artifacts
    accounting_partial = [
        artifact
        for artifact in partial_artifacts
        if artifact.role == "slurm_accounting"
    ]
    assert len(accounting_partial) == 1
    partial = accounting_partial[0]
    assert partial.artifact_id not in store.run(run_id).artifact_ids
    assert all(
        event.event_id != partial.producer_event_id for event in store.events(run_id)
    )
    assert store.scheduler_accounting(run_id) is not None

    monkeypatch.setattr(control_store_module, "commit_run_event", original_commit)
    retried = executor.reconcile(store, run_id)

    assert retried.accounting_created is False
    artifacts = store.artifact_manifest(run_id).artifacts
    assert [artifact.role for artifact in artifacts].count("slurm_accounting") == 1
    assert [artifact.role for artifact in artifacts].count("slurm_accounting_raw") == 1
    repaired = next(
        artifact for artifact in artifacts if artifact.role == "slurm_accounting"
    )
    assert repaired.artifact_id == partial.artifact_id
    run_after_retry = store.run(run_id)
    assert run_after_retry.artifact_ids.count(repaired.artifact_id) == 1
    producer_events = [
        event
        for event in store.events(run_id)
        if event.event_id == repaired.producer_event_id
    ]
    assert len(producer_events) == 1
    assert producer_events[0].actor == "slurm-reconciler"
    assert commands == [
        [
            "squeue",
            "--local",
            "--noheader",
            "--jobs",
            "742",
            "--format=%i|%T|%P|%N|%M|%C|%m|%r",
        ],
        [
            "sacct",
            "--local",
            "--jobs",
            "742",
            "--noheader",
            "--parsable2",
            "--units=K",
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,NodeList,Start,End,Partition",
        ],
    ]


@pytest.mark.parametrize(
    ("raw", "expected_code"),
    [
        (
            (
                "742|COMPLETED|0:0|not-an-integer|3|3072M||node-a|"
                "2026-07-14T01:00:00|2026-07-14T01:00:12|peerd\n"
            ),
            "SLURM_ACCOUNTING_INVALID",
        ),
        (
            _terminal_sacct("742", "COMPLETED", "0:0").replace(
                "|peerd", "|other"
            ),
            "SLURM_PARTITION_MISMATCH",
        ),
    ],
)
def test_malformed_or_wrong_partition_accounting_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected_code: str,
) -> None:
    store, run_id = _slurm_store(tmp_path)
    _bind_job(store, run_id, job_id="742")
    _, run = _terminal_runner(raw)
    executor = SlurmExecutor()
    monkeypatch.setattr(executor, "_run", run)

    with pytest.raises(ControlError) as exc_info:
        executor.reconcile(store, run_id)

    assert exc_info.value.code == expected_code
    assert store.scheduler_accounting(run_id) is None
    assert store.run(run_id).state == RunState.queued


def test_pre_worker_scheduler_failure_closes_run_and_records_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path)
    _bind_job(store, run_id, job_id="742")
    raw = _terminal_sacct("742", "FAILED", "1:0")
    _, run = _terminal_runner(raw)
    executor = SlurmExecutor()
    monkeypatch.setattr(executor, "_run", run)

    result = executor.reconcile(store, run_id)

    assert result.run_transition_applied is True
    assert result.run.state == RunState.failed
    assert result.run.end_reason == (
        "Slurm job 742 reached FAILED before the worker recorded a terminal outcome"
    )
    assert result.accounting_created is True
    assert result.accounting is not None
    assert result.accounting.consistent_with_run is True


def test_cancel_signals_exact_bound_job_and_reconciles_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path)
    _bind_job(store, run_id, job_id="742")
    raw = _terminal_sacct("742", "CANCELLED", "0:15")
    commands, run = _terminal_runner(raw)
    executor = SlurmExecutor()
    monkeypatch.setattr(executor, "_run", run)
    service = ExperimentService(store=store, slurm_executor=executor)

    cancelled = service.cancel(run_id, reason="unit cancellation")
    assert cancelled.applied is True
    assert cancelled.scheduler_signalled is True
    assert cancelled.run.state == RunState.cancelling
    assert commands == [["scancel", "742"]]

    reconciled = service.reconcile_scheduler(run_id)
    assert reconciled.run.state == RunState.cancelled
    assert reconciled.run_transition_applied is True
    assert reconciled.accounting_created is True
    assert reconciled.accounting is not None
    assert reconciled.accounting.consistent_with_run is True
    assert commands[1][0] == "squeue"
    assert commands[1][commands[1].index("--jobs") + 1] == "742"
    assert commands[2][0] == "sacct"
    assert commands[2][commands[2].index("--jobs") + 1] == "742"


def test_duplicate_and_stale_cancel_do_not_repeat_scancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path / "duplicate")
    _bind_job(store, run_id, job_id="742")
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command == ["scancel", "742"]
        return _completed(command)

    executor = SlurmExecutor()
    monkeypatch.setattr(executor, "_run", run)
    service = ExperimentService(store=store, slurm_executor=executor)

    first = service.cancel(run_id, reason="first")
    duplicate = service.cancel(run_id, reason="duplicate")
    assert first.applied is True
    assert first.scheduler_signalled is True
    assert duplicate.applied is False
    assert duplicate.scheduler_signalled is False
    assert commands == [["scancel", "742"]]

    completed_store, completed_run_id = _slurm_store(tmp_path / "terminal")
    _bind_job(completed_store, completed_run_id, job_id="743")
    monkeypatch.setenv("SLURM_JOB_ID", "743")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "peerd")
    assert execute(completed_store, completed_run_id) == 0
    completed_service = ExperimentService(
        store=completed_store, slurm_executor=executor
    )

    stale = completed_service.cancel(completed_run_id, reason="too late")
    assert stale.applied is False
    assert stale.scheduler_signalled is False
    assert commands == [["scancel", "742"]]


# --- Configurable partition: new work vs. already-submitted jobs -----------


def test_non_default_partition_flows_into_script_and_sbatch_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", "gpu_a100")
    store, run_id = _slurm_store(
        tmp_path, idempotency_key="gpu-partition", partition="gpu_a100"
    )
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            return _completed(command)
        if command[0] == "sbatch":
            return _completed(command, stdout="742;cluster\n")
        if command == ["scontrol", "release", "742"]:
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run)
    launched = executor.launch(store, run_id)

    assert launched.launched is True
    assert launched.handle.partition == "gpu_a100"
    sbatch = next(command for command in commands if command[0] == "sbatch")
    assert "--partition=gpu_a100" in sbatch
    assert not any(arg.startswith("--partition=peerd") for arg in sbatch)
    script = store.scheduler_script_path(run_id).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu_a100" in script
    assert "--partition=peerd" not in script


def test_new_submission_refuses_spec_partition_other_than_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", "gpu_a100")
    with pytest.raises(ControlError) as exc_info:
        _slurm_store(tmp_path, idempotency_key="spec-mismatch", partition="peerd")
    assert exc_info.value.code == "SLURM_PARTITION_MISMATCH"
    assert exc_info.value.details["partition"] == "peerd"
    assert exc_info.value.details["configured_partition"] == "gpu_a100"


def test_new_submission_requires_a_configured_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CARIBOU_SLURM_PARTITION")
    with pytest.raises(ControlError) as exc_info:
        _slurm_store(tmp_path, idempotency_key="unconfigured")
    assert exc_info.value.code == "SLURM_PARTITION_NOT_CONFIGURED"
    assert "caribou config set-slurm-partition" in exc_info.value.message


def test_launch_refuses_new_sbatch_after_partition_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="switch-before-launch")
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", "gpu_a100")
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run)
    with pytest.raises(ControlError) as exc_info:
        executor.launch(store, run_id)

    assert exc_info.value.code == "SLURM_PARTITION_MISMATCH"
    assert exc_info.value.details["partition"] == "peerd"
    assert exc_info.value.details["configured_partition"] == "gpu_a100"
    assert [command[0] for command in commands] == ["squeue"]
    assert store.scheduler_submission(run_id) is None
    assert store.scheduler_handle(run_id) is None
    failed = store.run(run_id)
    assert failed.scheduler_job_id is None
    assert failed.state == RunState.failed
    assert failed.end_reason == "Slurm submission failure: SLURM_PARTITION_MISMATCH"


def test_records_from_partition_a_still_load_after_switch_to_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="reload-after-switch")
    handle = _bind_job(store, run_id, job_id="742")
    experiment_id = store.run(run_id).experiment_id
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", "gpu_a100")

    reopened = ExperimentStore(store.root)
    assert reopened.run(run_id).partition == "peerd"
    assert reopened.spec(experiment_id).execution.partition == "peerd"
    loaded = reopened.scheduler_handle(run_id)
    assert loaded == handle
    assert loaded.partition == "peerd"


def test_status_cancel_and_accounting_use_recorded_partition_after_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="cancel-after-switch")
    _bind_job(store, run_id, job_id="742")
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", "gpu_a100")
    raw = _terminal_sacct("742", "CANCELLED", "0:15", partition="peerd")
    commands, run = _terminal_runner(raw)
    executor = SlurmExecutor()
    monkeypatch.setattr(executor, "_run", run)
    service = ExperimentService(store=store, slurm_executor=executor)

    observation = executor.inspect(store, run_id)
    assert observation.partition == "peerd"

    cancelled = service.cancel(run_id, reason="cancel after partition switch")
    assert cancelled.applied is True
    assert cancelled.scheduler_signalled is True

    reconciled = service.reconcile_scheduler(run_id)
    assert reconciled.run.state == RunState.cancelled
    assert reconciled.accounting is not None
    assert reconciled.accounting.partition == "peerd"
    assert store.scheduler_accounting(run_id).partition == "peerd"
    assert ["scancel", "742"] in commands


def test_partial_handle_recovery_after_switch_uses_recorded_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="recover-after-switch")
    _write_partial_handle(store, run_id)
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", "gpu_a100")
    executor = SlurmExecutor()
    commands: list[list[str]] = []

    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command == ["scontrol", "release", "742"]
        return _completed(command)

    monkeypatch.setattr(executor, "_run", run_command)
    result = executor.launch(store, run_id)

    assert result.launched is False
    assert result.handle.partition == "peerd"
    assert store.run(run_id).scheduler_job_id == "742"
    assert commands == [["scontrol", "release", "742"]]


def test_held_job_recovery_after_switch_matches_run_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="held-after-switch")
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", "gpu_a100")
    executor = SlurmExecutor()
    commands: list[list[str]] = []
    held_row = f"742|caribou_{run_id}|peerd|JobHeldUser|{os.geteuid()}\n"

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "squeue":
            return _completed(command, stdout=held_row)
        if command == ["scontrol", "release", "742"]:
            return _completed(command)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(executor, "_run", run)
    result = executor.launch(store, run_id)

    assert result.handle.job_id == "742"
    assert result.handle.partition == "peerd"
    assert not any(command[0] == "sbatch" for command in commands)
    assert store.run(run_id).scheduler_job_id == "742"


def test_bind_rejects_handle_partition_differing_from_run(
    tmp_path: Path,
) -> None:
    store, run_id = _slurm_store(tmp_path, idempotency_key="bind-mismatch")
    script_path, script_hash = store.write_scheduler_script(
        run_id, "#!/bin/bash\nexit 0\n"
    )
    handle = SlurmExecutionHandle(
        run_id=run_id,
        job_id="742",
        partition="gpu_a100",
        script_path=str(script_path.relative_to(store.run_dir(run_id))),
        script_hash=script_hash,
        stdout_path="slurm-%j.out",
    )
    with pytest.raises(ControlError) as exc_info:
        store.bind_scheduler_job(handle)
    assert exc_info.value.code == "SLURM_PARTITION_MISMATCH"
    assert store.scheduler_handle(run_id) is None
