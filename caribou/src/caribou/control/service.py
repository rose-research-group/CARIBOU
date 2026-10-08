"""One application service shared by machine CLI and future web adapters."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml  # type: ignore[import-untyped]

from caribou.domain.enums import ExecutorKind, InterfaceOrigin, RunState
from caribou.domain.ids import new_id
from caribou.domain.models import (
    Artifact,
    Checkpoint,
    CodeIdentity,
    ContentReference,
    Event,
    ExperimentEvaluatorSpec,
    ExperimentSpec,
    ModelSpec,
    Run,
    utc_now,
)
from caribou.domain.serialization import file_hash, model_hash

from .api import ControlError, ExitCode, code_commit
from .executor import LaunchResult, LocalProcessExecutor
from .slurm import ReconciliationResult, SchedulerObservation, SlurmExecutor
from .specs import build_local_plan, load_experiment_spec, validate_control_spec
from .records import CheckpointRequest
from .store import ExperimentStore, ResumeSubmission, Submission


@dataclass(frozen=True)
class SubmittedExperiment:
    submission: Submission
    launches: tuple[LaunchResult, ...]


@dataclass(frozen=True)
class CancellationResult:
    run: Run
    applied: bool
    scheduler_signalled: bool


@dataclass(frozen=True)
class CheckpointRequestResult:
    run: Run
    request: CheckpointRequest
    applied: bool


@dataclass(frozen=True)
class ResumedExperiment:
    submission: ResumeSubmission
    launches: tuple[LaunchResult, ...]


def _git_value(
    repository_root: Path, *arguments: str, allow_empty: bool = False
) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value or allow_empty else None


def _identity_override(name: str) -> str | None:
    configured = os.environ.get(name)
    if configured is None:
        return None
    value = configured.strip()
    if not value or any(ord(character) < 32 for character in value):
        raise ControlError(
            "CODE_IDENTITY_INVALID",
            f"{name} must be a non-empty value without control characters",
            exit_code=ExitCode.validation,
        )
    return value


def _safe_repository_identity(repository: str) -> str:
    """Remove clone credentials and non-identity URL fields before persistence."""

    if "://" in repository:
        parsed = urlsplit(repository)
        if not parsed.scheme or parsed.hostname is None:
            raise ControlError(
                "CODE_IDENTITY_INVALID",
                "CARIBOU repository identity is not a valid URL",
                exit_code=ExitCode.validation,
            )
        try:
            port = parsed.port
        except ValueError as exc:
            raise ControlError(
                "CODE_IDENTITY_INVALID",
                "CARIBOU repository identity has an invalid port",
                exit_code=ExitCode.validation,
            ) from exc
        hostname = parsed.hostname
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        network_location = hostname if port is None else f"{hostname}:{port}"
        return urlunsplit((parsed.scheme, network_location, parsed.path, "", ""))

    # Git's SCP-like clone syntax commonly includes a public SSH username. Drop
    # everything before the last @ so credentials or usernames are never emitted.
    host_and_path = repository.rsplit("@", maxsplit=1)[-1]
    return host_and_path.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]


def _unresolved_identity(field: str, environment_name: str) -> ControlError:
    return ControlError(
        "CODE_IDENTITY_UNRESOLVED",
        f"could not resolve executing {field}; set {environment_name} explicitly",
        exit_code=ExitCode.validation,
    )


def _configured_dirty(repository_root: Path) -> bool:
    configured = os.environ.get("CARIBOU_CODE_DIRTY")
    if configured is None:
        status = _git_value(
            repository_root, "status", "--porcelain=v1", allow_empty=True
        )
        if status is None:
            raise _unresolved_identity("worktree state", "CARIBOU_CODE_DIRTY")
        return bool(status)
    normalized = configured.strip().casefold()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ControlError(
        "CODE_IDENTITY_INVALID",
        "CARIBOU_CODE_DIRTY must be true or false",
        exit_code=ExitCode.validation,
    )


def _executing_code_identity() -> CodeIdentity:
    repository_root = Path(__file__).resolve().parents[4]
    commit = code_commit()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ControlError(
            "CODE_IDENTITY_UNRESOLVED",
            "set CARIBOU_CODE_COMMIT to the exact 40-character Git commit "
            "before initializing a spec",
            exit_code=ExitCode.validation,
        )
    branch = _identity_override("CARIBOU_CODE_BRANCH")
    if branch is None:
        branch = _git_value(
            repository_root, "branch", "--show-current", allow_empty=True
        )
        if branch is None:
            raise _unresolved_identity("Git branch", "CARIBOU_CODE_BRANCH")
        branch = branch or "detached"

    repository = _identity_override("CARIBOU_CODE_REPOSITORY")
    if repository is None:
        repository = _git_value(repository_root, "remote", "get-url", "origin")
        if repository is None:
            raise _unresolved_identity("Git repository", "CARIBOU_CODE_REPOSITORY")
    repository = _safe_repository_identity(repository)
    if not repository:
        raise ControlError(
            "CODE_IDENTITY_INVALID",
            "CARIBOU repository identity must not be empty after sanitization",
            exit_code=ExitCode.validation,
        )
    return CodeIdentity(
        repository=repository,
        branch=branch,
        commit=commit,
        dirty=_configured_dirty(repository_root),
    )


def executing_code_identity() -> CodeIdentity:
    """Resolve the exact repository identity used by new frozen specifications."""

    return _executing_code_identity()


class ExperimentService:
    """Authoritative local lifecycle operations with transport-free semantics."""

    def __init__(
        self,
        store: ExperimentStore | None = None,
        executor: LocalProcessExecutor | None = None,
        slurm_executor: SlurmExecutor | None = None,
    ) -> None:
        self.store = store or ExperimentStore()
        self.executor = executor or LocalProcessExecutor()
        self.slurm_executor = slurm_executor or SlurmExecutor()

    def validate(self, path: Path) -> ExperimentSpec:
        return load_experiment_spec(path)

    def initialize_spec(
        self, destination: Path, *, overwrite: bool = False
    ) -> tuple[ExperimentSpec, Path]:
        """Create one ready-to-run lifecycle template with current code identity."""

        target = Path(os.path.abspath(destination.expanduser()))
        if target.suffix.lower() not in {".yaml", ".yml"}:
            raise ControlError(
                "SPEC_OUTPUT_FORMAT_UNSUPPORTED",
                "experiment init currently writes YAML; use a .yaml or .yml path",
                exit_code=ExitCode.validation,
                details={"output": str(target)},
            )
        if target.is_symlink() or target.is_dir():
            raise ControlError(
                "OUTPUT_INVALID",
                "experiment specification output must be a regular file path",
                exit_code=ExitCode.conflict,
                details={"output": str(target)},
            )
        if target.exists() and not overwrite:
            raise ControlError(
                "OUTPUT_EXISTS",
                "experiment specification output exists; use --overwrite to replace it",
                exit_code=ExitCode.conflict,
                details={"output": str(target)},
            )

        template_text = (
            files("caribou.control")
            .joinpath("experiment-template.yaml")
            .read_text(encoding="utf-8")
        )
        template = ExperimentSpec.model_validate_json(
            json.dumps(yaml.safe_load(template_text))
        )
        candidate = template.model_copy(
            update={
                "spec_id": new_id("spec"),
                "code": _executing_code_identity(),
                "created_at": utc_now(),
            }
        )
        spec = ExperimentSpec.model_validate_json(candidate.model_dump_json())
        validate_control_spec(spec, require_submit_adapter=True)
        payload = yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=True).encode(
            "utf-8"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return spec, target

    def plan(self, spec: ExperimentSpec) -> dict:
        return build_local_plan(spec)

    def clone_with_evaluator(
        self,
        experiment_id: str,
        destination: Path,
        *,
        provider: str,
        model: str,
        reason: str | None = None,
        overwrite: bool = False,
    ) -> tuple[ExperimentSpec, Path]:
        """Create an unsubmitted v2 draft with immutable source lineage."""

        source = self.store.spec(experiment_id)
        target = Path(os.path.abspath(destination.expanduser()))
        if target.suffix.lower() not in {".yaml", ".yml"}:
            raise ControlError(
                "SPEC_OUTPUT_FORMAT_UNSUPPORTED",
                "experiment clone writes YAML; use a .yaml or .yml path",
                exit_code=ExitCode.validation,
            )
        if target.exists() and not overwrite:
            raise ControlError(
                "OUTPUT_EXISTS",
                "experiment specification output exists; use --overwrite to replace it",
                exit_code=ExitCode.conflict,
            )
        resolved_provider = provider.strip().casefold()
        resolved_model = model.strip()
        normalized_reason = (reason or "").strip() or None
        if not resolved_provider or not resolved_model:
            raise ControlError(
                "EVALUATOR_MODEL_REQUIRED",
                "evaluator provider and exact model identifier are required",
                exit_code=ExitCode.validation,
            )
        if normalized_reason is not None and len(normalized_reason) > 1000:
            raise ControlError(
                "MODEL_CHANGE_REASON_TOO_LONG",
                "model change reason cannot exceed 1000 characters",
                exit_code=ExitCode.validation,
            )
        if source.evaluator is not None:
            evaluator_agent = source.evaluator.agent
            evaluator_agent_name = source.evaluator.agent_name
        else:
            evaluator_path = (
                Path(__file__).resolve().parents[1] / "agents" / "evaluator_agent.json"
            )
            evaluator_agent = ContentReference(
                uri=evaluator_path.resolve().as_uri(),
                content_hash=file_hash(evaluator_path),
                size_bytes=evaluator_path.stat().st_size,
                media_type="application/json",
            )
            evaluator_agent_name = "evaluator"
        candidate = source.model_copy(
            update={
                "schema_version": "caribou.experiment_spec.v2",
                "spec_id": new_id("spec"),
                "spec_version": source.spec_version + 1,
                "parent_spec_id": source.spec_id,
                "model_change_reason": normalized_reason,
                "evaluator": ExperimentEvaluatorSpec(
                    agent_name=evaluator_agent_name,
                    agent=evaluator_agent,
                    model=ModelSpec(provider=resolved_provider, model=resolved_model),
                ),
                "created_at": utc_now(),
            }
        )
        clone = ExperimentSpec.model_validate_json(candidate.model_dump_json())
        validate_control_spec(clone)
        payload = yaml.safe_dump(clone.model_dump(mode="json"), sort_keys=True).encode(
            "utf-8"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return clone, target

    def compare(self, experiment_id: str) -> dict[str, Any]:
        """Return a deterministic read-only comparison of logical run leaves."""

        experiment = self.store.experiment(experiment_id)
        spec = self.store.spec(experiment_id)
        runs = [self.store.run(run_id) for run_id in experiment.run_ids]
        if model_hash(spec) != experiment.spec_hash or any(
            run.experiment_id != experiment_id or run.spec_hash != experiment.spec_hash
            for run in runs
        ):
            raise ControlError(
                "EXPERIMENT_GRAPH_MISMATCH",
                "the stored specification and runs do not match the experiment",
                exit_code=ExitCode.integrity,
                details={"experiment_id": experiment_id},
            )

        run_by_id = {run.run_id: run for run in runs}
        resumed = [run for run in runs if run.resumed_from_run_id is not None]
        if any(run.resumed_from_run_id not in run_by_id for run in resumed):
            raise ControlError(
                "RUN_LINEAGE_INVALID",
                "a resumed attempt references a missing source run",
                exit_code=ExitCode.integrity,
            )
        children_by_source: dict[str, list[Run]] = {}
        for child in resumed:
            assert child.resumed_from_run_id is not None
            children_by_source.setdefault(child.resumed_from_run_id, []).append(child)
        if any(len(children) != 1 for children in children_by_source.values()):
            raise ControlError(
                "RUN_LINEAGE_AMBIGUOUS",
                "comparison currently requires non-branching resume lineage",
                exit_code=ExitCode.integrity,
            )
        condition_order = {
            condition.condition_id: index
            for index, condition in enumerate(spec.conditions)
        }
        sorted_runs = sorted(
            runs,
            key=lambda run: (
                condition_order.get(run.condition_id, len(condition_order)),
                run.replicate_index,
                run.attempt_index,
                run.run_id,
            ),
        )
        superseded_ids = set(children_by_source)
        leaf_runs = [run for run in sorted_runs if run.run_id not in superseded_ids]

        def counts(values: list[str]) -> dict[str, int]:
            return {value: values.count(value) for value in sorted(set(values))}

        def run_summary(run: Run) -> dict[str, Any]:
            children = children_by_source.get(run.run_id, [])
            return {
                "run_id": run.run_id,
                "condition_id": run.condition_id,
                "replicate_index": run.replicate_index,
                "attempt_index": run.attempt_index,
                "state": run.state.value,
                "terminal_outcome": (
                    run.terminal_outcome.value
                    if run.terminal_outcome is not None
                    else None
                ),
                "resumed_from_run_id": run.resumed_from_run_id,
                "superseded_by_run_id": children[0].run_id if children else None,
                "resume_eligible": run.resume_eligible,
                "event_cursor": run.event_sequence,
                "current_turn": run.current_turn,
                "artifact_count": len(run.artifact_ids),
                "metric_record_count": len(run.metric_record_ids),
                "failure_record_count": len(run.failure_ids),
                "checkpoint_count": len(run.checkpoint_ids),
                "budget_record_count": len(run.budget_record_ids),
                "scheduler_job_id": run.scheduler_job_id,
            }

        conditions: list[dict[str, Any]] = []
        for condition in spec.conditions:
            condition_leaves = [
                run for run in leaf_runs if run.condition_id == condition.condition_id
            ]
            states = [run.state.value for run in condition_leaves]
            outcomes = [
                run.terminal_outcome.value
                if run.terminal_outcome is not None
                else "nonterminal"
                for run in condition_leaves
            ]
            conditions.append(
                {
                    "condition_id": condition.condition_id,
                    "label": condition.label,
                    "intervention": {
                        "provider": condition.model.provider,
                        "model": condition.model.model,
                        "topology": condition.blueprint.topology.value,
                        "blueprint_hash": condition.blueprint.topology_hash,
                        "prompt_hash": condition.prompt.content_hash,
                        "memory_strategy": condition.memory.strategy.value,
                    },
                    "expected_repetitions": spec.repetitions,
                    "leaf_run_ids": [run.run_id for run in condition_leaves],
                    "state_counts": counts(states),
                    "outcome_counts": counts(outcomes),
                }
            )

        logical_terminal_states = {
            RunState.succeeded,
            RunState.failed,
            RunState.cancelled,
            RunState.rejected,
        }
        all_logical_runs_terminal = all(
            run.state in logical_terminal_states for run in leaf_runs
        )
        comparison: dict[str, Any] = {
            "schema_version": "caribou.experiment_comparison.v1",
            "experiment_id": experiment.experiment_id,
            "spec_id": experiment.spec_id,
            "spec_version": experiment.spec_version,
            "spec_hash": experiment.spec_hash,
            "experiment_state": experiment.state.value,
            "status": "complete" if all_logical_runs_terminal else "partial",
            "scope": "run_lifecycle_lineage_and_record_inventory",
            "expected_logical_runs": len(spec.conditions) * spec.repetitions,
            "attempt_count": len(sorted_runs),
            "leaf_run_count": len(leaf_runs),
            "leaf_run_ids": [run.run_id for run in leaf_runs],
            "superseded_run_ids": [
                run.run_id for run in sorted_runs if run.run_id in superseded_ids
            ],
            "awaiting_resume_run_ids": [
                run.run_id for run in leaf_runs if run.state == RunState.resumable
            ],
            "all_logical_runs_terminal": all_logical_runs_terminal,
            "conditions": conditions,
            "attempts": [run_summary(run) for run in sorted_runs],
            "metric_values_aggregated": False,
        }
        canonical = json.dumps(
            comparison,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        comparison["comparison_hash"] = (
            "sha256:" + hashlib.sha256(canonical).hexdigest()
        )
        return comparison

    def submit(
        self,
        spec: ExperimentSpec,
        *,
        idempotency_key: str,
        expected_plan_hash: str | None = None,
        interface: InterfaceOrigin = InterfaceOrigin.cli,
    ) -> SubmittedExperiment:
        plan = self.plan(spec)
        if expected_plan_hash is not None and plan["plan_hash"] != expected_plan_hash:
            raise ControlError(
                "PLAN_CHANGED",
                "the current deterministic plan differs from the expected plan hash",
                exit_code=ExitCode.conflict,
                details={
                    "expected_plan_hash": expected_plan_hash,
                    "current_plan_hash": plan["plan_hash"],
                },
            )
        submission = self.store.submit(
            spec,
            idempotency_key,
            interface=interface,
        )
        selected_executor = (
            self.slurm_executor
            if spec.execution.executor == ExecutorKind.slurm
            else self.executor
        )
        launches = tuple(
            selected_executor.launch(self.store, run.run_id)
            for run in submission.runs
            if run.state.value == "queued"
        )
        if spec.execution.executor == ExecutorKind.slurm:
            # A failed held-job cleanup deliberately leaves the run cancelling.
            # Retrying the same idempotent submit must advance that cleanup rather
            # than returning a misleading no-op success.
            for submitted_run in submission.runs:
                current = self.store.run(submitted_run.run_id)
                if current.state == RunState.cancelling:
                    self.slurm_executor.cancel(self.store, current.run_id)
                    current = self.store.run(current.run_id)
                    if current.state == RunState.cancelled:
                        self.store.reconcile_experiment(current.experiment_id)
        refreshed = Submission(
            experiment=self.store.experiment(submission.experiment.experiment_id),
            runs=tuple(self.store.run(run.run_id) for run in submission.runs),
            plan=submission.plan,
            idempotent_replay=submission.idempotent_replay,
        )
        return SubmittedExperiment(submission=refreshed, launches=launches)

    def status(self, run_id: str) -> Run:
        return self.store.run(run_id)

    def events(self, run_id: str, *, after: int, limit: int) -> tuple[Event, ...]:
        return self.store.events(run_id, after=after, limit=limit)

    def cancel(
        self,
        run_id: str,
        *,
        reason: str,
        interface: InterfaceOrigin = InterfaceOrigin.cli,
    ) -> CancellationResult:
        run, applied = self.store.request_cancel(
            run_id,
            actor=interface.value,
            reason=reason,
        )
        scheduler_signalled = False
        if run.executor == ExecutorKind.slurm and run.state == RunState.cancelling:
            scheduler_signalled = self.slurm_executor.cancel(self.store, run_id)
            run = self.store.run(run_id)
            if run.state == RunState.cancelled:
                self.store.reconcile_experiment(run.experiment_id)
        return CancellationResult(
            run=run,
            applied=applied,
            scheduler_signalled=scheduler_signalled,
        )

    def request_checkpoint(
        self,
        run_id: str,
        *,
        idempotency_key: str,
        reason: str,
        interface: InterfaceOrigin = InterfaceOrigin.cli,
    ) -> CheckpointRequestResult:
        run, request, applied = self.store.request_checkpoint(
            run_id,
            idempotency_key=idempotency_key,
            actor=interface.value,
            reason=reason,
        )
        return CheckpointRequestResult(run=run, request=request, applied=applied)

    def checkpoints(self, run_id: str) -> tuple[Checkpoint, ...]:
        return self.store.checkpoints(run_id)

    def resume(
        self,
        run_id: str,
        *,
        checkpoint_id: str,
        idempotency_key: str,
        interface: InterfaceOrigin = InterfaceOrigin.cli,
    ) -> ResumedExperiment:
        submission = self.store.resume(
            run_id,
            checkpoint_id=checkpoint_id,
            idempotency_key=idempotency_key,
            interface=interface,
        )
        child = submission.child
        selected_executor = (
            self.slurm_executor
            if child.executor == ExecutorKind.slurm
            else self.executor
        )
        launches = (
            (selected_executor.launch(self.store, child.run_id),)
            if child.state == RunState.queued
            else ()
        )
        refreshed = ResumeSubmission(
            source=self.store.run(submission.source.run_id),
            checkpoint=submission.checkpoint,
            child=self.store.run(child.run_id),
            idempotent_replay=submission.idempotent_replay,
        )
        return ResumedExperiment(submission=refreshed, launches=launches)

    def inspect_scheduler(self, run_id: str) -> SchedulerObservation:
        return self.slurm_executor.inspect(self.store, run_id)

    def reconcile_scheduler(self, run_id: str) -> ReconciliationResult:
        return self.slurm_executor.reconcile(self.store, run_id)

    def artifacts(self, run_id: str) -> tuple[Artifact, ...]:
        self.store.run(run_id)
        return self.store.artifact_manifest(run_id).artifacts

    def verify_artifacts(self, run_id: str) -> tuple[Artifact, ...]:
        self.store.run(run_id)
        return self.store.verify_artifacts(run_id)

    def fetch_artifact(
        self,
        run_id: str,
        artifact_id: str,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> tuple[Artifact, Path]:
        manifest = self.store.artifact_manifest(run_id)
        artifact = manifest.artifact(artifact_id)
        if artifact is None:
            raise ControlError(
                "ARTIFACT_NOT_FOUND",
                f"artifact {artifact_id} is not linked to run {run_id}",
                exit_code=ExitCode.not_found,
            )
        source = self.store.artifact_path(artifact)
        self.store.verify_artifacts(run_id)
        target = Path(os.path.abspath(destination.expanduser()))
        if target.is_dir():
            raise ControlError(
                "OUTPUT_IS_DIRECTORY",
                "artifact destination must be a file path",
                exit_code=ExitCode.conflict,
                details={"output": str(target)},
            )
        if target.is_symlink() or (target.exists() and not overwrite):
            raise ControlError(
                "OUTPUT_EXISTS",
                "artifact destination exists; use --overwrite for a regular file",
                exit_code=ExitCode.conflict,
                details={"output": str(target)},
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                for block in iter(lambda: input_file.read(1024 * 1024), b""):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return artifact, target
