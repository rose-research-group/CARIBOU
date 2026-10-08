"""Non-interactive CLI adapters for software agents and automation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import typer
from pydantic import BaseModel

from caribou.control.api import (
    ControlError,
    ExitCode,
    emit_json,
    fail_json,
    machine_response,
)
from caribou.control.records import (
    CheckpointRequest,
    ProviderCallReceipt,
    ProviderCallReceiptV2,
)
from caribou.control.service import ExperimentService
from caribou.control.specs import (
    build_local_plan,
    load_experiment_spec,
    validate_control_spec,
)
from caribou.control.store import default_store_root
from caribou.domain.enums import ExecutorKind
from caribou.domain.models import (
    Aggregate,
    Artifact,
    BudgetRecord,
    Checkpoint,
    Event,
    Experiment,
    ExperimentSpec,
    FailureRecord,
    MetricRecord,
    Run,
)
from caribou.domain.serialization import model_hash


experiment_app = typer.Typer(
    name="experiment",
    help="Validate, plan, submit, and compare durable experiments.",
    no_args_is_help=True,
)
artifact_app = typer.Typer(
    name="artifact",
    help="List, fetch, and verify durable run artifacts.",
    no_args_is_help=True,
)
scheduler_app = typer.Typer(
    name="scheduler",
    help="Inspect and reconcile durable Slurm execution records.",
    no_args_is_help=True,
)


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "experiment": ExperimentSpec,
    "experiment-record": Experiment,
    "run": Run,
    "event": Event,
    "artifact": Artifact,
    "failure": FailureRecord,
    "metric": MetricRecord,
    "checkpoint": Checkpoint,
    "budget": BudgetRecord,
    "aggregate": Aggregate,
    "provider-call-receipt": ProviderCallReceipt,
    "provider-call-receipt-v2": ProviderCallReceiptV2,
    "checkpoint-request": CheckpointRequest,
}


def _require_json(json_output: bool) -> None:
    if not json_output:
        typer.echo(
            "This automation command requires --json; no interactive fallback is used.",
            err=True,
        )
        raise typer.Exit(int(ExitCode.usage))


def _machine_call(
    command: str,
    json_output: bool,
    operation: Callable[[], Mapping[str, Any]],
) -> None:
    _require_json(json_output)
    try:
        emit_json(operation())
    except ControlError as exc:
        fail_json(command, exc)
    except Exception as exc:  # pragma: no cover - defensive transport boundary
        fail_json(
            command,
            ControlError(
                "INTERNAL_ERROR",
                "CARIBOU could not complete the command",
                exit_code=ExitCode.internal,
                details={"exception_type": type(exc).__name__},
            ),
        )


def capabilities_command(
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Describe discoverable machine commands and maturity boundaries."""

    def operation() -> Mapping[str, Any]:
        commands = {
            "capabilities": {"status": "implemented", "streaming": False},
            "schema": {"status": "implemented", "streaming": False},
            "experiment.init": {"status": "implemented", "mutates": True},
            "experiment.clone": {"status": "implemented", "mutates": True},
            "experiment.validate": {"status": "implemented", "mutates": False},
            "experiment.plan": {"status": "implemented", "mutates": False},
            "experiment.submit": {"status": "implemented", "mutates": True},
            "experiment.compare": {"status": "implemented", "mutates": False},
            "run.status": {"status": "implemented", "mutates": False},
            "run.events": {"status": "implemented", "streaming": True},
            "run.cancel": {"status": "implemented", "mutates": True},
            "run.checkpoint": {"status": "implemented", "mutates": True},
            "run.checkpoints": {"status": "implemented", "mutates": False},
            "run.resume": {"status": "implemented", "mutates": True},
            "artifact.list": {"status": "implemented", "mutates": False},
            "artifact.fetch": {"status": "implemented", "mutates": True},
            "artifact.verify": {"status": "implemented", "mutates": False},
            "scheduler.inspect": {"status": "implemented", "mutates": False},
            "scheduler.reconcile": {"status": "implemented", "mutates": True},
        }
        return machine_response(
            "capabilities",
            object_type="caribou",
            object_id="caribou",
            state="available",
            data={
                "machine_response_schema": "caribou.machine_response.v1",
                "stream_schema": "domain Event JSON Lines",
                "stable_exit_codes": {
                    name: int(value) for name, value in ExitCode.__members__.items()
                },
                "commands": commands,
                "schema_names": sorted(SCHEMA_MODELS),
                "execution_boundaries": {
                    "local_lifecycle_smoke": "validated_control_plane_probe",
                    "scripted_agent_path": (
                        "validated_actual_runner_with_test_boundaries"
                    ),
                    "local_agent_analysis": (
                        "implemented_not_validated_real_provider_container"
                    ),
                    "experiment_comparison": (
                        "lifecycle_lineage_only_no_metric_aggregation"
                    ),
                    "slurm": "implemented_pending_cluster_validation",
                },
                "store_root": str(default_store_root()),
            },
            links={
                "experiment_schema": "caribou schema experiment --json",
                "init": "caribou experiment init --output experiment.yaml --json",
                "validate": "caribou experiment validate SPEC --json",
                "plan": "caribou experiment plan SPEC --json",
            },
        )

    _machine_call("capabilities", json_output, operation)


def schema_command(
    name: str = typer.Argument(..., help="Versioned schema name."),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Return a checked-in consumer contract from its canonical model."""

    def operation() -> Mapping[str, Any]:
        model = SCHEMA_MODELS.get(name)
        if model is None:
            raise ControlError(
                "SCHEMA_NOT_FOUND",
                f"unknown schema name: {name}",
                exit_code=ExitCode.not_found,
                details={"available": sorted(SCHEMA_MODELS)},
            )
        schema = model.model_json_schema()
        return machine_response(
            "schema",
            object_type="schema",
            object_id=name,
            state="available",
            data={"name": name, "schema": schema},
            links={"capabilities": "caribou capabilities --json"},
        )

    _machine_call("schema", json_output, operation)


@experiment_app.command("init")
def init_experiment_command(
    output: Path = typer.Option(..., "--output", resolve_path=False),
    overwrite: bool = typer.Option(False, "--overwrite"),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Initialize a valid, runnable lifecycle-smoke experiment specification."""

    def operation() -> Mapping[str, Any]:
        service = ExperimentService()
        spec, destination = service.initialize_spec(output, overwrite=overwrite)
        return machine_response(
            "experiment.init",
            object_type="experiment_spec",
            object_id=spec.spec_id,
            state="initialized",
            data={
                "spec_hash": model_hash(spec),
                "specification": str(destination),
                "template": "lifecycle_smoke",
                "submission_ready": not spec.code.dirty,
            },
            links={
                "validate": f"caribou experiment validate {destination} --json",
                "plan": f"caribou experiment plan {destination} --json",
                "submit": (
                    f"caribou experiment submit {destination} "
                    "--idempotency-key KEY --json"
                ),
            },
        )

    _machine_call("experiment.init", json_output, operation)


@experiment_app.command("clone")
def clone_experiment_command(
    experiment_id: str = typer.Argument(...),
    output: Path = typer.Option(..., "--output", resolve_path=False),
    evaluator_provider: str = typer.Option(..., "--evaluator-provider"),
    evaluator_model: str = typer.Option(..., "--evaluator-model"),
    reason: str | None = typer.Option(
        None, "--reason", help="Optional human explanation for the model change."
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Clone a submitted experiment into an unsubmitted evaluator-model revision."""

    def operation() -> Mapping[str, Any]:
        service = ExperimentService()
        spec, destination = service.clone_with_evaluator(
            experiment_id,
            output,
            provider=evaluator_provider,
            model=evaluator_model,
            reason=reason,
            overwrite=overwrite,
        )
        return machine_response(
            "experiment.clone",
            object_type="experiment_spec",
            object_id=spec.spec_id,
            state="initialized",
            data={
                "parent_experiment_id": experiment_id,
                "parent_spec_id": spec.parent_spec_id,
                "spec_hash": model_hash(spec),
                "specification": str(destination),
                "model_change_reason": spec.model_change_reason,
            },
            links={
                "validate": f"caribou experiment validate {destination} --json",
                "plan": f"caribou experiment plan {destination} --json",
            },
        )

    _machine_call("experiment.clone", json_output, operation)


@experiment_app.command("validate")
def validate_experiment_command(
    specification: Path = typer.Argument(..., exists=False, resolve_path=False),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Strictly validate a frozen experiment specification without mutation."""

    def operation() -> Mapping[str, Any]:
        spec = load_experiment_spec(specification)
        checks = validate_control_spec(spec)
        return machine_response(
            "experiment.validate",
            object_type="experiment_spec",
            object_id=spec.spec_id,
            state="validated",
            data={
                "spec_hash": model_hash(spec),
                "specification": str(specification.expanduser().resolve()),
                "checks": checks,
            },
            links={
                "plan": f"caribou experiment plan {specification} --json",
                "schema": "caribou schema experiment --json",
            },
        )

    _machine_call("experiment.validate", json_output, operation)


@experiment_app.command("plan")
def plan_experiment_command(
    specification: Path = typer.Argument(..., exists=False, resolve_path=False),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Build a deterministic resource plan without submitting work."""

    def operation() -> Mapping[str, Any]:
        spec = load_experiment_spec(specification)
        plan = build_local_plan(spec)
        return machine_response(
            "experiment.plan",
            object_type="experiment_spec",
            object_id=spec.spec_id,
            state="planned",
            data=plan,
            links={
                "validate": f"caribou experiment validate {specification} --json",
                "submit": f"caribou experiment submit {specification} --idempotency-key KEY --json",
            },
        )

    _machine_call("experiment.plan", json_output, operation)


@experiment_app.command("submit")
def submit_experiment_command(
    specification: Path = typer.Argument(..., exists=False, resolve_path=False),
    idempotency_key: str = typer.Option(
        ..., "--idempotency-key", help="Stable duplicate-protection key."
    ),
    expected_plan_hash: str | None = typer.Option(
        None,
        "--expected-plan-hash",
        help="Reject submission if the deterministic plan changed.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Persist an experiment and launch its declared execution transport."""

    def operation() -> Mapping[str, Any]:
        service = ExperimentService()
        spec = service.validate(specification)
        result = service.submit(
            spec,
            idempotency_key=idempotency_key,
            expected_plan_hash=expected_plan_hash,
        )
        submission = result.submission
        experiment = submission.experiment
        runs = list(submission.runs)
        return machine_response(
            "experiment.submit",
            object_type="experiment",
            object_id=experiment.experiment_id,
            state=experiment.state.value,
            data={
                "experiment": experiment.model_dump(mode="json"),
                "runs": [run.model_dump(mode="json") for run in runs],
                "run_ids": [run.run_id for run in runs],
                "plan_hash": submission.plan["plan_hash"],
                "idempotent_replay": submission.idempotent_replay,
                "workers_launched": sum(item.launched for item in result.launches),
            },
            links={
                "status": f"caribou run status {runs[0].run_id} --json",
                "events": f"caribou run events {runs[0].run_id} --after 0 --format jsonl",
            },
        )

    _machine_call("experiment.submit", json_output, operation)


@experiment_app.command("compare")
def compare_experiment_command(
    experiment_id: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Compare logical run leaves without mutating or rerunning an experiment."""

    def operation() -> Mapping[str, Any]:
        comparison = ExperimentService().compare(experiment_id)
        leaf_run_ids = comparison["leaf_run_ids"]
        links = {"capabilities": "caribou capabilities --json"}
        if leaf_run_ids:
            links.update(
                {
                    "first_run": f"caribou run status {leaf_run_ids[0]} --json",
                    "first_run_artifacts": (
                        f"caribou artifact list {leaf_run_ids[0]} --json"
                    ),
                }
            )
        return machine_response(
            "experiment.compare",
            object_type="experiment_comparison",
            object_id=experiment_id,
            state=str(comparison["status"]),
            data=comparison,
            links=links,
        )

    _machine_call("experiment.compare", json_output, operation)


def run_status_command(
    run_id: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Read one durable attempt snapshot without side effects."""

    def operation() -> Mapping[str, Any]:
        run = ExperimentService().status(run_id)
        return machine_response(
            "run.status",
            object_type="run",
            object_id=run.run_id,
            state=run.state.value,
            data={"run": run.model_dump(mode="json"), "cursor": run.event_sequence},
            links={
                "events": f"caribou run events {run.run_id} --after {run.event_sequence} --format jsonl",
                "artifacts": f"caribou artifact list {run.run_id} --json",
            },
        )

    _machine_call("run.status", json_output, operation)


def run_events_command(
    run_id: str = typer.Argument(...),
    after: int = typer.Option(0, "--after", min=0, help="Exclusive event cursor."),
    limit: int = typer.Option(1000, "--limit", min=1, max=10000),
    output_format: str = typer.Option("jsonl", "--format"),
) -> None:
    """Emit durable events as resumable JSON Lines."""

    if output_format != "jsonl":
        fail_json(
            "run.events",
            ControlError(
                "FORMAT_UNSUPPORTED",
                "run events supports only --format jsonl",
                exit_code=ExitCode.validation,
            ),
        )
    try:
        events = ExperimentService().events(run_id, after=after, limit=limit)
        for event in events:
            emit_json(
                {
                    "schema_version": "caribou.event_line.v1",
                    "run_id": run_id,
                    "cursor": event.sequence,
                    "event": event.model_dump(mode="json"),
                }
            )
    except ControlError as exc:
        fail_json("run.events", exc)


def run_cancel_command(
    run_id: str = typer.Argument(...),
    reason: str = typer.Option("cancel requested by CLI", "--reason"),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Request cooperative cancellation and preserve the attempt record."""

    def operation() -> Mapping[str, Any]:
        result = ExperimentService().cancel(run_id, reason=reason)
        run = result.run
        return machine_response(
            "run.cancel",
            object_type="run",
            object_id=run.run_id,
            state=run.state.value,
            data={
                "run": run.model_dump(mode="json"),
                "applied": result.applied,
                "scheduler_signalled": result.scheduler_signalled,
            },
            links={"status": f"caribou run status {run.run_id} --json"},
        )

    _machine_call("run.cancel", json_output, operation)


def run_checkpoint_command(
    run_id: str = typer.Argument(...),
    idempotency_key: str = typer.Option(
        ..., "--idempotency-key", help="Stable duplicate-protection key."
    ),
    reason: str = typer.Option("checkpoint requested by CLI", "--reason"),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Stop cooperatively after the next complete agent turn."""

    def operation() -> Mapping[str, Any]:
        result = ExperimentService().request_checkpoint(
            run_id,
            idempotency_key=idempotency_key,
            reason=reason,
        )
        run = result.run
        return machine_response(
            "run.checkpoint",
            object_type="run",
            object_id=run.run_id,
            state=run.state.value,
            data={
                "run": run.model_dump(mode="json"),
                "request": result.request.model_dump(mode="json"),
                "applied": result.applied,
                "safe_boundary": "completed_agent_turn",
            },
            links={
                "status": f"caribou run status {run.run_id} --json",
                "checkpoints": f"caribou run checkpoints {run.run_id} --json",
            },
        )

    _machine_call("run.checkpoint", json_output, operation)


def run_checkpoints_command(
    run_id: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """List the complete checkpoint envelopes attached to one attempt."""

    def operation() -> Mapping[str, Any]:
        service = ExperimentService()
        run = service.status(run_id)
        checkpoints = service.checkpoints(run_id)
        return machine_response(
            "run.checkpoints",
            object_type="run",
            object_id=run.run_id,
            state=run.state.value,
            data={
                "checkpoints": [
                    checkpoint.model_dump(mode="json") for checkpoint in checkpoints
                ],
                "count": len(checkpoints),
            },
            links={"status": f"caribou run status {run.run_id} --json"},
        )

    _machine_call("run.checkpoints", json_output, operation)


def run_resume_command(
    run_id: str = typer.Argument(..., help="Terminal resumable source attempt."),
    from_checkpoint: str = typer.Option(
        "latest", "--from-checkpoint", help="Checkpoint ID or 'latest'."
    ),
    idempotency_key: str = typer.Option(
        ..., "--idempotency-key", help="Stable duplicate-protection key."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Create and launch one linked child attempt from a complete checkpoint."""

    def operation() -> Mapping[str, Any]:
        service = ExperimentService()
        checkpoints = service.checkpoints(run_id)
        if from_checkpoint == "latest":
            if not checkpoints:
                raise ControlError(
                    "CHECKPOINT_NOT_FOUND",
                    "the source attempt has no complete checkpoint",
                    exit_code=ExitCode.not_found,
                )
            latest_turn = max(checkpoint.turn for checkpoint in checkpoints)
            latest = [
                checkpoint
                for checkpoint in checkpoints
                if checkpoint.turn == latest_turn
            ]
            if len(latest) != 1:
                raise ControlError(
                    "CHECKPOINT_LATEST_AMBIGUOUS",
                    "multiple checkpoints share the latest turn; select an exact ID",
                    exit_code=ExitCode.conflict,
                )
            checkpoint_id = latest[0].checkpoint_id
        else:
            checkpoint_id = from_checkpoint
        result = service.resume(
            run_id,
            checkpoint_id=checkpoint_id,
            idempotency_key=idempotency_key,
        )
        submission = result.submission
        child = submission.child
        return machine_response(
            "run.resume",
            object_type="run",
            object_id=child.run_id,
            state=child.state.value,
            data={
                "source_run": submission.source.model_dump(mode="json"),
                "checkpoint": submission.checkpoint.model_dump(mode="json"),
                "child_run": child.model_dump(mode="json"),
                "idempotent_replay": submission.idempotent_replay,
                "workers_launched": sum(item.launched for item in result.launches),
            },
            links={
                "status": f"caribou run status {child.run_id} --json",
                "events": (
                    f"caribou run events {child.run_id} --after 0 --format jsonl"
                ),
                "source": f"caribou run status {run_id} --json",
            },
        )

    _machine_call("run.resume", json_output, operation)


@scheduler_app.command("inspect")
def scheduler_inspect_command(
    run_id: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Read live or durable scheduler state without mutating the run."""

    def operation() -> Mapping[str, Any]:
        service = ExperimentService()
        run = service.status(run_id)
        if run.executor != ExecutorKind.slurm:
            service.inspect_scheduler(run_id)
        handle = service.store.scheduler_handle(run_id)
        submission = service.store.scheduler_submission(run_id)
        cancellation = service.store.scheduler_cancellation(run_id)
        observation = service.inspect_scheduler(run_id) if handle is not None else None
        return machine_response(
            "scheduler.inspect",
            object_type=(
                "scheduler_job" if handle is not None else "scheduler_submission"
            ),
            object_id=(handle.job_id if handle is not None else run_id),
            state=(
                observation.state.lower()
                if observation is not None
                else run.state.value
            ),
            data={
                "run_id": run_id,
                "handle": (
                    handle.model_dump(mode="json") if handle is not None else None
                ),
                "submission": (
                    submission.model_dump(mode="json")
                    if submission is not None
                    else None
                ),
                "observation": (
                    observation.as_dict() if observation is not None else None
                ),
                "cancellation": (
                    cancellation.model_dump(mode="json")
                    if cancellation is not None
                    else None
                ),
            },
            links={
                "run": f"caribou run status {run_id} --json",
                "reconcile": f"caribou scheduler reconcile {run_id} --json",
            },
        )

    _machine_call("scheduler.inspect", json_output, operation)


@scheduler_app.command("reconcile")
def scheduler_reconcile_command(
    run_id: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Persist terminal Slurm accounting and close pre-worker failures."""

    def operation() -> Mapping[str, Any]:
        result = ExperimentService().reconcile_scheduler(run_id)
        return machine_response(
            "scheduler.reconcile",
            object_type="run",
            object_id=run_id,
            state=result.run.state.value,
            data={
                "run": result.run.model_dump(mode="json"),
                "observation": result.observation.as_dict(),
                "accounting": (
                    result.accounting.model_dump(mode="json")
                    if result.accounting is not None
                    else None
                ),
                "accounting_created": result.accounting_created,
                "run_transition_applied": result.run_transition_applied,
            },
            links={
                "status": f"caribou run status {run_id} --json",
                "events": f"caribou run events {run_id} --after 0 --format jsonl",
            },
        )

    _machine_call("scheduler.reconcile", json_output, operation)


@artifact_app.command("list")
def artifact_list_command(
    run_id: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """List the durable manifest for one run."""

    def operation() -> Mapping[str, Any]:
        service = ExperimentService()
        run = service.status(run_id)
        artifacts = service.artifacts(run_id)
        return machine_response(
            "artifact.list",
            object_type="run",
            object_id=run_id,
            state=run.state.value,
            data={
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
                "count": len(artifacts),
            },
            links={"verify": f"caribou artifact verify {run_id} --json"},
        )

    _machine_call("artifact.list", json_output, operation)


@artifact_app.command("verify")
def artifact_verify_command(
    run_id: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Hash, size, symlink, and root-boundary verify every linked artifact."""

    def operation() -> Mapping[str, Any]:
        service = ExperimentService()
        run = service.status(run_id)
        artifacts = service.verify_artifacts(run_id)
        return machine_response(
            "artifact.verify",
            object_type="run",
            object_id=run_id,
            state=run.state.value,
            data={
                "verified": len(artifacts),
                "artifact_ids": [item.artifact_id for item in artifacts],
            },
            links={"list": f"caribou artifact list {run_id} --json"},
        )

    _machine_call("artifact.verify", json_output, operation)


@artifact_app.command("fetch")
def artifact_fetch_command(
    run_id: str = typer.Argument(...),
    artifact_id: str = typer.Argument(...),
    output: Path = typer.Option(..., "--output", resolve_path=False),
    overwrite: bool = typer.Option(False, "--overwrite"),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON object."),
) -> None:
    """Atomically copy one verified artifact to an explicit destination."""

    def operation() -> Mapping[str, Any]:
        service = ExperimentService()
        run = service.status(run_id)
        artifact, destination = service.fetch_artifact(
            run_id, artifact_id, output, overwrite=overwrite
        )
        return machine_response(
            "artifact.fetch",
            object_type="artifact",
            object_id=artifact.artifact_id,
            state="verified",
            data={
                "run_id": run.run_id,
                "output": str(destination),
                "content_hash": artifact.content_hash,
                "size_bytes": artifact.size_bytes,
            },
            links={"verify": f"caribou artifact verify {run_id} --json"},
        )

    _machine_call("artifact.fetch", json_output, operation)


def register_machine_commands(app: typer.Typer, run_app: typer.Typer) -> None:
    """Attach machine commands without changing interactive command behavior."""

    app.command("capabilities")(capabilities_command)
    app.command("schema")(schema_command)
    app.add_typer(experiment_app, name="experiment")
    app.add_typer(artifact_app, name="artifact")
    app.add_typer(scheduler_app, name="scheduler")
    run_app.command("status")(run_status_command)
    run_app.command("events")(run_events_command)
    run_app.command("cancel")(run_cancel_command)
    run_app.command("checkpoint")(run_checkpoint_command)
    run_app.command("checkpoints")(run_checkpoints_command)
    run_app.command("resume")(run_resume_command)
