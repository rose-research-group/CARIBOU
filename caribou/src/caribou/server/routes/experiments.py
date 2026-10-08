"""Thin HTTP adapter for the durable CARIBOU experiment control plane."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from caribou.control.api import (
    ControlError,
    ExitCode,
    error_response,
    machine_response,
)
from caribou.control.service import ExperimentService
from caribou.config import CARIBOU_HOME
from caribou.control.specs import validate_control_spec
from caribou.core.control_access import CONTROL_TOKEN_ENV, resolve_control_access_token
from caribou.domain.enums import InterfaceOrigin
from caribou.domain.models import ExperimentSpec
from caribou.domain.serialization import model_hash


def require_control_access(
    authorization: str | None = Header(default=None),
    x_caribou_control_token: str | None = Header(
        default=None,
        alias="X-Caribou-Control-Token",
    ),
) -> None:
    """Require the CARIBOU token without colliding with proxy authentication."""

    resolved = resolve_control_access_token(create=False)
    if resolved is None:
        error = ControlError(
            "CONTROL_API_DISABLED",
            "run 'caribou config get-control-token' or set "
            f"{CONTROL_TOKEN_ENV} before exposing the experiment control API",
            exit_code=ExitCode.permission,
        )
        raise HTTPException(
            status_code=503,
            detail=error_response("control.authorize", error),
        )
    supplied = (
        x_caribou_control_token if isinstance(x_caribou_control_token, str) else None
    )
    if supplied is None:
        scheme, _, legacy_token = (authorization or "").partition(" ")
        supplied = legacy_token if scheme.casefold() == "bearer" else ""
    if not supplied or not secrets.compare_digest(supplied, resolved.value):
        error = ControlError(
            "CONTROL_API_UNAUTHORIZED",
            "a valid experiment-control bearer token is required",
            exit_code=ExitCode.permission,
        )
        raise HTTPException(
            status_code=401,
            detail=error_response("control.authorize", error),
        )


router = APIRouter(
    prefix="/api/control",
    tags=["experiment-control"],
    dependencies=[Depends(require_control_access)],
)


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExperimentSubmissionRequest(_RequestModel):
    specification: ExperimentSpec
    idempotency_key: str = Field(min_length=1, max_length=256)
    expected_plan_hash: str | None = None


class ExperimentCloneRequest(_RequestModel):
    evaluator_provider: str = Field(min_length=1, max_length=64)
    evaluator_model: str = Field(min_length=1, max_length=256)
    reason: str | None = Field(default=None, max_length=1000)


class RunCancellationRequest(_RequestModel):
    reason: str = Field(
        default="cancel requested by web", min_length=1, max_length=1000
    )


class CheckpointCreationRequest(_RequestModel):
    idempotency_key: str = Field(min_length=1, max_length=256)
    reason: str = Field(
        default="checkpoint requested by web",
        min_length=1,
        max_length=1000,
    )


class RunResumeRequest(_RequestModel):
    checkpoint_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=256)


def get_experiment_service() -> ExperimentService:
    """Construct a request-local facade over the process-independent store."""

    return ExperimentService()


def _http_status(error: ControlError) -> int:
    return {
        ExitCode.usage: 400,
        ExitCode.validation: 422,
        ExitCode.not_found: 404,
        ExitCode.conflict: 409,
        ExitCode.permission: 403,
        ExitCode.budget: 402,
        ExitCode.transient: 503,
        ExitCode.execution: 500,
        ExitCode.cancelled: 409,
        ExitCode.internal: 500,
        ExitCode.integrity: 500,
    }[error.exit_code]


def _call(
    command: str,
    operation: Callable[[], dict[str, Any]],
    *,
    success_status: int = 200,
) -> JSONResponse:
    try:
        return JSONResponse(operation(), status_code=success_status)
    except ControlError as error:
        return JSONResponse(
            error_response(command, error),
            status_code=_http_status(error),
        )
    except Exception as error:  # pragma: no cover - defensive transport boundary
        failure = ControlError(
            "INTERNAL_ERROR",
            "CARIBOU could not complete the request",
            exit_code=ExitCode.internal,
            details={"exception_type": type(error).__name__},
        )
        return JSONResponse(
            error_response(command, failure),
            status_code=_http_status(failure),
        )


def request_validation_error_response(
    error: RequestValidationError,
) -> JSONResponse:
    """Keep FastAPI pre-handler validation inside the machine error contract."""

    issues = [
        {
            "location": [str(part) for part in issue.get("loc", ())],
            "message": str(issue.get("msg", "invalid request")),
            "type": str(issue.get("type", "validation_error")),
        }
        for issue in error.errors()
    ]
    failure = ControlError(
        "REQUEST_INVALID",
        "experiment control request failed validation",
        exit_code=ExitCode.validation,
        details={"issues": issues},
    )
    return JSONResponse(
        error_response("control.request", failure),
        status_code=422,
    )


def control_http_exception_response(error: StarletteHTTPException) -> JSONResponse:
    """Keep routing and method failures inside the sanitized machine contract."""

    detail = error.detail
    if (
        isinstance(detail, dict)
        and detail.get("schema_version") == "caribou.machine_response.v1"
    ):
        payload = detail
    else:
        code, message, exit_code = {
            404: (
                "CONTROL_ROUTE_NOT_FOUND",
                "experiment control route was not found",
                ExitCode.not_found,
            ),
            405: (
                "CONTROL_METHOD_NOT_ALLOWED",
                "HTTP method is not allowed for this experiment control route",
                ExitCode.usage,
            ),
        }.get(
            error.status_code,
            (
                "CONTROL_HTTP_ERROR",
                "experiment control request could not be completed",
                ExitCode.usage,
            ),
        )
        failure = ControlError(
            code,
            message,
            exit_code=exit_code,
            details={"status_code": error.status_code},
        )
        payload = error_response("control.request", failure)
    return JSONResponse(
        payload,
        status_code=error.status_code,
        headers=error.headers,
    )


@router.get("/schema/experiment")
def experiment_schema() -> JSONResponse:
    return _call(
        "schema",
        lambda: machine_response(
            "schema",
            object_type="schema",
            object_id="experiment",
            state="available",
            data={
                "name": "experiment",
                "schema": ExperimentSpec.model_json_schema(),
            },
            links={"submit": "/api/control/experiments"},
        ),
    )


@router.post("/experiments/validate")
def validate_experiment(specification: ExperimentSpec) -> JSONResponse:
    def operation() -> dict[str, Any]:
        checks = validate_control_spec(specification)
        return machine_response(
            "experiment.validate",
            object_type="experiment_spec",
            object_id=specification.spec_id,
            state="validated",
            data={"spec_hash": model_hash(specification), "checks": checks},
            links={
                "plan": "/api/control/experiments/plan",
                "submit": "/api/control/experiments",
            },
        )

    return _call("experiment.validate", operation)


@router.post("/experiments/plan")
def plan_experiment(
    specification: ExperimentSpec,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    return _call(
        "experiment.plan",
        lambda: machine_response(
            "experiment.plan",
            object_type="experiment_spec",
            object_id=specification.spec_id,
            state="planned",
            data=service.plan(specification),
            links={
                "validate": "/api/control/experiments/validate",
                "submit": "/api/control/experiments",
            },
        ),
    )


@router.post("/experiments")
def submit_experiment(
    request: ExperimentSubmissionRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    def operation() -> dict[str, Any]:
        result = service.submit(
            request.specification,
            idempotency_key=request.idempotency_key,
            expected_plan_hash=request.expected_plan_hash,
            interface=InterfaceOrigin.web,
        )
        submission = result.submission
        runs = list(submission.runs)
        first_run_id = runs[0].run_id
        return machine_response(
            "experiment.submit",
            object_type="experiment",
            object_id=submission.experiment.experiment_id,
            state=submission.experiment.state.value,
            data={
                "experiment": submission.experiment.model_dump(mode="json"),
                "runs": [run.model_dump(mode="json") for run in runs],
                "run_ids": [run.run_id for run in runs],
                "plan_hash": submission.plan["plan_hash"],
                "idempotent_replay": submission.idempotent_replay,
                "workers_launched": sum(item.launched for item in result.launches),
            },
            links={
                "status": f"/api/control/runs/{first_run_id}",
                "events": f"/api/control/runs/{first_run_id}/events?after=0",
            },
        )

    return _call("experiment.submit", operation, success_status=202)


@router.post("/experiments/{experiment_id}/clone")
def clone_experiment(
    experiment_id: str,
    request: ExperimentCloneRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    """Create an unsubmitted v2 draft with a changed evaluator model."""

    def operation() -> dict[str, Any]:
        destination = (
            CARIBOU_HOME / "experiment_drafts" / f"{experiment_id}-{uuid4().hex}.yaml"
        )
        spec, path = service.clone_with_evaluator(
            experiment_id,
            destination,
            provider=request.evaluator_provider,
            model=request.evaluator_model,
            reason=request.reason,
        )
        return machine_response(
            "experiment.clone",
            object_type="experiment_spec",
            object_id=spec.spec_id,
            state="initialized",
            data={
                "parent_experiment_id": experiment_id,
                "specification": spec.model_dump(mode="json"),
                "specification_path": str(path),
                "spec_hash": model_hash(spec),
                "model_change_reason": spec.model_change_reason,
            },
            links={
                "validate": "/api/control/experiments/validate",
                "plan": "/api/control/experiments/plan",
                "submit": "/api/control/experiments",
            },
        )

    return _call("experiment.clone", operation, success_status=201)


@router.get("/runs/{run_id}")
def run_status(
    run_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    def operation() -> dict[str, Any]:
        run = service.status(run_id)
        return machine_response(
            "run.status",
            object_type="run",
            object_id=run.run_id,
            state=run.state.value,
            data={"run": run.model_dump(mode="json"), "cursor": run.event_sequence},
            links={
                "events": (
                    f"/api/control/runs/{run.run_id}/events?after={run.event_sequence}"
                ),
                "artifacts": f"/api/control/runs/{run.run_id}/artifacts",
            },
        )

    return _call("run.status", operation)


@router.get("/runs/{run_id}/events")
def run_events(
    run_id: str,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=10_000),
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    def operation() -> dict[str, Any]:
        run = service.status(run_id)
        events = service.events(run_id, after=after, limit=limit)
        next_cursor = events[-1].sequence if events else after
        return machine_response(
            "run.events",
            object_type="run",
            object_id=run.run_id,
            state=run.state.value,
            data={
                "events": [event.model_dump(mode="json") for event in events],
                "after": after,
                "next_cursor": next_cursor,
                "current_cursor": run.event_sequence,
                "has_more": next_cursor < run.event_sequence,
            },
            links={
                "next": f"/api/control/runs/{run.run_id}/events?after={next_cursor}",
                "status": f"/api/control/runs/{run.run_id}",
            },
        )

    return _call("run.events", operation)


@router.post("/runs/{run_id}/cancel")
def cancel_run(
    run_id: str,
    request: RunCancellationRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    def operation() -> dict[str, Any]:
        result = service.cancel(
            run_id,
            reason=request.reason,
            interface=InterfaceOrigin.web,
        )
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
            links={"status": f"/api/control/runs/{run.run_id}"},
        )

    return _call("run.cancel", operation)


@router.post("/runs/{run_id}/checkpoint")
def checkpoint_run(
    run_id: str,
    request: CheckpointCreationRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    def operation() -> dict[str, Any]:
        result = service.request_checkpoint(
            run_id,
            idempotency_key=request.idempotency_key,
            reason=request.reason,
            interface=InterfaceOrigin.web,
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
                "status": f"/api/control/runs/{run.run_id}",
                "checkpoints": f"/api/control/runs/{run.run_id}/checkpoints",
            },
        )

    return _call("run.checkpoint", operation)


@router.get("/runs/{run_id}/checkpoints")
def list_checkpoints(
    run_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    def operation() -> dict[str, Any]:
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
            links={"status": f"/api/control/runs/{run.run_id}"},
        )

    return _call("run.checkpoints", operation)


@router.post("/runs/{run_id}/resume")
def resume_run(
    run_id: str,
    request: RunResumeRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    def operation() -> dict[str, Any]:
        result = service.resume(
            run_id,
            checkpoint_id=request.checkpoint_id,
            idempotency_key=request.idempotency_key,
            interface=InterfaceOrigin.web,
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
                "status": f"/api/control/runs/{child.run_id}",
                "events": f"/api/control/runs/{child.run_id}/events?after=0",
                "source": f"/api/control/runs/{run_id}",
            },
        )

    return _call("run.resume", operation, success_status=202)


@router.get("/runs/{run_id}/artifacts")
def list_artifacts(
    run_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    def operation() -> dict[str, Any]:
        run = service.status(run_id)
        artifacts = service.artifacts(run_id)
        return machine_response(
            "artifact.list",
            object_type="run",
            object_id=run.run_id,
            state=run.state.value,
            data={
                "artifacts": [
                    artifact.model_dump(mode="json") for artifact in artifacts
                ],
                "count": len(artifacts),
            },
            links={"verify": f"/api/control/runs/{run.run_id}/artifacts/verify"},
        )

    return _call("artifact.list", operation)


@router.post("/runs/{run_id}/artifacts/verify")
def verify_artifacts(
    run_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> JSONResponse:
    def operation() -> dict[str, Any]:
        run = service.status(run_id)
        artifacts = service.verify_artifacts(run_id)
        return machine_response(
            "artifact.verify",
            object_type="run",
            object_id=run.run_id,
            state=run.state.value,
            data={
                "verified": len(artifacts),
                "artifact_ids": [artifact.artifact_id for artifact in artifacts],
            },
            links={"list": f"/api/control/runs/{run.run_id}/artifacts"},
        )

    return _call("artifact.verify", operation)


@router.get("/runs/{run_id}/artifacts/{artifact_id}/download")
def download_artifact(
    run_id: str,
    artifact_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> Response:
    try:
        service.status(run_id)
        artifacts = service.verify_artifacts(run_id)
        artifact = next(
            (item for item in artifacts if item.artifact_id == artifact_id),
            None,
        )
        if artifact is None:
            raise ControlError(
                "ARTIFACT_NOT_FOUND",
                f"artifact {artifact_id} is not linked to run {run_id}",
                exit_code=ExitCode.not_found,
            )
        path: Path = service.store.artifact_path(artifact)
        return FileResponse(
            path,
            media_type=artifact.media_type,
            filename=artifact.filename,
            headers={
                "X-Caribou-Run-Id": run_id,
                "X-Caribou-Artifact-Id": artifact.artifact_id,
                "X-Caribou-Content-Hash": artifact.content_hash,
            },
        )
    except ControlError as error:
        return JSONResponse(
            error_response("artifact.fetch", error),
            status_code=_http_status(error),
        )
    except Exception as error:  # pragma: no cover - defensive transport boundary
        failure = ControlError(
            "INTERNAL_ERROR",
            "CARIBOU could not fetch the artifact",
            exit_code=ExitCode.internal,
            details={"exception_type": type(error).__name__},
        )
        return JSONResponse(
            error_response("artifact.fetch", failure),
            status_code=_http_status(failure),
        )
