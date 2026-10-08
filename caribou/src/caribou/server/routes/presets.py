"""Authenticated web adapter for typed experiment presets."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from caribou.control.api import machine_response
from caribou.control.presets import PresetResolver, get_preset_list
from caribou.control.specs import validate_control_spec
from caribou.domain.serialization import model_hash
from caribou.server.routes.experiments import _call, require_control_access


router = APIRouter(
    prefix="/api/control/presets",
    tags=["experiment-control"],
    dependencies=[Depends(require_control_access)],
)


class PresetResolveRequest(BaseModel):
    """Human selections required to freeze one canonical ExperimentSpec."""

    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_path: str = Field(min_length=1, max_length=4096)
    model_provider: Literal["openai", "deepseek", "openrouter"]
    model_name: str = Field(min_length=1, max_length=256)
    openrouter_endpoint: str | None = Field(default=None, min_length=1, max_length=256)
    profile: Literal["fast", "thorough"]
    max_turns: int | None = Field(default=None, ge=1, le=100)
    executor: Literal["local", "slurm"]
    owner: str = Field(min_length=1, max_length=256)
    reviewer: str = Field(min_length=1, max_length=256)
    evaluator_provider: Literal["openai", "deepseek", "openrouter"] | None = None
    evaluator_model_name: str | None = Field(default=None, min_length=1, max_length=256)


def get_preset_resolver() -> PresetResolver:
    return PresetResolver()


@router.get("")
def list_presets() -> JSONResponse:
    """Discover available presets without constructing or submitting a run."""

    return _call(
        "preset.list",
        lambda: machine_response(
            "preset.list",
            object_type="preset_catalog",
            object_id="caribou-presets",
            state="available",
            data={"presets": get_preset_list()},
            links={"resolve": "/api/control/presets/{preset_id}/resolve"},
        ),
    )


@router.post("/{preset_id}/resolve")
def resolve_preset(
    preset_id: str,
    request: PresetResolveRequest,
    resolver: PresetResolver = Depends(get_preset_resolver),
) -> JSONResponse:
    """Freeze a preset; planning and submission use the canonical endpoints."""

    def operation() -> dict[str, object]:
        specification = resolver.resolve(
            preset_id,
            dataset_path=request.dataset_path,
            model_provider=request.model_provider,
            model_name=request.model_name,
            openrouter_endpoint=request.openrouter_endpoint,
            profile=request.profile,
            max_turns=request.max_turns,
            executor=request.executor,
            owner=request.owner,
            reviewer=request.reviewer,
            evaluator_provider=request.evaluator_provider,
            evaluator_model_name=request.evaluator_model_name,
        )
        checks = validate_control_spec(
            specification,
            require_submit_adapter=True,
        )
        return machine_response(
            "preset.resolve",
            object_type="experiment_spec",
            object_id=specification.spec_id,
            state="validated",
            data={
                "preset_id": preset_id,
                "spec_hash": model_hash(specification),
                "specification": specification.model_dump(mode="json"),
                "checks": checks,
            },
            links={
                "plan": "/api/control/experiments/plan",
                "submit": "/api/control/experiments",
            },
        )

    return _call("preset.resolve", operation)
