"""Authenticated preset resolution remains a thin control-plane adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from caribou.control.presets import PresetResolver
from caribou.domain.models import CodeIdentity
from caribou.server.routes.experiments import require_control_access
from caribou.server.routes.presets import (
    PresetResolveRequest,
    list_presets,
    resolve_preset,
    router,
)


@pytest.fixture
def preset_client(
    tmp_path: Path,
) -> tuple[PresetResolver, Path]:
    package_root = Path(__file__).resolve().parents[2] / "src" / "caribou"
    container = tmp_path / "analysis.sif"
    container.write_bytes(b"frozen-test-container")
    dataset = tmp_path / "input.h5ad"
    dataset.write_bytes(b"frozen-test-dataset")
    resolver = PresetResolver(
        package_root=package_root,
        container_path=container,
        code_identity=CodeIdentity(
            repository="https://example.test/CARIBOU",
            branch="preset-tests",
            commit="c" * 40,
            dirty=False,
        ),
    )
    return resolver, dataset


def _request(dataset: Path) -> dict[str, object]:
    return {
        "dataset_path": str(dataset),
        "model_provider": "openai",
        "model_name": "gpt-test-snapshot",
        "profile": "fast",
        "max_turns": 10,
        "executor": "local",
        "owner": "test-operator",
        "reviewer": "test-reviewer",
    }


def test_preset_routes_require_control_authorization(
    preset_client: tuple[PresetResolver, Path],
) -> None:
    del preset_client

    route_dependencies = {
        route.path: {dependency.call for dependency in route.dependant.dependencies}
        for route in router.routes
    }

    assert route_dependencies
    assert all(
        require_control_access in dependencies
        for dependencies in route_dependencies.values()
    )


def test_preset_catalog_and_resolution_use_machine_contract(
    preset_client: tuple[PresetResolver, Path],
) -> None:
    resolver, dataset = preset_client

    catalog = list_presets()
    resolved = resolve_preset(
        "single_agent_qc",
        PresetResolveRequest(**_request(dataset)),
        resolver,
    )

    assert catalog.status_code == 200
    catalog_payload = json.loads(catalog.body)
    assert catalog_payload["schema_version"] == "caribou.machine_response.v1"
    assert catalog_payload["data"]["presets"]
    assert resolved.status_code == 200, resolved.body
    payload = json.loads(resolved.body)
    assert payload["schema_version"] == "caribou.machine_response.v1"
    assert payload["command"] == "preset.resolve"
    assert payload["object"]["state"] == "validated"
    assert (
        payload["data"]["specification"]["schema_version"]
        == "caribou.experiment_spec.v2"
    )
    assert payload["data"]["specification"]["execution"]["executor"] == "local"
    assert payload["links"]["submit"] == "/api/control/experiments"


def test_unknown_preset_is_a_sanitized_not_found_response(
    preset_client: tuple[PresetResolver, Path],
) -> None:
    resolver, dataset = preset_client

    response = resolve_preset(
        "missing",
        PresetResolveRequest(**_request(dataset)),
        resolver,
    )

    assert response.status_code == 404
    assert json.loads(response.body)["error"] == {
        "code": "PRESET_NOT_FOUND",
        "message": "experiment preset was not found",
        "retryable": False,
        "details": {"preset_id": "missing"},
    }


def test_wizard_uses_shared_authenticated_plan_and_submit_client() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    wizard = (
        repository_root
        / "frontend/src/app/pages/experiments-wizard/wizard.component.ts"
    ).read_text(encoding="utf-8")
    control = (
        repository_root / "frontend/src/app/core/services/experiment-control.service.ts"
    ).read_text(encoding="utf-8")

    assert ".resolvePreset(" in wizard
    assert ".plan(" in wizard
    assert ".submit(" in wizard
    assert "runQcPreset" not in wizard
    assert "api/control/presets" in control
    assert "api/experiments/preset" not in control
