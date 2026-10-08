"""Route tests for the slurm_partition field of GET/PATCH /api/settings.

ENV_FILE is redirected to tmp_path in both caribou.config and the route
module, and os.environ is snapshotted/restored, so the user's real .env and
environment are never touched.
"""

import asyncio
import os
from pathlib import Path
from unittest import mock

import pytest
from dotenv import dotenv_values
from fastapi import HTTPException
from pydantic import ValidationError

import caribou.config as caribou_config
import caribou.server.routes.config as config_routes
from caribou.server.routes.config import (
    UpdateSettingsRequest,
    get_settings,
    update_settings,
)


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "caribou_home" / ".env"
    monkeypatch.setattr(caribou_config, "ENV_FILE", path)
    monkeypatch.setattr(config_routes, "ENV_FILE", path)
    with mock.patch.dict(os.environ):
        os.environ.pop("CARIBOU_SLURM_PARTITION", None)
        yield path


def test_get_reports_null_when_unconfigured(env_file: Path) -> None:
    settings = asyncio.run(get_settings())

    assert settings.slurm_partition is None


def test_patch_writes_env_file_and_applies_to_running_process(env_file: Path) -> None:
    result = asyncio.run(
        update_settings(UpdateSettingsRequest(slurm_partition="  gpu_long  "))
    )

    assert result == {"updated": ["CARIBOU_SLURM_PARTITION"]}
    assert dotenv_values(env_file)["CARIBOU_SLURM_PARTITION"] == "gpu_long"
    assert os.environ["CARIBOU_SLURM_PARTITION"] == "gpu_long"
    assert caribou_config.get_caribou_slurm_partition() == "gpu_long"
    assert asyncio.run(get_settings()).slurm_partition == "gpu_long"


def test_patch_overrides_previous_process_environment_value(env_file: Path) -> None:
    # A server started with the variable exported must still switch to the
    # value saved through the settings page.
    os.environ["CARIBOU_SLURM_PARTITION"] = "old_partition"

    asyncio.run(update_settings(UpdateSettingsRequest(slurm_partition="new_partition")))

    assert caribou_config.get_caribou_slurm_partition() == "new_partition"


@pytest.mark.parametrize("bad", ["bad name", "x;y", "", "   "])
def test_patch_rejects_invalid_partition_with_422_and_writes_nothing(
    env_file: Path, bad: str
) -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            update_settings(
                UpdateSettingsRequest(slurm_partition=bad, ollama_model="some-model")
            )
        )

    assert exc.value.status_code == 422
    assert "plain identifier" in exc.value.detail
    assert not env_file.exists()
    assert "CARIBOU_SLURM_PARTITION" not in os.environ


def test_get_fails_loudly_on_invalid_configured_value(env_file: Path) -> None:
    env_file.parent.mkdir(parents=True)
    env_file.write_text('CARIBOU_SLURM_PARTITION="bad value"\n')

    with pytest.raises(HTTPException) as exc:
        asyncio.run(get_settings())

    assert exc.value.status_code == 500
    assert "plain identifier" in exc.value.detail


def test_patch_without_partition_leaves_it_unset(env_file: Path) -> None:
    asyncio.run(update_settings(UpdateSettingsRequest(ollama_model="m")))

    assert "CARIBOU_SLURM_PARTITION" not in dotenv_values(env_file)
    assert asyncio.run(get_settings()).slurm_partition is None


def test_request_model_rejects_non_string_partition() -> None:
    with pytest.raises(ValidationError):
        UpdateSettingsRequest(slurm_partition=["a"])
