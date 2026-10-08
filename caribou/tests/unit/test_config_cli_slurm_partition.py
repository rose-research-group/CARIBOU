"""CLI tests for `caribou config set-/show-slurm-partition`.

Every test points ENV_FILE at tmp_path and isolates os.environ, so the
user's real CARIBOU .env and shell environment are never read or written.
"""

import os
from pathlib import Path
from unittest import mock

import pytest
from dotenv import dotenv_values
from typer.testing import CliRunner

import caribou.cli.config_cli as config_cli
import caribou.config as caribou_config

runner = CliRunner()


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "caribou_home" / ".env"
    monkeypatch.setattr(caribou_config, "ENV_FILE", path)
    monkeypatch.setattr(config_cli, "ENV_FILE", path)
    with mock.patch.dict(os.environ):
        os.environ.pop("CARIBOU_SLURM_PARTITION", None)
        yield path


def test_show_reports_not_configured_and_exits_nonzero(env_file: Path) -> None:
    result = runner.invoke(config_cli.config_app, ["show-slurm-partition"])

    assert result.exit_code == 1
    assert "No Slurm partition is configured" in result.output
    assert "set-slurm-partition" in result.output


def test_set_creates_env_dir_and_show_reads_it_back(env_file: Path) -> None:
    assert not env_file.parent.exists()

    result = runner.invoke(config_cli.config_app, ["set-slurm-partition", "gpu-a100"])

    assert result.exit_code == 0, result.output
    assert dotenv_values(env_file)["CARIBOU_SLURM_PARTITION"] == "gpu-a100"
    assert oct(env_file.stat().st_mode & 0o777) == oct(0o600)
    assert "Restart" in result.output

    shown = runner.invoke(config_cli.config_app, ["show-slurm-partition"])
    assert shown.exit_code == 0, shown.output
    assert shown.output.strip() == "gpu-a100"


def test_environment_variable_wins_over_env_file(env_file: Path) -> None:
    runner.invoke(config_cli.config_app, ["set-slurm-partition", "from_file"])
    os.environ["CARIBOU_SLURM_PARTITION"] = "from_env"

    shown = runner.invoke(config_cli.config_app, ["show-slurm-partition"])

    assert shown.exit_code == 0, shown.output
    assert shown.output.strip() == "from_env"


def test_set_warns_when_shell_variable_shadows_new_value(env_file: Path) -> None:
    os.environ["CARIBOU_SLURM_PARTITION"] = "shadow"

    result = runner.invoke(config_cli.config_app, ["set-slurm-partition", "newpart"])

    assert result.exit_code == 0, result.output
    assert "takes precedence" in result.output
    assert dotenv_values(env_file)["CARIBOU_SLURM_PARTITION"] == "newpart"


@pytest.mark.parametrize("bad", ["bad name", "a;rm -rf /", "x\ny", ""])
def test_set_rejects_invalid_partition_without_writing(env_file: Path, bad: str) -> None:
    result = runner.invoke(config_cli.config_app, ["set-slurm-partition", bad])

    assert result.exit_code == 1
    assert "plain identifier" in result.output
    assert not env_file.exists()


def test_show_fails_loudly_on_invalid_configured_value(env_file: Path) -> None:
    os.environ["CARIBOU_SLURM_PARTITION"] = "not valid!"

    result = runner.invoke(config_cli.config_app, ["show-slurm-partition"])

    assert result.exit_code == 1
    assert "plain identifier" in result.output
