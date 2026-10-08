"""Resolution of the configured CARIBOU Slurm partition.

The autouse fixture in tests/conftest.py sets CARIBOU_SLURM_PARTITION and
points caribou.config.ENV_FILE at a per-test path; each test here adjusts
those explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import caribou.config as caribou_config
from caribou.config import (
    InvalidSlurmPartitionError,
    SlurmPartitionNotConfiguredError,
    get_caribou_slurm_partition,
    read_caribou_slurm_partition,
    validate_slurm_partition,
)


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setattr(caribou_config, "ENV_FILE", path)
    monkeypatch.delenv("CARIBOU_SLURM_PARTITION")
    return path


def test_conftest_isolates_from_real_environment() -> None:
    assert os.environ["CARIBOU_SLURM_PARTITION"] == "peerd"
    assert not caribou_config.ENV_FILE.exists()
    assert get_caribou_slurm_partition() == "peerd"


def test_unset_partition_reads_none_and_get_raises(env_file: Path) -> None:
    assert read_caribou_slurm_partition() is None
    with pytest.raises(SlurmPartitionNotConfiguredError) as exc_info:
        get_caribou_slurm_partition()
    message = str(exc_info.value)
    assert "caribou config set-slurm-partition <name>" in message
    assert "CARIBOU_SLURM_PARTITION" in message


def test_env_file_value_is_used_without_exporting_keys(env_file: Path) -> None:
    env_file.write_text(
        "CARIBOU_SLURM_PARTITION=gpu_a100\nCARIBOU_TEST_UNRELATED_KEY=secret\n",
        encoding="utf-8",
    )
    assert read_caribou_slurm_partition() == "gpu_a100"
    assert get_caribou_slurm_partition() == "gpu_a100"
    assert "CARIBOU_SLURM_PARTITION" not in os.environ
    assert "CARIBOU_TEST_UNRELATED_KEY" not in os.environ


def test_environment_variable_wins_over_env_file(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text("CARIBOU_SLURM_PARTITION=gpu_a100\n", encoding="utf-8")
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", "cpu-long")
    assert read_caribou_slurm_partition() == "cpu-long"
    assert get_caribou_slurm_partition() == "cpu-long"


def test_env_file_change_is_read_fresh(env_file: Path) -> None:
    env_file.write_text("CARIBOU_SLURM_PARTITION=first\n", encoding="utf-8")
    assert get_caribou_slurm_partition() == "first"
    env_file.write_text("CARIBOU_SLURM_PARTITION=second\n", encoding="utf-8")
    assert get_caribou_slurm_partition() == "second"


@pytest.mark.parametrize(
    "value", ["", "gpu a100", "gpu;rm", "peerd\n#SBATCH --qos=x", "gpu,cpu"]
)
def test_invalid_partition_names_are_rejected(
    value: str, env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(InvalidSlurmPartitionError):
        validate_slurm_partition(value)
    monkeypatch.setenv("CARIBOU_SLURM_PARTITION", value)
    with pytest.raises(InvalidSlurmPartitionError):
        read_caribou_slurm_partition()
    with pytest.raises(InvalidSlurmPartitionError):
        get_caribou_slurm_partition()


def test_invalid_env_file_value_is_rejected(env_file: Path) -> None:
    env_file.write_text('CARIBOU_SLURM_PARTITION="gpu a100"\n', encoding="utf-8")
    with pytest.raises(InvalidSlurmPartitionError):
        read_caribou_slurm_partition()


@pytest.mark.parametrize("value", ["peerd", "gpu_a100", "cpu-long", "A1"])
def test_valid_partition_names_are_accepted(value: str) -> None:
    assert validate_slurm_partition(value) == value
