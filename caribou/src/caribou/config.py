# caribou/config.py
import os
import re
from pathlib import Path

from dotenv import dotenv_values
from platformdirs import PlatformDirs

# Define app-specific identifiers for platformdirs
APP_NAME = "caribou"
APP_AUTHOR = "OpenTechBio"
dirs = PlatformDirs(APP_NAME, APP_AUTHOR)

# Define the root directory for all user-specific CARIBOU files.
# This respects the CARIBOU_HOME environment variable but has a sensible default.
CARIBOU_HOME = Path(os.environ.get("CARIBOU_HOME", dirs.user_data_dir)).expanduser()

# Define standard subdirectories
DEFAULT_AGENT_DIR = CARIBOU_HOME / "agent_systems"
DEFAULT_DATASETS_DIR = CARIBOU_HOME / "datasets"

# The benchmark-validated full multi-agent system used in end-to-end evaluation
DEFAULT_BLUEPRINT_NAME = "caribou_fully_connected_v2.json"

# Define the path to the environment file for storing secrets like API keys
ENV_FILE = CARIBOU_HOME / ".env"

SLURM_PARTITION_ENV_VAR = "CARIBOU_SLURM_PARTITION"

# Slurm partition names are simple identifiers. This value is rendered
# unescaped into generated `#SBATCH --partition=...` lines and `sbatch`
# argv, so it is validated wherever it is set or read rather than trusted
# as free-form text.
SLURM_PARTITION_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class InvalidSlurmPartitionError(ValueError):
    """Raised when a Slurm partition name fails the safe-identifier check."""


class SlurmPartitionNotConfiguredError(RuntimeError):
    """Raised when Slurm work is requested but no partition is configured."""

    def __init__(self) -> None:
        super().__init__(
            "No Slurm partition is configured for CARIBOU. Run "
            "`caribou config set-slurm-partition <name>` or set the "
            f"{SLURM_PARTITION_ENV_VAR} environment variable."
        )


def validate_slurm_partition(partition: str) -> str:
    if not isinstance(partition, str) or not SLURM_PARTITION_PATTERN.fullmatch(
        partition
    ):
        raise InvalidSlurmPartitionError(
            "Slurm partition must be a plain identifier "
            "(letters, digits, '-', '_' only); "
            f"got {partition!r}"
        )
    return partition


def read_caribou_slurm_partition() -> str | None:
    """Return the configured Slurm partition, or None when none is configured.

    The CARIBOU_SLURM_PARTITION environment variable wins; otherwise the value
    stored in the CARIBOU .env file (ENV_FILE) is used. There is no built-in
    default. The .env file is parsed without exporting its keys into
    os.environ, and is read fresh on every call (unlike CARIBOU_HOME, which is
    frozen at import time) so a changed setting applies without a restart.
    A configured value is validated and raises InvalidSlurmPartitionError if
    it is not a plain identifier.
    """
    if SLURM_PARTITION_ENV_VAR in os.environ:
        partition = os.environ[SLURM_PARTITION_ENV_VAR]
    else:
        partition = dotenv_values(ENV_FILE).get(SLURM_PARTITION_ENV_VAR)
        if partition is None:
            return None
    return validate_slurm_partition(partition)


def get_caribou_slurm_partition() -> str:
    """Return the configured Slurm partition that new Slurm work must target.

    Resolved like read_caribou_slurm_partition(), but raises
    SlurmPartitionNotConfiguredError when no partition is configured.
    """
    partition = read_caribou_slurm_partition()
    if partition is None:
        raise SlurmPartitionNotConfiguredError()
    return partition


def init_caribou_home():
    """Ensures the main CARIBOU directory and its subdirectories exist."""
    CARIBOU_HOME.mkdir(parents=True, exist_ok=True)
    DEFAULT_AGENT_DIR.mkdir(exist_ok=True)
    DEFAULT_DATASETS_DIR.mkdir(exist_ok=True)
