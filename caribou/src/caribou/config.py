# caribou/config.py
import os
import re
from pathlib import Path

from dotenv import load_dotenv
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

_DEFAULT_SLURM_PARTITION = "peerd"

# Slurm partition names are simple identifiers. This value is rendered
# unescaped into generated `#SBATCH --partition=...` lines and `sbatch`
# argv, so it is validated wherever it is set or read rather than trusted
# as free-form text.
SLURM_PARTITION_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class InvalidSlurmPartitionError(ValueError):
    """Raised when a Slurm partition name fails the safe-identifier check."""


def validate_slurm_partition(partition: str) -> str:
    if not SLURM_PARTITION_PATTERN.fullmatch(partition):
        raise InvalidSlurmPartitionError(
            "Slurm partition must be a plain identifier "
            "(letters, digits, '-', '_' only); "
            f"got {partition!r}"
        )
    return partition


def get_caribou_slurm_partition() -> str:
    """Return the Slurm partition CARIBOU is authorized to submit and bind jobs on.

    Resolved from the CARIBOU_SLURM_PARTITION environment variable, or the
    CARIBOU .env file, falling back to the historical default. A real
    environment variable always wins over the .env file, matching how every
    other CARIBOU secret/setting is resolved. Read fresh on every call (rather
    than frozen at import time, like CARIBOU_HOME) so the control plane can
    move clusters without a code change or process restart.
    """
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    return validate_slurm_partition(
        os.environ.get("CARIBOU_SLURM_PARTITION", _DEFAULT_SLURM_PARTITION)
    )


def init_caribou_home():
    """Ensures the main CARIBOU directory and its subdirectories exist."""
    CARIBOU_HOME.mkdir(parents=True, exist_ok=True)
    DEFAULT_AGENT_DIR.mkdir(exist_ok=True)
    DEFAULT_DATASETS_DIR.mkdir(exist_ok=True)
