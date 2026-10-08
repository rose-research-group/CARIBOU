"""
Session bootstrap helpers.

Isolated from `session_manager` so blueprint resolution and sandbox/LLM
construction can be reused (or tested) without touching the manager.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from caribou.config import DEFAULT_AGENT_DIR
from caribou.core.python_environments import PythonEnvironmentError
from caribou.server.models import (
    EvaluatorModelConfig,
    ResolvedModelInfo,
    SessionCreateRequest,
)
from caribou.server.session_state import (
    SANDBOX_DATA_PATH,
    SANDBOX_REF_DATA_PATH,
)

_log = logging.getLogger(__name__)


def resolve_model_binding(
    *,
    backend: str,
    model_name: str | None = None,
    ollama_model: str | None = None,
    resolved_model_name: str | None = None,
) -> ResolvedModelInfo | None:
    """Resolve one provider/model request into persisted provenance."""

    from caribou.core.deepseek import (
        deepseek_profile_for_backend,
        is_deepseek_backend,
    )

    if is_deepseek_backend(backend):
        profile = deepseek_profile_for_backend(backend)
        return ResolvedModelInfo(
            provider="deepseek",
            model=resolved_model_name or model_name or profile.model,
            parameters=profile.model_parameters(),
        )
    if backend == "chatgpt":
        return ResolvedModelInfo(
            provider="openai",
            model=resolved_model_name or model_name or "gpt-4o",
        )
    if backend == "claude":
        return ResolvedModelInfo(
            provider="anthropic",
            model=resolved_model_name or model_name or "claude-sonnet-4-6",
        )
    if backend == "openrouter":
        model = resolved_model_name or model_name or ""
        if model:
            return ResolvedModelInfo(
                provider="openrouter",
                model=model,
                parameters={
                    "routing": "flexible",
                    "zdr": True,
                    "data_collection": "deny",
                },
            )
    if backend.startswith("ollama"):
        model = (
            resolved_model_name
            or ollama_model
            or model_name
            or os.environ.get("OLLAMA_MODEL", "")
        )
        if model:
            return ResolvedModelInfo(provider="ollama", model=model)
    return None


def resolve_model_info(
    config: SessionCreateRequest,
    *,
    resolved_model_name: str | None = None,
) -> ResolvedModelInfo | None:
    """Resolve the exact model record shown to users and persisted on disk."""

    return resolve_model_binding(
        backend=config.llm_backend,
        model_name=getattr(config, "model_name", None),
        ollama_model=getattr(config, "ollama_model", None),
        resolved_model_name=resolved_model_name,
    )


def resolve_evaluator_model_info(
    config: SessionCreateRequest,
    *,
    worker_resolved: ResolvedModelInfo | None = None,
    resolved_model_name: str | None = None,
) -> ResolvedModelInfo | None:
    selection = config.evaluator_model
    if selection.mode == "inherit_worker":
        return worker_resolved or resolve_model_info(config)
    assert selection.llm_backend is not None
    return resolve_model_binding(
        backend=selection.llm_backend,
        model_name=selection.model_name,
        ollama_model=selection.ollama_model,
        resolved_model_name=resolved_model_name,
    )


class SandboxUnavailableError(RuntimeError):
    """
    Raised when the requested sandbox backend can't be started because a
    prerequisite is missing (binary, image, permissions). The server catches
    this and forwards `code` + `suggested_fix` to the UI so users see an
    actionable message instead of a stack trace.
    """

    def __init__(self, code: str, message: str, suggested_fix: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.suggested_fix = suggested_fix


def find_blueprint(name: str) -> Path:
    """Resolve a blueprint name to a JSON path."""
    from caribou.cli.run_cli import PACKAGE_AGENTS_DIR

    for search_dir in (DEFAULT_AGENT_DIR, PACKAGE_AGENTS_DIR):
        candidate = Path(search_dir) / name
        if candidate.exists():
            return candidate
        candidate = Path(search_dir) / f"{name}.json"
        if candidate.exists():
            return candidate

    # Absolute path provided
    p = Path(name)
    if p.exists():
        return p

    raise FileNotFoundError(
        f"Blueprint '{name}' not found in {DEFAULT_AGENT_DIR} or package agents dir."
    )


def build_model_client(
    *,
    backend: str,
    model_name: str | None = None,
    ollama_model: str | None = None,
):
    """Return ``(client, exact_model_name)`` for one model binding."""
    from openai import OpenAI

    from caribou.core.deepseek import (
        create_deepseek_client,
        deepseek_profile_for_backend,
        is_deepseek_backend,
    )

    if backend == "chatgpt":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        return OpenAI(api_key=key), model_name or "gpt-4o"

    if backend == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        from caribou.core.anthropic_wrapper import AnthropicClient

        return AnthropicClient(api_key=key), model_name or "claude-sonnet-4-6"

    if is_deepseek_backend(backend):
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise EnvironmentError("DEEPSEEK_API_KEY not set.")
        profile = deepseek_profile_for_backend(backend)
        return create_deepseek_client(key, profile=profile), model_name or profile.model

    if backend == "openrouter":
        from caribou.core.openrouter import (
            create_openrouter_client,
            validate_openrouter_model_id,
        )

        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise EnvironmentError("OPENROUTER_API_KEY not set.")
        if not model_name:
            raise ValueError("Select an OpenRouter model before starting the session.")
        model = validate_openrouter_model_id(model_name, strict=False)
        return create_openrouter_client(key), model

    if backend.startswith("ollama"):
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        env_model = os.environ.get("OLLAMA_MODEL", "")
        requested_model = ollama_model or model_name or env_model or ""
        from caribou.server.ollama_service import ensure_ollama_ready
        from caribou.core.ollama_wrapper import OllamaClient

        resolved_host, model_name = ensure_ollama_ready(host, requested_model)
        return OllamaClient(host=resolved_host, model=model_name), model_name

    raise ValueError(f"Unknown LLM backend: {backend!r}")


def build_llm_client(config: SessionCreateRequest):
    """Backward-compatible worker model resolver."""

    return build_model_client(
        backend=config.llm_backend,
        model_name=getattr(config, "model_name", None),
        ollama_model=getattr(config, "ollama_model", None),
    )


def build_evaluator_client(
    config: SessionCreateRequest,
    *,
    worker_client: object,
    worker_model_name: str,
) -> tuple[object, str, ResolvedModelInfo | None]:
    selection: EvaluatorModelConfig = config.evaluator_model
    if selection.mode == "inherit_worker":
        return (
            worker_client,
            worker_model_name,
            resolve_model_info(config, resolved_model_name=worker_model_name),
        )
    assert selection.llm_backend is not None
    client, exact_model = build_model_client(
        backend=selection.llm_backend,
        model_name=selection.model_name,
        ollama_model=selection.ollama_model,
    )
    return (
        client,
        exact_model,
        resolve_model_binding(
            backend=selection.llm_backend,
            model_name=selection.model_name,
            ollama_model=selection.ollama_model,
            resolved_model_name=exact_model,
        ),
    )


def _preflight_docker() -> None:
    if not shutil.which("docker"):
        raise SandboxUnavailableError(
            code="DOCKER_NOT_INSTALLED",
            message="Docker executable not found in PATH.",
            suggested_fix=(
                "Install Docker Desktop (macOS/Windows) or the docker CLI (Linux) "
                "and make sure the daemon is running, then retry."
            ),
        )


def _preflight_singularity() -> None:
    if not (shutil.which("apptainer") or shutil.which("singularity")):
        raise SandboxUnavailableError(
            code="SANDBOX_UNAVAILABLE",
            message="Singularity/Apptainer executable not found in PATH.",
            suggested_fix=(
                "Install Apptainer or Singularity, or load the module on this host "
                "(e.g. `module load singularity`), then restart the CARIBOU server."
            ),
        )


def build_sandbox(config: SessionCreateRequest, output_dir: Path):
    """
    Build and start a sandbox manager. Blocking — run in a thread.

    Raises SandboxUnavailableError with a suggested fix when a prerequisite
    is missing so the UI can render an actionable error instead of a stack
    trace. Also catches SystemExit — some sandbox helpers historically
    `sys.exit(1)` on missing binaries, which would otherwise crash the
    asyncio task and take down the server lifespan.
    """
    from rich.console import Console

    script_dir = Path(__file__).resolve().parent
    # Quiet console so sandbox init output doesn't go to stdout;
    # errors are surfaced via exceptions caught by the caller.
    console = Console(quiet=True)

    sandbox_type = config.sandbox_type.value

    try:
        if sandbox_type == "docker":
            _preflight_docker()
            from caribou.core.sandbox_management import init_docker

            manager_class, handle, copy_cmd, _, _ = init_docker(
                script_dir,
                subprocess,
                console,
                force_refresh=False,
                python_environment_path=config.python_environment_path,
            )
            sandbox = manager_class()
            if not sandbox.start_container():
                environment_error = getattr(sandbox, "last_start_error", None)
                raise SandboxUnavailableError(
                    code=(
                        "PYTHON_ENV_INCOMPATIBLE"
                        if config.python_environment_path
                        else "DOCKER_START_FAILED"
                    ),
                    message=environment_error or "Docker sandbox failed to start.",
                    suggested_fix=(
                        "Use an environment whose Python and ipykernel work inside the "
                        "CARIBOU Docker image, or select the bundled environment."
                        if config.python_environment_path
                        else "Confirm the Docker daemon is running (`docker info`), then retry."
                    ),
                )
            copy_cmd(config.dataset_path, f"{handle}:{SANDBOX_DATA_PATH}")
            if config.reference_dataset_path:
                copy_cmd(
                    config.reference_dataset_path, f"{handle}:{SANDBOX_REF_DATA_PATH}"
                )
            return sandbox

        if sandbox_type == "singularity":
            _preflight_singularity()
            from caribou.core.sandbox_management import init_singularity_exec

            manager_class, _, _, _, _ = init_singularity_exec(
                script_dir,
                SANDBOX_DATA_PATH,
                subprocess,
                console,
                force_refresh=False,
                python_environment_path=config.python_environment_path,
            )
            sandbox = manager_class()
            sandbox.set_data(
                [(Path(config.dataset_path), SANDBOX_DATA_PATH)]
                + (
                    [(Path(config.reference_dataset_path), SANDBOX_REF_DATA_PATH)]
                    if config.reference_dataset_path
                    else []
                ),
                output_dir,
            )
            if not sandbox.start_container():
                environment_error = getattr(sandbox, "last_start_error", None)
                raise SandboxUnavailableError(
                    code=(
                        "PYTHON_ENV_INCOMPATIBLE"
                        if config.python_environment_path
                        else "SINGULARITY_START_FAILED"
                    ),
                    message=environment_error or "Singularity sandbox failed to start.",
                    suggested_fix=(
                        "Use an environment built for a compatible Linux/CUDA runtime, "
                        "or select the bundled environment."
                        if config.python_environment_path
                        else "Check the singularity/apptainer install and try again."
                    ),
                )
            return sandbox

        raise SandboxUnavailableError(
            code="SANDBOX_TYPE_UNKNOWN",
            message=f"Unknown sandbox type: {sandbox_type}",
            suggested_fix="Pick 'docker' or 'singularity' for the session sandbox.",
        )
    except SandboxUnavailableError:
        raise
    except PythonEnvironmentError as exc:
        raise SandboxUnavailableError(
            code=exc.code,
            message=str(exc),
            suggested_fix=(
                "Choose a discovered environment or provide an absolute readable prefix "
                "containing bin/python."
            ),
        ) from exc
    except SystemExit as exc:
        # Legacy sandbox helpers call sys.exit(1) on missing binaries at
        # import time. Convert that into an actionable UI error rather than
        # letting SystemExit escape and take down the event loop.
        _log.warning(
            "Sandbox helper raised SystemExit(%s); converting to error.", exc.code
        )
        raise SandboxUnavailableError(
            code="SANDBOX_UNAVAILABLE",
            message=f"Sandbox helper exited unexpectedly (SystemExit {exc.code}).",
            suggested_fix=(
                "The sandbox backend was unable to initialize (typically a missing "
                "binary such as singularity/apptainer or docker). Install the "
                "required tool and restart the CARIBOU server."
            ),
        )
