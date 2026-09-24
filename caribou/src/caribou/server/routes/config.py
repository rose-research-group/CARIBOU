from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv, set_key
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import shutil

from caribou.config import CARIBOU_HOME, DEFAULT_AGENT_DIR, ENV_FILE
from caribou.core.python_environments import (
    PythonEnvironmentCandidate,
    PythonEnvironmentError,
    discover_python_environments,
    validate_python_environment_path,
)
from caribou.core.deepseek import DEEPSEEK_PROFILES
from caribou.core.openrouter import (
    OPENROUTER_CATALOG_URL,
    OpenRouterError,
    get_openrouter_catalogue,
    get_openrouter_endpoints,
)
from caribou.server.models import (
    AgentBlueprint,
    AgentConfig,
    BlueprintContent,
    CommandConfig,
    LLMBackend,
    OllamaModelsResponse,
    PythonEnvironmentPathRequest,
    SaveBlueprintRequest,
    ServerStatus,
)
from caribou.server.ollama_service import (
    DEFAULT_OLLAMA_MODEL,
    OllamaStartupError,
    normalize_host,
    probe_ollama,
    start_ollama,
)
from caribou.server.session_manager import session_manager, _SESSIONS_DIR

router = APIRouter(prefix="/api", tags=["config"])

_BACKENDS = [
    LLMBackend(
        id="chatgpt", provider="openai", display_name="GPT-4o (OpenAI)", available=False
    ),
    LLMBackend(
        id="claude",
        provider="anthropic",
        display_name="Claude Sonnet (Anthropic)",
        available=False,
    ),
    LLMBackend(
        id="openrouter",
        provider="openrouter",
        display_name="OpenRouter",
        available=False,
    ),
    *[
        LLMBackend(
            id=profile.backend_id,
            provider="deepseek",
            display_name=profile.display_name,
            available=False,
            model_name=profile.model,
            thinking=profile.thinking,
        )
        for profile in DEEPSEEK_PROFILES
    ],
    LLMBackend(
        id="ollama", provider="ollama", display_name="Ollama (local)", available=False
    ),
]

_KEY_MAP = {
    "chatgpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "deepseek-v4.1": "DEEPSEEK_API_KEY",
    "deepseek-thinking": "DEEPSEEK_API_KEY",
    "ollama": None,
}


@router.get("/status", response_model=ServerStatus)
async def get_status() -> ServerStatus:
    return ServerStatus(
        sandbox_type=_detect_sandbox(),
        active_sessions=len(
            [
                s
                for s in session_manager.list_sessions()
                if s.status.value in ("initializing", "idle", "running", "recovering")
            ]
        ),
    )


@router.get("/config/backends", response_model=List[LLMBackend])
async def get_backends() -> List[LLMBackend]:
    load_dotenv(dotenv_path=ENV_FILE)
    result = []
    for b in _BACKENDS:
        key_name = _KEY_MAP.get(b.id)
        if b.id == "ollama":
            ollama = probe_ollama(os.environ.get("OLLAMA_HOST"))
            available = ollama.status in {"ready", "not_running"}
            result.append(
                b.copy(
                    update={
                        "available": available,
                        "status": ollama.status,
                        "message": ollama.message,
                        "suggested_fix": ollama.suggested_fix,
                    }
                )
            )
        else:
            available = bool(os.environ.get(key_name)) if key_name else True
            update: dict[str, object] = {"available": available}
            if b.id == "openrouter" and not available:
                update.update(
                    {
                        "status": "not_configured",
                        "message": "OpenRouter API key is not configured.",
                        "suggested_fix": "Add OPENROUTER_API_KEY in Settings or run "
                        "'caribou config set-openrouter-key'.",
                    }
                )
            result.append(b.copy(update=update))
    return result


@router.get(
    "/config/python-environments",
    response_model=List[PythonEnvironmentCandidate],
)
async def get_python_environments() -> List[PythonEnvironmentCandidate]:
    """Discover usable Python prefixes visible to the CARIBOU server host."""

    return discover_python_environments()


@router.post(
    "/config/python-environments/validate",
    response_model=PythonEnvironmentCandidate,
)
async def validate_python_environment(
    body: PythonEnvironmentPathRequest,
) -> PythonEnvironmentCandidate:
    try:
        return validate_python_environment_path(body.path)
    except PythonEnvironmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/config/blueprints", response_model=List[AgentBlueprint])
async def get_blueprints() -> List[AgentBlueprint]:
    from caribou.agents.AgentSystem import AgentSystem
    from caribou.cli.run_cli import PACKAGE_AGENTS_DIR

    blueprints = []
    seen_names: set = set()
    for search_dir, is_pkg in (
        (DEFAULT_AGENT_DIR, False),
        (Path(PACKAGE_AGENTS_DIR), True),
    ):
        p = Path(search_dir)
        if not p.exists():
            continue
        for json_file in sorted(p.glob("*.json")):
            if json_file.stem in seen_names:
                continue
            seen_names.add(json_file.stem)
            try:
                sys = AgentSystem.load_from_json(str(json_file))
                blueprints.append(
                    AgentBlueprint(
                        name=json_file.stem,
                        description=getattr(sys, "description", json_file.stem),
                        agents=list(sys.agents.keys()),
                        has_rag=any(a.is_rag_enabled for a in sys.agents.values()),
                        path=str(json_file),
                        is_package_default=is_pkg,
                    )
                )
            except Exception:
                pass
    return blueprints


class ServerSettings(BaseModel):
    caribou_home: str
    sessions_dir: str
    uploads_dir: str
    env_file: str
    api_keys: Dict[str, str]  # key name → masked value
    ollama_host: str
    ollama_model: str


class UpdateSettingsRequest(BaseModel):
    sessions_dir: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    ollama_host: Optional[str] = None
    ollama_model: Optional[str] = None


@router.get("/settings", response_model=ServerSettings)
async def get_settings() -> ServerSettings:
    load_dotenv(dotenv_path=ENV_FILE, override=True)

    def _mask(val: str) -> str:
        if not val:
            return ""
        return val[:8] + "•" * min(len(val) - 8, 32) if len(val) > 8 else "•" * len(val)

    return ServerSettings(
        caribou_home=str(CARIBOU_HOME),
        sessions_dir=str(_SESSIONS_DIR),
        uploads_dir=str(CARIBOU_HOME / "server_uploads"),
        env_file=str(ENV_FILE),
        api_keys={
            "OPENAI_API_KEY": _mask(os.environ.get("OPENAI_API_KEY", "")),
            "ANTHROPIC_API_KEY": _mask(os.environ.get("ANTHROPIC_API_KEY", "")),
            "DEEPSEEK_API_KEY": _mask(os.environ.get("DEEPSEEK_API_KEY", "")),
            "OPENROUTER_API_KEY": _mask(os.environ.get("OPENROUTER_API_KEY", "")),
        },
        ollama_host=normalize_host(os.environ.get("OLLAMA_HOST")),
        ollama_model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    )


@router.patch("/settings")
async def update_settings(body: UpdateSettingsRequest) -> dict:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.touch()

    updated = []
    if body.openai_api_key is not None:
        set_key(str(ENV_FILE), "OPENAI_API_KEY", body.openai_api_key)
        updated.append("OPENAI_API_KEY")
    if body.anthropic_api_key is not None:
        set_key(str(ENV_FILE), "ANTHROPIC_API_KEY", body.anthropic_api_key)
        updated.append("ANTHROPIC_API_KEY")
    if body.deepseek_api_key is not None:
        set_key(str(ENV_FILE), "DEEPSEEK_API_KEY", body.deepseek_api_key)
        updated.append("DEEPSEEK_API_KEY")
    if body.openrouter_api_key is not None:
        set_key(str(ENV_FILE), "OPENROUTER_API_KEY", body.openrouter_api_key)
        updated.append("OPENROUTER_API_KEY")
    if body.ollama_host is not None:
        set_key(str(ENV_FILE), "OLLAMA_HOST", normalize_host(body.ollama_host))
        updated.append("OLLAMA_HOST")
    if body.ollama_model is not None:
        set_key(str(ENV_FILE), "OLLAMA_MODEL", body.ollama_model.strip())
        updated.append("OLLAMA_MODEL")

    if body.sessions_dir is not None:
        p = Path(body.sessions_dir).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise HTTPException(400, f"Cannot create sessions directory: {exc}")
        set_key(str(ENV_FILE), "CARIBOU_SESSIONS_DIR", str(p))
        updated.append("CARIBOU_SESSIONS_DIR")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return {"updated": updated}


@router.get("/config/openrouter/models")
async def get_openrouter_models(refresh: bool = False) -> dict[str, object]:
    """Proxy the account-filtered catalogue without exposing its API key."""

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured")
    try:
        return get_openrouter_catalogue(key, refresh=refresh).as_dict()
    except OpenRouterError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/config/openrouter/endpoints")
async def get_openrouter_model_endpoints(model_id: str) -> dict[str, object]:
    """Return selectable upstream endpoints for one canonical model."""

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured")
    try:
        endpoints = get_openrouter_endpoints(key, model_id)
    except OpenRouterError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {
        "model_id": model_id,
        "endpoints": [endpoint.as_dict() for endpoint in endpoints],
        "catalog_url": OPENROUTER_CATALOG_URL,
    }


@router.get("/config/ollama/models", response_model=OllamaModelsResponse)
async def get_ollama_models() -> OllamaModelsResponse:
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    default_model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    status = probe_ollama(os.environ.get("OLLAMA_HOST"))
    if default_model not in status.models and status.models:
        default_model = status.models[0]
    return OllamaModelsResponse(
        host=status.host,
        running=status.running,
        models=status.models,
        default_model=default_model,
        status=status.status,
        message=status.message,
        suggested_fix=status.suggested_fix,
    )


@router.post("/config/ollama/start", response_model=OllamaModelsResponse)
async def start_ollama_server() -> OllamaModelsResponse:
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    default_model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    try:
        status = start_ollama(os.environ.get("OLLAMA_HOST"))
    except OllamaStartupError as exc:
        raise HTTPException(
            400,
            {
                "code": exc.code,
                "message": str(exc),
                "suggested_fix": exc.suggested_fix,
            },
        )
    if default_model not in status.models and status.models:
        default_model = status.models[0]
    return OllamaModelsResponse(
        host=status.host,
        running=status.running,
        models=status.models,
        default_model=default_model,
        status=status.status,
        message=status.message,
        suggested_fix=status.suggested_fix,
    )


# ---------------------------------------------------------------------------
# Blueprint CRUD helpers
# ---------------------------------------------------------------------------


def _get_package_agents_dir() -> Path:
    from caribou.cli.run_cli import PACKAGE_AGENTS_DIR

    return Path(PACKAGE_AGENTS_DIR)


def _is_package_default(name: str) -> bool:
    return (_get_package_agents_dir() / f"{name}.json").exists()


def _resolve_blueprint_path(name: str) -> Path:
    """Return the path to a blueprint file, searching user dir then package dir."""
    for search_dir in (DEFAULT_AGENT_DIR, _get_package_agents_dir()):
        candidate = Path(search_dir) / f"{name}.json"
        if candidate.exists():
            return candidate
    raise HTTPException(404, f"Blueprint '{name}' not found.")


def _load_blueprint_content(name: str) -> BlueprintContent:
    """Parse a blueprint JSON file into BlueprintContent."""
    path = _resolve_blueprint_path(name)
    with open(path) as f:
        raw = json.load(f)

    agents: Dict[str, AgentConfig] = {}
    for agent_name, agent_raw in raw.get("agents", {}).items():
        neighbors = {
            cmd_name: CommandConfig(
                target_agent=cmd_data["target_agent"],
                description=cmd_data.get("description", ""),
            )
            for cmd_name, cmd_data in agent_raw.get("neighbors", {}).items()
        }
        agents[agent_name] = AgentConfig(
            prompt=agent_raw.get("prompt", ""),
            rag_enabled=agent_raw.get("rag", {}).get("enabled", False),
            neighbors=neighbors,
            code_samples=agent_raw.get("code_samples", []),
        )

    global_policy = raw.get("global_policy", "")
    if not isinstance(global_policy, str):
        global_policy = json.dumps(global_policy, indent=2)

    return BlueprintContent(
        name=name,
        global_policy=global_policy,
        agents=agents,
        is_package_default=_is_package_default(name),
        evaluator_agent=raw.get("evaluator_agent"),
        work_item_policy=raw.get("work_item_policy") or {"qc_mode": "optional"},
    )


def _to_disk_dict(req: SaveBlueprintRequest) -> dict:
    """Convert SaveBlueprintRequest to the on-disk JSON structure."""
    agents_dict = {}
    for agent_name, agent in req.agents.items():
        agents_dict[agent_name] = {
            "prompt": agent.prompt,
            "rag": {"enabled": agent.rag_enabled},
            "neighbors": {
                cmd_name: {
                    "target_agent": cmd.target_agent,
                    "description": cmd.description,
                }
                for cmd_name, cmd in agent.neighbors.items()
            },
            **({"code_samples": agent.code_samples} if agent.code_samples else {}),
        }
    return {
        "global_policy": req.global_policy,
        "evaluator_agent": req.evaluator_agent,
        "work_item_policy": req.work_item_policy.model_dump(),
        "agents": agents_dict,
    }


def _validate_blueprint(req: SaveBlueprintRequest) -> None:
    """Raise HTTPException 422 on structural validation failure."""
    if (
        not req.name
        or "/" in req.name
        or "\\" in req.name
        or req.name.endswith(".json")
    ):
        raise HTTPException(422, "Invalid blueprint name.")
    if not req.agents:
        raise HTTPException(422, "Blueprint must have at least one agent.")
    agent_keys = set(req.agents.keys())
    for agent_name, agent in req.agents.items():
        if not agent_name:
            raise HTTPException(422, "Agent names must be non-empty.")
        for cmd_name, cmd in agent.neighbors.items():
            if cmd.target_agent not in agent_keys:
                raise HTTPException(
                    422,
                    f"Agent '{agent_name}' command '{cmd_name}' references unknown agent '{cmd.target_agent}'.",
                )
    if req.evaluator_agent is not None and req.evaluator_agent not in agent_keys:
        raise HTTPException(
            422,
            f"evaluator_agent '{req.evaluator_agent}' does not match any defined agent.",
        )
    if req.work_item_policy.qc_mode == "required" and req.evaluator_agent is None:
        raise HTTPException(
            422, "Required work-item QC needs an evaluator_agent in this blueprint."
        )


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Blueprint CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("/config/blueprints/{name}", response_model=BlueprintContent)
async def get_blueprint(name: str) -> BlueprintContent:
    return _load_blueprint_content(name)


@router.post("/config/blueprints", response_model=BlueprintContent, status_code=201)
async def create_blueprint(req: SaveBlueprintRequest) -> BlueprintContent:
    _validate_blueprint(req)
    dest = DEFAULT_AGENT_DIR / f"{req.name}.json"
    if dest.exists():
        raise HTTPException(409, f"Blueprint '{req.name}' already exists.")
    _atomic_write(dest, _to_disk_dict(req))
    return _load_blueprint_content(req.name)


@router.put("/config/blueprints/{name}", response_model=BlueprintContent)
async def update_blueprint(name: str, req: SaveBlueprintRequest) -> BlueprintContent:
    if _is_package_default(name):
        raise HTTPException(
            403, f"Blueprint '{name}' is a package default and cannot be modified."
        )
    user_path = DEFAULT_AGENT_DIR / f"{name}.json"
    if not user_path.exists():
        raise HTTPException(404, f"Blueprint '{name}' not found in user blueprints.")
    _validate_blueprint(req)
    req = SaveBlueprintRequest(
        name=name,
        global_policy=req.global_policy,
        agents=req.agents,
        evaluator_agent=req.evaluator_agent,
        work_item_policy=req.work_item_policy,
    )
    _atomic_write(user_path, _to_disk_dict(req))
    return _load_blueprint_content(name)


@router.delete("/config/blueprints/{name}", status_code=204)
async def delete_blueprint(name: str) -> None:
    if _is_package_default(name):
        raise HTTPException(
            403, f"Blueprint '{name}' is a package default and cannot be deleted."
        )
    user_path = DEFAULT_AGENT_DIR / f"{name}.json"
    if not user_path.exists():
        raise HTTPException(404, f"Blueprint '{name}' not found in user blueprints.")
    user_path.unlink()


_USER_CODE_SAMPLES_DIR = CARIBOU_HOME / "code_samples"
_PACKAGE_CODE_SAMPLES_DIR = (
    Path(__file__).resolve().parent.parent.parent / "code_samples"
)


class ImportCodeSampleRequest(BaseModel):
    source_path: str


class ImportCodeSampleResponse(BaseModel):
    filename: str
    destination: str


class CodeSampleInfo(BaseModel):
    filename: str
    size_bytes: int
    is_builtin: bool


class CodeSampleContent(BaseModel):
    filename: str
    content: str
    is_builtin: bool


class CreateCodeSampleRequest(BaseModel):
    filename: str
    content: str


class UpdateCodeSampleRequest(BaseModel):
    content: str


def _is_builtin_sample(filename: str) -> bool:
    return (_PACKAGE_CODE_SAMPLES_DIR / filename).exists() and not (
        _USER_CODE_SAMPLES_DIR / filename
    ).exists()


@router.post("/config/code-samples/import", response_model=ImportCodeSampleResponse)
async def import_code_sample(req: ImportCodeSampleRequest) -> ImportCodeSampleResponse:
    src = Path(req.source_path)
    if not src.is_absolute():
        raise HTTPException(400, "source_path must be an absolute path.")
    if not src.exists():
        raise HTTPException(404, f"File not found: {src}")
    if not src.is_file():
        raise HTTPException(400, f"Path is not a regular file: {src}")

    _USER_CODE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    dest = _USER_CODE_SAMPLES_DIR / src.name
    if dest.exists():
        raise HTTPException(
            409,
            f"A code sample named '{src.name}' already exists. Rename the source file or remove the existing sample.",
        )

    shutil.copy2(src, dest)
    return ImportCodeSampleResponse(filename=src.name, destination=str(dest))


@router.get("/config/code-samples", response_model=List[CodeSampleInfo])
async def list_code_samples() -> List[CodeSampleInfo]:
    samples: dict[str, CodeSampleInfo] = {}

    if _PACKAGE_CODE_SAMPLES_DIR.exists():
        for f in sorted(_PACKAGE_CODE_SAMPLES_DIR.glob("*.py")):
            samples[f.name] = CodeSampleInfo(
                filename=f.name,
                size_bytes=f.stat().st_size,
                is_builtin=True,
            )

    if _USER_CODE_SAMPLES_DIR.exists():
        for f in sorted(_USER_CODE_SAMPLES_DIR.glob("*.py")):
            samples[f.name] = CodeSampleInfo(
                filename=f.name,
                size_bytes=f.stat().st_size,
                is_builtin=False,
            )

    return list(samples.values())


@router.get("/config/code-samples/{filename}", response_model=CodeSampleContent)
async def get_code_sample(filename: str) -> CodeSampleContent:
    user_path = _USER_CODE_SAMPLES_DIR / filename
    if user_path.exists():
        return CodeSampleContent(
            filename=filename, content=user_path.read_text(), is_builtin=False
        )

    pkg_path = _PACKAGE_CODE_SAMPLES_DIR / filename
    if pkg_path.exists():
        return CodeSampleContent(
            filename=filename, content=pkg_path.read_text(), is_builtin=True
        )

    raise HTTPException(404, f"Code sample '{filename}' not found.")


@router.post("/config/code-samples", response_model=CodeSampleContent, status_code=201)
async def create_code_sample(req: CreateCodeSampleRequest) -> CodeSampleContent:
    if not req.filename or "/" in req.filename or "\\" in req.filename:
        raise HTTPException(422, "Invalid filename.")
    dest = _USER_CODE_SAMPLES_DIR / req.filename
    if dest.exists():
        raise HTTPException(
            409, f"A code sample named '{req.filename}' already exists."
        )
    _USER_CODE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_text(req.content)
    return CodeSampleContent(
        filename=req.filename, content=req.content, is_builtin=False
    )


@router.put("/config/code-samples/{filename}", response_model=CodeSampleContent)
async def update_code_sample(
    filename: str, req: UpdateCodeSampleRequest
) -> CodeSampleContent:
    if _is_builtin_sample(filename):
        raise HTTPException(
            403,
            f"'{filename}' is a built-in sample and cannot be modified. Clone it first.",
        )
    dest = _USER_CODE_SAMPLES_DIR / filename
    if not dest.exists():
        raise HTTPException(404, f"Code sample '{filename}' not found.")
    dest.write_text(req.content)
    return CodeSampleContent(filename=filename, content=req.content, is_builtin=False)


@router.delete("/config/code-samples/{filename}", status_code=204)
async def delete_code_sample(filename: str) -> None:
    if _is_builtin_sample(filename):
        raise HTTPException(
            403, f"'{filename}' is a built-in sample and cannot be deleted."
        )
    dest = _USER_CODE_SAMPLES_DIR / filename
    if not dest.exists():
        raise HTTPException(404, f"Code sample '{filename}' not found.")
    dest.unlink()


def _detect_sandbox() -> str:
    if shutil.which("singularity"):
        return "singularity"
    if shutil.which("docker"):
        return "docker"
    return "offline"
