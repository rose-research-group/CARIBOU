"""Run the existing CARIBOU agent session through the durable control plane."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from rich.console import Console

from caribou.config import ENV_FILE
from caribou.control.api import ControlError, ExitCode
from caribou.domain.enums import ArtifactType, EventType, MemoryStrategy, RunState
from caribou.domain.models import (
    AgentSwitchPayload,
    Artifact,
    BlockChangedPayload,
    CodeResultPayload,
    CodeSubmittedPayload,
    ContentReference,
    Event,
    HeartbeatPayload,
    MessagePayload,
    RagPayload,
    WorkItemChangedPayload,
)
from caribou.domain.serialization import file_hash, sha256_bytes
from caribou.execution.runner import (
    AgentSessionCheckpointState,
    AgentSessionResult,
    RunnerEvent,
    run_agent_session,
)

from .records import ProviderCallReceipt, ProviderCallReceiptV2, ProviderCallUsage
from .specs import (
    AGENT_PATH_SMOKE_ADAPTER,
    AGENT_SMOKE_DELAY_PARAMETER,
    CARIBOU_AGENT_ADAPTER,
    _model_deepseek_options,
    _model_max_output_tokens,
    _model_openrouter_options,
)
from .store import ExperimentStore, SUPPORTED_RESUME_REQUIREMENTS


SANDBOX_DATA_PATH = "/workspace/dataset.h5ad"
FROZEN_RAG_CORPUS_ENV = "CARIBOU_RAG_CORPUS_FILE"
_SMOKE_CODE = 'print("CARIBOU_AGENT_PATH_OK")'
_CHECKPOINT_DATASET_FILENAME = "checkpoint-dataset.h5ad"
_CHECKPOINT_DATASET_CONTAINER_PATH = (
    f"/workspace/outputs/.{_CHECKPOINT_DATASET_FILENAME}"
)
_CHECKPOINT_DATASET_CAPTURE_CODE = f"""\
import os as _caribou_os
_caribou_candidate = globals().get("adata")
if _caribou_candidate is None or not hasattr(_caribou_candidate, "write_h5ad"):
    raise RuntimeError("CARIBOU checkpoint requires an AnnData global named adata")
_caribou_target = {_CHECKPOINT_DATASET_CONTAINER_PATH!r}
_caribou_temporary = _caribou_target + ".tmp"
_caribou_candidate.write_h5ad(_caribou_temporary)
_caribou_os.replace(_caribou_temporary, _caribou_target)
"""
_CHECKPOINT_DATASET_RESTORE_CODE = f"""\
import anndata as _caribou_anndata
adata = _caribou_anndata.read_h5ad({SANDBOX_DATA_PATH!r})
"""


@dataclass(frozen=True)
class _RestoredAgentCheckpoint:
    source_run_id: str
    checkpoint_id: str
    dataset_path: Path
    history: list[dict[str, str]]
    runner_state: AgentSessionCheckpointState


def _verify_code_identity(
    expected_commit: str,
    adapter: str,
    *,
    expected_dirty: bool = False,
) -> None:
    root_result = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).resolve().parent),
            "rev-parse",
            "--show-toplevel",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if root_result.returncode != 0:
        raise ControlError(
            "CODE_IDENTITY_UNAVAILABLE",
            "agent workloads require a verifiable Git checkout",
            exit_code=ExitCode.integrity,
        )
    repository_root = Path(root_result.stdout.strip())
    head_result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    status_result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if head_result.returncode != 0 or status_result.returncode != 0:
        raise ControlError(
            "CODE_IDENTITY_UNAVAILABLE",
            "the executing Git checkout could not be inspected",
            exit_code=ExitCode.integrity,
        )
    actual = head_result.stdout.strip()
    dirty = bool(status_result.stdout.strip())
    dirty_mismatch = dirty != expected_dirty
    dirty_not_allowed = dirty and adapter != AGENT_PATH_SMOKE_ADAPTER
    if actual != expected_commit or dirty_mismatch or dirty_not_allowed:
        raise ControlError(
            "CODE_COMMIT_MISMATCH",
            "the executing CARIBOU checkout does not match the frozen code identity",
            exit_code=ExitCode.integrity,
            details={
                "expected": expected_commit,
                "actual": actual,
                "expected_dirty": expected_dirty,
                "dirty": dirty,
            },
        )


def _local_file(
    reference: ContentReference, *, role: str, verify_hash: bool = True
) -> Path:
    parsed = urlparse(reference.uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ControlError(
            "CONTENT_URI_UNSUPPORTED",
            f"{role} must use a local file URI",
            exit_code=ExitCode.validation,
            details={"role": role, "uri": reference.uri},
        )
    path = Path(unquote(parsed.path)).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ControlError(
            "CONTENT_FILE_INVALID",
            f"{role} must resolve to a regular non-symlink file",
            exit_code=ExitCode.integrity,
            details={"role": role, "path": str(path)},
        )
    resolved = path.resolve()
    actual_hash = file_hash(resolved) if verify_hash else None
    if actual_hash is not None and actual_hash != reference.content_hash:
        raise ControlError(
            "CONTENT_HASH_MISMATCH",
            f"{role} does not match its frozen SHA-256",
            exit_code=ExitCode.integrity,
            details={
                "role": role,
                "path": str(resolved),
                "expected": reference.content_hash,
                "actual": actual_hash,
            },
        )
    if (
        reference.size_bytes is not None
        and resolved.stat().st_size != reference.size_bytes
    ):
        raise ControlError(
            "CONTENT_SIZE_MISMATCH",
            f"{role} does not match its frozen size",
            exit_code=ExitCode.integrity,
            details={
                "role": role,
                "path": str(resolved),
                "expected": reference.size_bytes,
                "actual": resolved.stat().st_size,
            },
        )
    return resolved


class _ScriptedCompletions:
    def __init__(self, delay_seconds: float, completed_turns: int = 0) -> None:
        self._delay_seconds = delay_seconds
        responses = (
            "delegate_to_general",
            f"```python\n{_SMOKE_CODE}\n```",
            "end_session",
        )
        self._responses = iter(responses[completed_turns:])

    def create(self, **_: Any) -> SimpleNamespace:
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        try:
            content = next(self._responses)
        except StopIteration:
            content = "end_session"
        message = SimpleNamespace(content=content, role="assistant")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _ScriptedClient:
    def __init__(self, delay_seconds: float, completed_turns: int = 0) -> None:
        self.chat = SimpleNamespace(
            completions=_ScriptedCompletions(delay_seconds, completed_turns)
        )


class _RecordingSandbox:
    def __init__(self) -> None:
        self.started = False

    def set_data(self, _: list[tuple[Path, str]], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def start_container(self) -> bool:
        self.started = True
        return True

    def stop_container(self) -> None:
        self.started = False

    def exec_code(self, code: str, timeout: int) -> dict[str, Any]:
        if not self.started:
            raise RuntimeError("scripted sandbox is not running")
        if timeout <= 0 or code.strip() != _SMOKE_CODE:
            raise RuntimeError("scripted sandbox received unexpected code")
        return {
            "status": "ok",
            "stdout": "CARIBOU_AGENT_PATH_OK\n",
            "stderr": "",
            "images": [],
        }


class _CancellationAwareSandbox:
    """Forward the real backend while turning durable cancellation into an event."""

    def __init__(self, backend: Any, should_cancel: Callable[[], bool]) -> None:
        self._backend = backend
        self._should_cancel = should_cancel

    def set_data(self, resources: list[tuple[Path, str]], output_dir: Path) -> None:
        self._backend.set_data(resources, output_dir)

    def start_container(self) -> bool:
        return bool(self._backend.start_container())

    def stop_container(self) -> None:
        self._backend.stop_container()

    def exec_code(self, code: str, timeout: int) -> dict[str, Any]:
        cancel_event = threading.Event()
        watcher_stop = threading.Event()

        def watch() -> None:
            while not watcher_stop.wait(0.05):
                if self._should_cancel():
                    cancel_event.set()
                    return

        if self._should_cancel():
            cancel_event.set()
        watcher = threading.Thread(
            target=watch,
            name="caribou-sandbox-cancel-watch",
            daemon=True,
        )
        watcher.start()
        try:
            result = self._backend.exec_code(
                code,
                timeout=timeout,
                cancel_event=cancel_event,
            )
            if not isinstance(result, dict):
                raise RuntimeError("sandbox returned a non-object result")
            return result
        finally:
            watcher_stop.set()
            watcher.join(timeout=1)


def _verify_blueprint_dependencies(agent_system: Any, run: Any) -> Path | None:
    actual_samples: dict[str, str] = {}
    rag_enabled_agents: list[str] = []
    for agent_name, agent in agent_system.get_all_agents().items():
        if agent.is_rag_enabled:
            rag_enabled_agents.append(str(agent_name))
        for filename, content in agent.code_samples.items():
            digest = sha256_bytes(content.encode("utf-8"))
            previous = actual_samples.get(str(filename))
            if previous is not None and previous != digest:
                raise ControlError(
                    "CODE_SAMPLE_AMBIGUOUS",
                    "agents loaded different content under one code-sample name",
                    exit_code=ExitCode.integrity,
                    details={"filename": str(filename)},
                )
            actual_samples[str(filename)] = digest
    expected_samples = dict(run.resolved_blueprint.code_sample_hashes)
    if actual_samples != expected_samples:
        raise ControlError(
            "CODE_SAMPLE_HASH_MISMATCH",
            "loaded code samples do not match the frozen blueprint manifest",
            exit_code=ExitCode.integrity,
            details={
                "expected": expected_samples,
                "actual": actual_samples,
            },
        )
    rag_reference = run.resolved_blueprint.rag_corpus
    if rag_enabled_agents and rag_reference is None:
        raise ControlError(
            "RAG_NOT_BOUND",
            "RAG-enabled agents require a frozen corpus in the resolved blueprint",
            exit_code=ExitCode.validation,
            details={"rag_enabled_agents": sorted(rag_enabled_agents)},
        )
    if not rag_enabled_agents and rag_reference is not None:
        raise ControlError(
            "RAG_CORPUS_UNUSED",
            "a frozen RAG corpus cannot be declared when no agent enables RAG",
            exit_code=ExitCode.validation,
        )
    if rag_reference is None:
        return None
    return _local_file(rag_reference, role="RAG corpus")


def _provider_client(provider: str, parameters: dict[str, Any]) -> object:
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    # Validate the frozen model request here as well as during preflight so a
    # direct worker invocation cannot silently discard unsupported parameters.
    _model_max_output_tokens(parameters)
    thinking, reasoning_effort = _model_deepseek_options(parameters)
    openrouter_endpoint = _model_openrouter_options(parameters)
    if provider == "openai":
        if thinking is not None or reasoning_effort is not None:
            raise RuntimeError("DeepSeek thinking controls require provider=deepseek")
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return OpenAI(api_key=key, max_retries=0)
    if provider == "deepseek":
        from caribou.core.deepseek import create_deepseek_client

        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        return create_deepseek_client(
            key,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_retries=0,
        )
    if provider == "openrouter":
        from caribou.core.openrouter import create_openrouter_client

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        if openrouter_endpoint is None:
            raise RuntimeError("OpenRouter experiment endpoint is not frozen")
        return create_openrouter_client(
            key,
            endpoint=openrouter_endpoint,
            max_retries=0,
        )
    raise RuntimeError(f"unsupported provider: {provider}")


def _receipt_text(observation: dict[str, object], key: str) -> str | None:
    value = observation.get(key)
    return value if isinstance(value, str) and value else None


def _required_receipt_text(observation: dict[str, object], key: str) -> str:
    value = _receipt_text(observation, key)
    if value is None:
        raise RuntimeError(f"provider observation field {key!r} is not text")
    return value


def _receipt_int(observation: dict[str, object], key: str) -> int | None:
    value = observation.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _required_receipt_int(observation: dict[str, object], key: str) -> int:
    value = _receipt_int(observation, key)
    if value is None:
        raise RuntimeError(f"provider observation field {key!r} is not an integer")
    return value


def _receipt_datetime(observation: dict[str, object], key: str) -> datetime:
    value = _required_receipt_text(observation, key)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(
            f"provider observation field {key!r} is not a timestamp"
        ) from exc
    return parsed


def _provider_receipt_recorder(
    store: ExperimentStore,
    run_id: str,
) -> Callable[[dict[str, object]], None]:
    run = store.run(run_id)

    def record(observation: dict[str, object]) -> None:
        turn = _required_receipt_int(observation, "turn")
        attempt = _required_receipt_int(observation, "attempt")
        maximum_attempts = _required_receipt_int(observation, "maximum_attempts")
        requested_model = _required_receipt_text(observation, "requested_model")
        if requested_model != run.resolved_model.model:
            raise RuntimeError(
                "provider observation requested model differs from frozen run"
            )
        if maximum_attempts != run.resolved_stop_rules.retry.maximum_attempts:
            raise RuntimeError(
                "provider observation retry policy differs from frozen run"
            )
        if (
            run.resolved_model.provider == "openrouter"
            and observation.get("outcome") == "succeeded"
        ):
            response_model = _required_receipt_text(observation, "response_model")
            if response_model != requested_model:
                raise RuntimeError(
                    "OpenRouter response model differs from the frozen canonical model"
                )
            endpoint = str(
                run.resolved_model.parameters.get("openrouter_endpoint", "")
            ).split("/", 1)[0]
            upstream = _required_receipt_text(observation, "upstream_provider")

            def normalize(value: str) -> str:
                return "".join(
                    character for character in value.casefold() if character.isalnum()
                )

            if normalize(endpoint) != normalize(upstream):
                raise RuntimeError(
                    "OpenRouter upstream provider differs from the frozen endpoint"
                )
        receipt_type = (
            ProviderCallReceiptV2
            if run.resolved_model.provider == "openrouter"
            else ProviderCallReceipt
        )
        receipt = receipt_type(
            call_id=(f"{run_id}:turn:{turn}:attempt:{attempt}"),
            run_id=run_id,
            turn=turn,
            agent_name=_required_receipt_text(observation, "agent_name"),
            attempt=attempt,
            maximum_attempts=maximum_attempts,
            provider=run.resolved_model.provider,
            requested_model=requested_model,
            outcome=_required_receipt_text(observation, "outcome"),  # type: ignore[arg-type]
            started_at=_receipt_datetime(observation, "started_at"),
            ended_at=_receipt_datetime(observation, "ended_at"),
            duration_ms=_required_receipt_int(observation, "duration_ms"),
            response_id=_receipt_text(observation, "response_id"),
            request_id=_receipt_text(observation, "request_id"),
            response_model=_receipt_text(observation, "response_model"),
            system_fingerprint=_receipt_text(observation, "system_fingerprint"),
            finish_reason=_receipt_text(observation, "finish_reason"),
            usage=ProviderCallUsage(
                prompt_tokens=_receipt_int(observation, "prompt_tokens"),
                completion_tokens=_receipt_int(observation, "completion_tokens"),
                total_tokens=_receipt_int(observation, "total_tokens"),
                cached_tokens=_receipt_int(observation, "cached_tokens"),
                cache_miss_tokens=_receipt_int(observation, "cache_miss_tokens"),
                reasoning_tokens=_receipt_int(observation, "reasoning_tokens"),
            ),
            failure_type=_receipt_text(observation, "failure_type"),
            http_status_code=_receipt_int(observation, "http_status_code"),
            **(
                {
                    "upstream_provider": _receipt_text(
                        observation, "upstream_provider"
                    ),
                    "cost_usd": observation.get("cost_usd"),
                    "upstream_cost_usd": observation.get("upstream_cost_usd"),
                }
                if receipt_type is ProviderCallReceiptV2
                else {}
            ),
        )
        store.record_idempotent_json_artifact(
            run_id,
            filename=(f"provider-call-turn-{turn}-attempt-{attempt}.json"),
            role="provider_call_receipt",
            value=receipt.model_dump(mode="json"),
            producer="provider-client",
            artifact_type=ArtifactType.manifest,
            schema_type="caribou.provider_call_receipt",
            schema_version_name=(
                "v2" if receipt_type is ProviderCallReceiptV2 else "v1"
            ),
            turn=turn,
            current_agent=receipt.agent_name,
        )

    return record


def _real_sandbox(
    container_path: Path,
    container_hash: str,
    console: Console,
    *,
    gpu_enabled: bool,
) -> object:
    from caribou.core.sandbox_management import init_singularity_exec

    manager_class, _, _, _, _ = init_singularity_exec(
        str(Path(__file__).resolve().parent),
        SANDBOX_DATA_PATH,
        subprocess,
        console,
        sif_path=container_path,
        sif_sha256=container_hash,
        no_pull=True,
        gpu_enabled=gpu_enabled,
        celltypist_cache_enabled=False,
    )
    return manager_class()


def _artifact_type(path: Path) -> ArtifactType:
    suffix = path.suffix.lower()
    if suffix == ".h5ad":
        return ArtifactType.dataset
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return ArtifactType.plot
    if suffix == ".ipynb":
        return ArtifactType.notebook
    if suffix in {".csv", ".tsv", ".parquet"}:
        return ArtifactType.metric_table
    if suffix in {".md", ".html", ".json"}:
        return ArtifactType.report
    return ArtifactType.other


def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"runner event field {key!r} is not an integer")
    return value


def _payload_block_id(payload: dict[str, object]) -> str | None:
    """Return the runner's block attribution for a code event.

    ``block_id`` is optional in the journal; when the runner sends it, it must
    be a non-empty string.
    """
    if "block_id" not in payload:
        return None
    value = payload["block_id"]
    if not isinstance(value, str) or not value:
        raise RuntimeError("runner event field 'block_id' is not a non-empty string")
    return value


def _event_recorder(
    store: ExperimentStore, run_id: str
) -> Callable[[RunnerEvent], None]:
    def record(event: RunnerEvent) -> None:
        event_type = event["event_type"]
        turn = event["turn"]
        agent_name = event["agent_name"]
        payload = event["payload"]
        if event_type == "turn_started":
            store.append_run_event(
                run_id,
                event_type=EventType.heartbeat,
                payload=HeartbeatPayload(message="agent turn started"),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
                stage="agent_turn",
            )
            return
        if event_type == "assistant_message":
            store.append_run_event(
                run_id,
                event_type=EventType.message,
                payload=MessagePayload(
                    role=str(payload["role"]),
                    agent_name=agent_name,
                    content=str(payload["content"]),
                ),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
            )
            return
        if event_type == "agent_switch":
            to_agent = str(payload["to_agent"])
            store.append_run_event(
                run_id,
                event_type=EventType.agent_switch,
                payload=AgentSwitchPayload(
                    from_agent=str(payload["from_agent"]),
                    to_agent=to_agent,
                    command=str(payload["command"]),
                ),
                actor="agent-runner",
                turn=turn,
                current_agent=to_agent,
            )
            return
        if event_type == "rag_attempt":
            store.append_run_event(
                run_id,
                event_type=EventType.heartbeat,
                payload=HeartbeatPayload(message="RAG request started"),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
                stage="rag",
            )
            return
        if event_type == "rag_result":
            result_text = str(payload.get("content", "")) or str(
                payload.get("error", "")
            )
            result_artifact_id = None
            if result_text:
                artifact = store.record_text_artifact(
                    run_id,
                    filename=f"turn-{turn}-rag.txt",
                    role="rag_result",
                    text=result_text,
                    producer="agent-runner",
                    artifact_type=ArtifactType.log,
                    turn=turn,
                    current_agent=agent_name,
                )
                result_artifact_id = artifact.artifact_id
            store.append_run_event(
                run_id,
                event_type=EventType.rag,
                payload=RagPayload(
                    query=str(payload["query"]),
                    result_artifact_id=result_artifact_id,
                    success=bool(payload["success"]),
                ),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
            )
            return
        if event_type == "code_submitted":
            block_index = _payload_int(payload, "block_index")
            source = store.record_text_artifact(
                run_id,
                filename=f"turn-{turn}-block-{block_index}.py",
                role="generated_code",
                text=str(payload["source"]),
                producer="agent-runner",
                artifact_type=ArtifactType.code,
                media_type="text/x-python",
                turn=turn,
                current_agent=agent_name,
            )
            store.append_run_event(
                run_id,
                event_type=EventType.code_submitted,
                payload=CodeSubmittedPayload(
                    action_id=str(payload["action_id"]),
                    source_artifact_id=source.artifact_id,
                    agent_name=agent_name,
                    block_index=block_index,
                    total_blocks=_payload_int(payload, "total_blocks"),
                    block_id=_payload_block_id(payload),
                ),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
            )
            return
        if event_type == "code_blocks_ignored":
            store.append_run_event(
                run_id,
                event_type=EventType.heartbeat,
                payload=HeartbeatPayload(
                    message=(
                        f"ignored {_payload_int(payload, 'ignored_blocks')} "
                        "additional provider code block(s); executed only the "
                        "first complete block"
                    )
                ),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
                stage="code_block_limit",
            )
            return
        if event_type == "code_result":
            block_index = _payload_int(payload, "block_index")
            stdout_id = None
            stderr_id = None
            if payload.get("stdout"):
                stdout = store.record_text_artifact(
                    run_id,
                    filename=f"turn-{turn}-block-{block_index}.stdout.txt",
                    role="code_stdout",
                    text=str(payload["stdout"]),
                    producer="agent-runner",
                    artifact_type=ArtifactType.log,
                    turn=turn,
                    current_agent=agent_name,
                )
                stdout_id = stdout.artifact_id
            if payload.get("stderr"):
                stderr = store.record_text_artifact(
                    run_id,
                    filename=f"turn-{turn}-block-{block_index}.stderr.txt",
                    role="code_stderr",
                    text=str(payload["stderr"]),
                    producer="agent-runner",
                    artifact_type=ArtifactType.log,
                    turn=turn,
                    current_agent=agent_name,
                )
                stderr_id = stderr.artifact_id
            store.append_run_event(
                run_id,
                event_type=EventType.code_result,
                payload=CodeResultPayload(
                    action_id=str(payload["action_id"]),
                    success=bool(payload["success"]),
                    duration_ms=_payload_int(payload, "duration_ms"),
                    stdout_artifact_id=stdout_id,
                    stderr_artifact_id=stderr_id,
                    block_id=_payload_block_id(payload),
                ),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
            )
            return
        if event_type == "session_end":
            store.append_run_event(
                run_id,
                event_type=EventType.heartbeat,
                payload=HeartbeatPayload(
                    message=f"agent session ended: {payload['end_reason']}"
                ),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
                stage="agent_session_end",
            )
            return
        if event_type == "work_item_changed":
            item = payload["item"]
            if not isinstance(item, dict):
                raise RuntimeError("runner event field 'item' is not an object")
            store.append_run_event(
                run_id,
                event_type=EventType.work_item_changed,
                payload=WorkItemChangedPayload(
                    item_id=item["id"],
                    status=item["status"],
                    owner=item["owner"],
                    item=item,
                ),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
            )
            return
        if event_type == "block_changed":
            block = payload["block"]
            if not isinstance(block, dict):
                raise RuntimeError("runner event field 'block' is not an object")
            store.append_run_event(
                run_id,
                event_type=EventType.block_changed,
                payload=BlockChangedPayload(block=block),
                actor="agent-runner",
                turn=turn,
                current_agent=agent_name,
            )
            return
        raise RuntimeError(f"unsupported runner event: {event_type}")

    return record


def _checkpoint_artifact(
    store: ExperimentStore, run_id: str, artifact_id: str, role: str
) -> Artifact:
    artifact = store.artifact_manifest(run_id).artifact(artifact_id)
    if artifact is None or artifact.role != role:
        raise ControlError(
            "CHECKPOINT_COMPONENT_INVALID",
            "checkpoint component is missing or has the wrong role",
            exit_code=ExitCode.integrity,
            details={
                "run_id": run_id,
                "artifact_id": artifact_id,
                "expected_role": role,
            },
        )
    return artifact


def _checkpoint_json(
    store: ExperimentStore, artifact: Artifact, *, expected_keys: set[str]
) -> dict[str, object]:
    try:
        value = json.loads(store.artifact_path(artifact).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlError(
            "CHECKPOINT_COMPONENT_INVALID",
            "checkpoint JSON component could not be decoded",
            exit_code=ExitCode.integrity,
            details={"artifact_id": artifact.artifact_id},
        ) from exc
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ControlError(
            "CHECKPOINT_COMPONENT_INVALID",
            "checkpoint JSON component has an unexpected schema shape",
            exit_code=ExitCode.integrity,
            details={"artifact_id": artifact.artifact_id},
        )
    return value


def _durable_event_ledger(store: ExperimentStore, run_id: str) -> list[Event]:
    """Read the complete authoritative event stream without a hidden page cap."""

    events: list[Event] = []
    after = 0
    while True:
        page = store.events(run_id, after=after, limit=10_000)
        if not page:
            break
        events.extend(page)
        after = page[-1].sequence
        if len(page) < 10_000:
            break
    return events


def _checkpoint_value_hash(value: object) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _checkpoint_component_reference(artifact: Artifact) -> dict[str, object]:
    return {
        "artifact_id": artifact.artifact_id,
        "role": artifact.role,
        "content_hash": artifact.content_hash,
        "size_bytes": artifact.size_bytes,
        "producer_event_id": artifact.producer_event_id,
    }


def _load_restored_checkpoint(
    store: ExperimentStore, run: Any
) -> _RestoredAgentCheckpoint | None:
    if run.resumed_from_run_id is None:
        return None
    if run.resume_checkpoint_id is None:
        raise ControlError(
            "CHECKPOINT_LINEAGE_INVALID",
            "resumed run is missing its checkpoint identity",
            exit_code=ExitCode.integrity,
        )
    source = store.run(run.resumed_from_run_id)
    checkpoint = store.checkpoint(source.run_id, run.resume_checkpoint_id)
    if (
        source.experiment_id != run.experiment_id
        or source.owner != run.owner
        or checkpoint.checkpoint_id not in source.checkpoint_ids
        or frozenset(checkpoint.resume_requirements) != SUPPORTED_RESUME_REQUIREMENTS
    ):
        raise ControlError(
            "CHECKPOINT_LINEAGE_INVALID",
            "resume lineage or requirements do not match the frozen child",
            exit_code=ExitCode.integrity,
        )
    store.verify_artifacts(source.run_id)
    durable_events = _durable_event_ledger(store, source.run_id)

    dataset_artifact = _checkpoint_artifact(
        store,
        source.run_id,
        str(checkpoint.dataset_artifact_id),
        "checkpoint_dataset_state",
    )
    message_artifact = _checkpoint_artifact(
        store,
        source.run_id,
        str(checkpoint.message_history_artifact_id),
        "checkpoint_message_history",
    )
    state_artifact = _checkpoint_artifact(
        store,
        source.run_id,
        str(checkpoint.agent_state_artifact_id),
        "checkpoint_agent_state",
    )
    actions_artifact = _checkpoint_artifact(
        store,
        source.run_id,
        str(checkpoint.executed_actions_artifact_id),
        "checkpoint_executed_actions",
    )
    manifest_artifact = _checkpoint_artifact(
        store,
        source.run_id,
        checkpoint.artifact_manifest_id,
        "checkpoint_artifact_manifest",
    )

    message_value = _checkpoint_json(
        store,
        message_artifact,
        expected_keys={"schema_version", "run_id", "messages"},
    )
    if (
        message_value["schema_version"] != "caribou.message_history.v1"
        or message_value["run_id"] != source.run_id
        or not isinstance(message_value["messages"], list)
    ):
        raise ControlError(
            "CHECKPOINT_MESSAGE_HISTORY_INVALID",
            "checkpoint message history identity is invalid",
            exit_code=ExitCode.integrity,
        )
    history: list[dict[str, str]] = []
    for item in message_value["messages"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or not isinstance(item["role"], str)
            or not isinstance(item["content"], str)
        ):
            raise ControlError(
                "CHECKPOINT_MESSAGE_HISTORY_INVALID",
                "checkpoint message history contains an invalid message",
                exit_code=ExitCode.integrity,
            )
        history.append({"role": item["role"], "content": item["content"]})

    state_value = _checkpoint_json(
        store,
        state_artifact,
        expected_keys={"schema_version", "run_id", "state"},
    )
    state_payload = state_value["state"]
    if (
        state_value["schema_version"] != "caribou.checkpoint_agent_state.v1"
        or state_value["run_id"] != source.run_id
        or not isinstance(state_payload, dict)
        or set(state_payload) != set(AgentSessionCheckpointState.__dataclass_fields__)
    ):
        raise ControlError(
            "CHECKPOINT_AGENT_STATE_INVALID",
            "checkpoint agent state schema or identity is invalid",
            exit_code=ExitCode.integrity,
        )
    actions = state_payload.get("action_space_past_actions")
    if not isinstance(actions, list) or any(
        not isinstance(item, dict) for item in actions
    ):
        raise ControlError(
            "CHECKPOINT_AGENT_STATE_INVALID",
            "checkpoint action-space state is invalid",
            exit_code=ExitCode.integrity,
        )
    try:
        runner_state = AgentSessionCheckpointState(
            **{
                **state_payload,
                "action_space_past_actions": tuple(dict(item) for item in actions),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ControlError(
            "CHECKPOINT_AGENT_STATE_INVALID",
            "checkpoint agent state failed validation",
            exit_code=ExitCode.integrity,
        ) from exc
    if (
        runner_state.turns_completed != checkpoint.turn
        or runner_state.current_agent_name != source.current_agent
        or run.current_turn != checkpoint.turn
        or run.current_agent != runner_state.current_agent_name
    ):
        raise ControlError(
            "CHECKPOINT_CURSOR_MISMATCH",
            "checkpoint runner state differs from durable attempt lineage",
            exit_code=ExitCode.integrity,
        )

    action_value = _checkpoint_json(
        store,
        actions_artifact,
        expected_keys={
            "schema_version",
            "run_id",
            "through_event_sequence",
            "through_turn",
            "event_ids",
            "events_hash",
        },
    )
    through_sequence = action_value["through_event_sequence"]
    if (
        action_value["schema_version"] != "caribou.executed_action_ledger.v1"
        or action_value["run_id"] != source.run_id
        or action_value["through_turn"] != checkpoint.turn
        or isinstance(through_sequence, bool)
        or not isinstance(through_sequence, int)
        or through_sequence < 0
        or through_sequence >= checkpoint.event_sequence
    ):
        raise ControlError(
            "CHECKPOINT_ACTION_LEDGER_INVALID",
            "checkpoint action ledger cursor is invalid",
            exit_code=ExitCode.integrity,
        )
    action_event_types = {
        EventType.agent_switch,
        EventType.rag,
        EventType.code_submitted,
        EventType.code_result,
    }
    expected_action_events = [
        event.model_dump(mode="json")
        for event in durable_events
        if event.sequence <= through_sequence and event.event_type in action_event_types
    ]
    if action_value["event_ids"] != [
        event["event_id"] for event in expected_action_events
    ] or action_value["events_hash"] != _checkpoint_value_hash(expected_action_events):
        raise ControlError(
            "CHECKPOINT_ACTION_LEDGER_INVALID",
            "checkpoint action ledger differs from the durable event stream",
            exit_code=ExitCode.integrity,
        )

    manifest_value = _checkpoint_json(
        store,
        manifest_artifact,
        expected_keys={
            "schema_version",
            "run_id",
            "frontier_event_sequence",
            "components",
        },
    )
    expected_components = [
        _checkpoint_component_reference(artifact)
        for artifact in (
            dataset_artifact,
            message_artifact,
            state_artifact,
            actions_artifact,
        )
    ]
    frontier = manifest_value["frontier_event_sequence"]
    if (
        manifest_value["schema_version"] != "caribou.checkpoint_artifact_manifest.v1"
        or manifest_value["run_id"] != source.run_id
        or manifest_value["components"] != expected_components
        or isinstance(frontier, bool)
        or not isinstance(frontier, int)
        or frontier >= checkpoint.event_sequence
    ):
        raise ControlError(
            "CHECKPOINT_MANIFEST_INVALID",
            "checkpoint artifact frontier differs from its component records",
            exit_code=ExitCode.integrity,
        )
    frontier_event_ids = {
        event.event_id for event in durable_events if event.sequence <= frontier
    }
    if any(
        artifact.producer_event_id not in frontier_event_ids
        for artifact in (
            dataset_artifact,
            message_artifact,
            state_artifact,
            actions_artifact,
        )
    ):
        raise ControlError(
            "CHECKPOINT_MANIFEST_INVALID",
            "checkpoint artifact frontier differs from its component records",
            exit_code=ExitCode.integrity,
        )
    return _RestoredAgentCheckpoint(
        source_run_id=source.run_id,
        checkpoint_id=checkpoint.checkpoint_id,
        dataset_path=store.artifact_path(dataset_artifact),
        history=history,
        runner_state=runner_state,
    )


def _checkpoint_action_events(
    store: ExperimentStore, run_id: str
) -> list[dict[str, object]]:
    action_event_types = {
        EventType.agent_switch,
        EventType.rag,
        EventType.code_submitted,
        EventType.code_result,
    }
    return [
        event.model_dump(mode="json")
        for event in _durable_event_ledger(store, run_id)
        if event.event_type in action_event_types
    ]


def _capture_checkpoint_dataset(
    *,
    store: ExperimentStore,
    run_id: str,
    adapter: str,
    actor: str,
    sandbox: object,
    input_path: Path,
    output_dir: Path,
    turn: int,
    current_agent: str,
) -> Path:
    destination = output_dir / f".{_CHECKPOINT_DATASET_FILENAME}"
    if adapter == AGENT_PATH_SMOKE_ADAPTER:
        shutil.copyfile(input_path, destination)
        store.append_run_event(
            run_id,
            event_type=EventType.heartbeat,
            payload=HeartbeatPayload(message="scripted checkpoint dataset copied"),
            actor=actor,
            turn=turn,
            current_agent=current_agent,
            stage="checkpoint_capture",
        )
        return destination
    capture_source = store.record_idempotent_json_artifact(
        run_id,
        filename="checkpoint-capture-code.json",
        role="checkpoint_capture_code",
        value={
            "schema_version": "caribou.checkpoint_capture_code.v1",
            "source": _CHECKPOINT_DATASET_CAPTURE_CODE,
        },
        producer=actor,
        artifact_type=ArtifactType.checkpoint,
        turn=turn,
        current_agent=current_agent,
    )
    result = sandbox.exec_code(_CHECKPOINT_DATASET_CAPTURE_CODE, timeout=600)  # type: ignore[attr-defined]
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ControlError(
            "CHECKPOINT_DATASET_CAPTURE_FAILED",
            "the live AnnData state could not be written at the safe boundary",
            exit_code=ExitCode.execution,
            details={"capture_artifact_id": capture_source.artifact_id},
        )
    store.record_idempotent_json_artifact(
        run_id,
        filename="checkpoint-capture-result.json",
        role="checkpoint_capture_result",
        value={
            "schema_version": "caribou.checkpoint_capture_result.v1",
            "capture_artifact_id": capture_source.artifact_id,
            "status": "ok",
        },
        producer=actor,
        artifact_type=ArtifactType.checkpoint,
        turn=turn,
        current_agent=current_agent,
    )
    if not destination.is_file():
        raise ControlError(
            "CHECKPOINT_DATASET_CAPTURE_MISSING",
            "checkpoint capture reported success without a dataset file",
            exit_code=ExitCode.integrity,
        )
    return destination


def _persist_agent_checkpoint(
    *,
    store: ExperimentStore,
    run_id: str,
    adapter: str,
    actor: str,
    sandbox: object,
    input_path: Path,
    output_dir: Path,
    history: list[dict[str, str]],
    state: AgentSessionCheckpointState,
) -> str:
    turn = state.turns_completed
    dataset_path = _capture_checkpoint_dataset(
        store=store,
        run_id=run_id,
        adapter=adapter,
        actor=actor,
        sandbox=sandbox,
        input_path=input_path,
        output_dir=output_dir,
        turn=turn,
        current_agent=state.current_agent_name,
    )
    dataset_artifact = store.record_idempotent_file_artifact(
        run_id,
        source=dataset_path,
        filename=_CHECKPOINT_DATASET_FILENAME,
        role="checkpoint_dataset_state",
        producer=actor,
        artifact_type=ArtifactType.dataset,
        media_type="application/x-hdf5",
        schema_type="anndata",
        schema_version_name="h5ad",
        turn=turn,
        current_agent=state.current_agent_name,
    )
    message_artifact = store.record_idempotent_json_artifact(
        run_id,
        filename="checkpoint-message-history.json",
        role="checkpoint_message_history",
        value={
            "schema_version": "caribou.message_history.v1",
            "run_id": run_id,
            "messages": history,
        },
        producer=actor,
        artifact_type=ArtifactType.message_history,
        schema_type="caribou.message_history",
        schema_version_name="v1",
        turn=turn,
        current_agent=state.current_agent_name,
    )
    state_artifact = store.record_idempotent_json_artifact(
        run_id,
        filename="checkpoint-agent-state.json",
        role="checkpoint_agent_state",
        value={
            "schema_version": "caribou.checkpoint_agent_state.v1",
            "run_id": run_id,
            "state": asdict(state),
        },
        producer=actor,
        artifact_type=ArtifactType.checkpoint,
        schema_type="caribou.checkpoint_agent_state",
        schema_version_name="v1",
        turn=turn,
        current_agent=state.current_agent_name,
    )
    action_frontier = store.run(run_id).event_sequence
    action_events = _checkpoint_action_events(store, run_id)
    actions_artifact = store.record_idempotent_json_artifact(
        run_id,
        filename="checkpoint-executed-actions.json",
        role="checkpoint_executed_actions",
        value={
            "schema_version": "caribou.executed_action_ledger.v1",
            "run_id": run_id,
            "through_event_sequence": action_frontier,
            "through_turn": turn,
            "event_ids": [event["event_id"] for event in action_events],
            "events_hash": _checkpoint_value_hash(action_events),
        },
        producer=actor,
        artifact_type=ArtifactType.checkpoint,
        schema_type="caribou.executed_action_ledger",
        schema_version_name="v1",
        turn=turn,
        current_agent=state.current_agent_name,
    )
    manifest_frontier = store.run(run_id).event_sequence
    manifest_artifact = store.record_idempotent_json_artifact(
        run_id,
        filename="checkpoint-artifact-manifest.json",
        role="checkpoint_artifact_manifest",
        value={
            "schema_version": "caribou.checkpoint_artifact_manifest.v1",
            "run_id": run_id,
            "frontier_event_sequence": manifest_frontier,
            "components": [
                _checkpoint_component_reference(artifact)
                for artifact in (
                    dataset_artifact,
                    message_artifact,
                    state_artifact,
                    actions_artifact,
                )
            ],
        },
        producer=actor,
        artifact_type=ArtifactType.manifest,
        schema_type="caribou.checkpoint_artifact_manifest",
        schema_version_name="v1",
        turn=turn,
        current_agent=state.current_agent_name,
    )
    checkpoint = store.record_checkpoint(
        run_id,
        stage="agent_turn_boundary",
        turn=turn,
        current_agent=state.current_agent_name,
        dataset_artifact_id=dataset_artifact.artifact_id,
        message_history_artifact_id=message_artifact.artifact_id,
        agent_state_artifact_id=state_artifact.artifact_id,
        executed_actions_artifact_id=actions_artifact.artifact_id,
        artifact_manifest_id=manifest_artifact.artifact_id,
        resume_requirements=sorted(SUPPORTED_RESUME_REQUIREMENTS),
        actor=actor,
    )
    return checkpoint.checkpoint_id


def _restore_checkpoint_dataset(
    *,
    store: ExperimentStore,
    run_id: str,
    adapter: str,
    actor: str,
    sandbox: object,
    restored: _RestoredAgentCheckpoint,
) -> None:
    run = store.run(run_id)
    if adapter == AGENT_PATH_SMOKE_ADAPTER:
        store.append_run_event(
            run_id,
            event_type=EventType.heartbeat,
            payload=HeartbeatPayload(message="scripted checkpoint dataset rebound"),
            actor=actor,
            turn=run.current_turn,
            current_agent=run.current_agent,
            stage="checkpoint_restore",
        )
        return
    restore_source = store.record_idempotent_json_artifact(
        run_id,
        filename="checkpoint-restore-code.json",
        role="checkpoint_restore_code",
        value={
            "schema_version": "caribou.checkpoint_restore_code.v1",
            "source_checkpoint_id": restored.checkpoint_id,
            "source": _CHECKPOINT_DATASET_RESTORE_CODE,
        },
        producer=actor,
        artifact_type=ArtifactType.checkpoint,
        turn=run.current_turn,
        current_agent=run.current_agent,
    )
    result = sandbox.exec_code(_CHECKPOINT_DATASET_RESTORE_CODE, timeout=600)  # type: ignore[attr-defined]
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise ControlError(
            "CHECKPOINT_DATASET_RESTORE_FAILED",
            "the checkpoint AnnData state could not be loaded into the fresh REPL",
            exit_code=ExitCode.execution,
            details={"restore_artifact_id": restore_source.artifact_id},
        )
    store.record_idempotent_json_artifact(
        run_id,
        filename="checkpoint-restore-result.json",
        role="checkpoint_restore_result",
        value={
            "schema_version": "caribou.checkpoint_restore_result.v1",
            "source_checkpoint_id": restored.checkpoint_id,
            "restore_artifact_id": restore_source.artifact_id,
            "status": "ok",
        },
        producer=actor,
        artifact_type=ArtifactType.checkpoint,
        turn=run.current_turn,
        current_agent=run.current_agent,
    )


def execute_agent_workload(
    store: ExperimentStore,
    run_id: str,
    *,
    adapter: str,
    actor: str = "local-worker",
) -> AgentSessionResult | None:
    """Execute one frozen agent condition; ``None`` means pre-run cancellation."""

    from caribou.agents.AgentSystem import AgentSystem

    run = store.run(run_id)
    _verify_code_identity(
        run.code.commit,
        adapter,
        expected_dirty=run.code.dirty,
    )
    restored = _load_restored_checkpoint(store, run)
    blueprint_path = _local_file(run.resolved_blueprint.source, role="blueprint")
    prompt_path = _local_file(run.resolved_prompt, role="prompt")
    input_path = (
        restored.dataset_path
        if restored is not None
        else _local_file(run.resolved_inputs[0], role="input")
    )
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError("frozen analysis prompt is empty")

    agent_system = AgentSystem.load_from_json(str(blueprint_path))
    driver = agent_system.get_agent(run.resolved_blueprint.driver_agent)
    if driver is None:
        raise RuntimeError(
            f"driver agent not found: {run.resolved_blueprint.driver_agent}"
        )
    rag_corpus_path = None
    if adapter == CARIBOU_AGENT_ADAPTER:
        rag_corpus_path = _verify_blueprint_dependencies(agent_system, run)

    condition = next(
        item
        for item in store.spec(run.experiment_id).conditions
        if item.condition_id == run.condition_id
    )
    if adapter == AGENT_PATH_SMOKE_ADAPTER:
        if "delegate_to_general" not in driver.commands:
            raise RuntimeError("agent path smoke blueprint lacks delegate_to_general")
        delay_value = condition.parameters.get(AGENT_SMOKE_DELAY_PARAMETER, 0.0)
        if isinstance(delay_value, bool) or not isinstance(delay_value, (int, float)):
            raise RuntimeError("validated agent smoke delay is not numeric")
        delay = float(delay_value)
        llm_client: object = _ScriptedClient(
            delay,
            restored.runner_state.turns_completed if restored is not None else 0,
        )
        sandbox: object = _RecordingSandbox()
        llm_attempt_callback: Callable[[dict[str, object]], None] | None = None
        max_output_tokens = None
    else:
        # The backend verifies this large image immediately before launch; avoid
        # reading a multi-gigabyte SIF twice in the same worker.
        container_path = _local_file(
            run.container.image, role="container", verify_hash=False
        )
        model_parameters = dict(run.resolved_model.parameters)
        llm_client = _provider_client(run.resolved_model.provider, model_parameters)
        max_output_tokens = _model_max_output_tokens(model_parameters)
        llm_attempt_callback = _provider_receipt_recorder(store, run_id)
        sandbox = _CancellationAwareSandbox(
            _real_sandbox(
                container_path,
                run.container.image.content_hash,
                Console(stderr=True),
                gpu_enabled=run.container.gpu_enabled,
            ),
            lambda: store.cancel_requested(run_id),
        )

    output_dir = store.run_dir(run_id) / "workload-output"
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    sandbox.set_data([(input_path, SANDBOX_DATA_PATH)], output_dir)  # type: ignore[attr-defined]
    if store.cancel_requested(run_id):
        return None
    result: AgentSessionResult | None = None
    history = (
        [dict(message) for message in restored.history]
        if restored is not None
        else [
            {
                "role": "system",
                "content": f"**GLOBAL POLICY**: {agent_system.global_policy}\n",
            },
            {
                "role": "system",
                "content": (
                    driver.get_full_prompt(None)
                    + f"\n\nPrimary dataset path: **{SANDBOX_DATA_PATH}**\n"
                    + "Save generated files under `/workspace/outputs/`."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    checkpoint_states: list[AgentSessionCheckpointState] = []
    previous_rag_corpus = os.environ.pop(FROZEN_RAG_CORPUS_ENV, None)
    if rag_corpus_path is not None:
        os.environ[FROZEN_RAG_CORPUS_ENV] = str(rag_corpus_path)
    try:
        if not sandbox.start_container():  # type: ignore[attr-defined]
            raise RuntimeError("agent sandbox failed to start")
        if store.cancel_requested(run_id):
            return None
        store.transition_run(
            run_id,
            RunState.running,
            reason="CARIBOU agent workload initialized",
            actor=actor,
        )
        if restored is not None:
            _restore_checkpoint_dataset(
                store=store,
                run_id=run_id,
                adapter=adapter,
                actor=actor,
                sandbox=sandbox,
                restored=restored,
            )
        result = run_agent_session(
            console=Console(stderr=True),
            agent_system=agent_system,
            driver_agent=driver,
            analysis_context=(
                f"Primary dataset path: **{SANDBOX_DATA_PATH}**\n"
                "Save generated files under `/workspace/outputs/`."
            ),
            llm_client=llm_client,
            sandbox_manager=sandbox,  # type: ignore[arg-type]
            history=history,
            is_auto=True,
            compress_memory=run.resolved_memory.strategy == MemoryStrategy.episodic,
            agent_report_memory=(
                run.resolved_memory.strategy == MemoryStrategy.agent_report
            ),
            max_turns=run.resolved_stop_rules.maximum_turns,
            model_name=run.resolved_model.model,
            model_parameters=dict(run.resolved_model.parameters),
            output_dir=output_dir,
            durable_run_id=run_id,
            should_cancel=lambda: store.cancel_requested(run_id),
            event_callback=_event_recorder(store, run_id),
            llm_attempt_callback=llm_attempt_callback,
            should_checkpoint=lambda: store.checkpoint_requested(run_id),
            checkpoint_callback=checkpoint_states.append,
            resume_state=(restored.runner_state if restored is not None else None),
            timeout_seconds=run.resolved_stop_rules.timeout_seconds,
            max_consecutive_no_action=(
                run.resolved_stop_rules.maximum_consecutive_no_action
            ),
            max_consecutive_exec_failures=(
                run.resolved_stop_rules.maximum_consecutive_execution_failures
            ),
            llm_retry_attempts=run.resolved_stop_rules.retry.maximum_attempts,
            llm_retry_base_delay=(run.resolved_stop_rules.retry.base_delay_seconds),
            llm_retry_max_delay=(run.resolved_stop_rules.retry.maximum_delay_seconds),
            max_output_tokens=max_output_tokens,
        )
        checkpointed = result.end_reason == "checkpointed"
        if checkpointed != (len(checkpoint_states) == 1):
            raise ControlError(
                "CHECKPOINT_STATE_MISSING",
                "runner checkpoint outcome and captured state disagree",
                exit_code=ExitCode.integrity,
                details={"captured_states": len(checkpoint_states)},
            )
        if not checkpointed:
            store.record_json_artifact(
                run_id,
                filename="message-history.json",
                role="message_history",
                value={
                    "schema_version": "caribou.message_history.v1",
                    "run_id": run_id,
                    "messages": history,
                },
                producer="agent-runner",
                artifact_type=ArtifactType.message_history,
                schema_type="caribou.message_history",
                schema_version_name="v1",
                turn=result.final_turn,
                current_agent=result.current_agent_name,
            )
        store.record_json_artifact(
            run_id,
            filename="agent-session-result.json",
            role="agent_session_result",
            value=asdict(result),
            producer="agent-runner",
            artifact_type=ArtifactType.report,
            schema_type="caribou.agent_session_result",
            schema_version_name="v1",
            turn=result.final_turn,
            current_agent=result.current_agent_name,
        )
        for path in sorted(output_dir.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(output_dir)
            safe_name = "__".join(relative.parts)
            store.record_file_artifact(
                run_id,
                source=path,
                filename=safe_name,
                role=f"analysis_output:{relative.as_posix()}",
                producer="agent-runner",
                artifact_type=_artifact_type(path),
                media_type=mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                turn=result.final_turn,
                current_agent=result.current_agent_name,
            )
        if checkpointed:
            _persist_agent_checkpoint(
                store=store,
                run_id=run_id,
                adapter=adapter,
                actor=actor,
                sandbox=sandbox,
                input_path=input_path,
                output_dir=output_dir,
                history=history,
                state=checkpoint_states[0],
            )
        return result
    finally:
        os.environ.pop(FROZEN_RAG_CORPUS_ENV, None)
        if previous_rag_corpus is not None:
            os.environ[FROZEN_RAG_CORPUS_ENV] = previous_rag_corpus
        sandbox.stop_container()  # type: ignore[attr-defined]
