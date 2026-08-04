"""Typed, provenance-complete experiment presets.

Presets resolve into the same frozen :class:`ExperimentSpec` accepted by the CLI
and durable control service. They do not submit work themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from caribou.config import get_caribou_slurm_partition
from caribou.core.deepseek import (
    DEEPSEEK_MODEL_IDS,
    deepseek_profile_for_model,
)
from caribou.domain.enums import (
    ExecutorKind,
    MemoryStrategy,
    MetricRole,
    SandboxKind,
    StudyClass,
    TopologyKind,
)
from caribou.domain.ids import new_id
from caribou.domain.models import (
    BlueprintSpec,
    BudgetAllocation,
    BudgetCounter,
    CodeIdentity,
    ConditionSpec,
    ContainerSpec,
    ContentReference,
    ExecutionSpec,
    ExperimentSpec,
    MemorySpec,
    MetricDefinition,
    ModelSpec,
    ResourceRequest,
    StopRules,
    utc_now,
)
from caribou.domain.serialization import file_hash, sha256_bytes

from .api import ControlError, ExitCode
from .specs import ADAPTER_PARAMETER, CARIBOU_AGENT_ADAPTER


PresetProfile = Literal["fast", "thorough"]


@dataclass(frozen=True)
class ResourceProfile:
    cpu_cores: int
    memory_bytes: int
    wall_seconds: int
    scratch_bytes: int
    storage_bytes: int


@dataclass(frozen=True)
class PresetDefinition:
    preset_id: str
    name: str
    description: str
    title: str
    question: str
    hypothesis: str
    negative_interpretation: str
    blueprint_file: str
    topology: TopologyKind
    driver_agent: str
    prompt_file: str
    default_profile: PresetProfile
    default_max_turns: int
    maximum_max_turns: int


RESOURCE_PROFILES: dict[PresetProfile, ResourceProfile] = {
    "fast": ResourceProfile(
        cpu_cores=4,
        memory_bytes=16_000_000_000,
        wall_seconds=3_600,
        scratch_bytes=10_000_000_000,
        storage_bytes=10_000_000_000,
    ),
    "thorough": ResourceProfile(
        cpu_cores=8,
        memory_bytes=32_000_000_000,
        wall_seconds=7_200,
        scratch_bytes=20_000_000_000,
        storage_bytes=20_000_000_000,
    ),
}


PRESETS: dict[str, PresetDefinition] = {
    "single_agent_qc": PresetDefinition(
        preset_id="single_agent_qc",
        name="Single-Agent QC",
        description="A bounded single-agent quality-control pilot.",
        title="Single-agent quality-control pilot",
        question="Can one CARIBOU agent produce a traceable QC assessment for this dataset?",
        hypothesis="The single agent will inspect the dataset and preserve a concise QC report and plots.",
        negative_interpretation="Failure or an incomplete report identifies a data, model, container, or workflow limitation and remains part of the run record.",
        blueprint_file="caribou_single_agent.json",
        topology=TopologyKind.single_agent,
        driver_agent="solo_agent",
        prompt_file="single-agent-qc.txt",
        default_profile="fast",
        default_max_turns=10,
        maximum_max_turns=20,
    ),
    "multi_agent_batch": PresetDefinition(
        preset_id="multi_agent_batch",
        name="Batch-Correction Team",
        description="A bounded routing and batch-correction pilot.",
        title="Multi-agent batch-correction pilot",
        question="Can a CARIBOU agent team assess batch structure and produce a traceable correction plan or result?",
        hypothesis="The orchestrator will route the task to the appropriate analysis agent and preserve its outputs.",
        negative_interpretation="Failure, unnecessary delegation, or an invalid correction remains a pilot outcome and does not establish integration quality.",
        blueprint_file="integration_system.json",
        topology=TopologyKind.multi_agent,
        driver_agent="master_agent",
        prompt_file="multi-agent-integration.txt",
        default_profile="thorough",
        default_max_turns=20,
        maximum_max_turns=30,
    ),
    "benchmark_qc": PresetDefinition(
        preset_id="benchmark_qc",
        name="QC Contract Probe",
        description="A single-run pilot of the QC output contract; not a powered benchmark.",
        title="QC output-contract pilot",
        question="Can the agent produce the declared QC summary artifacts for this dataset?",
        hypothesis="The run will complete with a machine-readable QC summary and supporting plots.",
        negative_interpretation="Any missing artifact, execution failure, or unsupported conclusion is retained and blocks a stronger benchmark claim.",
        blueprint_file="caribou_single_agent.json",
        topology=TopologyKind.single_agent,
        driver_agent="solo_agent",
        prompt_file="benchmark-qc.txt",
        default_profile="fast",
        default_max_turns=10,
        maximum_max_turns=15,
    ),
}


def get_preset(preset_id: str) -> PresetDefinition:
    """Return one known preset or a typed control-plane error."""

    try:
        return PRESETS[preset_id]
    except KeyError as exc:
        raise ControlError(
            "PRESET_NOT_FOUND",
            "experiment preset was not found",
            exit_code=ExitCode.not_found,
            details={"preset_id": preset_id},
        ) from exc


def get_preset_list() -> list[dict[str, Any]]:
    """Return stable, JSON-compatible preset discovery metadata."""

    profiles = {
        name: {
            "cpu_cores": profile.cpu_cores,
            "memory_bytes": profile.memory_bytes,
            "wall_seconds": profile.wall_seconds,
        }
        for name, profile in RESOURCE_PROFILES.items()
    }
    return [
        {
            "id": definition.preset_id,
            "name": definition.name,
            "description": definition.description,
            "default_profile": definition.default_profile,
            "default_max_turns": definition.default_max_turns,
            "maximum_max_turns": definition.maximum_max_turns,
            "resource_profiles": profiles,
        }
        for definition in PRESETS.values()
    ]


@lru_cache(maxsize=256)
def _cached_file_hash(
    path_text: str,
    size_bytes: int,
    modified_ns: int,
    inode: int,
) -> str:
    del size_bytes, modified_ns, inode
    return file_hash(Path(path_text))


def _content_reference(
    path: Path,
    *,
    role: str,
    media_type: str,
    required_suffix: str | None = None,
) -> ContentReference:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ControlError(
            "PRESET_CONTENT_INVALID",
            f"{role} must not be a symbolic link",
            exit_code=ExitCode.validation,
        )
    try:
        resolved = candidate.resolve(strict=True)
        before = resolved.stat()
    except (OSError, ValueError) as exc:
        raise ControlError(
            "PRESET_CONTENT_INVALID",
            f"{role} must be a readable regular file",
            exit_code=ExitCode.validation,
        ) from exc
    if not resolved.is_file() or (
        required_suffix is not None and resolved.suffix.casefold() != required_suffix
    ):
        raise ControlError(
            "PRESET_CONTENT_INVALID",
            f"{role} must be a readable {required_suffix or 'regular'} file",
            exit_code=ExitCode.validation,
        )
    try:
        digest = _cached_file_hash(
            str(resolved), before.st_size, before.st_mtime_ns, before.st_ino
        )
        after = resolved.stat()
    except OSError as exc:
        raise ControlError(
            "PRESET_CONTENT_INVALID",
            f"{role} could not be read while resolving provenance",
            exit_code=ExitCode.validation,
        ) from exc
    if (
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
    ) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ControlError(
            "PRESET_CONTENT_CHANGED",
            f"{role} changed while its provenance was being resolved",
            exit_code=ExitCode.conflict,
            retryable=True,
        )
    return ContentReference(
        uri=resolved.as_uri(),
        content_hash=digest,
        size_bytes=after.st_size,
        media_type=media_type,
    )


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _unlimited_budget() -> BudgetAllocation:
    def counter(unit: str) -> BudgetCounter:
        return BudgetCounter(unit=unit, limit=None)

    return BudgetAllocation(
        api_calls=counter("calls"),
        input_tokens=counter("tokens"),
        output_tokens=counter("tokens"),
        cached_tokens=counter("tokens"),
        cost=counter("usd"),
        cpu_seconds=counter("cpu_seconds"),
        gpu_seconds=counter("gpu_seconds"),
        memory_byte_seconds=counter("byte_seconds"),
        storage_bytes=counter("bytes"),
        wall_seconds=counter("seconds"),
        concurrency=counter("runs"),
    )


class PresetResolver:
    """Resolve a preset and actual local files into one frozen ExperimentSpec."""

    def __init__(
        self,
        *,
        package_root: Path | None = None,
        container_path: Path | None = None,
        code_identity: CodeIdentity | None = None,
    ) -> None:
        self.package_root = package_root or Path(__file__).resolve().parents[1]
        self.container_path = container_path or (
            self.package_root / "sandbox" / "sandbox.sif"
        )
        self.code_identity = code_identity

    def resolve(
        self,
        preset_id: str,
        *,
        dataset_path: str,
        model_provider: str,
        model_name: str,
        openrouter_endpoint: str | None = None,
        profile: PresetProfile,
        max_turns: int | None,
        executor: str,
        owner: str,
        reviewer: str,
    ) -> ExperimentSpec:
        definition = get_preset(preset_id)
        if profile not in RESOURCE_PROFILES:
            raise ControlError(
                "PRESET_PROFILE_INVALID",
                "preset resource profile must be fast or thorough",
                exit_code=ExitCode.validation,
            )
        provider = model_provider.strip().casefold()
        if provider not in {"openai", "deepseek", "openrouter"}:
            raise ControlError(
                "PRESET_PROVIDER_UNSUPPORTED",
                "preset experiments support openai, deepseek, or openrouter",
                exit_code=ExitCode.validation,
                details={"provider": provider},
            )
        resolved_model_name = model_name.strip()
        resolved_owner = owner.strip()
        resolved_reviewer = reviewer.strip()
        if not resolved_model_name or not resolved_owner or not resolved_reviewer:
            raise ControlError(
                "PRESET_FIELD_REQUIRED",
                "model name, owner, and reviewer must be non-empty",
                exit_code=ExitCode.validation,
            )
        model_parameters: dict[str, object] = {"max_output_tokens": 4_096}
        if provider == "deepseek":
            try:
                deepseek_profile = deepseek_profile_for_model(resolved_model_name)
            except ValueError as exc:
                raise ControlError(
                    "PRESET_DEEPSEEK_MODEL_UNSUPPORTED",
                    "select a supported exact DeepSeek V4 model",
                    exit_code=ExitCode.validation,
                    details={"supported_models": list(DEEPSEEK_MODEL_IDS)},
                ) from exc
            model_parameters.update(deepseek_profile.model_parameters())
        if provider == "openrouter":
            from caribou.core.openrouter import (
                OpenRouterError,
                validate_openrouter_model_id,
            )

            try:
                resolved_model_name = validate_openrouter_model_id(
                    resolved_model_name, strict=True
                )
            except OpenRouterError as exc:
                raise ControlError(
                    "PRESET_OPENROUTER_MODEL_INVALID",
                    str(exc),
                    exit_code=ExitCode.validation,
                ) from exc
            endpoint = (openrouter_endpoint or "").strip()
            if not endpoint:
                raise ControlError(
                    "PRESET_OPENROUTER_ENDPOINT_REQUIRED",
                    "select an OpenRouter provider endpoint",
                    exit_code=ExitCode.validation,
                )
            model_parameters.update(
                {
                    "openrouter_endpoint": endpoint,
                    "openrouter_allow_fallbacks": False,
                    "openrouter_zdr": True,
                    "openrouter_data_collection": "deny",
                }
            )
        turns = definition.default_max_turns if max_turns is None else max_turns
        if (
            isinstance(turns, bool)
            or not isinstance(turns, int)
            or not (1 <= turns <= definition.maximum_max_turns)
        ):
            raise ControlError(
                "PRESET_TURNS_INVALID",
                "maximum turns is outside the preset's supported range",
                exit_code=ExitCode.validation,
                details={"maximum": definition.maximum_max_turns},
            )
        try:
            executor_kind = ExecutorKind(executor)
        except ValueError as exc:
            raise ControlError(
                "PRESET_EXECUTOR_INVALID",
                "executor must be local or slurm",
                exit_code=ExitCode.validation,
            ) from exc

        code = self.code_identity
        if code is None:
            from .service import executing_code_identity

            code = executing_code_identity()
        if code.dirty:
            raise ControlError(
                "PRESET_CODE_DIRTY",
                "preset experiments require a clean frozen CARIBOU commit",
                exit_code=ExitCode.validation,
            )

        dataset = _content_reference(
            Path(dataset_path),
            role="dataset",
            media_type="application/x-hdf5",
            required_suffix=".h5ad",
        )
        blueprint = self._blueprint(definition)
        prompt = _content_reference(
            self.package_root / "control" / "templates" / definition.prompt_file,
            role="analysis prompt",
            media_type="text/plain",
        )
        evaluator = _content_reference(
            self.package_root
            / "control"
            / "templates"
            / "run-completion-evaluator.json",
            role="metric evaluator declaration",
            media_type="application/json",
        )
        container_image = _content_reference(
            self.container_path,
            role="analysis container",
            media_type="application/vnd.sylabs.sif",
            required_suffix=".sif",
        )
        resources = RESOURCE_PROFILES[profile]
        spec_id = new_id("spec")
        return ExperimentSpec(
            spec_id=spec_id,
            title=definition.title,
            study_class=StudyClass.pilot,
            question=definition.question,
            hypothesis=definition.hypothesis,
            negative_interpretation=definition.negative_interpretation,
            owner=resolved_owner,
            reviewers=[resolved_reviewer],
            code=code,
            inputs=[dataset],
            conditions=[
                ConditionSpec(
                    condition_id=preset_id.replace("_", "-"),
                    label=definition.name,
                    blueprint=blueprint,
                    model=ModelSpec(
                        provider=provider,
                        model=resolved_model_name,
                        parameters=model_parameters,
                    ),
                    memory=MemorySpec(strategy=MemoryStrategy.full),
                    prompt=prompt,
                    parameters={ADAPTER_PARAMETER: CARIBOU_AGENT_ADAPTER},
                )
            ],
            repetitions=1,
            execution=ExecutionSpec(
                executor=executor_kind,
                resources=ResourceRequest(
                    cpu_cores=resources.cpu_cores,
                    gpu_count=0,
                    memory_bytes=resources.memory_bytes,
                    wall_seconds=resources.wall_seconds,
                    scratch_bytes=resources.scratch_bytes,
                    storage_bytes=resources.storage_bytes,
                ),
                container=ContainerSpec(
                    sandbox=SandboxKind.singularity,
                    image=container_image,
                    gpu_enabled=False,
                    network_enabled=False,
                ),
                partition=(
                    get_caribou_slurm_partition()
                    if executor_kind == ExecutorKind.slurm
                    else None
                ),
                output_root=f"runs/presets/{preset_id}-{spec_id.removeprefix('spec_')}",
            ),
            budget=_unlimited_budget(),
            metrics=[
                MetricDefinition(
                    metric_key="run_completion",
                    name="Declared preset run completed",
                    role=MetricRole.diagnostic,
                    evaluator=evaluator,
                    unit="boolean",
                    direction="target",
                    denominator_definition="The one submitted preset attempt, including every terminal outcome.",
                    acceptance_rule="The durable run reaches succeeded and its artifacts pass control-plane verification.",
                )
            ],
            stop_rules=StopRules(
                maximum_turns=turns,
                timeout_seconds=resources.wall_seconds,
                maximum_consecutive_execution_failures=3,
                maximum_consecutive_no_action=2,
            ),
            randomization="Not applicable to this single-attempt pilot; every terminal outcome is retained.",
            created_at=utc_now(),
        )

    def _blueprint(self, definition: PresetDefinition) -> BlueprintSpec:
        path = self.package_root / "agents" / definition.blueprint_file
        source = _content_reference(
            path,
            role="agent blueprint",
            media_type="application/json",
        )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            agents = value["agents"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ControlError(
                "PRESET_BLUEPRINT_INVALID",
                "preset agent blueprint is not a valid CARIBOU blueprint",
                exit_code=ExitCode.integrity,
            ) from exc
        if not isinstance(agents, dict) or definition.driver_agent not in agents:
            raise ControlError(
                "PRESET_BLUEPRINT_INVALID",
                "preset driver agent is absent from its blueprint",
                exit_code=ExitCode.integrity,
            )

        prompt_hashes: dict[str, str] = {}
        code_sample_hashes: dict[str, str] = {}
        rag_enabled = False
        topology_manifest: dict[str, Any] = {}
        for agent_name, raw_agent in agents.items():
            if not isinstance(agent_name, str) or not isinstance(raw_agent, dict):
                raise ControlError(
                    "PRESET_BLUEPRINT_INVALID",
                    "preset blueprint contains an invalid agent declaration",
                    exit_code=ExitCode.integrity,
                )
            prompt = raw_agent.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ControlError(
                    "PRESET_BLUEPRINT_INVALID",
                    "every preset agent requires a non-empty prompt",
                    exit_code=ExitCode.integrity,
                )
            prompt_hashes[agent_name] = sha256_bytes(prompt.encode("utf-8"))
            neighbors = raw_agent.get("neighbors", {})
            topology_manifest[agent_name] = neighbors
            rag = raw_agent.get("rag", {})
            rag_enabled = rag_enabled or (
                isinstance(rag, dict) and rag.get("enabled") is True
            )
            samples = raw_agent.get("code_samples", [])
            if not isinstance(samples, list):
                raise ControlError(
                    "PRESET_BLUEPRINT_INVALID",
                    "preset code-sample declaration must be a list",
                    exit_code=ExitCode.integrity,
                )
            for sample_name in samples:
                if (
                    not isinstance(sample_name, str)
                    or Path(sample_name).name != sample_name
                ):
                    raise ControlError(
                        "PRESET_BLUEPRINT_INVALID",
                        "preset code-sample names must be path-safe",
                        exit_code=ExitCode.integrity,
                    )
                sample_path = self.package_root / "code_samples" / sample_name
                reference = _content_reference(
                    sample_path,
                    role="agent code sample",
                    media_type="text/x-python",
                )
                previous = code_sample_hashes.get(sample_name)
                if previous is not None and previous != reference.content_hash:
                    raise ControlError(
                        "PRESET_BLUEPRINT_INVALID",
                        "one code-sample name resolved to different content",
                        exit_code=ExitCode.integrity,
                    )
                code_sample_hashes[sample_name] = reference.content_hash

        rag_corpus = None
        if rag_enabled:
            rag_corpus = _content_reference(
                self.package_root / "rag" / "functions.jsonl",
                role="RAG corpus",
                media_type="application/x-ndjson",
            )
        return BlueprintSpec(
            source=source,
            topology=definition.topology,
            driver_agent=definition.driver_agent,
            global_policy_hash=_canonical_hash(value.get("global_policy", "")),
            topology_hash=_canonical_hash(topology_manifest),
            prompt_hashes=prompt_hashes,
            code_sample_hashes=code_sample_hashes,
            rag_corpus=rag_corpus,
        )
