"""Preset resolution uses the canonical experiment domain and real provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from caribou.control.api import ControlError
from caribou.control.presets import PRESETS, PresetResolver, get_preset_list
from caribou.control.specs import validate_control_spec
from caribou.core.deepseek import (
    DEEPSEEK_FAST_MODEL,
    DEEPSEEK_FLASH_V41_MODEL,
    DEEPSEEK_THINKING_MODEL,
)
from caribou.domain.enums import ExecutorKind, TopologyKind
from caribou.domain.models import CodeIdentity
from caribou.domain.serialization import file_hash, model_hash


COMMIT = "c" * 40


@pytest.fixture
def package_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "caribou"


@pytest.fixture
def resolver(package_root: Path, tmp_path: Path) -> PresetResolver:
    container = tmp_path / "analysis.sif"
    container.write_bytes(b"frozen-test-container")
    return PresetResolver(
        package_root=package_root,
        container_path=container,
        code_identity=CodeIdentity(
            repository="https://example.test/CARIBOU",
            branch="preset-tests",
            commit=COMMIT,
            dirty=False,
        ),
    )


@pytest.mark.parametrize(
    ("preset_id", "expected_topology", "profile"),
    [
        ("single_agent_qc", TopologyKind.single_agent, "fast"),
        ("multi_agent_batch", TopologyKind.multi_agent, "thorough"),
        ("benchmark_qc", TopologyKind.single_agent, "fast"),
    ],
)
def test_presets_resolve_to_submit_ready_canonical_specs(
    resolver: PresetResolver,
    tmp_path: Path,
    preset_id: str,
    expected_topology: TopologyKind,
    profile: str,
) -> None:
    dataset = tmp_path / f"{preset_id}.h5ad"
    dataset.write_bytes(b"frozen-dataset")

    spec = resolver.resolve(
        preset_id,
        dataset_path=str(dataset),
        model_provider="openai",
        model_name="gpt-test-snapshot",
        profile=profile,  # type: ignore[arg-type]
        max_turns=None,
        executor="slurm",
        owner="test-operator",
        reviewer="test-reviewer",
    )

    assert spec.inputs[0].content_hash == file_hash(dataset)
    assert spec.inputs[0].size_bytes == dataset.stat().st_size
    assert spec.conditions[0].blueprint.topology == expected_topology
    assert spec.conditions[0].blueprint.prompt_hashes
    assert spec.conditions[0].blueprint.rag_corpus is not None
    assert spec.execution.executor == ExecutorKind.slurm
    assert spec.execution.partition == "peerd"
    assert spec.execution.output_root.startswith(f"runs/presets/{preset_id}-")
    assert spec.code.commit == COMMIT
    assert model_hash(spec).startswith("sha256:")
    assert validate_control_spec(spec, require_submit_adapter=True)


def test_preset_catalog_exposes_only_supported_resolution_controls() -> None:
    catalog = get_preset_list()

    assert [item["id"] for item in catalog] == list(PRESETS)
    assert all(item["default_profile"] in {"fast", "thorough"} for item in catalog)
    assert all(
        item["maximum_max_turns"] >= item["default_max_turns"] for item in catalog
    )
    assert all(
        set(item["resource_profiles"]) == {"fast", "thorough"} for item in catalog
    )


@pytest.mark.parametrize(
    ("model_name", "expected_parameters"),
    [
        (
            DEEPSEEK_FAST_MODEL,
            {"max_output_tokens": 4_096, "thinking": False},
        ),
        (
            DEEPSEEK_FLASH_V41_MODEL,
            {"max_output_tokens": 4_096, "thinking": False},
        ),
        (
            DEEPSEEK_THINKING_MODEL,
            {
                "max_output_tokens": 4_096,
                "thinking": True,
                "reasoning_effort": "high",
            },
        ),
    ],
)
def test_deepseek_presets_freeze_exact_model_and_mode(
    resolver: PresetResolver,
    tmp_path: Path,
    model_name: str,
    expected_parameters: dict[str, object],
) -> None:
    dataset = tmp_path / f"{model_name}.h5ad"
    dataset.write_bytes(b"frozen-dataset")

    spec = resolver.resolve(
        "single_agent_qc",
        dataset_path=str(dataset),
        model_provider="deepseek",
        model_name=model_name,
        profile="fast",
        max_turns=10,
        executor="local",
        owner="test-operator",
        reviewer="test-reviewer",
    )

    assert spec.conditions[0].model.model == model_name
    assert dict(spec.conditions[0].model.parameters) == expected_parameters
    assert validate_control_spec(spec, require_submit_adapter=True)


def test_deepseek_preset_rejects_retiring_alias(
    resolver: PresetResolver,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "input.h5ad"
    dataset.write_bytes(b"frozen-dataset")

    with pytest.raises(ControlError) as failure:
        resolver.resolve(
            "single_agent_qc",
            dataset_path=str(dataset),
            model_provider="deepseek",
            model_name="deepseek-chat",
            profile="fast",
            max_turns=10,
            executor="local",
            owner="test-operator",
            reviewer="test-reviewer",
        )

    assert failure.value.code == "PRESET_DEEPSEEK_MODEL_UNSUPPORTED"
    assert failure.value.details == {
        "supported_models": [
            DEEPSEEK_FAST_MODEL,
            DEEPSEEK_FLASH_V41_MODEL,
            DEEPSEEK_THINKING_MODEL,
        ]
    }


def test_openrouter_preset_freezes_strict_private_routing(
    resolver: PresetResolver,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "openrouter.h5ad"
    dataset.write_bytes(b"frozen-dataset")
    spec = resolver.resolve(
        "single_agent_qc",
        dataset_path=str(dataset),
        model_provider="openrouter",
        model_name="anthropic/claude-fixed-20260720",
        openrouter_endpoint="anthropic",
        profile="fast",
        max_turns=10,
        executor="local",
        owner="test-operator",
        reviewer="test-reviewer",
    )

    parameters = dict(spec.conditions[0].model.parameters)
    assert parameters["openrouter_endpoint"] == "anthropic"
    assert parameters["openrouter_allow_fallbacks"] is False
    assert parameters["openrouter_zdr"] is True
    assert parameters["openrouter_data_collection"] == "deny"
    assert validate_control_spec(spec, require_submit_adapter=True)


def test_openrouter_preset_rejects_dynamic_alias(
    resolver: PresetResolver,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "openrouter-alias.h5ad"
    dataset.write_bytes(b"frozen-dataset")
    with pytest.raises(ControlError) as failure:
        resolver.resolve(
            "single_agent_qc",
            dataset_path=str(dataset),
            model_provider="openrouter",
            model_name="~anthropic/claude-latest",
            openrouter_endpoint="anthropic",
            profile="fast",
            max_turns=10,
            executor="local",
            owner="test-operator",
            reviewer="test-reviewer",
        )
    assert failure.value.code == "PRESET_OPENROUTER_MODEL_INVALID"


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    [
        ({"preset_id": "missing"}, "PRESET_NOT_FOUND"),
        ({"max_turns": 999}, "PRESET_TURNS_INVALID"),
        ({"model_provider": "unsupported"}, "PRESET_PROVIDER_UNSUPPORTED"),
        ({"executor": "kubernetes"}, "PRESET_EXECUTOR_INVALID"),
    ],
)
def test_preset_resolution_rejects_unsupported_inputs(
    resolver: PresetResolver,
    tmp_path: Path,
    updates: dict[str, object],
    expected_code: str,
) -> None:
    dataset = tmp_path / "input.h5ad"
    dataset.write_bytes(b"dataset")
    values: dict[str, object] = {
        "preset_id": "single_agent_qc",
        "dataset_path": str(dataset),
        "model_provider": "openai",
        "model_name": "gpt-test-snapshot",
        "profile": "fast",
        "max_turns": 10,
        "executor": "local",
        "owner": "test-operator",
        "reviewer": "test-reviewer",
    }
    values.update(updates)

    with pytest.raises(ControlError) as failure:
        resolver.resolve(**values)  # type: ignore[arg-type]

    assert failure.value.code == expected_code


def test_preset_resolution_rejects_non_h5ad_dataset(
    resolver: PresetResolver,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "input.txt"
    dataset.write_bytes(b"not-h5ad")

    with pytest.raises(ControlError) as failure:
        resolver.resolve(
            "single_agent_qc",
            dataset_path=str(dataset),
            model_provider="openai",
            model_name="gpt-test-snapshot",
            profile="fast",
            max_turns=10,
            executor="local",
            owner="test-operator",
            reviewer="test-reviewer",
        )

    assert failure.value.code == "PRESET_CONTENT_INVALID"


def test_preset_resolution_rejects_dirty_code_before_hashing_inputs(
    package_root: Path,
    tmp_path: Path,
) -> None:
    container = tmp_path / "analysis.sif"
    container.write_bytes(b"frozen-test-container")
    resolver = PresetResolver(
        package_root=package_root,
        container_path=container,
        code_identity=CodeIdentity(
            repository="https://example.test/CARIBOU",
            branch="dirty-tests",
            commit=COMMIT,
            dirty=True,
        ),
    )

    with pytest.raises(ControlError) as failure:
        resolver.resolve(
            "single_agent_qc",
            dataset_path=str(tmp_path / "missing.h5ad"),
            model_provider="openai",
            model_name="gpt-test-snapshot",
            profile="fast",
            max_turns=10,
            executor="local",
            owner="test-operator",
            reviewer="test-reviewer",
        )

    assert failure.value.code == "PRESET_CODE_DIRTY"
