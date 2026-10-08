from __future__ import annotations

from pathlib import Path

import pytest

from caribou.control.api import ControlError
from caribou.control.service import ExperimentService
from caribou.control.specs import ADAPTER_PARAMETER, LOCAL_LIFECYCLE_ADAPTER
from caribou.control.store import ExperimentStore
from caribou.domain.models import ExperimentSpec

from .test_domain_models import make_spec


def _submittable_spec() -> ExperimentSpec:
    source = make_spec()
    condition = source.conditions[0].model_copy(
        update={"parameters": {ADAPTER_PARAMETER: LOCAL_LIFECYCLE_ADAPTER}}
    )
    return ExperimentSpec.model_validate_json(
        source.model_copy(update={"conditions": [condition]}).model_dump_json()
    )


def test_clone_changes_evaluator_without_mutating_submitted_spec(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / "store")
    source = _submittable_spec()
    submission = store.submit(source, "source-experiment")
    destination = tmp_path / "drafts" / "stronger-evaluator.yaml"

    clone, written = ExperimentService(store=store).clone_with_evaluator(
        submission.experiment.experiment_id,
        destination,
        provider="anthropic",
        model="claude-opus-4-1",
        reason="  higher-confidence final review  ",
    )

    assert written == destination
    assert written.is_file()
    assert clone.schema_version == "caribou.experiment_spec.v2"
    assert clone.spec_id != source.spec_id
    assert clone.parent_spec_id == source.spec_id
    assert clone.model_change_reason == "higher-confidence final review"
    assert clone.evaluator is not None
    assert clone.evaluator.model.provider == "anthropic"
    assert clone.evaluator.model.model == "claude-opus-4-1"
    assert store.spec(submission.experiment.experiment_id) == source


def test_clone_rejects_overlong_optional_reason(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "store")
    submission = store.submit(_submittable_spec(), "source-experiment")

    with pytest.raises(ControlError, match="cannot exceed 1000"):
        ExperimentService(store=store).clone_with_evaluator(
            submission.experiment.experiment_id,
            tmp_path / "draft.yaml",
            provider="openai",
            model="gpt-5",
            reason="x" * 1001,
        )
