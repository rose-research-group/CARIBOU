"""Adversarial tests for the shared CARIBOU experiment-domain contract."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from caribou.domain.enums import (
    AggregateStatus,
    ArtifactType,
    BudgetStatus,
    CheckpointComponent,
    EventType,
    ExecutorKind,
    ExperimentState,
    FailureCategory,
    FailureDisposition,
    InterfaceOrigin,
    MemoryStrategy,
    MetricRole,
    MetricStatus,
    RetentionPolicy,
    RunOutcome,
    RunState,
    SandboxKind,
    StudyClass,
    TopologyKind,
    UncertaintyRole,
)
from caribou.domain.models import (
    Aggregate,
    Artifact,
    BlueprintSpec,
    BudgetAllocation,
    BudgetBreach,
    BudgetCounter,
    BudgetRecord,
    Checkpoint,
    CodeIdentity,
    ConditionSpec,
    ContainerSpec,
    ContentReference,
    Event,
    Experiment,
    ExperimentSpec,
    ExperimentTransitionRecord,
    FailureRecord,
    HeartbeatPayload,
    MemorySpec,
    MessagePayload,
    MetricDefinition,
    MetricRecord,
    ModelSpec,
    ResourceRequest,
    Run,
    StateTransitionPayload,
    StopRules,
)
from caribou.domain.ids import new_id
from caribou.domain.migrations import inspect_legacy_record
from caribou.domain.serialization import ExperimentJournal, RunJournal

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
COMMIT = "c" * 40
EXP_ID = "exp_" + "1" * 32
RUN_ID = "run_" + "2" * 32
EVENT_ID = "evt_" + "3" * 32
ARTIFACT_ID = "art_" + "4" * 32
CHECKPOINT_ID = "chk_" + "5" * 32
FAILURE_ID = "fail_" + "6" * 32


def ref(
    uri: str = "artifact://input/data.h5ad", digest: str = HASH_A
) -> ContentReference:
    return ContentReference(uri=uri, content_hash=digest)


def model_spec() -> ModelSpec:
    return ModelSpec(provider="ollama", model="model@sha256:abc", context_length=8192)


def blueprint() -> BlueprintSpec:
    return BlueprintSpec(
        source=ref("artifact://blueprints/agent.json"),
        topology=TopologyKind.single_agent,
        driver_agent="analyst",
        global_policy_hash=HASH_A,
        topology_hash=HASH_B,
        prompt_hashes={"analyst": HASH_A},
    )


def resources() -> ResourceRequest:
    return ResourceRequest(memory_bytes=4_000_000_000, wall_seconds=3600)


def container() -> ContainerSpec:
    return ContainerSpec(
        sandbox=SandboxKind.apptainer, image=ref("artifact://images/a.sif")
    )


def budget() -> BudgetAllocation:
    def counter(unit: str, limit: int = 1000) -> BudgetCounter:
        return BudgetCounter(unit=unit, limit=limit)

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


def make_run(**updates: object) -> Run:
    values = {
        "run_id": RUN_ID,
        "experiment_id": EXP_ID,
        "spec_hash": HASH_A,
        "condition_id": "local-single-agent",
        "replicate_index": 0,
        "idempotency_key": "exp/condition/0/1",
        "interface": InterfaceOrigin.cli,
        "owner": "researcher",
        "executor": ExecutorKind.local,
        "code": CodeIdentity(
            repository="OpenTechBio/caribou", branch="study", commit=COMMIT
        ),
        "resolved_model": model_spec(),
        "resolved_blueprint": blueprint(),
        "resolved_prompt": ref("artifact://prompts/task.txt"),
        "resolved_memory": MemorySpec(strategy=MemoryStrategy.full),
        "resolved_inputs": [ref()],
        "resolved_stop_rules": StopRules(
            maximum_turns=20,
            timeout_seconds=3600,
            maximum_consecutive_execution_failures=3,
            maximum_consecutive_no_action=3,
        ),
        "resolved_budget": budget(),
        "container": container(),
        "resources": resources(),
    }
    values.update(updates)
    return Run.model_validate(values)


def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        title="Matched agent pilot",
        study_class=StudyClass.pilot,
        question="Does delegation change contract success?",
        hypothesis="Delegation has task-dependent effects.",
        negative_interpretation="Equivalent performance supports the simpler topology.",
        owner="researcher",
        reviewers=["independent-reviewer"],
        code=CodeIdentity(
            repository="OpenTechBio/caribou", branch="study", commit=COMMIT
        ),
        inputs=[ref()],
        conditions=[
            ConditionSpec(
                condition_id="single",
                label="Matched single agent",
                blueprint=blueprint(),
                model=model_spec(),
                memory=MemorySpec(strategy=MemoryStrategy.full),
                prompt=ref("artifact://prompts/task.txt"),
            )
        ],
        repetitions=5,
        execution={
            "executor": ExecutorKind.local,
            "resources": resources(),
            "container": container(),
            "output_root": "runs/pilot",
        },
        budget=budget(),
        metrics=[
            MetricDefinition(
                metric_key="contract_success",
                name="Contract success",
                role=MetricRole.primary,
                evaluator=ref("artifact://evaluators/contract.py"),
                direction="maximize",
            )
        ],
        stop_rules=StopRules(
            maximum_turns=20,
            timeout_seconds=3600,
            maximum_consecutive_execution_failures=3,
            maximum_consecutive_no_action=3,
        ),
        randomization="replicate seed = sha256(spec, condition, replicate)",
    )


def test_complete_experiment_spec_is_strict_and_versioned() -> None:
    spec = make_spec()
    assert spec.schema_version == "caribou.experiment_spec.v1"
    assert spec.conditions[0].condition_id == "single"
    assert spec.execution.executor == ExecutorKind.local


def top_level_records():
    spec = make_spec()
    experiment = Experiment(
        experiment_id=EXP_ID,
        spec_id=spec.spec_id,
        spec_version=spec.spec_version,
        spec_hash=HASH_A,
        owner=spec.owner,
        run_ids=[RUN_ID],
    )
    run = make_run()
    event = Event(
        event_id=EVENT_ID,
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        sequence=1,
        event_type=EventType.heartbeat,
        actor="runner",
        payload=HeartbeatPayload(message="alive"),
    )
    artifact = Artifact(
        artifact_id=ARTIFACT_ID,
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        producer_event_id=EVENT_ID,
        producer="runner",
        artifact_type=ArtifactType.manifest,
        role="checkpoint_manifest",
        filename="manifest.json",
        storage_uri="artifacts/manifest.json",
        content_hash=HASH_A,
        media_type="application/json",
        size_bytes=10,
        owner="researcher",
    )
    failure = FailureRecord(
        failure_id=FAILURE_ID,
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        event_id=EVENT_ID,
        category=FailureCategory.execution,
        stage="analysis",
        code="python_error",
        message="example failure",
        fatal=False,
        retryable=True,
        attempt=1,
        detected_by="runner",
        downstream_effect="none; action rolled back",
        disposition=FailureDisposition.retry,
    )
    metric = MetricRecord(
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        event_id=EVENT_ID,
        condition_id="single",
        replicate_index=0,
        metric_key="success",
        metric_name="Success",
        evaluator=ref("artifact://evaluators/success.py"),
        status=MetricStatus.measured,
        value=True,
        role=MetricRole.primary,
        uncertainty_role=UncertaintyRole.not_applicable,
        input_artifact_ids=[ARTIFACT_ID],
    )
    checkpoint = Checkpoint(
        checkpoint_id=CHECKPOINT_ID,
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        event_id=EVENT_ID,
        event_sequence=1,
        stage="analysis",
        turn=0,
        components=[CheckpointComponent.artifact_manifest],
        artifact_manifest_id=ARTIFACT_ID,
        spec_hash=HASH_A,
        code_commit=COMMIT,
        container_digest=HASH_A,
        model_identity="ollama:model@sha256:abc",
        integrity_hash=HASH_B,
    )
    budget_record = BudgetRecord(
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        event_id=EVENT_ID,
        allocation=budget(),
        status=BudgetStatus.within_limit,
        source="measured",
    )
    aggregate = Aggregate(
        experiment_id=EXP_ID,
        spec_hash=HASH_A,
        status=AggregateStatus.complete,
        included_run_ids=[RUN_ID],
        method=ref("artifact://aggregation/method.json"),
    )
    migration_report = inspect_legacy_record(
        {"id": "legacy-session", "status": "complete"},
        source_kind="web_session",
        source_uri="legacy://sessions/legacy-session/session.json",
    )
    run_journal = RunJournal(run=run, events=[])
    experiment_transition = ExperimentTransitionRecord(
        experiment_id=EXP_ID,
        sequence=1,
        from_state=ExperimentState.draft,
        to_state=ExperimentState.validated,
        reason="validated",
        actor="reviewer",
    )
    experiment_journal = ExperimentJournal(experiment=experiment, transitions=[])
    return {
        "experiment-spec": spec,
        "experiment": experiment,
        "run": run,
        "event": event,
        "artifact": artifact,
        "failure": failure,
        "metric": metric,
        "checkpoint": checkpoint,
        "budget": budget_record,
        "aggregate": aggregate,
        "migration-report": migration_report,
        "run-journal": run_journal,
        "experiment-transition": experiment_transition,
        "experiment-journal": experiment_journal,
    }


def test_every_top_level_record_round_trips_without_losing_type_information() -> None:
    for record in top_level_records().values():
        assert type(record).model_validate_json(record.model_dump_json()) == record


def test_domain_models_are_deeply_immutable_and_hash_stable() -> None:
    from caribou.domain.serialization import model_hash

    spec = make_spec()
    original_hash = model_hash(spec)
    with pytest.raises((AttributeError, TypeError)):
        spec.inputs.append(ref("artifact://inputs/extra.h5ad"))
    with pytest.raises(TypeError):
        spec.conditions[0].model.parameters["temperature"] = 0.5
    assert model_hash(spec) == original_hash


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update({"unknown": True}),
        lambda data: data["conditions"][0].update({"ignored": "field"}),
        lambda data: data.update({"schema_version": "caribou.experiment_spec.v2"}),
    ],
)
def test_unknown_and_wrong_version_fields_are_rejected_recursively(mutation) -> None:
    data = make_spec().model_dump(mode="json")
    mutation(data)
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate_json(json.dumps(data))


def test_unsafe_coercions_and_non_utc_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ResourceRequest(cpu_cores="1", memory_bytes=1, wall_seconds=1)
    with pytest.raises(ValidationError):
        make_run(created_at=datetime(2026, 7, 13, 12, 0, 0))


def test_identifiers_hashes_reviewers_and_slurm_partition_are_validated() -> None:
    with pytest.raises(ValidationError):
        make_run(run_id="run-not-valid")
    with pytest.raises(ValidationError):
        ContentReference(uri="input", content_hash="not-a-hash")
    data = make_spec().model_dump(mode="json")
    data["reviewers"] = []
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate_json(json.dumps(data))
    with pytest.raises(ValidationError, match="invalid Slurm partition"):
        make_run(executor=ExecutorKind.slurm, partition="gpu --qos=x")
    with pytest.raises(ValidationError, match="requires a Slurm partition"):
        make_run(executor=ExecutorKind.slurm, partition=None)
    # A stored run keeps whichever valid partition it was created on; it is
    # never compared against the currently configured partition on load.
    assert make_run(executor=ExecutorKind.slurm, partition="gpu").partition == "gpu"


def test_generated_ids_are_type_distinguishable_path_safe_and_collision_resistant() -> (
    None
):
    generated = {new_id("run") for _ in range(10_000)}
    assert len(generated) == 10_000
    assert all(identifier.startswith("run_") for identifier in generated)
    assert all(
        "/" not in identifier and "\\" not in identifier for identifier in generated
    )


def test_nonfinite_metrics_and_non_utc_offsets_are_rejected() -> None:
    metric_args = dict(
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        event_id=EVENT_ID,
        condition_id="single",
        replicate_index=0,
        metric_key="loss",
        metric_name="Loss",
        evaluator=ref("artifact://evaluators/loss.py"),
        status=MetricStatus.measured,
        role=MetricRole.primary,
        uncertainty_role=UncertaintyRole.point_estimate,
        input_artifact_ids=[ARTIFACT_ID],
    )
    with pytest.raises(ValidationError):
        MetricRecord(**metric_args, value=float("nan"))
    data = make_run().model_dump(mode="json")
    data["created_at"] = "2026-07-13T12:00:00-04:00"
    with pytest.raises(ValidationError, match="UTC"):
        Run.model_validate_json(json.dumps(data))


def test_terminal_and_checkpointed_run_invariants() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        make_run(state=RunState.failed)
    with pytest.raises(ValidationError):
        make_run(state=RunState.running, end_reason="premature")
    with pytest.raises(ValidationError):
        make_run(state=RunState.checkpointed)
    failed = make_run(
        state=RunState.failed,
        created_at=now,
        ended_at=now,
        updated_at=now,
        terminal_outcome=RunOutcome.failed,
        end_reason="executor error",
        exit_code=1,
    )
    assert failed.resume_eligible is False


def test_event_payload_must_match_type_and_tokens_are_ephemeral() -> None:
    common = dict(
        event_id=EVENT_ID,
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        sequence=1,
        actor="runner",
    )
    with pytest.raises(ValidationError):
        Event(
            **common,
            event_type=EventType.message,
            payload=StateTransitionPayload(
                from_state="draft", to_state="validated", reason="ok"
            ),
        )
    with pytest.raises(ValidationError, match="ephemeral"):
        Event(
            **common,
            event_type=EventType.token,
            payload={"agent_name": "analyst", "token": "x"},
        )
    event = Event(
        **common,
        event_type=EventType.message,
        payload=MessagePayload(role="assistant", content="complete"),
    )
    assert event.durable


def test_artifact_paths_and_checkpoint_components_are_closed() -> None:
    common = dict(
        artifact_id=ARTIFACT_ID,
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        producer_event_id=EVENT_ID,
        producer="runner",
        artifact_type=ArtifactType.log,
        role="stdout",
        content_hash=HASH_A,
        media_type="text/plain",
        size_bytes=1,
        retention=RetentionPolicy.experiment,
        owner="researcher",
    )
    with pytest.raises(ValidationError):
        Artifact(**common, filename="../log.txt", storage_uri="artifacts/log.txt")
    with pytest.raises(ValidationError):
        Artifact(**common, filename="log.txt", storage_uri="../outside/log.txt")
    with pytest.raises(ValidationError, match="own parent"):
        Artifact(
            **common,
            filename="log.txt",
            storage_uri="artifacts/log.txt",
            parent_artifact_ids=[ARTIFACT_ID],
        )
    with pytest.raises(ValidationError, match="artifact manifest"):
        Checkpoint(
            experiment_id=EXP_ID,
            run_id=RUN_ID,
            event_id=EVENT_ID,
            event_sequence=1,
            stage="analysis",
            turn=1,
            components=[],
            artifact_manifest_id=ARTIFACT_ID,
            spec_hash=HASH_A,
            code_commit=COMMIT,
            container_digest=HASH_B,
            model_identity="ollama:model@digest",
            integrity_hash=HASH_A,
        )


def test_metric_budget_and_aggregate_exclusions_are_explicit() -> None:
    metric_args = dict(
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        event_id=EVENT_ID,
        condition_id="single",
        replicate_index=0,
        metric_key="success",
        metric_name="Success",
        evaluator=ref("artifact://evaluators/success.py"),
        role=MetricRole.primary,
        uncertainty_role=UncertaintyRole.not_applicable,
        input_artifact_ids=[ARTIFACT_ID],
    )
    with pytest.raises(ValidationError):
        MetricRecord(**metric_args, status=MetricStatus.measured)
    with pytest.raises(ValidationError):
        MetricRecord(**metric_args, status=MetricStatus.excluded)
    excluded = MetricRecord(
        **metric_args,
        status=MetricStatus.excluded,
        exclusion_reason="predeclared corrupt input",
    )
    assert excluded.value is None
    fractional = BudgetCounter(unit="usd", limit=0.3, consumed=0.1, reserved=0.2)
    assert not fractional.is_over_limit()
    over_limit = budget().model_dump(mode="json")
    over_limit["api_calls"]["limit"] = 1
    over_limit["api_calls"]["consumed"] = 2
    measured_overage = BudgetAllocation.model_validate_json(json.dumps(over_limit))
    bad_budget = budget().model_dump(mode="json")
    bad_budget["cost"]["unit"] = "dollars"
    with pytest.raises(ValidationError, match="cost budget unit"):
        BudgetAllocation.model_validate_json(json.dumps(bad_budget))
    with pytest.raises(ValidationError):
        BudgetRecord(
            experiment_id=EXP_ID,
            allocation=measured_overage,
            status=BudgetStatus.exceeded,
            source="measured",
        )
    exceeded = BudgetRecord(
        experiment_id=EXP_ID,
        allocation=measured_overage,
        status=BudgetStatus.exceeded,
        source="measured",
        breaches=[
            BudgetBreach(
                counter="api_calls",
                limit=1,
                observed=2,
                detail="two calls consumed against a one-call limit",
            )
        ],
    )
    assert exceeded.status == BudgetStatus.exceeded
    with pytest.raises(ValidationError):
        Aggregate(
            experiment_id=EXP_ID,
            spec_hash=HASH_A,
            status=AggregateStatus.partial,
            included_run_ids=[RUN_ID],
            excluded_run_ids=["run_" + "7" * 32],
            exclusion_reasons={},
            method=ref("artifact://aggregation/method.json"),
        )


def test_contradictory_failure_records_are_rejected() -> None:
    common = dict(
        experiment_id=EXP_ID,
        run_id=RUN_ID,
        event_id=EVENT_ID,
        category=FailureCategory.execution,
        stage="analysis",
        code="failure",
        message="failure",
        attempt=1,
        detected_by="runner",
        downstream_effect="none",
    )
    with pytest.raises(ValidationError, match="fatal failure cannot be retryable"):
        FailureRecord(
            **common,
            fatal=True,
            retryable=True,
            disposition=FailureDisposition.retry,
        )
    with pytest.raises(ValidationError, match="correction"):
        FailureRecord(
            **common,
            fatal=False,
            retryable=False,
            disposition=FailureDisposition.corrected,
        )
    corrected = FailureRecord(
        **common,
        fatal=False,
        retryable=False,
        correction_attempted=True,
        correction_status="succeeded",
        disposition=FailureDisposition.corrected,
    )
    assert corrected.correction_attempted
    with pytest.raises(ValidationError, match="cause itself"):
        FailureRecord(
            **common,
            failure_id=FAILURE_ID,
            fatal=False,
            retryable=False,
            caused_by_failure_id=FAILURE_ID,
            disposition=FailureDisposition.investigate,
        )
