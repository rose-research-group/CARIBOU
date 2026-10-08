"""Block attribution in the control-plane typed journal.

The payload union on ``Event`` has no discriminator, so every test that writes
a block event reloads the journal from disk to prove it parses back into the
right payload model.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from caribou.control.agent_workload import _event_recorder
from caribou.domain.enums import EventType
from caribou.domain.models import (
    BlockChangedPayload,
    CodeResultPayload,
    CodeSubmittedPayload,
)
from caribou.domain.serialization import read_run_journal

from .test_control_runner_event_coverage import (
    _MINIMAL_PAYLOADS,
    _event,
    _running_store,
)


def _block_record() -> dict[str, object]:
    return copy.deepcopy(_MINIMAL_PAYLOADS["block_changed"]["block"])  # type: ignore[index]


def test_block_changed_round_trips_into_its_payload_model(tmp_path: Path) -> None:
    store, run_id = _running_store(tmp_path)
    block = _block_record()

    _event_recorder(store, run_id)(_event(run_id, "block_changed", {"block": block}))

    event = read_run_journal(store.run_journal_path(run_id)).events[-1]
    assert event.event_type == EventType.block_changed
    assert isinstance(event.payload, BlockChangedPayload)
    assert event.payload.block == block
    assert event.turn == 1


def test_implicit_block_with_null_work_item_round_trips(tmp_path: Path) -> None:
    store, run_id = _running_store(tmp_path)
    block = _block_record()
    block.update(
        work_item_id=None, implicit=True, title="analyst (no work item)", status="ok"
    )

    _event_recorder(store, run_id)(_event(run_id, "block_changed", {"block": block}))

    event = read_run_journal(store.run_journal_path(run_id)).events[-1]
    assert isinstance(event.payload, BlockChangedPayload)
    assert event.payload.block["work_item_id"] is None
    assert event.payload.block["implicit"] is True


def test_code_events_carry_block_id_through_the_journal(tmp_path: Path) -> None:
    store, run_id = _running_store(tmp_path)
    record = _event_recorder(store, run_id)

    record(_event(run_id, "code_submitted", dict(_MINIMAL_PAYLOADS["code_submitted"])))
    record(_event(run_id, "code_result", dict(_MINIMAL_PAYLOADS["code_result"])))

    events = read_run_journal(store.run_journal_path(run_id)).events
    (submitted,) = [e for e in events if e.event_type == EventType.code_submitted]
    (result,) = [e for e in events if e.event_type == EventType.code_result]
    assert isinstance(submitted.payload, CodeSubmittedPayload)
    assert submitted.payload.block_id == "blk-0001"
    assert isinstance(result.payload, CodeResultPayload)
    assert result.payload.block_id == "blk-0001"


def test_code_events_without_block_id_stay_valid(tmp_path: Path) -> None:
    # block_id is optional: journals written before block attribution still load.
    store, run_id = _running_store(tmp_path)
    payload = dict(_MINIMAL_PAYLOADS["code_submitted"])
    del payload["block_id"]

    _event_recorder(store, run_id)(_event(run_id, "code_submitted", payload))

    event = read_run_journal(store.run_journal_path(run_id)).events[-1]
    assert isinstance(event.payload, CodeSubmittedPayload)
    assert event.payload.block_id is None


@pytest.mark.parametrize("bad", ["", 7, None])
def test_recorder_rejects_a_bad_block_id(tmp_path: Path, bad: object) -> None:
    store, run_id = _running_store(tmp_path)
    payload = dict(_MINIMAL_PAYLOADS["code_result"])
    payload["block_id"] = bad
    with pytest.raises(RuntimeError, match="'block_id' is not a non-empty string"):
        _event_recorder(store, run_id)(_event(run_id, "code_result", payload))


def test_recorder_rejects_a_non_object_block(tmp_path: Path) -> None:
    store, run_id = _running_store(tmp_path)
    with pytest.raises(RuntimeError, match="'block' is not an object"):
        _event_recorder(store, run_id)(
            _event(run_id, "block_changed", {"block": "blk-0001"})
        )


def test_block_changed_payload_requires_a_block_record() -> None:
    block = _block_record()
    block["schema_version"] = "caribou.block.v0"
    with pytest.raises(ValueError, match="schema_version"):
        BlockChangedPayload(block=block)
    block = _block_record()
    del block["block_id"]
    with pytest.raises(ValueError, match="block_id"):
        BlockChangedPayload(block=block)


def test_block_changed_payload_is_not_mistaken_for_another_payload() -> None:
    # A bare {"block": ...} must not parse as e.g. HeartbeatPayload, whose only
    # field is optional; extra="forbid" on every payload guarantees that.
    from caribou.domain.models import HeartbeatPayload

    with pytest.raises(ValueError):
        HeartbeatPayload.model_validate({"block": _block_record()})


def test_legacy_code_payload_dumps_stay_byte_identical() -> None:
    # Checkpoint action ledgers hash event.model_dump(mode="json"); an absent
    # block_id must not add a "block_id": null key to pre-existing events.
    submitted = {
        "action_id": "run:turn:1:block:1",
        "source_artifact_id": "art_" + "0" * 32,
        "agent_name": "analyst",
        "block_index": 1,
        "total_blocks": 1,
    }
    result = {
        "action_id": "run:turn:1:block:1",
        "success": True,
        "duration_ms": 5,
        "stdout_artifact_id": None,
        "stderr_artifact_id": None,
    }
    assert CodeSubmittedPayload(**submitted).model_dump(mode="json") == submitted
    assert CodeResultPayload(**result).model_dump(mode="json") == result
    attributed = CodeResultPayload(**result, block_id="blk-0002")
    assert attributed.model_dump(mode="json") == {**result, "block_id": "blk-0002"}
    assert CodeResultPayload.model_validate_json(attributed.model_dump_json()) == (
        attributed
    )


def test_journaled_legacy_code_event_has_no_block_id_key(tmp_path: Path) -> None:
    # The ledger and journal dump whole Events, so the omission must survive
    # serialization through the undiscriminated payload union.
    store, run_id = _running_store(tmp_path)
    payload = dict(_MINIMAL_PAYLOADS["code_result"])
    del payload["block_id"]

    _event_recorder(store, run_id)(_event(run_id, "code_result", payload))

    event = read_run_journal(store.run_journal_path(run_id)).events[-1]
    assert event.event_type == EventType.code_result
    assert "block_id" not in event.model_dump(mode="json")["payload"]
    assert '"block_id"' not in store.run_journal_path(run_id).read_text("utf-8")
