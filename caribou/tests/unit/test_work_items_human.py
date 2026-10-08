"""Human reviews and human-opened tickets on WorkItemStore (workbench Tier 1):
`HUMAN_REVIEWER`, the optional-QC human-reject reopen, `open(opened_by,
anchor)` and the v3 schema with v2 read compatibility."""

from __future__ import annotations

import json

import pytest

from caribou.execution.work_item_runtime import (
    copy_work_items,
    render_work_item_state,
)
from caribou.execution.work_items import (
    HUMAN_REVIEWER,
    WORK_ITEM_SCHEMA,
    WorkItemConflict,
    WorkItemPolicy,
    WorkItemStore,
)


def _store(tmp_path, *, qc_mode="optional", name="work-items"):
    return WorkItemStore(
        tmp_path / name, session_id="sess", policy=WorkItemPolicy(qc_mode=qc_mode)
    )


def _done(store):
    item = store.open("QC", "filter cells", "coder", 1)
    return store.close(item["id"], "filtered", "coder", 2)


def test_human_reviewer_is_user():
    assert HUMAN_REVIEWER == "user"


def test_optional_qc_human_reject_reopens_a_done_item(tmp_path):
    store = _store(tmp_path)
    item = _done(store)
    assert item["closed_turn"] == 2 and item["completed_turn"] == 2

    reopened = store.record_review(
        item["id"], turn=5, verdict="reject", assessment="thresholds too loose"
    )

    assert reopened["status"] == "In progress"
    for key in ("completed_turn", "completed_at", "closed_turn", "closed_at"):
        assert reopened[key] is None
    assert reopened["reviews"][-1]["evaluator"] == HUMAN_REVIEWER
    assert reopened["reviews"][-1]["verdict"] == "reject"
    transition = reopened["transitions"][-1]
    assert {k: v for k, v in transition.items() if k != "timestamp"} == {
        "kind": "reopened",
        "actor": "user",
        "turn": 5,
        "from_status": "Done",
        "to_status": "In progress",
        "from_owner": "coder",
        "to_owner": "coder",
    }
    assert reopened["latest_commit"] != item["latest_commit"]
    assert store.list()[0]["status"] == "In progress"
    assert store.list()[0]["completed_turn"] is None
    # Durable: a fresh store instance reads the reopened state from Git.
    assert _store(tmp_path).read(item["id"])["status"] == "In progress"
    # The owner can close it again.
    again = store.close(item["id"], "tightened", "coder", 6)
    assert again["status"] == "Done"
    assert again["completed_turn"] == 6


def test_optional_qc_human_approve_keeps_the_item_done(tmp_path):
    store = _store(tmp_path)
    item = _done(store)
    approved = store.record_review(item["id"], turn=3, verdict="approve", assessment="")
    assert approved["status"] == "Done"
    assert approved["transitions"][-1]["kind"] == "closed"
    assert approved["reviews"][-1]["evaluator"] == "user"


def test_optional_qc_evaluator_reject_does_not_reopen(tmp_path):
    store = _store(tmp_path)
    item = _done(store)
    reviewed = store.record_review(
        item["id"], evaluator="qc", turn=3, verdict="reject", assessment="no"
    )
    assert reviewed["status"] == "Done"
    assert reviewed["completed_turn"] == 2
    assert reviewed["transitions"][-1]["kind"] == "closed"


@pytest.mark.parametrize("assessment", ["", "   "])
def test_human_reject_needs_an_assessment(tmp_path, assessment):
    store = _store(tmp_path)
    item = _done(store)
    with pytest.raises(WorkItemConflict, match="assessment"):
        store.record_review(item["id"], turn=3, verdict="reject", assessment=assessment)
    assert store.read(item["id"])["reviews"] == []


def test_human_review_of_an_open_item_in_optional_mode_conflicts(tmp_path):
    store = _store(tmp_path)
    item = store.open("QC", "b", "coder", 1)
    with pytest.raises(WorkItemConflict, match="Done status"):
        store.record_review(item["id"], turn=2, verdict="reject", assessment="no")


def test_required_qc_human_reject_is_unchanged(tmp_path):
    store = _store(tmp_path, qc_mode="required")
    item = store.open("QC", "b", "coder", 1)
    store.close(item["id"], "s", "coder", 2)
    rejected = store.record_review(item["id"], turn=3, verdict="reject", assessment="no")
    assert rejected["status"] == "In progress"
    transition = rejected["transitions"][-1]
    assert (transition["kind"], transition["actor"]) == ("reviewed", "user")
    with pytest.raises(WorkItemConflict, match="In review"):
        store.record_review(item["id"], turn=4, verdict="approve", assessment="")


def test_open_defaults_opened_by_to_owner_and_has_no_anchor(tmp_path):
    store = _store(tmp_path)
    item = store.open("QC", "b", "coder", 1)
    assert item["schema_version"] == WORK_ITEM_SCHEMA == "caribou.work_item.v3"
    assert item["anchor"] is None
    assert item["transitions"][0]["actor"] == "coder"
    on_disk = json.loads((store.items_dir / "0.json").read_text())
    assert on_disk["anchor"] is None


def test_human_ticket_records_opened_by_and_anchor(tmp_path):
    store = _store(tmp_path)
    anchor = {"block_id": "blk-0003", "action_id": None, "artifact_path": "qc.png"}
    item = store.open(
        "Check doublets", "look at the scrublet scores", "coder", 4,
        opened_by=HUMAN_REVIEWER, anchor=anchor,
    )
    assert item["owner"] == "coder"
    assert item["anchor"] == anchor
    opened = item["transitions"][0]
    assert (opened["kind"], opened["actor"], opened["to_owner"]) == (
        "opened", "user", "coder",
    )
    # A ticket blocks its owner like any other item.
    assert [i["id"] for i in store.blocking_for_owner("coder")] == [item["id"]]


def test_anchor_absent_keys_are_null(tmp_path):
    store = _store(tmp_path)
    item = store.open("T", "b", "coder", 1, anchor={"action_id": "sess-t1-a1"})
    assert item["anchor"] == {
        "block_id": None, "action_id": "sess-t1-a1", "artifact_path": None,
    }


@pytest.mark.parametrize(
    "anchor",
    [
        {},
        {"block_id": None, "action_id": None, "artifact_path": None},
        {"block_id": "blk-0001", "extra": "x"},
        {"block_id": ""},
        {"block_id": 3},
        "blk-0001",
    ],
)
def test_invalid_anchor_is_refused_and_nothing_is_committed(tmp_path, anchor):
    store = _store(tmp_path)
    with pytest.raises(WorkItemConflict, match="anchor"):
        store.open("T", "b", "coder", 1, anchor=anchor)
    assert store.list() == []


def _write_v2_item(store):
    """Rewrite item 0 the way a v2 store committed it (no anchor key)."""
    path = store.items_dir / "0.json"
    raw = json.loads(path.read_text())
    raw["schema_version"] = "caribou.work_item.v2"
    del raw["anchor"]
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    store._git("add", "items/0.json")
    store._git("commit", "--quiet", "-m", "legacy v2 item")
    store._index_cache = None


def test_v2_items_read_with_a_null_anchor_and_keep_working(tmp_path):
    store = _store(tmp_path)
    store.open("Legacy", "b", "coder", 1)
    _write_v2_item(store)

    item = store.read(0)
    assert item["schema_version"] == "caribou.work_item.v2"
    assert item["anchor"] is None
    assert "WORK ITEMS" in render_work_item_state(store, "coder")
    assert "Legacy" in render_work_item_state(store, "coder")

    closed = store.close(0, "done", "coder", 2)
    assert closed["anchor"] is None
    reopened = store.record_review(0, turn=3, verdict="reject", assessment="redo")
    assert reopened["status"] == "In progress"
    # Mutations keep the legacy file as it was written: no anchor is invented.
    on_disk = json.loads((store.items_dir / "0.json").read_text())
    assert on_disk["schema_version"] == "caribou.work_item.v2"
    assert "anchor" not in on_disk
    assert "reopened" in store.review_diff(0)

    child = copy_work_items(
        store.root,
        tmp_path / "child-items",
        child_session_id="child",
        forked_from_session_id="sess",
    )
    assert child.read(0)["anchor"] is None
    assert child.read(0)["session_id"] == "child"
