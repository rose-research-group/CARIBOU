"""BlockTracker: attribution of code actions to work-item blocks, statuses,
close conditions and blocks.json persistence, against a real git-backed
WorkItemStore (summaries from `list()` lack reviews and transitions, so a
mock would hide the re-read the attribution rule depends on)."""

from __future__ import annotations

import json

import pytest

from caribou.execution.blocks import (
    BLOCK_INDEX_SCHEMA,
    BlockError,
    BlockTracker,
    blocks_path_for,
    fork_blocks,
    load_blocks,
)
from caribou.execution.work_items import WorkItemPolicy, WorkItemStore


def _store(tmp_path, *, qc_mode="optional", session_id="sess"):
    return WorkItemStore(
        tmp_path / "work-items",
        session_id=session_id,
        policy=WorkItemPolicy(qc_mode=qc_mode),
    )


def _tracker(store, *, session_id="sess"):
    changes: list = []
    tracker = BlockTracker(
        blocks_path_for(store), session_id, store, on_change=changes.append
    )
    return tracker, changes


def _by_id(tracker):
    return {block["block_id"]: block for block in tracker.blocks()}


def test_code_with_no_work_item_goes_to_one_implicit_block_per_agent_run(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)

    assert tracker.begin_action("coder", 1, "a1") == "blk-0001"
    tracker.finish_action("a1", True)
    assert tracker.begin_action("coder", 2, "a2") == "blk-0001"

    block = tracker.blocks()[0]
    assert block["implicit"] is True
    assert block["work_item_id"] is None
    assert block["attempt"] == 1
    assert block["title"] == "coder (no work item)"
    assert block["kind"] is None
    assert block["status"] == "running"
    assert (block["turn_start"], block["turn_end"]) == (1, 2)
    assert block["action_ids"] == ["a1", "a2"]
    assert changes[-1] == block


def test_another_agent_starts_a_new_implicit_block_and_closes_the_previous(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)

    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", True)
    assert tracker.begin_action("reviewer", 2, "a2") == "blk-0002"

    blocks = _by_id(tracker)
    assert blocks["blk-0001"]["status"] == "ok"
    assert blocks["blk-0002"]["status"] == "running"
    assert blocks["blk-0002"]["agents"] == ["reviewer"]


def test_code_goes_to_the_owners_work_item_and_closes_the_implicit_block(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", False)
    item = store.open("QC", "filter cells", "coder", 2)

    block_id = tracker.begin_action("coder", 3, "a2")

    blocks = _by_id(tracker)
    assert blocks["blk-0001"]["status"] == "error"
    block = blocks[block_id]
    assert block["work_item_id"] == item["id"]
    assert block["implicit"] is False
    assert block["title"] == "QC"
    assert block["attempt"] == 1


def test_most_recently_transitioned_item_wins_then_highest_id(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    first = store.open("first", "b", "coder", 1)
    second = store.open("second", "b", "coder", 1)
    tracker.begin_action("coder", 2, "a1")
    assert tracker.blocks()[-1]["work_item_id"] == second["id"]

    # A transfer round-trip at a later turn makes `first` the most recent.
    store.transfer(first["id"], "coder", "other", 5)
    store.transfer(first["id"], "other", "coder", 5)
    tracker.begin_action("coder", 6, "a2")
    assert tracker.blocks()[-1]["work_item_id"] == first["id"]


def test_done_closes_the_block_ok_and_warn_when_an_earlier_action_failed(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", False)
    tracker.begin_action("coder", 2, "a2")
    tracker.finish_action("a2", True)

    done = store.close(item["id"], "done", "coder", 3)
    tracker.on_work_item_changed(done)

    block = tracker.blocks()[0]
    assert block["status"] == "warn"
    assert block["failed_action_ids"] == ["a1"]
    assert changes[-1]["status"] == "warn"


def test_last_action_failing_closes_as_error(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", False)
    tracker.on_work_item_changed(store.close(item["id"], "s", "coder", 2))
    assert tracker.blocks()[0]["status"] == "error"


def test_reject_closes_the_attempt_and_the_next_action_opens_attempt_two(tmp_path):
    store = _store(tmp_path, qc_mode="required")
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", True)
    store.close(item["id"], "s", "coder", 2)
    rejected = store.record_review(
        item["id"], evaluator="qc", turn=3, verdict="reject", assessment="no"
    )

    tracker.on_work_item_changed(rejected)
    assert tracker.blocks()[0]["status"] == "warn"

    assert tracker.begin_action("coder", 4, "a2") == "blk-0002"
    block = tracker.blocks()[1]
    assert (block["work_item_id"], block["attempt"]) == (item["id"], 2)


def test_errored_review_does_not_count_as_a_reject(tmp_path):
    store = _store(tmp_path, qc_mode="required")
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    store.close(item["id"], "s", "coder", 2)
    store.record_review(
        item["id"],
        evaluator="qc",
        turn=3,
        verdict=None,
        assessment="",
        error="provider down",
    )
    assert tracker.begin_action("coder", 4, "a2") == "blk-0001"
    assert tracker.blocks()[0]["status"] == "running"


def test_review_made_outside_the_loop_is_picked_up_by_sync(tmp_path):
    store = _store(tmp_path, qc_mode="required")
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    store.close(item["id"], "s", "coder", 2)
    store.record_review(
        item["id"], evaluator="qc", turn=3, verdict="approve", assessment="ok"
    )

    tracker.sync()
    assert tracker.blocks()[0]["status"] == "ok"
    tracker.sync()  # idempotent


def test_close_all_marks_a_rejected_attempt_warn_not_ok(tmp_path):
    store = _store(tmp_path, qc_mode="required")
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", True)
    store.close(item["id"], "s", "coder", 2)
    store.record_review(
        item["id"], evaluator="qc", turn=3, verdict="reject", assessment="no"
    )

    tracker.close_all()
    assert tracker.blocks()[0]["status"] == "warn"


def test_transfer_adds_the_new_agent_to_the_same_block(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "driver", 1)
    tracker.begin_action("driver", 1, "a1")
    for moved in store.transfer_active("driver", "coder", 2):
        tracker.on_work_item_changed(moved)
    assert tracker.begin_action("coder", 2, "a2") == "blk-0001"
    assert tracker.blocks()[0]["agents"] == ["driver", "coder"]
    assert tracker.blocks()[0]["work_item_id"] == item["id"]


def test_artifacts_attach_to_the_producing_block_once(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    tracker.begin_action("coder", 1, "a1")

    assert tracker.record_artifact("plots/umap.png", "a1") == "blk-0001"
    count = len(changes)
    assert tracker.record_artifact("plots/umap.png", "a1") == "blk-0001"
    assert len(changes) == count
    assert tracker.record_artifact("x.png", None) is None
    assert tracker.blocks()[0]["artifact_paths"] == ["plots/umap.png"]
    with pytest.raises(BlockError):
        tracker.record_artifact("y.png", "unknown")


def test_unknown_or_repeated_actions_raise(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    with pytest.raises(BlockError):
        tracker.finish_action("nope", True)
    tracker.begin_action("coder", 1, "a1")
    with pytest.raises(BlockError):
        tracker.begin_action("coder", 1, "a1")


def test_blocks_json_is_written_and_reloaded_so_ids_continue(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.close_all()

    path = tmp_path / "blocks.json"
    index = load_blocks(path)
    assert index["schema_version"] == BLOCK_INDEX_SCHEMA
    assert index["session_id"] == "sess"
    assert [b["block_id"] for b in index["blocks"]] == ["blk-0001"]
    assert not (tmp_path / "blocks.json.tmp").exists()

    resumed, _ = _tracker(store)
    assert resumed.block_for_action("a1")["block_id"] == "blk-0001"
    assert resumed.begin_action("coder", 2, "a2") == "blk-0002"


def test_a_work_item_block_closed_by_session_end_reopens_on_resume(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.close_all()
    assert tracker.blocks()[0]["status"] == "ok"

    resumed, _ = _tracker(store)
    assert resumed.begin_action("coder", 2, "a2") == "blk-0001"
    assert resumed.blocks()[0]["status"] == "running"


def test_load_blocks_missing_file_is_none_and_malformed_raises(tmp_path):
    path = tmp_path / "blocks.json"
    assert load_blocks(path) is None
    path.write_text("{not json")
    with pytest.raises(BlockError):
        load_blocks(path)
    path.write_text(json.dumps({"schema_version": "other", "session_id": "s"}))
    with pytest.raises(BlockError):
        load_blocks(path)
    path.write_text(
        json.dumps(
            {
                "schema_version": BLOCK_INDEX_SCHEMA,
                "session_id": "s",
                "blocks": [
                    {
                        "schema_version": "caribou.block.v1",
                        "block_id": "blk-0002",
                        "index": 2,
                    }
                ],
            }
        )
    )
    with pytest.raises(BlockError):
        load_blocks(path)


def test_reused_dir_continues_under_the_recorded_session(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.close_all()

    reused, changes = _tracker(store, session_id="later")
    assert reused.session_id == "sess"
    assert reused.begin_action("coder", 1, "b1") == "blk-0002"
    assert changes[-1]["session_id"] == "sess"
    assert load_blocks(tmp_path / "blocks.json")["session_id"] == "sess"


def test_fork_blocks_restamps(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")

    child_dir = tmp_path / "child"
    child_store = store.copy_to(
        child_dir / "work-items",
        child_session_id="child",
        forked_from_session_id="sess",
    )
    assert fork_blocks(
        blocks_path_for(store), blocks_path_for(child_store), child_session_id="child"
    )
    child, _ = _tracker(child_store, session_id="child")
    assert child.blocks()[0]["session_id"] == "child"
    assert fork_blocks(tmp_path / "absent.json", tmp_path / "x.json", child_session_id="c") is False


def test_emitted_records_are_snapshots_not_live_views(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.begin_action("coder", 2, "a2")
    assert changes[0]["action_ids"] == ["a1"]
    tracker.blocks()[0]["action_ids"].append("tampered")
    assert tracker.blocks()[0]["action_ids"] == ["a1", "a2"]
