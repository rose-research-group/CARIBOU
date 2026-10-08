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


def test_a_reject_after_an_ok_close_turns_the_block_warn(tmp_path):
    # Optional QC: an evaluator reject leaves the item Done, but the attempt
    # that already closed ok is now known to be rejected.
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", True)
    tracker.on_work_item_changed(store.close(item["id"], "s", "coder", 2))
    assert tracker.blocks()[0]["status"] == "ok"

    store.record_review(
        item["id"], evaluator="qc", turn=3, verdict="reject", assessment="no"
    )
    tracker.sync()
    assert tracker.blocks()[0]["status"] == "warn"
    assert changes[-1]["block_id"] == "blk-0001"
    assert changes[-1]["status"] == "warn"
    count = len(changes)
    tracker.sync()  # idempotent
    assert len(changes) == count


def test_an_approve_after_an_ok_close_leaves_it_ok(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.on_work_item_changed(store.close(item["id"], "s", "coder", 2))
    count = len(changes)
    tracker.on_work_item_changed(
        store.record_review(item["id"], turn=3, verdict="approve", assessment="")
    )
    assert tracker.blocks()[0]["status"] == "ok"
    assert len(changes) == count


def test_an_error_block_stays_error_after_a_reject(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", False)
    tracker.on_work_item_changed(store.close(item["id"], "s", "coder", 2))
    tracker.on_work_item_changed(
        store.record_review(item["id"], turn=3, verdict="reject", assessment="no")
    )
    assert tracker.blocks()[0]["status"] == "error"


def test_human_reopen_moves_the_next_run_to_attempt_two(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    item = store.open("QC", "b", "coder", 1)

    # 1. The item reaches Done; its attempt-1 block closes ok.
    assert tracker.begin_action("coder", 1, "a1") == "blk-0001"
    tracker.finish_action("a1", True)
    tracker.on_work_item_changed(store.close(item["id"], "s", "coder", 2))
    first = tracker.blocks()[0]
    assert (first["work_item_id"], first["attempt"], first["status"]) == (
        item["id"], 1, "ok",
    )

    # 2. A human rejects it (made outside the loop, e.g. the REST route):
    #    the item goes back to In progress, and sync turns attempt 1 warn.
    reopened = store.record_review(
        item["id"], turn=3, verdict="reject", assessment="redo the thresholds"
    )
    assert reopened["status"] == "In progress"
    tracker.sync()
    assert tracker.blocks()[0]["status"] == "warn"
    assert changes[-1]["block_id"] == "blk-0001"
    assert changes[-1]["status"] == "warn"

    # 3. The next code run goes to a new attempt-2 block.
    assert tracker.begin_action("coder", 4, "a2") == "blk-0002"
    tracker.finish_action("a2", True)
    second = tracker.blocks()[1]
    assert (second["work_item_id"], second["attempt"], second["status"]) == (
        item["id"], 2, "running",
    )
    assert second["action_ids"] == ["a2"]
    assert tracker.blocks()[0]["status"] == "warn"
    assert tracker.blocks()[0]["action_ids"] == ["a1"]

    # Closing again ends attempt 2 ok; attempt 1 stays warn.
    tracker.on_work_item_changed(store.close(item["id"], "s2", "coder", 5))
    assert [b["status"] for b in tracker.blocks()] == ["warn", "ok"]


def test_human_reopen_without_an_explicit_sync_is_picked_up_by_begin_action(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.on_work_item_changed(store.close(item["id"], "s", "coder", 2))
    store.record_review(item["id"], turn=3, verdict="reject", assessment="redo")

    assert tracker.begin_action("coder", 4, "a2") == "blk-0002"
    assert [b["status"] for b in tracker.blocks()] == ["warn", "running"]


# -- workbench focus and delegation handoffs --------------------------------


def test_focus_on_unknown_block_raises(tmp_path):
    tracker, _ = _tracker(_store(tmp_path))
    with pytest.raises(BlockError):
        tracker.set_focus("blk-0009")


def test_clear_focus_without_focus_is_a_no_op(tmp_path):
    tracker, changes = _tracker(_store(tmp_path))
    tracker.clear_focus()
    assert changes == []


def test_focus_reopens_a_closed_block_and_clear_recloses_it(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", True)
    tracker.begin_action("reviewer", 2, "a2")  # closes blk-0001 ok
    assert _by_id(tracker)["blk-0001"]["status"] == "ok"

    tracker.set_focus("blk-0001")
    # set_focus alone changes nothing.
    assert _by_id(tracker)["blk-0001"]["status"] == "ok"

    assert tracker.begin_action("reviewer", 3, "a3") == "blk-0001"
    blocks = _by_id(tracker)
    assert blocks["blk-0001"]["status"] == "running"
    assert blocks["blk-0001"]["agents"] == ["coder", "reviewer"]
    # The reviewer's implicit block closed when focus moved its code away.
    assert blocks["blk-0002"]["status"] == "ok"
    assert changes[-1]["block_id"] == "blk-0001"
    assert changes[-1]["status"] == "running"
    tracker.finish_action("a3", False)

    tracker.clear_focus()
    block = _by_id(tracker)["blk-0001"]
    assert block["status"] == "error"  # last action failed
    assert changes[-1] == block
    on_disk = load_blocks(blocks_path_for(store))["blocks"][0]
    assert on_disk["status"] == "error"
    assert on_disk["action_ids"] == ["a1", "a3"]

    # After the focus, attribution is normal again.
    assert tracker.begin_action("reviewer", 4, "a4") == "blk-0003"


def test_reclosing_a_reopened_block_warns_when_an_earlier_action_failed(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", False)
    tracker.begin_action("coder", 1, "a2")
    tracker.finish_action("a2", True)
    tracker.close_all()
    tracker.set_focus("blk-0001")
    tracker.begin_action("coder", 2, "a3")
    tracker.finish_action("a3", True)
    tracker.clear_focus()
    assert tracker.blocks()[0]["status"] == "warn"


def test_reclosing_a_reopened_rejected_attempt_is_warn(tmp_path):
    store = _store(tmp_path, qc_mode="required")
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", True)
    store.close(item["id"], "s", "coder", 2)
    tracker.on_work_item_changed(
        store.record_review(
            item["id"], evaluator="qc", turn=3, verdict="reject", assessment="no"
        )
    )
    assert tracker.blocks()[0]["status"] == "warn"

    tracker.set_focus("blk-0001")
    assert tracker.begin_action("coder", 4, "a2") == "blk-0001"
    tracker.finish_action("a2", True)
    # A sync during focus does not close the focused block.
    tracker.sync()
    assert tracker.blocks()[0]["status"] == "running"
    tracker.clear_focus()
    assert tracker.blocks()[0]["status"] == "warn"
    assert len(tracker.blocks()) == 1


def test_focus_overrides_work_item_attribution(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")  # implicit blk-0001
    store.open("QC", "b", "coder", 2)
    tracker.begin_action("coder", 2, "a2")  # work-item blk-0002
    tracker.set_focus("blk-0001")
    assert tracker.begin_action("coder", 3, "a3") == "blk-0001"
    blocks = _by_id(tracker)
    # The work-item block is not implicit, so it stays open.
    assert blocks["blk-0002"]["status"] == "running"
    tracker.clear_focus()
    assert _by_id(tracker)["blk-0001"]["status"] == "ok"
    assert tracker.begin_action("coder", 4, "a4") == "blk-0002"


def test_a_block_open_before_the_focus_stays_open_after_clear(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.set_focus("blk-0001")
    tracker.begin_action("coder", 2, "a2")
    count = len(changes)
    tracker.clear_focus()
    assert len(changes) == count
    assert tracker.blocks()[0]["status"] == "running"
    # It is still the coder's current implicit block.
    assert tracker.begin_action("coder", 3, "a3") == "blk-0001"


def test_an_open_work_item_block_whose_item_finished_during_focus_closes_on_clear(
    tmp_path,
):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", True)
    tracker.set_focus("blk-0001")
    tracker.on_work_item_changed(store.close(item["id"], "s", "coder", 2))
    assert tracker.blocks()[0]["status"] == "running"
    tracker.clear_focus()
    assert tracker.blocks()[0]["status"] == "ok"


def test_focus_set_without_any_action_leaves_a_closed_block_closed(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.close_all()
    count = len(changes)
    tracker.set_focus("blk-0001")
    tracker.clear_focus()
    assert len(changes) == count
    assert tracker.blocks()[0]["status"] == "ok"


def test_focusing_another_block_clears_the_first(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.begin_action("reviewer", 2, "a2")
    tracker.close_all()
    tracker.set_focus("blk-0001")
    tracker.begin_action("coder", 3, "a3")
    tracker.set_focus("blk-0002")
    blocks = _by_id(tracker)
    assert blocks["blk-0001"]["status"] == "ok"
    assert tracker.begin_action("coder", 4, "a4") == "blk-0002"


def test_close_all_ends_the_focus(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.set_focus("blk-0001")
    tracker.close_all()
    assert tracker.begin_action("coder", 2, "a2") == "blk-0002"


def test_note_delegation_titles_the_next_implicit_block_once(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("input_agent", 1, "a1")
    tracker.note_delegation("input_agent", "qc_agent", "delegate_to_QC_metrics")
    tracker.begin_action("qc_agent", 2, "a2")
    tracker.begin_action("input_agent", 3, "a3")
    tracker.begin_action("qc_agent", 4, "a4")
    titles = [block["title"] for block in tracker.blocks()]
    assert titles == [
        "input_agent (no work item)",
        "QC metrics (from input_agent)",
        "input_agent (no work item)",
        "qc_agent (no work item)",
    ]


def test_note_delegation_does_not_retitle_a_work_item_block(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.note_delegation("input_agent", "qc_agent", "delegate_to_QC_metrics")
    store.open("QC", "b", "qc_agent", 1)
    tracker.begin_action("qc_agent", 1, "a1")
    assert tracker.blocks()[0]["title"] == "QC"
    # Still pending: the next implicit block for qc_agent gets it.
    store.close(0, "s", "qc_agent", 2)
    tracker.begin_action("qc_agent", 3, "a2")
    assert tracker.blocks()[1]["title"] == "QC metrics (from input_agent)"


@pytest.mark.parametrize("command", ["QC_metrics", "delegate_to_", "delegate_to___"])
def test_note_delegation_rejects_malformed_commands(tmp_path, command):
    tracker, _ = _tracker(_store(tmp_path))
    with pytest.raises(ValueError):
        tracker.note_delegation("a", "b", command)
