"""BlockTracker: attribution of code actions to work-item blocks, statuses,
close conditions and blocks.json persistence, against a real git-backed
WorkItemStore (summaries from `list()` lack reviews and transitions, so a
mock would hide the re-read the attribution rule depends on)."""

from __future__ import annotations

import json

import pytest

from caribou.execution.blocks import (
    BLOCK_INDEX_SCHEMA,
    BLOCK_SCHEMA,
    BlockError,
    BlockTracker,
    blocks_path_for,
    fork_blocks,
    inherit_blocks,
    init_blocks,
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
    # The verdict lives on the review gate; the block reflects its code only.
    assert tracker.blocks()[0]["status"] == "ok"

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


def test_close_all_closes_a_rejected_attempt_by_its_code_alone(tmp_path):
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
    assert tracker.blocks()[0]["status"] == "ok"


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


def test_a_reject_after_an_ok_close_leaves_the_block_status_alone(tmp_path):
    # Optional QC: an evaluator reject leaves the item Done. The verdict is
    # shown on the review gate, not folded into the block's status.
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
    count = len(changes)
    tracker.sync()
    assert tracker.blocks()[0]["status"] == "ok"
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
    #    the item goes back to In progress. Attempt 1's block keeps its
    #    code-based status; the reject shows on its review gate.
    reopened = store.record_review(
        item["id"], turn=3, verdict="reject", assessment="redo the thresholds"
    )
    assert reopened["status"] == "In progress"
    tracker.sync()
    assert tracker.blocks()[0]["status"] == "ok"

    # 3. The next code run goes to a new attempt-2 block.
    assert tracker.begin_action("coder", 4, "a2") == "blk-0002"
    tracker.finish_action("a2", True)
    second = tracker.blocks()[1]
    assert (second["work_item_id"], second["attempt"], second["status"]) == (
        item["id"], 2, "running",
    )
    assert second["action_ids"] == ["a2"]
    assert tracker.blocks()[0]["status"] == "ok"
    assert tracker.blocks()[0]["action_ids"] == ["a1"]

    # Closing again ends attempt 2 ok.
    tracker.on_work_item_changed(store.close(item["id"], "s2", "coder", 5))
    assert [b["status"] for b in tracker.blocks()] == ["ok", "ok"]


def test_human_reopen_without_an_explicit_sync_is_picked_up_by_begin_action(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    item = store.open("QC", "b", "coder", 1)
    tracker.begin_action("coder", 1, "a1")
    tracker.on_work_item_changed(store.close(item["id"], "s", "coder", 2))
    store.record_review(item["id"], turn=3, verdict="reject", assessment="redo")

    assert tracker.begin_action("coder", 4, "a2") == "blk-0002"
    assert [b["status"] for b in tracker.blocks()] == ["ok", "running"]


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


def test_reclosing_a_reopened_rejected_attempt_uses_its_code_alone(tmp_path):
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
    assert tracker.blocks()[0]["status"] == "ok"

    tracker.set_focus("blk-0001")
    assert tracker.begin_action("coder", 4, "a2") == "blk-0001"
    tracker.finish_action("a2", True)
    # A sync during focus does not close the focused block.
    tracker.sync()
    assert tracker.blocks()[0]["status"] == "running"
    tracker.clear_focus()
    assert tracker.blocks()[0]["status"] == "ok"
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


# -- block record v2: entry, inherited_from ---------------------------------


def _entry(turn=1, checkpoint_id="checkpoint_x"):
    return {
        "turn": turn,
        "checkpoint_id": checkpoint_id,
        "checkpoint_complete": True,
        "fingerprint": None,
        "work_items_commit": None,
    }


def test_new_blocks_are_v2_with_null_entry_and_inherited_from(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")

    block = load_blocks(blocks_path_for(store))["blocks"][0]
    assert BLOCK_SCHEMA == "caribou.block.v2"
    assert block["schema_version"] == BLOCK_SCHEMA
    assert block["entry"] is None
    assert block["inherited_from"] is None


def test_on_new_block_runs_once_per_new_block_before_begin_action_returns(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    calls: list = []

    def hook(block_id):
        # Called before the block is first persisted or emitted.
        calls.append((block_id, len(changes)))
        return _entry(checkpoint_id=f"cp-{block_id}")

    assert tracker.begin_action("coder", 1, "a1", on_new_block=hook) == "blk-0001"
    tracker.finish_action("a1", True)
    assert tracker.begin_action("coder", 1, "a2", on_new_block=hook) == "blk-0001"
    item = store.open("QC", "b", "coder", 2)
    assert tracker.begin_action("coder", 2, "a3", on_new_block=hook) == "blk-0002"

    assert calls == [("blk-0001", 0), ("blk-0002", 2)]
    blocks = _by_id(tracker)
    assert blocks["blk-0001"]["entry"]["checkpoint_id"] == "cp-blk-0001"
    assert blocks["blk-0002"]["entry"]["checkpoint_id"] == "cp-blk-0002"
    assert blocks["blk-0002"]["work_item_id"] == item["id"]
    assert changes[0]["entry"]["checkpoint_id"] == "cp-blk-0001"
    persisted = load_blocks(blocks_path_for(store))["blocks"]
    assert persisted[1]["entry"] == _entry(checkpoint_id="cp-blk-0002")


def test_on_new_block_is_not_called_for_a_reopened_or_focused_block(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    tracker.begin_action("coder", 1, "a1")
    tracker.finish_action("a1", True)
    tracker.close_all()
    tracker.set_focus("blk-0001")

    def hook(block_id):
        raise AssertionError("no new block")

    assert tracker.begin_action("coder", 2, "a2", on_new_block=hook) == "blk-0001"
    assert tracker.blocks()[0]["entry"] is None


def test_a_failing_on_new_block_discards_the_block_and_propagates(tmp_path):
    store = _store(tmp_path)
    tracker, changes = _tracker(store)
    tracker.note_delegation("planner", "coder", "delegate_to_QC_metrics")

    def hook(block_id):
        raise FileNotFoundError("no dataset")

    with pytest.raises(FileNotFoundError):
        tracker.begin_action("coder", 1, "a1", on_new_block=hook)
    assert tracker.blocks() == []
    assert changes == []
    tracker.close_all()
    assert load_blocks(blocks_path_for(store))["blocks"] == []
    # The handoff title is still pending, and the action id was not consumed.
    assert tracker.begin_action("coder", 1, "a1") == "blk-0001"
    assert tracker.blocks()[0]["title"] == "QC metrics (from planner)"


def test_on_new_block_must_return_a_dict(tmp_path):
    store = _store(tmp_path)
    tracker, _ = _tracker(store)
    with pytest.raises(BlockError):
        tracker.begin_action("coder", 1, "a1", on_new_block=lambda block_id: None)
    assert tracker.blocks() == []


def _v1_index(session_id="sess"):
    return {
        "schema_version": BLOCK_INDEX_SCHEMA,
        "session_id": session_id,
        "blocks": [
            {
                "schema_version": "caribou.block.v1",
                "block_id": "blk-0001",
                "session_id": session_id,
                "index": 1,
                "work_item_id": None,
                "attempt": 1,
                "implicit": True,
                "title": "coder (no work item)",
                "kind": None,
                "agents": ["coder"],
                "status": "ok",
                "turn_start": 1,
                "turn_end": 1,
                "action_ids": ["old-1"],
                "failed_action_ids": [],
                "artifact_paths": [],
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ],
    }


def test_v1_blocks_load_with_null_entry_and_are_rewritten_as_v2(tmp_path):
    store = _store(tmp_path)
    path = blocks_path_for(store)
    path.write_text(json.dumps(_v1_index()))

    loaded = load_blocks(path)["blocks"][0]
    assert loaded["schema_version"] == "caribou.block.v1"
    assert "entry" not in loaded and "inherited_from" not in loaded

    tracker, _ = _tracker(store)
    assert tracker.begin_action("coder", 2, "a2") == "blk-0002"
    persisted = load_blocks(path)["blocks"]
    assert [block["schema_version"] for block in persisted] == [BLOCK_SCHEMA] * 2
    assert persisted[0]["entry"] is None


def test_v2_block_missing_its_fields_is_malformed(tmp_path):
    path = tmp_path / "blocks.json"
    index = _v1_index()
    index["blocks"][0]["schema_version"] = BLOCK_SCHEMA
    path.write_text(json.dumps(index))
    with pytest.raises(BlockError):
        load_blocks(path)
    index["blocks"][0].update(entry="nope", inherited_from=None)
    path.write_text(json.dumps(index))
    with pytest.raises(BlockError):
        load_blocks(path)


def test_fork_blocks_upgrades_v1_records(tmp_path):
    src = tmp_path / "parent" / "blocks.json"
    src.parent.mkdir()
    src.write_text(json.dumps(_v1_index("parent")))
    dst = tmp_path / "child" / "blocks.json"
    assert fork_blocks(src, dst, child_session_id="child") is True
    block = load_blocks(dst)["blocks"][0]
    assert block["schema_version"] == BLOCK_SCHEMA
    assert block["session_id"] == "child"
    assert block["entry"] is None and block["inherited_from"] is None


# -- inherit_blocks and immutability (A5) -----------------------------------


def _parent_with_blocks(tmp_path):
    """A parent session with three closed blocks: an implicit one, a work
    item block (attempt 1, still In progress), and another implicit one."""
    store = _store(tmp_path / "parent", session_id="parent")
    tracker, _ = _tracker(store, session_id="parent")
    tracker.begin_action("coder", 1, "p1", on_new_block=lambda b: _entry(1, "cp1"))
    tracker.finish_action("p1", True)
    store.open("QC", "b", "coder", 2)
    tracker.begin_action("coder", 2, "p2", on_new_block=lambda b: _entry(2, "cp2"))
    tracker.finish_action("p2", False)
    tracker.begin_action("reviewer", 3, "p3", on_new_block=lambda b: _entry(3, "cp3"))
    tracker.finish_action("p3", True)
    tracker.close_all()
    return store, blocks_path_for(store)


def test_inherit_blocks_copies_blocks_below_the_index_and_stamps_lineage(tmp_path):
    _, src = _parent_with_blocks(tmp_path)
    dst = tmp_path / "child" / "blocks.json"
    init_blocks(dst, "child")

    assert inherit_blocks(
        src, dst, below_index=3, parent_session_id="parent", child_session_id="child"
    ) == 2

    index = load_blocks(dst)
    assert index["session_id"] == "child"
    assert [block["block_id"] for block in index["blocks"]] == ["blk-0001", "blk-0002"]
    for block in index["blocks"]:
        assert block["session_id"] == "child"
        assert block["schema_version"] == BLOCK_SCHEMA
        assert block["inherited_from"] == {
            "session_id": "parent",
            "block_id": block["block_id"],
        }
    assert index["blocks"][1]["entry"]["checkpoint_id"] == "cp2"
    # The parent is untouched.
    assert all(block["inherited_from"] is None for block in load_blocks(src)["blocks"])


def test_inherit_blocks_with_below_index_one_copies_nothing(tmp_path):
    _, src = _parent_with_blocks(tmp_path)
    dst = tmp_path / "child" / "blocks.json"
    assert inherit_blocks(
        src, dst, below_index=1, parent_session_id="parent", child_session_id="child"
    ) == 0
    assert load_blocks(dst)["blocks"] == []


def test_inherit_blocks_rejects_bad_inputs(tmp_path):
    _, src = _parent_with_blocks(tmp_path)
    dst = tmp_path / "child" / "blocks.json"
    kwargs = {"parent_session_id": "parent", "child_session_id": "child"}
    with pytest.raises(BlockError):
        inherit_blocks(tmp_path / "missing.json", dst, below_index=1, **kwargs)
    with pytest.raises(BlockError):
        inherit_blocks(src, dst, below_index=5, **kwargs)
    with pytest.raises(BlockError):
        inherit_blocks(src, dst, below_index=0, **kwargs)
    with pytest.raises(BlockError):
        inherit_blocks(
            src, dst, below_index=2, parent_session_id="other", child_session_id="child"
        )
    assert not dst.exists()
    inherit_blocks(src, dst, below_index=2, **kwargs)
    with pytest.raises(BlockError):
        inherit_blocks(src, dst, below_index=2, **kwargs)


def test_inherit_blocks_refuses_a_running_block(tmp_path):
    store = _store(tmp_path / "parent", session_id="parent")
    tracker, _ = _tracker(store, session_id="parent")
    tracker.begin_action("coder", 1, "p1")
    with pytest.raises(BlockError, match="running"):
        inherit_blocks(
            blocks_path_for(store),
            tmp_path / "child" / "blocks.json",
            below_index=2,
            parent_session_id="parent",
            child_session_id="child",
        )


def _child_from(tmp_path, parent_store, src, *, below_index):
    """A branch child whose work items are a copy of the parent's store."""
    child_store = parent_store.copy_to(
        tmp_path / "child" / "work-items",
        child_session_id="child",
        forked_from_session_id="parent",
    )
    inherit_blocks(
        src,
        blocks_path_for(child_store),
        below_index=below_index,
        parent_session_id="parent",
        child_session_id="child",
    )
    tracker, changes = _tracker(child_store, session_id="child")
    return child_store, tracker, changes


def test_the_child_numbers_new_blocks_after_the_inherited_ones(tmp_path):
    store, src = _parent_with_blocks(tmp_path)
    _, tracker, _ = _child_from(tmp_path, store, src, below_index=2)
    # blk-0001 is an inherited implicit block of the same agent: a new one.
    assert tracker.begin_action("coder", 4, "c1") == "blk-0002"
    assert tracker.blocks()[1]["inherited_from"] is None


def test_an_inherited_work_item_attempt_continues_in_a_new_block(tmp_path):
    store, src = _parent_with_blocks(tmp_path)
    _, tracker, _ = _child_from(tmp_path, store, src, below_index=3)

    assert tracker.begin_action("coder", 4, "c1") == "blk-0003"
    blocks = _by_id(tracker)
    assert blocks["blk-0003"]["work_item_id"] == blocks["blk-0002"]["work_item_id"]
    assert blocks["blk-0003"]["attempt"] == 1
    assert blocks["blk-0002"]["action_ids"] == ["p2"]
    assert blocks["blk-0002"]["status"] == "error"
    assert tracker.begin_action("coder", 4, "c2") == "blk-0003"


def test_inherited_blocks_are_never_focused_synced_or_closed(tmp_path):
    store, src = _parent_with_blocks(tmp_path)
    child_store, tracker, changes = _child_from(tmp_path, store, src, below_index=3)
    before = {block["block_id"]: block for block in load_blocks(blocks_path_for(child_store))["blocks"]}

    with pytest.raises(BlockError, match="inherited"):
        tracker.set_focus("blk-0002")
    # A reject of the inherited attempt would downgrade an ok block to warn;
    # a Done would close an open one. Neither touches an inherited block.
    item_id = before["blk-0002"]["work_item_id"]
    child_store.close(item_id, "done", "coder", 5)
    tracker.on_work_item_changed(child_store.read(item_id))
    tracker.sync()
    tracker.close_all()

    after = {block["block_id"]: block for block in load_blocks(blocks_path_for(child_store))["blocks"]}
    assert after == before
    assert changes == []


def test_an_artifact_cannot_be_added_to_an_inherited_block(tmp_path):
    store, src = _parent_with_blocks(tmp_path)
    _, tracker, _ = _child_from(tmp_path, store, src, below_index=2)
    with pytest.raises(BlockError, match="immutable"):
        tracker.record_artifact("plots/new.png", "p1")
