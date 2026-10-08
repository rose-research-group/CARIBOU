"""BlockTracker's real blocks.json must pass the server's strict BlockRecord / BlocksResponse models."""

from caribou.execution.blocks import BlockTracker, blocks_path_for, load_blocks
from caribou.execution.work_items import WorkItemPolicy, WorkItemStore
from caribou.server.models import BlockRecord, BlocksResponse


def _tracker(tmp_path):
    store = WorkItemStore(
        tmp_path / "work-items", session_id="sess", policy=WorkItemPolicy()
    )
    changes = []
    tracker = BlockTracker(blocks_path_for(store), "sess", store, changes.append)
    return store, tracker, changes


def _response(path):
    index = load_blocks(path)
    assert index is not None
    return BlocksResponse(
        recorded=True,
        blocks=[BlockRecord.model_validate(block) for block in index["blocks"]],
    )


def test_new_tracker_writes_an_empty_index_up_front(tmp_path):
    store, _, changes = _tracker(tmp_path)
    response = _response(blocks_path_for(store))
    assert response.recorded is True
    assert response.blocks == []
    assert changes == []


def test_tracker_output_validates_against_the_route_models(tmp_path):
    store, tracker, changes = _tracker(tmp_path)

    tracker.begin_action("coordinator", 1, "sess:turn:1:block:1")
    tracker.finish_action("sess:turn:1:block:1", True)

    item = store.open("QC", "Filter low-quality cells", "qc_agent", 2)
    tracker.on_work_item_changed(item)
    tracker.begin_action("qc_agent", 2, "sess:turn:2:block:1")
    tracker.finish_action("sess:turn:2:block:1", False)
    tracker.record_artifact("plots/qc.png", "sess:turn:2:block:1")
    tracker.close_all()

    response = _response(blocks_path_for(store))
    implicit, qc = response.blocks
    assert implicit.implicit is True and implicit.work_item_id is None
    assert qc.work_item_id == item["id"] and qc.status == "error"
    assert qc.artifact_paths == ["plots/qc.png"]
    # Every live block_changed record also passes the route model.
    for record in changes:
        BlockRecord.model_validate(record)
