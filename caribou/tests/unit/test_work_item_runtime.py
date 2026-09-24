from __future__ import annotations

from caribou.execution.work_item_runtime import (
    apply_command,
    end_session_block,
    render_work_item_state,
    stall_report,
    transfer_on_delegation,
)
from caribou.execution.work_items import WorkItemPolicy, WorkItemStore


def _store(tmp_path, name="run", **policy_kwargs) -> WorkItemStore:
    return WorkItemStore(
        tmp_path / "work-items", session_id=name, policy=WorkItemPolicy(**policy_kwargs)
    )


def test_apply_command_returns_none_for_non_command_messages(tmp_path) -> None:
    store = _store(tmp_path)
    assert apply_command(store, "just some prose", owner="coder", turn=1) is None
    result = apply_command(
        store, 'open_work_item "Title" "Body"', owner="coder", turn=1
    )
    assert result is not None
    assert result.success
    assert store.list()[0]["title"] == "Title"


def test_apply_command_corrects_a_command_with_narration_instead_of_silently_dropping_it(
    tmp_path,
) -> None:
    # Observed in practice: an agent prefaces the command with prose, the
    # strict one-command-per-message grammar rejects it, and with no
    # feedback the agent repeats the identical mistake forever. This must
    # not silently return None — the agent needs a reason to self-correct.
    store = _store(tmp_path)
    message = (
        "I'll open a work item to track this.\n\n"
        'open_work_item "Load dataset" "Run the full pipeline"'
    )
    result = apply_command(store, message, owner="coder", turn=1)
    assert result is not None
    assert result.success is False
    assert "ENTIRE message" in result.feedback
    assert store.list() == []


def test_apply_command_still_silently_ignores_unrelated_prose(tmp_path) -> None:
    store = _store(tmp_path)
    assert apply_command(store, "Let's proceed with QC filtering.", owner="coder", turn=1) is None


def test_end_session_block_matches_blocking_for_owner(tmp_path) -> None:
    store = _store(tmp_path)
    store.open("Item", "Body", "coder", 1)
    assert [item["id"] for item in end_session_block(store, "coder")] == [0]
    assert end_session_block(store, "other") == []


def test_transfer_on_delegation_refuses_to_hand_items_to_the_evaluator(tmp_path) -> None:
    store = _store(tmp_path, qc_mode="required")
    store.open("Item", "Body", "driver", 1)

    # Delegating to the evaluator must not transfer ownership: otherwise the
    # evaluator could close the item into "In review" as its own owner, and
    # record_review's self-review refusal would deadlock it there forever.
    transferred = transfer_on_delegation(
        store, "driver", "reviewer", turn=2, evaluator_agent_name="reviewer"
    )
    assert transferred == []
    assert store.read(0)["owner"] == "driver"

    # Delegating to anyone else still transfers normally.
    transferred = transfer_on_delegation(
        store, "driver", "coder", turn=3, evaluator_agent_name="reviewer"
    )
    assert [item["id"] for item in transferred] == [0]
    assert store.read(0)["owner"] == "coder"


def test_render_work_item_state_empty_mixed_and_blocking(tmp_path) -> None:
    store = _store(tmp_path)
    assert render_work_item_state(store, "coder") == "WORK ITEMS: none open."

    store.open("Mine", "Body", "coder", 1)
    store.open("Theirs", "Body", "other", 2)
    rendered = render_work_item_state(store, "coder")
    assert "#0" in rendered and "#1" in rendered
    assert "owner coder (yours)" in rendered
    assert "You own 1 item" in rendered
    assert "end_session is blocked" in rendered


def test_render_work_item_state_omits_done_items(tmp_path) -> None:
    store = _store(tmp_path)
    store.open("Item", "Body", "coder", 1)
    store.close(0, "Finished", "coder", 2)
    assert render_work_item_state(store, "coder") == "WORK ITEMS: none open."


def test_render_work_item_state_truncates_many_items(tmp_path) -> None:
    store = _store(tmp_path)
    for i in range(20):
        store.open(f"Item {i}", "Body", "coder", 1)
    rendered = render_work_item_state(store, "coder")
    assert "+" in rendered and "more" in rendered


def test_stall_report_thresholds(tmp_path) -> None:
    store = _store(tmp_path)
    store.open("Idle item", "Body", "coder", turn=1)

    assert stall_report(store, "coder", turn=3, stall_turns=4) is None

    report = stall_report(store, "coder", turn=5, stall_turns=4)
    assert report is not None
    assert report["item_id"] == 0
    assert report["idle_turns"] == 4

    # Only items the caller owns count.
    assert stall_report(store, "other", turn=10, stall_turns=4) is None


def test_stall_report_ignores_done_items(tmp_path) -> None:
    store = _store(tmp_path)
    store.open("Item", "Body", "coder", 1)
    store.close(0, "Done", "coder", 1)
    assert stall_report(store, "coder", turn=20, stall_turns=4) is None
