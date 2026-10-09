from __future__ import annotations

from datetime import datetime, timezone

import pytest

from caribou.execution.session_brief import SessionBrief
from caribou.execution.work_item_runtime import (
    NOT_APPLIED_AFTER_REFUSAL,
    apply_command,
    apply_commands,
    delegation_allowed,
    end_session_block,
    freeze_brief,
    render_brief_pin,
    render_work_item_state,
    seed_brief_item,
    stall_report,
    transfer_on_delegation,
)
from caribou.execution.work_items import WorkItemPolicy, WorkItemStore


def _store(tmp_path, name="run", **policy_kwargs) -> WorkItemStore:
    return WorkItemStore(
        tmp_path / "work-items", session_id=name, policy=WorkItemPolicy(**policy_kwargs)
    )


def _brief(**overrides) -> SessionBrief:
    base = dict(
        deliverable="Ship the parser",
        in_scope=["parser module"],
        out_of_scope=["the UI"],
        done_when=["tests pass"],
        risks=["schema drift"],
        created_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return SessionBrief(**base)


def test_apply_command_returns_none_for_non_command_messages(tmp_path) -> None:
    store = _store(tmp_path)
    assert apply_command(store, "just some prose", owner="coder", turn=1) is None
    result = apply_command(
        store, 'open_work_item "Title" "Body"', owner="coder", turn=1
    )
    assert result is not None
    assert result.success
    assert store.list()[0]["title"] == "Title"


def test_apply_command_accepts_a_command_after_narration(tmp_path) -> None:
    # Observed in practice: agents preface the command with prose. The last
    # non-empty line may be the command.
    store = _store(tmp_path)
    message = (
        "I'll open a work item to track this.\n\n"
        'open_work_item "Load dataset" "Run the full pipeline"\n'
    )
    result = apply_command(store, message, owner="coder", turn=1)
    assert result is not None
    assert result.success is True
    assert store.list()[0]["title"] == "Load dataset"


def test_apply_command_corrects_a_malformed_last_line_command(tmp_path) -> None:
    # Without feedback the agent repeats the identical mistake forever.
    store = _store(tmp_path)
    message = "I'll open a work item.\nopen_work_item \"Load dataset\""
    result = apply_command(store, message, owner="coder", turn=1)
    assert result is not None
    assert result.success is False
    assert "not recognized" in result.feedback
    assert "'open_work_item \"Load dataset\"'" in result.feedback
    assert "block at the END of the message" in result.feedback
    assert store.list() == []


def test_apply_command_corrects_a_command_that_is_not_in_the_block(
    tmp_path,
) -> None:
    # The command is not applied, but the agent is told why instead of it
    # vanishing; the feedback states the block rule, delegation line included.
    store = _store(tmp_path)
    for message in (
        'open_work_item "Load" "Body"\nNow running the loader.',
        'close_work_item 0 "Done"\ndelegate_to_QC_metrics\nThanks.',
        'close_work_item 0 "Done"\ndelegate_to_QC_metrics now',
        "open_work_item Load\nNow running the loader.",
    ):
        result = apply_command(store, message, owner="coder", turn=1)
        assert result is not None and result.success is False
        assert "not applied" in result.feedback
        assert "block at the END of the message" in result.feedback
        assert "delegate_to_<agent>" in result.feedback
    assert store.list() == []


def test_apply_commands_applies_a_close_and_an_open_in_one_message(tmp_path) -> None:
    # Session ac59f7b8: close + open in one message applied only the last
    # line, so each agent saw "the close didn't register" and closed again.
    store = _store(tmp_path)
    store.open("Load", "Body", "coder", 1)
    results = apply_commands(
        store,
        "Loaded.\n```python\nx = 1\n```\n"
        'close_work_item 0 "loaded"\n'
        'open_work_item "QC" "filter cells"\n',
        owner="coder",
        turn=2,
    )
    assert [(r.success, r.command.name) for r in results] == [
        (True, "close_work_item"),
        (True, "open_work_item"),
    ]
    assert store.read(0)["status"] == "Done"
    assert store.read(1)["title"] == "QC" and store.read(1)["status"] == "In progress"
    assert apply_commands(store, "just prose", owner="coder", turn=3) == []


def test_apply_commands_stops_at_the_first_refusal(tmp_path) -> None:
    store = _store(tmp_path)
    store.open("Theirs", "Body", "other", 1)
    results = apply_commands(
        store,
        'close_work_item 0 "not mine"\nopen_work_item "Next" "Body"\nlist_work_items',
        owner="coder",
        turn=2,
    )
    assert [r.success for r in results] == [False, False, False]
    assert results[0].feedback.startswith("Work-item command refused")
    assert results[1].feedback == NOT_APPLIED_AFTER_REFUSAL
    assert results[1].command is not None and results[1].command.name == "open_work_item"
    assert results[2].feedback == NOT_APPLIED_AFTER_REFUSAL
    assert [item["id"] for item in store.list()] == [0]


def test_apply_commands_refuses_a_block_with_a_malformed_line_whole(tmp_path) -> None:
    store = _store(tmp_path)
    store.open("Load", "Body", "coder", 1)
    results = apply_commands(
        store,
        'close_work_item 0 "loaded"\nopen_work_item "QC"',
        owner="coder",
        turn=2,
    )
    assert len(results) == 1 and results[0].success is False
    assert results[0].command is None
    assert "none of this message's commands were applied" in results[0].feedback
    assert store.read(0)["status"] == "In progress"


def test_apply_commands_reports_a_stray_command_line_after_applying_the_block(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    results = apply_commands(
        store,
        'open_work_item "Early" "typed too soon"\n'
        "Now some prose about it.\n"
        'open_work_item "Late" "the real one"',
        owner="coder",
        turn=1,
    )
    assert [r.success for r in results] == [True, False]
    assert results[1].command is None
    assert "'open_work_item \"Early\" \"typed too soon\"'" in results[1].feedback
    assert "not in the command block" in results[1].feedback
    assert [item["title"] for item in store.list()] == ["Late"]
    # The one-result convenience refuses to drop any of them.
    with pytest.raises(ValueError, match="apply_commands"):
        apply_command(
            store, 'list_work_items\nread_work_item 0', owner="coder", turn=2
        )


def test_apply_command_accepts_a_command_in_either_order_with_a_delegation(
    tmp_path,
) -> None:
    # Observed in practice (session ef4aea40): `open_work_item ...` followed
    # by `delegate_to_X` on the last line was refused. Both orders apply.
    store = _store(tmp_path)
    opened = apply_command(
        store,
        'open_work_item "Load" "Body"\ndelegate_to_QC_metrics',
        owner="coder",
        turn=1,
    )
    assert opened is not None and opened.success is True
    closed = apply_command(
        store,
        'Done loading.\ndelegate_to_QC_metrics\nclose_work_item 0 "Loaded"',
        owner="coder",
        turn=2,
    )
    assert closed is not None and closed.success is True
    assert store.read(0)["status"] == "Done"


def test_apply_command_ignores_command_names_inside_code(tmp_path) -> None:
    store = _store(tmp_path)
    for message in (
        "```python\nprint(1)\nopen_work_item Load",
        "```python\nopen_work_item = 1\n```\nDone.",
    ):
        assert apply_command(store, message, owner="coder", turn=1) is None
    assert store.list() == []


def test_apply_command_still_silently_ignores_unrelated_prose(tmp_path) -> None:
    store = _store(tmp_path)
    assert apply_command(store, "Let's proceed with QC filtering.", owner="coder", turn=1) is None


def test_end_session_block_matches_blocking_for_owner(tmp_path) -> None:
    store = _store(tmp_path)
    store.open("Item", "Body", "coder", 1)
    assert [item["id"] for item in end_session_block(store, "coder")] == [0]
    assert end_session_block(store, "other") == []


def _block(item_id, agents, action_ids, block_id="blk-0001", inherited=None):
    return {
        "block_id": block_id,
        "work_item_id": item_id,
        "agents": list(agents),
        "action_ids": list(action_ids),
        "inherited_from": inherited,
    }


def test_delegation_allowed_refuses_self_delegation(tmp_path) -> None:
    store = _store(tmp_path)
    assert delegation_allowed(store, [], "coder", "coder") == (
        "Delegation refused: you cannot delegate to yourself."
    )
    assert delegation_allowed(store, [], "coder", "qc") is None


def test_delegation_allowed_refuses_while_a_worked_item_is_open(tmp_path) -> None:
    store = _store(tmp_path)
    store.open("QC filtering", "Body", "coder", 1)
    blocks = [_block(0, ["coder"], ["run_1_1"])]
    reason = delegation_allowed(store, blocks, "coder", "qc")
    assert reason == (
        "Delegation refused: close work item #0 first — you did the work, so "
        "you write its completion summary (close_work_item 0 \"<summary>\"). "
        "Delegate in your next message."
    )
    # Closed, it no longer blocks.
    store.close(0, "filtered", "coder", 2)
    assert delegation_allowed(store, blocks, "coder", "qc") is None


def test_delegation_allowed_ignores_items_the_owner_did_not_work_on(tmp_path) -> None:
    store = _store(tmp_path)
    store.open("Plan", "Body", "master", 1)
    # No block at all (a planning item): still hands over.
    assert delegation_allowed(store, [], "master", "coder") is None
    # A block of the item with no actions yet, or with another agent's
    # actions only, is not the owner's work either.
    assert (
        delegation_allowed(store, [_block(0, ["master"], [])], "master", "coder")
        is None
    )
    assert (
        delegation_allowed(
            store, [_block(0, ["coder"], ["run_1_1"])], "master", "coder"
        )
        is None
    )
    # Another agent's open item never blocks this owner.
    store.open("Theirs", "Body", "other", 2)
    assert (
        delegation_allowed(
            store, [_block(1, ["other"], ["run_2_1"], "blk-0002")], "master", "coder"
        )
        is None
    )


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


def test_seed_brief_item_opens_a_top_level_item_naming_the_deliverable(tmp_path) -> None:
    store = _store(tmp_path)
    brief = _brief()
    item = seed_brief_item(store, brief, owner="driver", turn=1)
    assert item["title"] == "Ship the parser"
    assert "tests pass" in item["body"]
    assert item["owner"] == "driver"
    assert store.blocking_for_owner("driver")[0]["id"] == item["id"]


def test_freeze_brief_context_mode_writes_provenance_without_seeding_an_item(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    brief = _brief()
    brief_path = tmp_path / "brief.json"

    result = freeze_brief(
        brief,
        brief_path=brief_path,
        store=store,
        brief_mode="context",
        owner="driver",
        turn=1,
    )

    assert result is None
    assert store.list() == []
    assert brief_path.exists()
    assert "Ship the parser" in brief_path.read_text()
    # The brief is committed into the store's own git history, not just a
    # side file — record_review/list/etc. all keep working afterward.
    assert (store.root / "brief.json").exists()


def test_freeze_brief_seed_item_mode_opens_the_deliverable_item(tmp_path) -> None:
    store = _store(tmp_path)
    brief = _brief()
    brief_path = tmp_path / "brief.json"

    result = freeze_brief(
        brief,
        brief_path=brief_path,
        store=store,
        brief_mode="seed_item",
        owner="driver",
        turn=1,
    )

    assert result is not None
    assert result["title"] == "Ship the parser"
    assert [item["id"] for item in store.list()] == [0]


def test_render_brief_pin_includes_deliverable_scope_and_done_when() -> None:
    brief = _brief()
    rendered = render_brief_pin(brief)
    assert "Ship the parser" in rendered
    assert "parser module" in rendered
    assert "the UI" in rendered
    assert "tests pass" in rendered
    assert "schema drift" in rendered
