from __future__ import annotations

import subprocess
import threading
from types import SimpleNamespace

import pytest

from caribou.agents.AgentSystem import Agent
from caribou.execution.evaluation import evaluate_work_item, parse_work_item_review
from caribou.execution.work_items import (
    WorkItemConflict,
    WorkItemPolicy,
    WorkItemStore,
    parse_work_item_command,
)


def test_agent_command_grammar_is_exact_and_quoted() -> None:
    opened = parse_work_item_command(
        'open_work_item "Fix parser" "Handle quoted input"'
    )
    assert opened is not None
    assert opened.title == "Fix parser"
    assert opened.body == "Handle quoted input"

    closed = parse_work_item_command('close_work_item 0 "Added parser tests"')
    assert closed is not None
    assert closed.item_id == 0
    assert closed.completion_summary == "Added parser tests"

    assert parse_work_item_command("open_work_item title-only") is None
    assert parse_work_item_command("list_work_items\nextra prose") is None
    assert parse_work_item_command("```\nlist_work_items\n```") is None


def test_optional_qc_store_commits_each_transition_and_restarts(tmp_path) -> None:
    store = WorkItemStore(tmp_path / "work-items", session_id="run-1", policy=WorkItemPolicy(qc_mode="optional"))
    opened = store.open("Implement", "Build the feature", "planner", 1)
    assert opened["id"] == 0
    assert opened["status"] == "In progress"

    transferred = store.transfer(0, "planner", "coder", 2)
    assert transferred["owner"] == "coder"
    completed = store.close(0, "Feature and tests added", "coder", 3)
    assert completed["status"] == "Done"
    assert completed["completed_turn"] == 3
    assert store.blocking_for_owner("coder") == []

    commits = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=store.root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    assert commits == "3"

    restarted = WorkItemStore(tmp_path / "work-items", session_id="run-1", policy=WorkItemPolicy(qc_mode="optional"))
    assert restarted.read(0)["completion_summary"] == "Feature and tests added"
    assert restarted.open("Second", "Another item", "coder", 4)["id"] == 1


def test_transfer_active_hands_off_all_in_progress_items(tmp_path) -> None:
    store = WorkItemStore(tmp_path / "work-items", session_id="run-transfer-active", policy=WorkItemPolicy())
    store.open("First", "Do the first thing", "planner", 1)
    store.open("Second", "Do the second thing", "planner", 2)
    store.open("Third", "Someone else's work", "other", 2)

    transferred = store.transfer_active("planner", "coder", 3)
    assert [item["id"] for item in transferred] == [0, 1]
    assert [item["owner"] for item in transferred] == ["coder", "coder"]

    # No in-progress items left for the delegator; the other agent's item is
    # untouched and a second transfer is a no-op.
    assert store.blocking_for_owner("planner") == []
    assert store.blocking_for_owner("other")[0]["id"] == 2
    assert store.transfer_active("planner", "coder", 4) == []


def test_required_qc_rejection_returns_item_to_owner(tmp_path) -> None:
    store = WorkItemStore(tmp_path / "work-items", session_id="run-2", policy=WorkItemPolicy(qc_mode="required"))
    store.open("Validate", "Needs review", "analyst", 1)
    submitted = store.close(0, "Candidate result", "analyst", 2)
    assert submitted["status"] == "In review"
    assert submitted["completed_at"] is None

    rejected = store.record_review(
        0,
        evaluator="evaluator",
        turn=2,
        verdict="reject",
        assessment="Missing evidence",
    )
    assert rejected["status"] == "In progress"
    assert rejected["owner"] == "analyst"
    assert store.blocking_for_owner("analyst")[0]["id"] == 0

    store.close(0, "Added evidence", "analyst", 3)
    approved = store.record_review(
        0,
        evaluator="evaluator",
        turn=3,
        verdict="approve",
        assessment="Complete",
    )
    assert approved["status"] == "Done"
    assert approved["completed_turn"] == 3


def test_invalid_owner_and_state_are_refused(tmp_path) -> None:
    store = WorkItemStore(tmp_path / "work-items", session_id="run-3", policy=WorkItemPolicy())
    store.open("Owned", "Only the owner can close", "a", 1)
    with pytest.raises(WorkItemConflict, match="owned by a"):
        store.close(0, "Not mine", "b", 2)
    store.close(0, "Done", "a", 2)
    with pytest.raises(WorkItemConflict, match="already Done"):
        store.transfer(0, "a", "b", 3)


def test_self_review_is_refused(tmp_path) -> None:
    store = WorkItemStore(
        tmp_path / "work-items", session_id="run-self", policy=WorkItemPolicy(qc_mode="required")
    )
    store.open("Check result", "Validate the output", "worker", 1)
    store.close(0, "Done work", "worker", 2)
    with pytest.raises(WorkItemConflict, match="own owner"):
        store.record_review(
            0, evaluator="worker", turn=2, verdict="approve", assessment="Looks fine"
        )


def test_index_cache_is_invalidated_by_mutations(tmp_path) -> None:
    store = WorkItemStore(tmp_path / "work-items", session_id="run-cache", policy=WorkItemPolicy())
    assert store.list() == []
    store.open("First", "Body", "owner", 1)
    # The cache populated by the empty list() call above must not mask the
    # item just committed.
    assert [item["id"] for item in store.list()] == [0]
    store.close(0, "Done", "owner", 2)
    assert store.list()[0]["status"] == "Done"


def test_copy_to_preserves_items_and_diverges_independently(tmp_path) -> None:
    parent = WorkItemStore(
        tmp_path / "parent" / "work-items", session_id="parent-session", policy=WorkItemPolicy()
    )
    parent.open("Shared item", "Present in both", "coder", 1)

    child = parent.copy_to(
        tmp_path / "child" / "work-items",
        child_session_id="child-session",
        forked_from_session_id="parent-session",
    )
    assert [item["id"] for item in child.list()] == [0]
    assert child.read(0)["session_id"] == "child-session"

    child.open("Child-only item", "Not in the parent", "coder", 2)
    assert [item["id"] for item in child.list()] == [0, 1]
    assert [item["id"] for item in parent.list()] == [0]


def test_index_reconciliation_preserves_stall_knobs(tmp_path) -> None:
    store = WorkItemStore(
        tmp_path / "work-items",
        session_id="run-stall-knobs",
        policy=WorkItemPolicy(qc_mode="optional", stall_turns=2, stall_halt_turns=6),
    )
    store.open("Item", "Body", "coder", 1)  # commits the index, populating it

    # A fresh store instance reading the same committed index must not
    # silently reset stall_turns/stall_halt_turns back to WorkItemPolicy()'s
    # defaults (4/8) — only qc_mode's own reconciliation was checked before.
    restarted = WorkItemStore(
        tmp_path / "work-items", session_id="run-stall-knobs", policy=WorkItemPolicy()
    )
    restarted.list()  # forces _index() to reconcile self.policy
    assert restarted.policy.stall_turns == 2
    assert restarted.policy.stall_halt_turns == 6


def test_last_transition_turn_and_note(tmp_path) -> None:
    store = WorkItemStore(tmp_path / "work-items", session_id="run-stall", policy=WorkItemPolicy())
    opened = store.open("Stuck item", "Body", "coder", 1)
    assert WorkItemStore.last_transition_turn(opened) == 1
    noted = store.note(0, "runner", 9, "Halted: stalled for 8 turns.")
    assert noted["notes"][0]["text"] == "Halted: stalled for 8 turns."
    # note() does not itself count as a status transition.
    assert WorkItemStore.last_transition_turn(noted) == 1


def test_separate_store_instances_serialize_id_allocation(tmp_path) -> None:
    first = WorkItemStore(tmp_path / "work-items", session_id="run-shared", policy=WorkItemPolicy())
    second = WorkItemStore(tmp_path / "work-items", session_id="run-shared", policy=WorkItemPolicy())
    opened = []

    def create(store, title):
        opened.append(store.open(title, "body", "agent", 1)["id"])

    threads = [
        threading.Thread(target=create, args=(first, "one")),
        threading.Thread(target=create, args=(second, "two")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(opened) == [0, 1]
    assert [item["id"] for item in first.list()] == [0, 1]


def test_work_item_review_json_contract() -> None:
    assert parse_work_item_review(
        '{"verdict":"approve","assessment":"Looks good"}'
    ) == ("approve", "Looks good")
    assert parse_work_item_review(
        '```json\n{"verdict":"reject","assessment":"Needs tests"}\n```'
    ) == ("reject", "Needs tests")
    with pytest.raises(ValueError, match="verdict"):
        parse_work_item_review('{"verdict":"maybe","assessment":"Unsure"}')


def test_evaluator_review_is_bounded_and_updates_required_lifecycle(tmp_path) -> None:
    store = WorkItemStore(tmp_path / "work-items", session_id="run-review", policy=WorkItemPolicy(qc_mode="required"))
    store.open("Check result", "Validate the claimed output", "worker", 1)
    store.close(0, "Produced the expected table", "worker", 2)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="review-model",
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"verdict":"approve","assessment":"Evidence is sufficient"}'
                    )
                )
            ],
        )

    result = evaluate_work_item(
        store=store,
        item_id=0,
        run_id="run-review",
        turn=2,
        evaluator_agent=Agent("reviewer", "Review carefully", {}, {}),
        llm_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model_name="review-model",
    )

    assert result["verdict"] == "approve"
    assert result["item"]["status"] == "Done"
    payload = calls[0]["messages"][1]["content"]
    assert '"kind": "work_item_review"' in payload
    assert '"history"' not in payload
