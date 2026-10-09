"""Review evidence: what the evaluator sees for a work-item review.

The builder (`execution/review_evidence.py`), its place in the review payload
(`execution/evaluation.py`), the budget fitting, and an end-to-end review of
a real `BlockTracker` run whose code reaches the evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from caribou.agents.AgentSystem import Agent
from caribou.execution.blocks import BlockTracker, blocks_path_for, load_blocks
from caribou.execution.evaluation import (
    EvaluationContextTooLarge,
    build_work_item_review_payload,
    evaluate_work_item,
    fit_review_payload_to_budget,
)
from caribou.execution.review_evidence import (
    SOURCE_CAP,
    STDERR_CAP,
    STDOUT_CAP,
    build_block_evidence,
    cap_text,
    code_lookup_from_cli_event_log,
    code_lookup_from_records,
    drop_oldest_outputs,
    read_cli_event_log,
)
from caribou.execution.work_items import WorkItemPolicy, WorkItemStore

from .test_blocks_route import _block


def _index(*blocks: dict) -> dict:
    return {
        "schema_version": "caribou.block_index.v1",
        "session_id": "source-id",
        "blocks": list(blocks),
    }


def _record(**overrides) -> dict:
    record = {
        "agent": "analyst",
        "success": True,
        "source": "print('qc')",
        "stdout": "qc\n",
        "stderr": "",
    }
    record.update(overrides)
    return record


# --- build_block_evidence -------------------------------------------------------


def test_evidence_has_the_contract_shape_in_index_order() -> None:
    index = _index(
        _block(1, work_item_id=2, attempt=1, action_ids=["a1", "a2"], failed_action_ids=["a2"]),
        _block(2, work_item_id=5, action_ids=["other"]),
        _block(3, work_item_id=2, attempt=2, action_ids=["a3"], artifact_paths=["figures/v2.png"], status="warn"),
    )
    records = {
        "a1": _record(source="x = 1"),
        "a2": _record(success=False, source="raise ValueError", stdout="", stderr="ValueError"),
        "a3": _record(agent="coder", source="x = 2"),
    }

    evidence = build_block_evidence(index, 2, records.get)

    assert evidence == {
        "blocks": [
            {
                "block_id": "blk-0001",
                "attempt": 1,
                "status": "ok",
                "agents": ["analyst"],
                "turn_start": 3,
                "turn_end": 5,
                "actions": [
                    {
                        "action_id": "a1",
                        "agent": "analyst",
                        "success": True,
                        "source": "x = 1",
                        "stdout": "qc\n",
                        "stderr": "",
                    },
                    {
                        "action_id": "a2",
                        "agent": "analyst",
                        "success": False,
                        "source": "raise ValueError",
                        "stdout": "",
                        "stderr": "ValueError",
                    },
                ],
                "artifact_paths": ["figures/qc.png"],
            },
            {
                "block_id": "blk-0003",
                "attempt": 2,
                "status": "warn",
                "agents": ["analyst"],
                "turn_start": 3,
                "turn_end": 5,
                "actions": [
                    {
                        "action_id": "a3",
                        "agent": "coder",
                        "success": True,
                        "source": "x = 2",
                        "stdout": "qc\n",
                        "stderr": "",
                    }
                ],
                "artifact_paths": ["figures/v2.png"],
            },
        ],
        "truncated": False,
    }


def test_item_without_blocks_gives_empty_evidence() -> None:
    assert build_block_evidence(_index(_block(1, work_item_id=2)), 9, lambda _a: None) == {
        "blocks": [],
        "truncated": False,
    }


def test_missing_record_is_listed_and_flagged_not_skipped() -> None:
    index = _index(_block(1, action_ids=["known", "lost"], failed_action_ids=["lost"]))
    records = {"known": _record()}

    evidence = build_block_evidence(index, 2, records.get)

    actions = evidence["blocks"][0]["actions"]
    assert [a["action_id"] for a in actions] == ["known", "lost"]
    assert "missing_record" not in actions[0]
    assert actions[1] == {
        "action_id": "lost",
        "agent": None,
        "success": False,  # from the block's failed_action_ids
        "source": None,
        "stdout": None,
        "stderr": None,
        "missing_record": True,
    }
    assert evidence["truncated"] is False


def test_incomplete_record_raises() -> None:
    index = _index(_block(1, action_ids=["a1"]))
    with pytest.raises(KeyError, match="lacks \\['stderr'\\]"):
        build_block_evidence(index, 2, lambda _a: {"agent": "x", "success": True, "source": "", "stdout": ""})


# --- caps ---------------------------------------------------------------------


def test_cap_text_keeps_head_and_tail_with_a_marker() -> None:
    text = "H" * 10 + "M" * 100 + "T" * 10
    capped, cut = cap_text(text, 20)
    assert cut is True
    assert capped.startswith("H" * 10)
    assert capped.endswith("T" * 10)
    assert "[100 characters truncated]" in capped
    assert cap_text("short", 20) == ("short", False)
    assert cap_text("x" * 20, 20) == ("x" * 20, False)


@pytest.mark.parametrize(
    ("field", "cap"),
    [("source", SOURCE_CAP), ("stdout", STDOUT_CAP), ("stderr", STDERR_CAP)],
)
def test_each_field_is_capped_and_sets_truncated(field: str, cap: int) -> None:
    index = _index(_block(1, action_ids=["a1"]))
    long = "y" * (cap * 3)
    evidence = build_block_evidence(index, 2, lambda _a: _record(**{field: long}))

    action = evidence["blocks"][0]["actions"][0]
    assert evidence["truncated"] is True
    assert len(action[field]) < cap + 80
    assert "characters truncated" in action[field]
    for other in {"source", "stdout", "stderr"} - {field}:
        assert "truncated" not in action[other]


# --- drop_oldest_outputs ------------------------------------------------------------


def test_drop_oldest_outputs_strips_one_block_at_a_time_oldest_first() -> None:
    index = _index(
        _block(1, attempt=1, action_ids=["a1"]),
        _block(2, attempt=2, action_ids=["a2"], artifact_paths=["figures/v2.png"]),
    )
    evidence = build_block_evidence(index, 2, lambda _a: _record())

    assert drop_oldest_outputs(evidence) == "blk-0001"
    first, second = evidence["blocks"]
    assert first["outputs_dropped"] is True
    assert first["actions"][0]["source"] is None
    assert first["actions"][0]["stdout"] is None
    assert first["actions"][0]["stderr"] is None
    assert first["actions"][0]["success"] is True
    assert first["artifact_paths"] == ["figures/qc.png"]
    assert "outputs_dropped" not in second
    assert second["actions"][0]["source"] == "print('qc')"

    assert drop_oldest_outputs(evidence) == "blk-0002"
    assert drop_oldest_outputs(evidence) is None


# --- lookups ------------------------------------------------------------------------


def test_record_lookup_skips_legacy_records_without_action_id() -> None:
    lookup = code_lookup_from_records(
        [
            {"action_id": None, "agent": "old", "success": True, "source": "legacy", "stdout": "", "stderr": ""},
            {"action_id": "a1", "agent": "analyst", "success": False, "source": "new", "stdout": "o", "stderr": "e", "extra": 1},
        ]
    )
    assert lookup("a1") == {"agent": "analyst", "success": False, "source": "new", "stdout": "o", "stderr": "e"}
    assert lookup("legacy") is None


def _runner_event(event_type: str, agent: str, payload: dict, turn: int = 3) -> dict:
    return {
        "schema_version": "caribou.runner_event.v1",
        "event_type": event_type,
        "occurred_at": "2026-10-09T00:00:00Z",
        "run_id": "source-id",
        "turn": turn,
        "agent_name": agent,
        "payload": payload,
    }


def _write_event_log(path: Path, events: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_cli_event_log_lookup_joins_submission_and_result(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    _write_event_log(
        log,
        [
            _runner_event("message_complete", "analyst", {"message": {}}),
            _runner_event("code_submitted", "analyst", {"action_id": "a1", "block_id": "blk-0001", "source": "x = 1"}),
            _runner_event("code_result", "analyst", {"action_id": "a1", "block_id": "blk-0001", "success": True, "stdout": "ok", "stderr": ""}),
            _runner_event("code_submitted", "coder", {"action_id": "a2", "block_id": "blk-0001", "source": "boom"}),
            _runner_event("code_result", "coder", {"action_id": "a2", "block_id": "blk-0001", "success": False, "stdout": "", "stderr": "Traceback"}),
            _runner_event("code_submitted", "coder", {"action_id": "a3", "block_id": "blk-0001", "source": "never finished"}),
        ],
    )

    lookup = code_lookup_from_cli_event_log(log)

    assert lookup("a1") == {"agent": "analyst", "success": True, "source": "x = 1", "stdout": "ok", "stderr": ""}
    assert lookup("a2") == {"agent": "coder", "success": False, "source": "boom", "stdout": "", "stderr": "Traceback"}
    assert lookup("a3") is None  # submitted, no result: no record to show
    assert lookup("unknown") is None


def test_cli_event_log_reader_rejects_missing_or_malformed_logs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_cli_event_log(tmp_path / "events.jsonl")
    log = tmp_path / "events.jsonl"
    log.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_cli_event_log(log)
    _write_event_log(
        log,
        [_runner_event("code_result", "a", {"action_id": "orphan", "success": True, "stdout": "", "stderr": ""})],
    )
    with pytest.raises(ValueError, match="without a code_submitted"):
        read_cli_event_log(log)


# --- payload -----------------------------------------------------------------------


def _item() -> dict:
    return {"id": 2, "title": "QC", "body": "Compute QC metrics", "status": "Done"}


def test_payload_carries_evidence_and_explains_it() -> None:
    evidence = build_block_evidence(_index(_block(1, action_ids=["a1"])), 2, lambda _a: _record())
    payload = build_work_item_review_payload(run_id="r", item=_item(), diff="diff", evidence=evidence)

    assert payload["evidence"] is evidence
    instructions = payload["instructions"]
    assert "evidence" in instructions
    assert "git_diff only covers the work-item metadata" in instructions
    assert "stdout and stderr" in instructions


def test_payload_without_evidence_is_unchanged() -> None:
    payload = build_work_item_review_payload(run_id="r", item=_item(), diff="diff")
    assert "evidence" not in payload
    assert "evidence" not in payload["instructions"]


def test_budget_fitting_drops_oldest_outputs_and_notes_it() -> None:
    index = _index(
        _block(1, attempt=1, action_ids=["a1"]),
        _block(2, attempt=2, action_ids=["a2"]),
        _block(3, attempt=3, action_ids=["a3"]),
    )
    evidence = build_block_evidence(index, 2, lambda _a: _record(source="z = 1\n" * 300))
    payload = build_work_item_review_payload(run_id="r", item=_item(), diff="", evidence=evidence)
    # Fits once the two oldest blocks' outputs are gone but not before.
    budget = 1400

    dropped = fit_review_payload_to_budget(payload, "review carefully", budget)

    assert dropped == ["blk-0001", "blk-0002"]
    assert evidence["blocks"][0]["outputs_dropped"] is True
    assert evidence["blocks"][1]["outputs_dropped"] is True
    assert "outputs_dropped" not in evidence["blocks"][2]
    assert evidence["blocks"][2]["actions"][0]["source"].startswith("z = 1")
    assert "blk-0001, blk-0002" in evidence["note"]


def test_budget_fitting_is_a_no_op_when_the_payload_fits() -> None:
    evidence = build_block_evidence(_index(_block(1, action_ids=["a1"])), 2, lambda _a: _record())
    payload = build_work_item_review_payload(run_id="r", item=_item(), diff="", evidence=evidence)
    assert fit_review_payload_to_budget(payload, "prompt", 100_000) == []
    assert "note" not in evidence
    assert fit_review_payload_to_budget({"kind": "x"}, "prompt", 1) == []


def test_review_still_raises_when_evidence_cannot_fit(tmp_path: Path) -> None:
    store = WorkItemStore(tmp_path / "work-items", session_id="s", policy=WorkItemPolicy())
    item = store.open("QC", "b" * 5000, "analyst", 1)
    store.close(item["id"], "done", "analyst", 2)
    evidence = build_block_evidence(_index(_block(1, work_item_id=item["id"], action_ids=["a1"])), item["id"], lambda _a: _record())

    with pytest.raises(EvaluationContextTooLarge):
        evaluate_work_item(
            store=store,
            item_id=item["id"],
            run_id="s",
            turn=2,
            evaluator_agent=Agent("reviewer", "Review", {}, {}),
            llm_client=SimpleNamespace(),
            model_name="m",
            evidence=evidence,
            max_context_tokens=50,
        )
    # Every block's outputs were dropped before giving up, and that is stated.
    assert evidence["blocks"][0]["outputs_dropped"] is True
    assert "blk-0001" in evidence["note"]
    assert store.read(item["id"])["reviews"][-1]["error"]


# --- end to end with a real BlockTracker run ------------------------------------------


def _fake_client(calls: list, verdict: str = "approve"):
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="response-1",
            model="review-model",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"verdict": verdict, "assessment": "Saw the code"})
                    )
                )
            ],
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_review_of_a_block_tracker_run_sends_its_code(tmp_path: Path) -> None:
    store = WorkItemStore(tmp_path / "work-items", session_id="s", policy=WorkItemPolicy())
    tracker = BlockTracker(blocks_path_for(store), "s", store, on_change=lambda _b: None)
    item = store.open("QC metrics", "Compute per-cell QC", "analyst", 1)
    # The engine's code records, keyed by action id, as a web session or the
    # CLI event log would hold them.
    records: dict = {}

    def run(action_id: str, source: str, success: bool, stdout: str, stderr: str = "") -> None:
        tracker.begin_action("analyst", 2, action_id)
        tracker.finish_action(action_id, success)
        records[action_id] = {"agent": "analyst", "success": success, "source": source, "stdout": stdout, "stderr": stderr}

    run("s:turn:2:block:1", "adata.obs['n_genes'] = (adata.X > 0).sum(1)", True, "computed\n")
    run("s:turn:2:block:2", "plt.savefig('figures/qc.png')", False, "", "OSError")
    tracker.record_artifact("figures/qc.png", "s:turn:2:block:1")
    store.close(item["id"], "QC metrics computed and plotted", "analyst", 3)
    tracker.sync()

    evidence = build_block_evidence(load_blocks(blocks_path_for(store)), item["id"], records.get)
    calls: list = []
    result = evaluate_work_item(
        store=store,
        item_id=item["id"],
        run_id="s",
        turn=3,
        evaluator_agent=Agent("reviewer", "Review carefully", {}, {}),
        llm_client=_fake_client(calls),
        model_name="review-model",
        evidence=evidence,
    )

    assert result["verdict"] == "approve"
    assert result["evidence"] is evidence
    sent = json.loads(calls[0]["messages"][1]["content"])
    assert sent["evidence"]["truncated"] is False
    [block] = sent["evidence"]["blocks"]
    assert block["block_id"] == "blk-0001"
    assert block["status"] == "error"  # its last action failed
    assert block["artifact_paths"] == ["figures/qc.png"]
    assert [a["source"] for a in block["actions"]] == [
        "adata.obs['n_genes'] = (adata.X > 0).sum(1)",
        "plt.savefig('figures/qc.png')",
    ]
    assert [a["success"] for a in block["actions"]] == [True, False]
    assert block["actions"][1]["stderr"] == "OSError"
    assert "evidence field holds what was actually executed" in sent["instructions"]
