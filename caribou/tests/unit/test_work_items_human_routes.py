"""Workbench REST routes for human tickets and human reviews of work items."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from caribou.execution.work_items import HUMAN_REVIEWER, WorkItemPolicy
from caribou.server.models import CodeEventRecord
from caribou.server.routes import sessions as session_routes

from .test_session_resume_fork_lifecycle import _manager, _stopped_session

BASE = "/api/sessions/source-id/work-items"
ANCHOR = {"block_id": "blk-0003", "action_id": None, "artifact_path": "figures/qc.png"}


def _running(session, qc_mode: str = "optional"):
    session.agent_system = SimpleNamespace(
        agents={"analyst": object(), "coder": object()},
        work_item_policy=WorkItemPolicy(qc_mode=qc_mode),
    )
    return session


@pytest.fixture
def session(tmp_path: Path):
    return _running(_stopped_session(tmp_path))


@pytest.fixture
def manager(session):
    return _manager(session)


@pytest.fixture
def client(manager, monkeypatch) -> TestClient:
    monkeypatch.setattr(session_routes, "session_manager", manager)
    app = FastAPI()
    app.include_router(session_routes.router)
    return TestClient(app)


def _ticket(**overrides) -> dict:
    body = {"title": "Recheck QC", "body": "Use a stricter mito cutoff", "owner": "analyst", "anchor": ANCHOR}
    body.update(overrides)
    return body


def _changed_events(session) -> list:
    return [e for e in session.events if e["type"] == "work_item_changed"]


def _done_item(manager, session) -> int:
    store = manager._work_item_store(session)
    item = store.open("Implement", "Build it", "analyst", 1)
    store.close(item["id"], "Built it", "analyst", 2)
    return item["id"]


# --- POST /work-items ---------------------------------------------------------


def test_create_opens_a_human_ticket_and_emits_an_event(client, session) -> None:
    response = client.post(BASE, json=_ticket())

    assert response.status_code == 201
    item = response.json()
    assert item["owner"] == "analyst"
    assert item["status"] == "In progress"
    assert item["created_turn"] == session.current_turn
    assert item["anchor"] == ANCHOR
    assert item["transitions"][0]["kind"] == "opened"
    assert item["transitions"][0]["actor"] == HUMAN_REVIEWER

    events = _changed_events(session)
    assert len(events) == 1
    assert isinstance(events[0]["seq"], int)
    assert events[0]["data"]["item"]["id"] == item["id"]
    assert events[0]["data"]["item"]["anchor"] == ANCHOR


def test_create_goes_through_the_cached_store(client, session) -> None:
    item = client.post(BASE, json=_ticket()).json()
    assert session.work_item_store is not None
    assert session.work_item_store.read(item["id"])["anchor"] == ANCHOR
    fetched = client.get(f"{BASE}/{item['id']}").json()
    assert fetched["session_id"] == session.id
    assert fetched["anchor"] == ANCHOR


def test_create_without_anchor_is_allowed(client) -> None:
    response = client.post(BASE, json=_ticket(anchor=None))
    assert response.status_code == 201
    assert response.json()["anchor"] is None


def test_create_with_owner_outside_the_blueprint_is_422(client, session) -> None:
    response = client.post(BASE, json=_ticket(owner="nobody"))
    assert response.status_code == 422
    assert "nobody" in response.json()["detail"]
    assert _changed_events(session) == []
    assert client.get(BASE).json() == []


def test_create_for_unknown_session_is_404(client) -> None:
    response = client.post("/api/sessions/nope/work-items", json=_ticket())
    assert response.status_code == 404


def test_create_on_a_session_without_blueprint_is_409(client, session) -> None:
    session.agent_system = None
    assert client.post(BASE, json=_ticket()).status_code == 409


def test_create_with_all_null_anchor_is_409(client, session) -> None:
    empty = {"block_id": None, "action_id": None, "artifact_path": None}
    assert client.post(BASE, json=_ticket(anchor=empty)).status_code == 409
    assert _changed_events(session) == []


def test_create_with_unknown_anchor_field_is_422(client) -> None:
    anchor = dict(ANCHOR, line=4)
    assert client.post(BASE, json=_ticket(anchor=anchor)).status_code == 422


def test_create_with_empty_title_is_409(client) -> None:
    assert client.post(BASE, json=_ticket(title="  ")).status_code == 409


def test_create_requires_owner(client) -> None:
    body = _ticket()
    del body["owner"]
    assert client.post(BASE, json=body).status_code == 422


# --- POST /work-items/{n}/human-review -----------------------------------------


def test_human_approve_records_a_user_review(client, manager, session) -> None:
    item_id = _done_item(manager, session)

    response = client.post(
        f"{BASE}/{item_id}/human-review",
        json={"verdict": "approve", "assessment": "Looks right"},
    )

    assert response.status_code == 200
    item = response.json()
    assert item["status"] == "Done"
    assert item["reviews"][-1]["evaluator"] == HUMAN_REVIEWER
    assert item["reviews"][-1]["verdict"] == "approve"
    assert item["reviews"][-1]["turn"] == session.current_turn
    events = _changed_events(session)
    assert len(events) == 1
    assert events[0]["data"]["item"]["reviews"][-1]["evaluator"] == HUMAN_REVIEWER


def test_human_reject_reopens_a_done_item(client, manager, session) -> None:
    item_id = _done_item(manager, session)

    response = client.post(
        f"{BASE}/{item_id}/human-review",
        json={"verdict": "reject", "assessment": "Cutoff is too loose"},
    )

    assert response.status_code == 200
    item = response.json()
    assert item["status"] == "In progress"
    assert item["completed_turn"] is None
    assert item["transitions"][-1]["kind"] == "reopened"
    assert item["transitions"][-1]["actor"] == HUMAN_REVIEWER
    assert _changed_events(session)[-1]["data"]["item"]["status"] == "In progress"


def test_human_reject_with_empty_assessment_is_409(client, manager, session) -> None:
    item_id = _done_item(manager, session)
    response = client.post(
        f"{BASE}/{item_id}/human-review",
        json={"verdict": "reject", "assessment": ""},
    )
    assert response.status_code == 409
    assert _changed_events(session) == []


def test_human_review_in_wrong_status_is_409(client, manager, session) -> None:
    item = manager._work_item_store(session).open("Implement", "Build it", "analyst", 1)
    response = client.post(
        f"{BASE}/{item['id']}/human-review",
        json={"verdict": "approve", "assessment": "ok"},
    )
    assert response.status_code == 409


def test_human_review_of_unknown_item_is_404(client) -> None:
    response = client.post(
        f"{BASE}/99/human-review", json={"verdict": "approve", "assessment": "ok"}
    )
    assert response.status_code == 404


def test_human_review_of_unknown_session_is_404(client) -> None:
    response = client.post(
        "/api/sessions/nope/work-items/0/human-review",
        json={"verdict": "approve", "assessment": "ok"},
    )
    assert response.status_code == 404


def test_human_review_with_bad_verdict_is_422(client, manager, session) -> None:
    item_id = _done_item(manager, session)
    response = client.post(
        f"{BASE}/{item_id}/human-review",
        json={"verdict": "maybe", "assessment": "ok"},
    )
    assert response.status_code == 422


def test_human_review_on_a_session_without_blueprint_is_409(
    client, manager, session
) -> None:
    item_id = _done_item(manager, session)
    session.agent_system = None
    response = client.post(
        f"{BASE}/{item_id}/human-review",
        json={"verdict": "approve", "assessment": "ok"},
    )
    assert response.status_code == 409


def test_detail_models_validate_a_real_store_item(manager, session) -> None:
    # WorkItemDetail used to require `run_id`, which the store never writes,
    # so GET detail and the evaluator /review result failed on real items.
    from caribou.server.models import WorkItemDetail, WorkItemReviewResult

    item = manager._work_item_store(session).read(_done_item(manager, session))
    detail = WorkItemDetail.model_validate(item)
    assert detail.session_id == session.id
    assert detail.origin_run_id == session.id
    WorkItemReviewResult.model_validate(
        {"item": item, "verdict": "approve", "assessment": "ok"}
    )


# --- POST /work-items/{n}/review: the evaluator sees the item's blocks ----------


def test_evaluator_review_route_returns_the_evidence_it_sent(
    client, manager, session, monkeypatch
) -> None:
    from types import SimpleNamespace as NS

    from .test_blocks_route import _block, _write_blocks

    item_id = _done_item(manager, session)
    session.evaluator_llm_client = object()
    session.evaluator_model_name = "judge"
    _write_blocks(
        session.output_dir.parent / "blocks.json",
        [_block(1, work_item_id=item_id, action_ids=["a1"], artifact_paths=["figures/qc.png"])],
    )
    session.code_events.append(
        CodeEventRecord(
            session_id=session.id, turn=2, agent_name="analyst", source="print('qc')",
            stdout="qc", success=True, action_id="a1", block_id="blk-0001",
        )
    )
    monkeypatch.setattr(
        "caribou.server.session_manager.resolve_evaluator_agent",
        lambda _system: (NS(name="evaluator", get_full_prompt=lambda _p: "review"), "test"),
    )
    sent: dict = {}

    def fake_run_evaluation(**kwargs):
        sent.update(kwargs["payload"])
        return '{"verdict": "approve", "assessment": "The code ran and produced the plot"}'

    monkeypatch.setattr("caribou.execution.evaluation.run_evaluation", fake_run_evaluation)

    response = client.post(f"{BASE}/{item_id}/review")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verdict"] == "approve"
    # The response model fills the optional flags the builder leaves out.
    assert [b["block_id"] for b in body["evidence"]["blocks"]] == [
        b["block_id"] for b in sent["evidence"]["blocks"]
    ]
    assert sent["evidence"]["blocks"][0]["actions"][0]["source"] == "print('qc')"
    [block] = body["evidence"]["blocks"]
    assert block["block_id"] == "blk-0001"
    assert block["artifact_paths"] == ["figures/qc.png"]
    assert block["actions"] == [
        {
            "action_id": "a1", "agent": "analyst", "success": True,
            "source": "print('qc')", "stdout": "qc", "stderr": "",
            "missing_record": False,
        }
    ]
    assert body["evidence"]["truncated"] is False
    assert body["evidence"]["note"] is None
