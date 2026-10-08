"""Saving a blueprint through the config API must keep its brief_policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from caribou.agents.AgentSystem import AgentSystem
from caribou.server.routes import config as config_routes

BRIEF_POLICY = {"enabled": True, "mode": "seed_item", "require_confirmation": False}


def _blueprint(**extra) -> dict:
    return {
        "global_policy": "Be careful.",
        "evaluator_agent": None,
        "work_item_policy": {"qc_mode": "optional"},
        **extra,
        "agents": {
            "driver": {"prompt": "Drive.", "rag": {"enabled": False}, "neighbors": {}},
        },
    }


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr(config_routes, "DEFAULT_AGENT_DIR", tmp_path)
    app = FastAPI()
    app.include_router(config_routes.router)
    return TestClient(app)


def _get_then_put(client: TestClient, name: str) -> dict:
    response = client.get(f"/api/config/blueprints/{name}")
    assert response.status_code == 200, response.text
    payload = response.json()
    put = client.put(f"/api/config/blueprints/{name}", json=payload)
    assert put.status_code == 200, put.text
    return payload


def test_brief_policy_round_trips_through_get_and_save(client, tmp_path: Path) -> None:
    path = tmp_path / "with_brief.json"
    path.write_text(json.dumps(_blueprint(brief_policy=BRIEF_POLICY)))

    payload = _get_then_put(client, "with_brief")

    assert payload["brief_policy"] == BRIEF_POLICY
    on_disk = json.loads(path.read_text())
    assert on_disk["brief_policy"] == BRIEF_POLICY
    loaded = AgentSystem.load_from_json(str(path)).brief_policy
    assert loaded.to_dict() == BRIEF_POLICY


def test_blueprint_without_brief_policy_stays_without_one(client, tmp_path: Path) -> None:
    path = tmp_path / "no_brief.json"
    path.write_text(json.dumps(_blueprint()))

    payload = _get_then_put(client, "no_brief")

    assert payload["brief_policy"] is None
    assert "brief_policy" not in json.loads(path.read_text())


def test_create_blueprint_accepts_brief_policy(client, tmp_path: Path) -> None:
    body = {
        "name": "created",
        "global_policy": "",
        "agents": {"driver": {"prompt": "Drive."}},
        "brief_policy": BRIEF_POLICY,
    }
    response = client.post("/api/config/blueprints", json=body)
    assert response.status_code == 201, response.text
    assert response.json()["brief_policy"] == BRIEF_POLICY
    assert json.loads((tmp_path / "created.json").read_text())["brief_policy"] == BRIEF_POLICY


def test_invalid_brief_policy_is_rejected(client) -> None:
    body = {
        "name": "bad",
        "global_policy": "",
        "agents": {"driver": {"prompt": "Drive."}},
        "brief_policy": {"enabled": "yes", "mode": "context"},
    }
    response = client.post("/api/config/blueprints", json=body)
    assert response.status_code == 422
