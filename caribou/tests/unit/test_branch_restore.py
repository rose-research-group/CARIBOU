"""Branch restore building blocks (session_recovery), driven against a fake
sandbox that really executes Python on a real AnnData: block-entry
checkpoints with fingerprints, pinned retention, fingerprint verification,
literal replay from the original dataset (including a branch of a branch)
and copying unchanged artifacts."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import anndata
import numpy as np
import pytest

from caribou.execution.session_recovery import (
    CONTAINER_LIVE_DATASET,
    LEDGER_BASE_ORIGINAL,
    ROLLING_CHECKPOINT_RETENTION,
    bootstrap_anndata,
    capture_checkpoint,
    checkpoint_dataset_path,
    copy_unchanged_artifacts,
    literal_replay,
    load_checkpoint,
    verify_fingerprint,
)


class PythonSandbox:
    """Executes code in one persistent namespace, mapping /workspace paths."""

    def __init__(self, dataset: Path, output_dir: Path) -> None:
        self.dataset = dataset
        self.output_dir = output_dir
        self.namespace: dict = {}

    def exec_code(self, source: str, timeout: int) -> dict:
        source = source.replace(CONTAINER_LIVE_DATASET, str(self.output_dir / Path(CONTAINER_LIVE_DATASET).name))
        source = source.replace("/workspace/dataset.h5ad", str(self.dataset))
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(source, self.namespace)
        except Exception as exc:  # the sandbox reports failures as results
            return {"status": "error", "stdout": stdout.getvalue(), "stderr": repr(exc)}
        return {"status": "ok", "stdout": stdout.getvalue(), "stderr": ""}


def _dataset(path: Path) -> Path:
    adata = anndata.AnnData(np.arange(12, dtype=float).reshape(4, 3))
    adata.obs["sample"] = ["a", "a", "b", "b"]
    adata.write_h5ad(path)
    return path


def _session(tmp_path: Path, name: str, dataset: Path, sandbox=None) -> SimpleNamespace:
    output_dir = tmp_path / name / "outputs"
    output_dir.mkdir(parents=True)
    return SimpleNamespace(
        id=name,
        output_dir=output_dir,
        config=SimpleNamespace(dataset_path=str(dataset)),
        current_agent="analyst",
        current_turn=0,
        sandbox_manager=sandbox,
        memory_manager=None,
        events=[],
        attempts=[],
        forked_from_checkpoint_id=None,
        checkpoint_id=None,
        checkpoint_turn=None,
        checkpoint_healthy=False,
    )


def _run(sandbox: PythonSandbox, ledger: list, session_id: str, turn: int, source: str) -> None:
    result = sandbox.exec_code(source, timeout=600)
    ledger.append(
        {
            "action_id": f"{session_id}:{turn}:{len(ledger)}",
            "turn": turn,
            "agent_name": "analyst",
            "source": source,
            "recorded_result": {
                "success": result["status"] == "ok",
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "duration_ms": 0,
            },
        }
    )


def _state(ledger: list, turns_completed: int) -> dict:
    return {
        "schema_version": "caribou.web_runner_checkpoint_state.v1",
        "current_agent_name": "analyst",
        "turns_completed": turns_completed,
        "next_turn": turns_completed + 1,
        "action_ledger": [dict(item) for item in ledger],
    }


LOAD = "import anndata\nadata = anndata.read_h5ad('/workspace/dataset.h5ad')\n"
# A failed attempt that changes state before it raises (A3).
PARTIAL = "adata.var['mt'] = [True, False, False]\nraise TypeError('late failure')\n"
LAYER = "adata.layers['counts'] = adata.X.copy()\n"
EMBED = "adata.obsm['X_pca'] = adata.X[:, :2]\n"


def test_block_entry_capture_is_pinned_unpublished_and_fingerprinted(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    session = _session(tmp_path, "src", dataset)
    sandbox = PythonSandbox(dataset, session.output_dir)
    session.sandbox_manager = sandbox
    published = capture_checkpoint(session=session, history=[], runner_state=_state([], 0))
    ledger: list = []
    _run(sandbox, ledger, "src", 1, LOAD)
    _run(sandbox, ledger, "src", 1, PARTIAL)

    # Mid-turn: turns_completed is still 0, yet the sandbox write runs.
    entry = capture_checkpoint(
        session=session,
        history=[{"role": "user", "content": "go"}],
        runner_state=_state(ledger, 0),
        pin_block_id="blk-0002",
        publish=False,
    )

    assert entry["pin_block_id"] == "blk-0002"
    assert entry["complete"] is True
    assert entry["capture_error"] is None
    assert entry["dataset"]["kind"] == "working_anndata"
    assert entry["ledger_base"] == LEDGER_BASE_ORIGINAL
    assert entry["actions"] == ledger
    assert entry["fingerprint"] == {
        "n_obs": 4,
        "n_vars": 3,
        "obs_keys": ["sample"],
        "var_keys": ["mt"],
        "obsm_keys": [],
        "layers_keys": [],
    }
    # Not published: latest and the session's checkpoint fields are unchanged.
    assert load_checkpoint(session.output_dir)["checkpoint_id"] == published["checkpoint_id"]
    assert session.checkpoint_id == published["checkpoint_id"]
    assert session.checkpoint_turn == 0
    assert load_checkpoint(session.output_dir, entry["checkpoint_id"]) == entry
    restored = anndata.read_h5ad(checkpoint_dataset_path(session.output_dir, entry))
    assert list(restored.var.columns) == ["mt"]


def test_block_entry_capture_requires_the_runner_ledger(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    session = _session(tmp_path, "src", dataset)
    with pytest.raises(ValueError, match="action_ledger"):
        capture_checkpoint(
            session=session,
            history=[],
            runner_state={"turns_completed": 1},
            pin_block_id="blk-0001",
            publish=False,
        )


def test_block_entry_capture_without_adata_is_incomplete_with_null_fingerprint(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    session = _session(tmp_path, "src", dataset)
    sandbox = PythonSandbox(dataset, session.output_dir)
    session.sandbox_manager = sandbox
    ledger: list = []
    _run(sandbox, ledger, "src", 1, "x = 1\n")

    entry = capture_checkpoint(
        session=session,
        history=[],
        runner_state=_state(ledger, 1),
        pin_block_id="blk-0002",
        publish=False,
    )

    assert entry["fingerprint"] is None
    assert "adata" in entry["capture_error"]
    assert entry["complete"] is False
    assert entry["dataset"]["kind"] == "original_dataset"


def test_first_block_entry_with_no_actions_is_complete_on_the_original(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    session = _session(tmp_path, "src", dataset)
    session.sandbox_manager = PythonSandbox(dataset, session.output_dir)

    entry = capture_checkpoint(
        session=session,
        history=[],
        runner_state=_state([], 0),
        pin_block_id="blk-0001",
        publish=False,
    )

    assert entry["complete"] is True
    assert entry["fingerprint"] is None
    assert entry["dataset"]["kind"] == "original_dataset"
    assert entry["actions"] == []


def test_pinned_checkpoints_survive_rolling_retention(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    session = _session(tmp_path, "src", dataset)
    published = capture_checkpoint(session=session, history=[], runner_state=_state([], 0))
    pinned = [
        capture_checkpoint(
            session=session,
            history=[],
            runner_state=_state([], 0),
            pin_block_id=f"blk-{index:04d}",
            publish=False,
        )["checkpoint_id"]
        for index in range(1, 6)
    ]
    rolling = [
        capture_checkpoint(session=session, history=[], runner_state=_state([], 0))[
            "checkpoint_id"
        ]
        for _ in range(5)
    ]

    root = session.output_dir.parent / ".checkpoints"
    retained = {path.parent.name for path in root.glob("checkpoint_*/checkpoint.json")}
    assert set(pinned) <= retained
    assert published["checkpoint_id"] not in retained
    assert retained - set(pinned) == set(rolling[-ROLLING_CHECKPOINT_RETENTION:])


def test_unpublished_captures_never_prune_the_published_latest(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    session = _session(tmp_path, "src", dataset)
    published = capture_checkpoint(session=session, history=[], runner_state=_state([], 0))
    for _ in range(5):
        capture_checkpoint(
            session=session, history=[], runner_state=_state([], 0), publish=False
        )
    assert load_checkpoint(session.output_dir)["checkpoint_id"] == published["checkpoint_id"]


def test_legacy_capture_stores_the_event_ledger_in_runner_state(tmp_path):
    """A4: a runner resumed from a checkpoint captured without a runner
    ledger inherits the event-derived ledger instead of restarting at []."""
    dataset = _dataset(tmp_path / "input.h5ad")
    session = _session(tmp_path, "src", dataset)
    session.events = [
        {"type": "code_submitted", "turn": 1, "data": {"action_id": "a1", "source": LOAD}},
        {"type": "code_result", "turn": 1, "data": {"action_id": "a1", "success": True}},
    ]

    checkpoint = capture_checkpoint(
        session=session,
        history=[],
        runner_state={"current_agent_name": "analyst", "turns_completed": 1},
    )

    assert [item["action_id"] for item in checkpoint["actions"]] == ["a1"]
    assert checkpoint["runner_state"]["action_ledger"] == checkpoint["actions"]
    assert checkpoint["ledger_base"] == LEDGER_BASE_ORIGINAL


def _replay_into_new_sandbox(tmp_path, name, dataset, checkpoint):
    sandbox = PythonSandbox(dataset, tmp_path / name)
    (tmp_path / name).mkdir()
    assert bootstrap_anndata(sandbox)[0] is True
    progress: list = []
    replayed, detail = literal_replay(sandbox=sandbox, checkpoint=checkpoint, emit=progress.append)
    return sandbox, replayed, detail


def test_replay_from_the_original_dataset_matches_the_entry_fingerprint(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    session = _session(tmp_path, "src", dataset)
    sandbox = PythonSandbox(dataset, session.output_dir)
    session.sandbox_manager = sandbox
    ledger: list = []
    _run(sandbox, ledger, "src", 1, LOAD)
    _run(sandbox, ledger, "src", 1, PARTIAL)
    _run(sandbox, ledger, "src", 2, LAYER)
    entry = capture_checkpoint(
        session=session, history=[], runner_state=_state(ledger, 2),
        pin_block_id="blk-0003", publish=False,
    )

    child_sandbox, replayed, detail = _replay_into_new_sandbox(tmp_path, "child", dataset, entry)
    assert replayed is True, detail
    # The failed attempt's partial state change is part of the result.
    assert verify_fingerprint(child_sandbox, entry["fingerprint"]) == (
        True,
        "Restored AnnData matches the checkpoint fingerprint (4 observations x 3 variables).",
    )


def test_a_branch_of_a_branch_replays_its_inherited_and_own_attempts(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    source = _session(tmp_path, "src", dataset)
    source_sandbox = PythonSandbox(dataset, source.output_dir)
    source.sandbox_manager = source_sandbox
    ledger: list = []
    _run(source_sandbox, ledger, "src", 1, LOAD)
    _run(source_sandbox, ledger, "src", 1, PARTIAL)
    entry = capture_checkpoint(
        session=source, history=[], runner_state=_state(ledger, 1),
        pin_block_id="blk-0002", publish=False,
    )
    _run(source_sandbox, ledger, "src", 2, "adata.obs['dropped'] = 1\n")

    # The branch restores E by replay; its runner is seeded with E's runner
    # state, so its ledger continues E's ledger.
    branch_sandbox, replayed, _ = _replay_into_new_sandbox(tmp_path, "b1-sandbox", dataset, entry)
    assert replayed is True
    branch = _session(tmp_path, "b1", dataset, branch_sandbox)
    branch_sandbox.output_dir = branch.output_dir
    branch_ledger = [dict(item) for item in entry["runner_state"]["action_ledger"]]
    _run(branch_sandbox, branch_ledger, "b1", 2, LAYER)
    branch_entry = capture_checkpoint(
        session=branch, history=[], runner_state=_state(branch_ledger, 2),
        pin_block_id="blk-0003", publish=False,
    )
    _run(branch_sandbox, branch_ledger, "b1", 3, EMBED)

    assert branch_entry["ledger_base"] == LEDGER_BASE_ORIGINAL
    assert [item["source"] for item in branch_entry["actions"]] == [LOAD, PARTIAL, LAYER]

    grandchild, replayed, detail = _replay_into_new_sandbox(tmp_path, "b2", dataset, branch_entry)
    assert replayed is True, detail
    ok, detail = verify_fingerprint(grandchild, branch_entry["fingerprint"])
    assert ok is True, detail
    assert branch_entry["fingerprint"]["layers_keys"] == ["counts"]
    assert "dropped" not in grandchild.namespace["adata"].obs


def test_verify_fingerprint_lists_every_differing_field(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    sandbox = PythonSandbox(dataset, tmp_path)
    bootstrap_anndata(sandbox)
    expected = {
        "n_obs": 5,
        "n_vars": 3,
        "obs_keys": ["leiden"],
        "var_keys": [],
        "obsm_keys": [],
        "layers_keys": [],
    }
    ok, detail = verify_fingerprint(sandbox, expected)
    assert ok is False
    assert "n_obs: expected 5, got 4" in detail
    assert "obs_keys: missing ['leiden'], unexpected ['sample']" in detail
    assert "n_vars" not in detail


def test_verify_fingerprint_fails_without_adata_and_refuses_a_null_expectation(tmp_path):
    sandbox = PythonSandbox(tmp_path / "none.h5ad", tmp_path)
    ok, detail = verify_fingerprint(
        sandbox,
        {"n_obs": 1, "n_vars": 1, "obs_keys": [], "var_keys": [], "obsm_keys": [], "layers_keys": []},
    )
    assert ok is False and "adata" in detail
    with pytest.raises(ValueError):
        verify_fingerprint(sandbox, None)


def test_copy_unchanged_artifacts_skips_changed_and_missing_files(tmp_path):
    dataset = _dataset(tmp_path / "input.h5ad")
    session = _session(tmp_path, "src", dataset)
    (session.output_dir / "plots").mkdir()
    (session.output_dir / "plots" / "umap.png").write_bytes(b"umap")
    (session.output_dir / "table.csv").write_text("a,b\n")
    (session.output_dir / "gone.txt").write_text("bye")
    entry = capture_checkpoint(session=session, history=[], runner_state=_state([], 0))
    assert {item["path"] for item in entry["artifacts"]} == {
        "plots/umap.png", "table.csv", "gone.txt",
    }
    assert all(len(item["sha256"]) == 64 for item in entry["artifacts"])
    (session.output_dir / "table.csv").write_text("a,b\n1,2\n")
    (session.output_dir / "gone.txt").unlink()
    (session.output_dir / "after.txt").write_text("new")

    destination = tmp_path / "child" / "outputs"
    skipped = copy_unchanged_artifacts(
        session.output_dir, destination, manifest=entry["artifacts"]
    )

    assert sorted(skipped) == ["gone.txt", "table.csv"]
    copied = sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file())
    assert copied == ["plots/umap.png"]
    with pytest.raises(FileExistsError):
        copy_unchanged_artifacts(session.output_dir, destination, manifest=entry["artifacts"])


def test_copy_unchanged_artifacts_rejects_paths_outside_the_output_dir(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "secret").write_text("x")
    with pytest.raises(ValueError):
        copy_unchanged_artifacts(
            tmp_path / "outputs",
            tmp_path / "dst",
            manifest=[{"path": "../secret", "sha256": "0" * 64, "size_bytes": 1}],
        )
