"""A8: branching end to end through the real web runner and SessionManager.

Sandbox: the sessions use `sandbox_type="offline"`, but the server's
`session_setup.build_sandbox` has no `offline` branch (it raises
SANDBOX_TYPE_UNKNOWN), and the only real kernel launcher runs
`/opt/offline_kernel.py` inside a Singularity SIF with `--containall --net
--network none`, which unit tests cannot rely on. So `build_sandbox` is
replaced with `OfflineKernelSandbox`: it executes every chunk with the real
offline kernel's `_run` in a persistent per-sandbox namespace (exactly what
`offline_kernel.py --repl` does inside the container), and rewrites the two
container paths to the host paths the container binds them to:
`/workspace/dataset.h5ad` -> the session's dataset, `/workspace/outputs` ->
the session's output dir.

LLM: `build_llm_client` returns a scripted streaming fake per session name;
no network is used.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import anndata
import numpy as np
import pytest

from caribou.execution.blocks import load_blocks
from caribou.server.models import (
    BranchRequest,
    RecoveryStatus,
    SessionCreateRequest,
    SessionStatus,
)
from caribou.server.session_manager import SessionManager

_EXEC_LOCK = threading.Lock()


class OfflineKernelSandbox:
    """The offline kernel's executor, run in-process with container-path binds."""

    def __init__(self, dataset_path: str, output_dir: Path) -> None:
        from caribou.sandbox import offline_kernel

        self._run = offline_kernel._run
        self.namespace: dict = {"__builtins__": __builtins__}
        self.dataset_path = str(dataset_path)
        self.output_dir = Path(output_dir)
        self.stopped = False

    def _bind(self, code: str) -> str:
        return code.replace("/workspace/outputs", str(self.output_dir)).replace(
            "/workspace/dataset.h5ad", self.dataset_path
        )

    def exec_code(self, code: str, timeout: int = 600) -> dict:
        if self.stopped:
            raise RuntimeError("sandbox was stopped")
        # redirect_stdout is process-wide: one chunk at a time.
        with _EXEC_LOCK:
            return self._run(self._bind(code), self.namespace)

    def stop_container(self) -> None:
        self.stopped = True


class ScriptedLLM:
    """Streams one scripted response per call; running out fails loudly."""

    def __init__(self, name: str, responses: list[str]) -> None:
        self.name = name
        self.responses = list(responses)
        self.requests: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if not kwargs.get("stream"):
            raise AssertionError(f"{self.name}: unexpected non-streaming call")
        self.requests.append([dict(item) for item in kwargs["messages"]])
        if not self.responses:
            raise AssertionError(f"{self.name}: script exhausted")
        text = self.responses.pop(0)
        return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])]


def _code(body: str, command: str) -> str:
    return f"```python\n{body}\n```\n{command}"


SOURCE_SCRIPT = [
    'open_work_item "QC" "flag cells"',
    _code(
        "import anndata\n"
        "adata = anndata.read_h5ad('/workspace/dataset.h5ad')\n"
        "adata.obs['qc'] = 1\n"
        "print('SOURCE_A')",
        'close_work_item 0 "flagged"',
    ),
    'open_work_item "Normalize" "add a layer"',
    _code(
        "adata.layers['norm'] = adata.X.copy()\nprint('SOURCE_B_MARKER')",
        'close_work_item 1 "layered"',
    ),
    "Source analysis complete.",
]

BRANCH_TEXT = "Use a log layer instead."


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    import caribou.server.session_manager as sm
    from caribou.sandbox import offline_kernel  # noqa: F401  (patches json once)

    original_json_default = json.JSONEncoder.default
    rag_stub = ModuleType("caribou.execution.rag_client")
    rag_stub.get_rag_client = lambda _console: None
    monkeypatch.setitem(__import__("sys").modules, "caribou.execution.rag_client", rag_stub)

    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(sm, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(sm, "_SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(sm, "ENV_FILE", tmp_path / "caribou.env")

    blueprint = tmp_path / "blueprint.json"
    blueprint.write_text(
        json.dumps(
            {
                "global_policy": "Be accurate.",
                "agents": {
                    "analyst": {
                        "prompt": "You analyze single-cell data.",
                        "neighbors": {},
                        "code_samples": [],
                    }
                },
            }
        )
    )
    dataset = tmp_path / "input.h5ad"
    anndata.AnnData(X=np.arange(15, dtype=np.float32).reshape(5, 3)).write_h5ad(dataset)

    scripts: dict[str, list[str]] = {"source": SOURCE_SCRIPT}
    llms: dict[str, list[ScriptedLLM]] = {}
    sandboxes: list[OfflineKernelSandbox] = []

    def build_llm_client(config):
        llm = ScriptedLLM(config.name, scripts[config.name])
        llms.setdefault(config.name, []).append(llm)
        return llm, "scripted-model"

    def build_sandbox(config, output_dir):
        assert config.sandbox_type.value == "offline"
        sandbox = OfflineKernelSandbox(config.dataset_path, output_dir)
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr(sm, "build_llm_client", build_llm_client)
    monkeypatch.setattr(sm, "build_sandbox", build_sandbox)

    manager = object.__new__(SessionManager)
    manager._sessions = {}
    manager._deleted_session_ids = set()
    manager._lock = asyncio.Lock()

    config = SessionCreateRequest(
        name="source",
        mode="interactive",
        agent_system=str(blueprint),
        llm_backend="openrouter",
        model_name="scripted/model",
        sandbox_type="offline",
        dataset_path=str(dataset),
    )
    yield SimpleNamespace(
        manager=manager,
        config=config,
        scripts=scripts,
        llms=llms,
        sandboxes=sandboxes,
        dataset=dataset,
    )
    json.JSONEncoder.default = original_json_default


async def _until(predicate, *, timeout: float = 60.0, what: str = "") -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.05)


def _tree_digest(root: Path) -> dict[str, str]:
    """Content hash of every file under a session dir except its live log."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "session.log"
    }


def _fatal_errors(session) -> list:
    return [e["data"] for e in session.events if e["type"] == "error" and e["data"].get("fatal")]


async def _run_to_idle(env, session, calls: int) -> None:
    def done() -> bool:
        if _fatal_errors(session):
            raise AssertionError(f"runner error in {session.name}: {_fatal_errors(session)}")
        llm = (env.llms.get(session.config.name) or [None])[-1]
        return (
            llm is not None
            and len(llm.requests) == calls
            and session.status == SessionStatus.idle
        )

    await _until(done, what=f"{session.name} to finish {calls} LLM calls")


async def _stop(manager: SessionManager, session) -> None:
    await manager.stop_session(session.id)
    await asyncio.wait_for(session.runner_task, timeout=30)
    assert session.status == SessionStatus.stopped


async def _build_source(env):
    response = await env.manager.create_session(env.config)
    source = env.manager.get_session(response.id)
    await _until(
        lambda: source.status == SessionStatus.idle and source.sandbox_manager is not None,
        what="source initialization",
    )
    assert await env.manager.start_run(source.id, "Start the analysis.")
    await _run_to_idle(env, source, len(SOURCE_SCRIPT))
    await _stop(env.manager, source)
    return source


async def _branch(env, parent, block_id: str, mode: str, name: str, script: list[str]):
    env.scripts[name] = script
    response = await env.manager.branch_session(
        parent.id,
        block_id,
        BranchRequest(instruction=BRANCH_TEXT, restore_mode=mode, name=name),
    )
    assert response.forked_from_block_id == block_id
    assert response.branch_restore_mode.value == mode
    child = env.manager.get_session(response.id)
    await _until(
        lambda: child.recovery_status
        in {RecoveryStatus.recovered, RecoveryStatus.failed, RecoveryStatus.partial},
        what=f"{name} restore",
    )
    assert child.recovery_status == RecoveryStatus.recovered, child.recovery_detail
    await _run_to_idle(env, child, len(script))
    return child


def _blocks(session) -> dict[str, dict]:
    index = load_blocks(session.output_dir.parent / "blocks.json")
    return {block["block_id"]: block for block in index["blocks"]}


def _first_request(env, child) -> list[dict]:
    return env.llms[child.config.name][-1].requests[0]


def _branch_turn(block_id: str, title: str, mode: str) -> str:
    return f'[Branch from {block_id} "{title}" · {mode}]\n{BRANCH_TEXT}'


def _assert_branch_start(env, source, child, *, mode: str, old_marker: str) -> None:
    expected_turn = _branch_turn("blk-0002", "Normalize", mode)
    request = _first_request(env, child)
    # The child's model history does not contain blk-0002's old attempt...
    assert not any(old_marker in message["content"] for message in request)
    # ...and its first user turn is the branch instruction.
    user_turns = [message for message in request if message["role"] == "user"]
    assert user_turns[-1]["content"] == expected_turn
    child_user_messages = [m.content for m in child.messages if m.role == "user"]
    assert child_user_messages[-1] == expected_turn
    assert not any(old_marker in m.content for m in child.messages)

    source_blocks = _blocks(source)
    child_blocks = _blocks(child)
    assert sorted(child_blocks) == ["blk-0001", "blk-0002"]
    inherited = child_blocks["blk-0001"]
    assert inherited["inherited_from"] == {"session_id": source.id, "block_id": "blk-0001"}
    # Immutable: the child's run changed nothing on the inherited block.
    for field in ("action_ids", "failed_action_ids", "status", "turn_end", "updated_at", "entry"):
        assert inherited[field] == source_blocks["blk-0001"][field]
    fresh = child_blocks["blk-0002"]
    assert fresh["inherited_from"] is None
    assert fresh["session_id"] == child.id
    assert fresh["action_ids"] and all(a.startswith(child.id) for a in fresh["action_ids"])
    assert fresh["entry"]["checkpoint_id"] is not None
    assert child.code_events[-1].success
    assert child.parent_session_id == source.id
    assert child.forked_from_checkpoint_id == source_blocks["blk-0002"]["entry"]["checkpoint_id"]


def test_replay_and_checkpoint_branches_end_to_end(env) -> None:
    async def scenario() -> None:
        source = await _build_source(env)
        source_blocks = _blocks(source)
        assert sorted(source_blocks) == ["blk-0001", "blk-0002"]
        entry = source_blocks["blk-0002"]["entry"]
        assert entry["checkpoint_id"] and entry["checkpoint_complete"]
        assert entry["fingerprint"]["obs_keys"] == ["qc"]
        assert entry["fingerprint"]["layers_keys"] == []
        source_dir = source.output_dir.parent
        before = _tree_digest(source_dir)
        source_event_count = len(source.events)

        replay = await _branch(
            env,
            source,
            "blk-0002",
            "replay",
            "child-replay",
            [
                _code(
                    "assert 'qc' in adata.obs and 'norm' not in adata.layers\n"
                    "adata.layers['log'] = adata.X.copy()\nprint('CHILD_REPLAY')",
                    'close_work_item 1 "logged"',
                ),
                "Replay branch complete.",
            ],
        )
        assert "matches the checkpoint fingerprint" in replay.recovery_detail
        assert "Replayed 1 recorded code attempts" in replay.recovery_detail
        _assert_branch_start(env, source, replay, mode="replay", old_marker="SOURCE_B_MARKER")
        assert all(event.success for event in replay.code_events)
        assert "CHILD_REPLAY" in replay.code_events[-1].stdout
        # Actively try to touch the inherited block: refused, runner alive.
        inherited_before = _blocks(replay)["blk-0001"]
        assert not await env.manager.send_user_message(
            replay.id, "redo QC", block_id="blk-0001"
        )
        assert replay.events[-1]["data"]["code"] == "INHERITED_BLOCK"
        await asyncio.sleep(1.5)
        assert not replay.runner_task.done()
        assert replay.status == SessionStatus.idle
        assert _blocks(replay)["blk-0001"] == inherited_before

        checkpoint = await _branch(
            env,
            source,
            "blk-0002",
            "checkpoint",
            "child-checkpoint",
            [
                _code(
                    "assert 'qc' in adata.obs and 'norm' not in adata.layers\n"
                    "adata.obs['ck'] = 1\nprint('CHILD_CHECKPOINT')",
                    'close_work_item 1 "ck"',
                ),
                "Checkpoint branch complete.",
            ],
        )
        assert "Restored the AnnData saved at the block's entry" in checkpoint.recovery_detail
        _assert_branch_start(
            env, source, checkpoint, mode="checkpoint", old_marker="SOURCE_B_MARKER"
        )
        assert "CHILD_CHECKPOINT" in checkpoint.code_events[-1].stdout

        # The source is untouched by either branch.
        assert _tree_digest(source_dir) == before
        assert len(source.events) == source_event_count
        assert source.status == SessionStatus.stopped
        assert [b.session_id for b in env.manager.list_branches(source.id)] == [
            replay.id,
            checkpoint.id,
        ]
        for child in (replay, checkpoint):
            await _stop(env.manager, child)

    asyncio.run(scenario())


def test_branch_of_a_branch_replays_the_inherited_and_appended_ledger(env) -> None:
    async def scenario() -> None:
        source = await _build_source(env)
        child = await _branch(
            env,
            source,
            "blk-0002",
            "replay",
            "child-one",
            [
                _code("adata.obs['c1'] = 2\nprint('C1_A')", 'close_work_item 1 "c1"'),
                'open_work_item "Extra" "more"',
                _code("adata.obs['c1b'] = 3\nprint('C1_B_MARKER')", 'close_work_item 2 "extra"'),
                "Child one complete.",
            ],
        )
        await _stop(env.manager, child)
        child_blocks = _blocks(child)
        assert sorted(child_blocks) == ["blk-0001", "blk-0002", "blk-0003"]
        entry = child_blocks["blk-0003"]["entry"]
        assert entry["fingerprint"]["obs_keys"] == ["c1", "qc"]
        child_before = _tree_digest(child.output_dir.parent)

        grandchild = await _branch(
            env,
            child,
            "blk-0003",
            "replay",
            "child-two",
            [
                _code(
                    "assert 'c1' in adata.obs and 'c1b' not in adata.obs\nadata.obs['c2'] = 4",
                    'close_work_item 2 "again"',
                ),
                "Child two complete.",
            ],
        )
        # The ledger was inherited from the source AND appended to by the
        # child: both attempts replay from the original dataset.
        assert "Replayed 2 recorded code attempts" in grandchild.recovery_detail
        assert "matches the checkpoint fingerprint" in grandchild.recovery_detail
        request = _first_request(env, grandchild)
        assert not any("C1_B_MARKER" in message["content"] for message in request)
        assert [m for m in request if m["role"] == "user"][-1]["content"] == _branch_turn(
            "blk-0003", "Extra", "replay"
        )
        blocks = _blocks(grandchild)
        assert sorted(blocks) == ["blk-0001", "blk-0002", "blk-0003"]
        # inherited_from names the immediate parent, also for blk-0001.
        assert blocks["blk-0001"]["inherited_from"] == {"session_id": child.id, "block_id": "blk-0001"}
        assert blocks["blk-0002"]["inherited_from"] == {"session_id": child.id, "block_id": "blk-0002"}
        assert blocks["blk-0003"]["inherited_from"] is None
        assert blocks["blk-0003"]["action_ids"][0].startswith(grandchild.id)
        assert _tree_digest(child.output_dir.parent) == child_before
        await _stop(env.manager, grandchild)

    asyncio.run(scenario())
