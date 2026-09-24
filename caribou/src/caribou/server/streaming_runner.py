"""
Async-compatible event-emitting runner for the CARIBOU web server.

Runs the agent session loop in a background thread and emits structured
events via a thread-safe callback. The server's WebSocket route forwards
those events to the browser.

This is a parallel implementation to execution/runner.py — it shares the
same helpers and execution model but replaces Console output with events.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

# Consecutive no-action / code-exec failures we tolerate before ending the run.
MAX_CONSECUTIVE_NO_ACTION = 3
MAX_CONSECUTIVE_EXEC_FAILURES = 5


# ---------------------------------------------------------------------------
# Token streaming helper
# ---------------------------------------------------------------------------


def _stream_tokens(llm_client, model: str, messages: List[Dict]) -> Any:
    """
    Generator yielding text tokens from the LLM.
    Works with AnthropicClient (has stream_chat) and OpenAI-compatible clients.
    """
    if hasattr(llm_client, "stream_chat"):
        yield from llm_client.stream_chat(
            model=model, messages=messages, temperature=0.0
        )
    else:
        stream = llm_client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, stream=True
        )
        if hasattr(stream, "choices"):
            content = stream.choices[0].message.content
            if content:
                yield content
            return
        for chunk in stream:
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                yield token


# ---------------------------------------------------------------------------
# Sync runner (runs inside a ThreadPoolExecutor thread)
# ---------------------------------------------------------------------------


def run_session_sync(
    *,
    session_id: str,
    agent_system,
    driver_agent,
    analysis_context: str,
    llm_client,
    sandbox_manager,
    history: List[Dict],
    is_auto: bool,
    max_turns: int,
    model_name: str,
    output_dir: Path,
    emit: Callable[[Dict], None],
    stop_flag: threading.Event,
    cancel_response_flag: Optional[threading.Event] = None,
    user_input_queue: Optional[queue.Queue] = None,
    control_message_queue: Optional[queue.Queue] = None,
    logger: Optional[logging.Logger] = None,
    memory_manager: Any = None,
    report_memory: Any = None,
    checkpoint_callback: Optional[Callable[[List[Dict], Dict[str, Any]], None]] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    start_waiting: bool = False,
    work_item_store: Optional["WorkItemStore"] = None,
) -> None:
    """
    Main agent session loop. Replaces Console output with emit() calls.
    Designed to be run in a thread via asyncio.to_thread or ThreadPoolExecutor.

    `work_item_store`: pass the session's cached singleton instance so the
    turn loop and REST work-item routes see the same in-memory index cache
    (see WS-0). Constructed locally only when not supplied, e.g. by direct
    unit tests of this function.
    """
    from caribou.execution.ActionSpace import AgentActionSpace
    from caribou.execution.agent_management import (
        _apply_agent_switch,
        _extract_possible_actions,
    )
    from caribou.execution.message_utils import (
        _code_preview,
        _count_code_blocks,
        detect_delegation,
        detect_end_session,
        detect_rag,
    )
    from caribou.execution.rag_client import get_rag_client
    from caribou.execution.work_items import (
        WorkItemError,
        WorkItemPolicy,
        WorkItemStore,
        parse_work_item_command,
        render_work_item_prompt,
    )
    from caribou.execution.work_item_runtime import (
        apply_command as apply_work_item_command,
        end_session_block,
        render_work_item_state,
        stall_report,
        transfer_on_delegation,
    )
    from caribou.core.io_helpers import (
        extract_python_code_blocks,
        format_execute_response,
    )
    from rich.console import Console

    console = Console(quiet=True)
    cancel_response_flag = cancel_response_flag or threading.Event()
    output_dir.mkdir(parents=True, exist_ok=True)
    emitted_artifacts: Set[str] = set()
    work_item_policy = getattr(agent_system, "work_item_policy", WorkItemPolicy())
    evaluator_agent_name = getattr(agent_system, "evaluator_agent_name", None)
    # `output_dir` is the sandbox mount: agent code executing in it can edit
    # or delete files there, and checkpoint restore can roll it back. Work
    # items live in a sibling directory outside the sandbox so neither can
    # silently revert work-item state. Constructed unconditionally — see the
    # matching comment in runner.py for why auto mode also gets a store.
    work_items = work_item_store or WorkItemStore(
        output_dir.parent / "work-items",
        session_id=session_id,
        policy=work_item_policy,
        origin_run_id=session_id,
    )

    def _agent_prompt(agent: Any) -> str:
        return agent.get_full_prompt(None) + render_work_item_prompt(work_item_policy)

    def _emit(event_type: str, data: Dict, turn: int = 0) -> None:
        if event_type == "system_message" and "id" not in data:
            # Every one of this file's ~12 system_message call sites omits
            # `id`, so the frontend's SessionResponse `id: undefined` was
            # identical across all of them. `upsertMessage` on the frontend
            # dedupes system messages by id — with every id equal to
            # `undefined`, only the FIRST system message a session ever
            # emits was ever shown; every later one (work-item feedback,
            # delegation-transfer confirmations, RAG failures, ...) was
            # silently dropped as an apparent duplicate. Assign a real id
            # here, once, instead of touching every call site.
            data = {**data, "id": str(uuid4())}
        emit(
            {
                "type": event_type,
                "session_id": session_id,
                "turn": turn,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            }
        )

    def _scan_new_artifacts(turn: int) -> None:
        """Emit artifact events for any new files in output_dir."""
        mime_map = {
            ".png": ("plot", "image/png"),
            ".jpg": ("plot", "image/jpeg"),
            ".svg": ("plot", "image/svg+xml"),
            ".pdf": ("plot", "application/pdf"),
            ".csv": ("data", "text/csv"),
            ".h5ad": ("data", "application/x-hdf5"),
            ".txt": ("report", "text/plain"),
        }
        if not output_dir.exists():
            return
        for fpath in sorted(output_dir.iterdir()):
            if not fpath.is_file() or fpath.name in emitted_artifacts:
                continue
            suffix = fpath.suffix.lower()
            art_type, mime_type = mime_map.get(
                suffix, ("data", "application/octet-stream")
            )
            emitted_artifacts.add(fpath.name)
            _emit(
                "artifact",
                {
                    "artifact": {
                        "filename": fpath.name,
                        "type": art_type,
                        "mime_type": mime_type,
                        "size_bytes": fpath.stat().st_size,
                        "local_path": str(fpath),
                        "turn": turn,
                    }
                },
                turn=turn,
            )

    current_agent = driver_agent
    if resume_state and resume_state.get("current_agent_name"):
        restored = agent_system.get_agent(resume_state["current_agent_name"])
        if restored is not None:
            current_agent = restored
    action_space = AgentActionSpace(current_agent.name)
    action_space.set_possible_actions(_extract_possible_actions(current_agent))
    if resume_state:
        action_space.past_actions = [
            dict(item) for item in resume_state.get("action_space_past_actions", [])
        ]
    else:
        action_init_msg = action_space.to_message()
        history.append({"role": "system", "content": action_init_msg})
        _emit(
            "system_message",
            {"content": action_init_msg, "category": "Action space"},
        )
        if memory_manager is not None:
            memory_manager.add_message("system", action_init_msg)

    turns_completed = int((resume_state or {}).get("turns_completed", 0))
    consecutive_failures = int((resume_state or {}).get("consecutive_exec_failures", 0))
    consecutive_no_action = int((resume_state or {}).get("consecutive_no_action", 0))
    action_ledger = [
        dict(item) for item in (resume_state or {}).get("action_ledger", [])
    ]

    # Report-memory tracking: which history index belongs to the current agent.
    current_agent_history_start = int(
        (resume_state or {}).get("current_agent_history_start", len(history))
    )
    # At most one automatic "continue" per user message, consumed after the agent
    # opens a work item so it can proceed with the work instead of stopping.
    auto_continue_budget = 1

    def _checkpoint_boundary() -> None:
        if checkpoint_callback is None:
            return
        checkpoint_callback(
            [dict(item) for item in history],
            {
                "schema_version": "caribou.web_runner_checkpoint_state.v1",
                "current_agent_name": current_agent.name,
                "turns_completed": turns_completed,
                "next_turn": turns_completed + 1,
                "consecutive_exec_failures": consecutive_failures,
                "consecutive_no_action": consecutive_no_action,
                "action_space_past_actions": [
                    dict(item) for item in action_space.past_actions
                ],
                "action_ledger": [dict(item) for item in action_ledger],
                "current_agent_history_start": current_agent_history_start,
            },
        )

    def _drain_control_messages(turn: int) -> None:
        if control_message_queue is None:
            return
        while True:
            try:
                content = control_message_queue.get_nowait()
            except queue.Empty:
                return
            history.append({"role": "system", "content": content})
            if memory_manager is not None:
                memory_manager.add_message("system", content)
            _emit(
                "system_message",
                {"content": content, "category": "Configuration change"},
                turn=turn,
            )

    def _wait_for_user(turn: int, reason: Optional[str] = None) -> bool:
        """Wait for one interactive message; return false if the session stops."""
        nonlocal auto_continue_budget
        _emit("status_change", {"status": "idle", "reason": reason}, turn=turn)
        if logger:
            logger.info("Waiting for user input | turn: %s", turn)
        if user_input_queue is None:
            return False

        while True:
            _drain_control_messages(turns_completed)
            if stop_flag.is_set():
                if logger:
                    logger.info("Session stopped while waiting for user input")
                _emit(
                    "status_change", {"status": "stopped", "reason": "user_requested"}
                )
                return False
            try:
                user_msg = user_input_queue.get(timeout=1.0)
                if logger:
                    logger.info(
                        "User message received | turn: %s | length: %s chars",
                        turn,
                        len(user_msg),
                    )
                history.append({"role": "user", "content": user_msg})
                if memory_manager is not None:
                    memory_manager.add_message("user", user_msg)
                auto_continue_budget = 1
                next_turn = turns_completed + 1
                _emit(
                    "message_complete",
                    {
                        "message": {
                            "id": f"msg_{session_id}_user_{next_turn}",
                            "turn": next_turn,
                            "role": "user",
                            "agent_name": "",
                            "content": user_msg,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    },
                    turn=next_turn,
                )
                return True
            except queue.Empty:
                continue

    try:
        if start_waiting and not is_auto:
            if not _wait_for_user(turns_completed, "recovered_ready"):
                return
        while True:
            _drain_control_messages(turns_completed)
            if stop_flag.is_set():
                if logger:
                    logger.info("Session stopped (user requested)")
                _emit(
                    "status_change", {"status": "stopped", "reason": "user_requested"}
                )
                return

            if is_auto and turns_completed >= max_turns:
                if logger:
                    logger.info("Session stopped — max turns reached (%s)", max_turns)
                _emit(
                    "status_change",
                    {"status": "stopped", "reason": f"max turns reached ({max_turns})"},
                )
                return

            turn = turns_completed + 1
            _emit(
                "status_change",
                {
                    "status": "running",
                    "reason": f"turn {turn} — {current_agent.name}",
                },
                turn=turn,
            )

            # --- Build cleaned context ---
            if report_memory is not None:
                working_history = history[current_agent_history_start:]
                cleaned_context = report_memory.build_context(working_history)
                if logger:
                    logger.info(
                        "Turn %s | agent: %s | context_length: %s (report_memory)",
                        turn,
                        current_agent.name,
                        len(cleaned_context),
                    )
            elif memory_manager is not None:
                cleaned_context = memory_manager.get_context()
                if logger:
                    logger.info(
                        "Turn %s | agent: %s | context_length: %s (episodic)",
                        turn,
                        current_agent.name,
                        len(cleaned_context),
                    )
            else:
                cleaned_context = []
                for msg in history:
                    m = msg.copy()
                    if isinstance(m.get("content"), str):
                        m["content"] = m["content"].rstrip()
                    cleaned_context.append(m)

            # Ambient work-item state (WS-2): recomputed from the store each
            # turn and appended after context assembly, same as runner.py.
            state_block = render_work_item_state(work_items, current_agent.name)
            if state_block:
                cleaned_context.append({"role": "system", "content": state_block})

            # Item-level stall detection (WS-4).
            stalled = stall_report(
                work_items,
                current_agent.name,
                turn=turn,
                stall_turns=work_item_policy.stall_turns,
            )
            if stalled is not None:
                if stalled["idle_turns"] >= work_item_policy.stall_halt_turns:
                    halt_note = (
                        f"Halted: work item {stalled['item_id']} "
                        f"('{stalled['title']}') stalled for "
                        f"{stalled['idle_turns']} turns."
                    )
                    work_items.note(stalled["item_id"], "runner", turn, halt_note)
                    _checkpoint_boundary()
                    _emit(
                        "status_change",
                        {"status": "stopped", "reason": "work_item_stalled"},
                        turn=turn,
                    )
                    return
                cleaned_context.append(
                    {
                        "role": "system",
                        "content": (
                            f"[SYSTEM] Work item {stalled['item_id']} "
                            f"('{stalled['title']}') has had no status change "
                            f"for {stalled['idle_turns']} turns. Close it, "
                            "transfer it, or explain why it is still open."
                        ),
                    }
                )

            if logger and (memory_manager is None and report_memory is None):
                logger.info(
                    "Turn %s | agent: %s | messages_in_context: %s",
                    turn,
                    current_agent.name,
                    len(cleaned_context),
                )

            # A user turn issues exactly one provider request. A failed stream
            # is surfaced instead of silently replaying the prompt.
            try:
                if logger:
                    logger.info("LLM request | turn: %s | model: %s", turn, model_name)
                llm_started = time.monotonic()
                first_token_at: Optional[float] = None
                full_msg = ""
                token_chars = 0
                for token in _stream_tokens(llm_client, model_name, cleaned_context):
                    if stop_flag.is_set() or cancel_response_flag.is_set():
                        break
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                        if logger:
                            logger.info(
                                "First token received | turn: %s | latency: %sms",
                                turn,
                                int((first_token_at - llm_started) * 1000),
                            )
                    full_msg += token
                    token_chars += len(token)
                    _emit(
                        "token",
                        {"agent_name": current_agent.name, "token": token},
                        turn=turn,
                    )
                msg = full_msg
                if logger:
                    logger.info(
                        "LLM response complete | turn: %s | duration: %sms | ~%s chars",
                        turn,
                        int((time.monotonic() - llm_started) * 1000),
                        token_chars,
                    )
            except Exception as exc:
                if logger:
                    logger.error("LLM error | turn: %s | %s", turn, exc, exc_info=True)
                _emit(
                    "error", {"code": "LLM_ERROR", "message": str(exc), "fatal": True}
                )
                _emit("status_change", {"status": "error", "reason": str(exc)})
                return

            if stop_flag.is_set():
                _emit(
                    "status_change", {"status": "stopped", "reason": "user_requested"}
                )
                return

            if cancel_response_flag.is_set() and not is_auto:
                cancel_response_flag.clear()
                if logger:
                    logger.info("Response cancelled | turn: %s", turn)
                if not _wait_for_user(turn, "response_cancelled"):
                    return
                continue

            history.append({"role": "assistant", "content": msg})
            if memory_manager is not None:
                memory_manager.add_message("assistant", msg)
            turns_completed += 1

            _emit(
                "message_complete",
                {
                    "message": {
                        "id": f"msg_{session_id}_{turn}",
                        "turn": turn,
                        "role": "assistant",
                        "agent_name": current_agent.name,
                        "content": msg,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                },
                turn=turn,
            )

            _action_fired = False
            _delegated = False
            _opened_work_item = False

            # --- Enforced work-item commands ---
            work_command = parse_work_item_command(msg)
            work_result = apply_work_item_command(
                work_items, msg, owner=current_agent.name, turn=turn
            )
            if work_result is not None:
                _action_fired = True
                if (
                    work_command is not None
                    and work_command.name == "open_work_item"
                    and work_result.success
                ):
                    _opened_work_item = True
                feedback = work_result.feedback
                history.append({"role": "system", "content": feedback})
                if memory_manager is not None:
                    memory_manager.add_message("system", feedback)
                _emit(
                    "system_message",
                    {"content": feedback, "category": "Work item"},
                    turn=turn,
                )
                if work_result.changed_item is not None:
                    _emit(
                        "work_item_changed",
                        {"item": work_result.changed_item},
                        turn=turn,
                    )

            # --- End session detection ---
            has_delegation = detect_delegation(msg) is not None
            end_session_refused = False
            if detect_end_session(msg):
                blocking = end_session_block(work_items, current_agent.name)
                if blocking:
                    end_session_refused = True
                    feedback = (
                        f"end_session refused. Current owner {current_agent.name} "
                        "still owns non-Done work items: "
                        + ", ".join(
                            f"{item['id']} ({item['status']}, owner={item['owner']})"
                            for item in blocking
                        )
                    )
                    history.append({"role": "system", "content": feedback})
                    if memory_manager is not None:
                        memory_manager.add_message("system", feedback)
                    _emit(
                        "system_message",
                        {"content": feedback, "category": "Work item"},
                        turn=turn,
                    )
            if (
                detect_end_session(msg)
                and _count_code_blocks(msg) == 0
                and not has_delegation
                and not end_session_refused
            ):
                if is_auto:
                    if logger:
                        logger.info(
                            "Session finished — agent signalled end | turn: %s", turn
                        )
                    _checkpoint_boundary()
                    _emit(
                        "status_change",
                        {"status": "stopped", "reason": "agent_finished"},
                    )
                    return
                else:
                    if logger:
                        logger.info("Agent requested session end | turn: %s", turn)
                    _checkpoint_boundary()
                    _emit(
                        "status_change",
                        {"status": "stopped", "reason": "agent_requested_end"},
                    )
                    return

            # --- RAG ---
            query_str = detect_rag(msg)
            if query_str and current_agent.is_rag_enabled:
                _action_fired = True
                try:
                    rag_client = get_rag_client(console)
                    docs = rag_client.query(query_str)
                    if docs:
                        history.append({"role": "system", "content": docs})
                        _emit(
                            "system_message",
                            {"content": docs, "category": "RAG context"},
                            turn=turn,
                        )
                        if memory_manager is not None:
                            memory_manager.add_message("system", docs)
                except Exception as rag_exc:  # noqa: BLE001 — surface, don't swallow
                    err = (
                        f"[SYSTEM] RAG query for '{query_str}' failed: {rag_exc}. "
                        f"Proceed without retrieved context."
                    )
                    history.append({"role": "system", "content": err})
                    _emit(
                        "system_message",
                        {"content": err, "category": "RAG failure"},
                        turn=turn,
                    )
                    if memory_manager is not None:
                        memory_manager.add_message("system", err)
                    _emit(
                        "error",
                        {
                            "code": "RAG_ERROR",
                            "message": str(rag_exc),
                            "fatal": False,
                        },
                        turn=turn,
                    )

            # --- Delegation ---
            cmd = detect_delegation(msg)
            if cmd and cmd in current_agent.commands:
                target_name = current_agent.commands[cmd].target_agent
                new_agent = agent_system.get_agent(target_name)
                if new_agent is not None:
                    try:
                        transferred_items = transfer_on_delegation(
                            work_items,
                            current_agent.name,
                            target_name,
                            turn=turn,
                            evaluator_agent_name=evaluator_agent_name,
                        )
                    except WorkItemError as exc:
                        feedback = (
                            f"Delegation refused: work-item transfer failed: {exc}"
                        )
                        history.append({"role": "system", "content": feedback})
                        if memory_manager is not None:
                            memory_manager.add_message("system", feedback)
                        _emit(
                            "system_message",
                            {"content": feedback, "category": "Work item"},
                            turn=turn,
                        )
                        new_agent = None
                    else:
                        for transferred_item in transferred_items:
                            _emit(
                                "work_item_changed",
                                {"item": transferred_item},
                                turn=turn,
                            )
                if new_agent:
                    _action_fired = True
                    if logger:
                        logger.info(
                            "Agent switch: %s -> %s | command: %s | turn: %s",
                            current_agent.name,
                            target_name,
                            cmd,
                            turn,
                        )
                    _emit(
                        "agent_switch",
                        {
                            "from_agent": current_agent.name,
                            "to_agent": target_name,
                            "command": cmd,
                            "reason": None,
                        },
                        turn=turn,
                    )
                    history.append(
                        {
                            "role": "assistant",
                            "content": f"Routing to **{target_name}** (command `{cmd}`)",
                        }
                    )
                    if memory_manager is not None:
                        memory_manager.add_message(
                            "assistant",
                            f"Routing to **{target_name}** (command `{cmd}`)",
                        )
                    # Generate handoff report for the departing agent
                    if report_memory is not None:
                        from caribou.execution.report_generation import (
                            _generate_agent_report,
                        )

                        agent_slice = history[current_agent_history_start:]
                        agent_report = _generate_agent_report(
                            console,
                            llm_client=llm_client,
                            model_name=model_name,
                            agent_name=current_agent.name,
                            history_slice=agent_slice,
                        )
                        if agent_report:
                            report_memory.add_report(current_agent.name, agent_report)
                            if logger:
                                logger.info(
                                    "Agent report generated for %s | length: %s chars",
                                    current_agent.name,
                                    len(agent_report),
                                )
                        report_memory.update_agent_prompt(_agent_prompt(new_agent))
                        current_agent_history_start = len(history)
                    switch_history_start = len(history)
                    refreshed_agent_prompt = (
                        _agent_prompt(new_agent) + "\n\n" + analysis_context
                    )
                    _emit(
                        "system_message",
                        {"content": refreshed_agent_prompt, "category": "Agent prompt"},
                        turn=turn,
                    )
                    _apply_agent_switch(
                        new_agent_prompt=_agent_prompt(new_agent),
                        analysis_context=analysis_context,
                        history=history,
                        memory_manager=memory_manager,
                        action_space=action_space,
                        new_agent=new_agent,
                    )
                    for system_item in history[switch_history_start:]:
                        if system_item.get("role") == "system":
                            _emit(
                                "system_message",
                                {
                                    "content": system_item.get("content", ""),
                                    "category": "Agent switch",
                                },
                                turn=turn,
                            )
                    current_agent = new_agent
                    _delegated = True

            # --- Code execution ---
            code_blocks = extract_python_code_blocks(msg)
            if code_blocks:
                _action_fired = True
                for idx, code in enumerate(code_blocks, start=1):
                    if stop_flag.is_set() or cancel_response_flag.is_set():
                        break

                    if logger:
                        logger.info(
                            "Code block %s/%s -> sandbox | turn: %s | lines: %s",
                            idx,
                            len(code_blocks),
                            turn,
                            code.count("\n") + 1,
                        )

                    _emit(
                        "code_submitted",
                        {
                            "action_id": f"{session_id}:{turn}:{idx}",
                            "agent_name": current_agent.name,
                            "source": code,
                            "block_index": idx,
                            "total_blocks": len(code_blocks),
                        },
                        turn=turn,
                    )

                    ledger_entry = {
                        "action_id": f"{session_id}:{turn}:{idx}",
                        "turn": turn,
                        "agent_name": current_agent.name,
                        "source": code,
                        "recorded_result": None,
                    }
                    action_ledger.append(ledger_entry)

                    t0 = time.time()
                    try:
                        exec_result = sandbox_manager.exec_code(code, timeout=600)
                    except Exception as exc:
                        duration_ms = int((time.time() - t0) * 1000)
                        ledger_entry["recorded_result"] = {
                            "success": False,
                            "stdout": "",
                            "stderr": str(exc),
                            "duration_ms": duration_ms,
                        }
                        _emit(
                            "code_result",
                            {
                                "action_id": ledger_entry["action_id"],
                                "agent_name": current_agent.name,
                                "stdout": "",
                                "stderr": str(exc),
                                "success": False,
                                "duration_ms": duration_ms,
                                "block_index": idx,
                            },
                            turn=turn,
                        )
                        _checkpoint_boundary()
                        raise
                    duration_ms = int((time.time() - t0) * 1000)

                    success = exec_result.get("status") == "ok"
                    consecutive_failures = 0 if success else consecutive_failures + 1
                    if logger:
                        logger.info(
                            "Code block %s/%s <- sandbox | turn: %s | duration: %sms | success: %s",
                            idx,
                            len(code_blocks),
                            turn,
                            duration_ms,
                            success,
                        )

                    _emit(
                        "code_result",
                        {
                            "action_id": ledger_entry["action_id"],
                            "agent_name": current_agent.name,
                            "stdout": exec_result.get("stdout", ""),
                            "stderr": exec_result.get("stderr", ""),
                            "success": success,
                            "duration_ms": duration_ms,
                            "block_index": idx,
                        },
                        turn=turn,
                    )
                    ledger_entry["recorded_result"] = {
                        "success": success,
                        "stdout": exec_result.get("stdout", ""),
                        "stderr": exec_result.get("stderr", ""),
                        "duration_ms": duration_ms,
                    }

                    _scan_new_artifacts(turn)

                    feedback = format_execute_response(exec_result, output_dir)
                    action_space.add_action(
                        "code_execution",
                        f"Ran code block {idx}/{len(code_blocks)}:\n{_code_preview(code)}",
                        status=exec_result.get("status"),
                    )
                    action_state_message = action_space.to_message()
                    history.append({"role": "system", "content": action_state_message})
                    _emit(
                        "system_message",
                        {"content": action_state_message, "category": "Action result"},
                        turn=turn,
                    )
                    history.append({"role": "assistant", "content": feedback})
                    if memory_manager is not None:
                        memory_manager.add_message("system", action_state_message)
                        memory_manager.add_message("assistant", feedback)
                        if success:
                            memory_manager.add_pivotal_code(code)

            if cancel_response_flag.is_set() and not is_auto:
                cancel_response_flag.clear()
                _checkpoint_boundary()
                if logger:
                    logger.info("Response cancelled after action | turn: %s", turn)
                if not _wait_for_user(turn, "response_cancelled"):
                    return
                continue

            if _delegated and is_auto:
                consecutive_no_action = 0
                _checkpoint_boundary()
                continue

            # Track / escalate stuck-loop conditions in auto mode.
            if is_auto and consecutive_failures >= MAX_CONSECUTIVE_EXEC_FAILURES:
                _checkpoint_boundary()
                _emit(
                    "status_change",
                    {
                        "status": "stopped",
                        "reason": f"stuck: {consecutive_failures} consecutive code failures",
                    },
                    turn=turn,
                )
                return

            if is_auto and not _action_fired:
                consecutive_no_action += 1
                if consecutive_no_action >= MAX_CONSECUTIVE_NO_ACTION:
                    _checkpoint_boundary()
                    _emit(
                        "status_change",
                        {
                            "status": "stopped",
                            "reason": f"stuck: {consecutive_no_action} consecutive no-action turns",
                        },
                        turn=turn,
                    )
                    return
                no_action_msg = (
                    "[SYSTEM] No action was recognised in your last message. "
                    "Please write executable Python code in a ```python ... ``` block "
                    "or issue a delegation command. "
                    f"(Attempt {consecutive_no_action}/{MAX_CONSECUTIVE_NO_ACTION}; "
                    "the run will halt after that.)"
                )
                history.append({"role": "system", "content": no_action_msg})
                _emit(
                    "system_message",
                    {"content": no_action_msg, "category": "Runner guidance"},
                    turn=turn,
                )
                if memory_manager is not None:
                    memory_manager.add_message("system", no_action_msg)
            elif _action_fired:
                consecutive_no_action = 0

            _checkpoint_boundary()

            if is_auto:
                history.append(
                    {"role": "user", "content": "Please continue with the next step."}
                )
                if memory_manager is not None:
                    memory_manager.add_message(
                        "user", "Please continue with the next step."
                    )
                continue

            # Interactive mode: after opening a work item, let the agent keep going
            # for one extra turn instead of stopping to wait for the user.
            # `not is_auto` is explicit here (rather than relying on the
            # `if is_auto: ... continue` above always firing first in auto
            # mode) because that ordering is incidental, not a contract —
            # work_items is no longer `None` in auto mode, so nothing else
            # here still depends on is_auto to disable this branch.
            if not is_auto and _opened_work_item and auto_continue_budget > 0:
                auto_continue_budget -= 1
                history.append(
                    {"role": "user", "content": "Please continue with the next step."}
                )
                if memory_manager is not None:
                    memory_manager.add_message(
                        "user", "Please continue with the next step."
                    )
                _emit(
                    "system_message",
                    {
                        "content": "Continuing automatically after opening a work item.",
                        "category": "Runner guidance",
                    },
                    turn=turn,
                )
                continue

            # --- Interactive: wait for next user message ---
            if not _wait_for_user(turn):
                return

    except Exception as exc:
        if logger:
            logger.error("Runner error: %s", exc, exc_info=True)
        _emit(
            "error",
            {
                "code": "RUNNER_ERROR",
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "fatal": True,
            },
        )
        _emit("status_change", {"status": "error", "reason": str(exc)})


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


async def run_session_async(
    *,
    session_id: str,
    agent_system,
    driver_agent,
    analysis_context: str,
    llm_client,
    sandbox_manager,
    history: List[Dict],
    is_auto: bool,
    max_turns: int,
    model_name: str,
    output_dir: Path,
    event_callback: Callable[[Dict], None],
    stop_flag: threading.Event,
    cancel_response_flag: Optional[threading.Event] = None,
    user_input_queue: Optional[queue.Queue] = None,
    control_message_queue: Optional[queue.Queue] = None,
    logger: Optional[logging.Logger] = None,
    memory_manager: Any = None,
    report_memory: Any = None,
    checkpoint_callback: Optional[Callable[[List[Dict], Dict[str, Any]], None]] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    start_waiting: bool = False,
    work_item_store: Optional["WorkItemStore"] = None,
) -> None:
    """
    Runs run_session_sync in a thread so it doesn't block the event loop.
    event_callback is called from the background thread — use
    loop.call_soon_threadsafe() internally if needed.
    """
    loop = asyncio.get_running_loop()

    def _emit(event: Dict) -> None:
        loop.call_soon_threadsafe(event_callback, event)

    await asyncio.to_thread(
        run_session_sync,
        session_id=session_id,
        agent_system=agent_system,
        driver_agent=driver_agent,
        analysis_context=analysis_context,
        llm_client=llm_client,
        sandbox_manager=sandbox_manager,
        history=history,
        is_auto=is_auto,
        max_turns=max_turns,
        model_name=model_name,
        output_dir=output_dir,
        emit=_emit,
        stop_flag=stop_flag,
        cancel_response_flag=cancel_response_flag,
        user_input_queue=user_input_queue,
        control_message_queue=control_message_queue,
        logger=logger,
        memory_manager=memory_manager,
        report_memory=report_memory,
        checkpoint_callback=checkpoint_callback,
        resume_state=resume_state,
        start_waiting=start_waiting,
        work_item_store=work_item_store,
    )
