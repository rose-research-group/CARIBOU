"""
WebSocket endpoint for real-time session streaming.

Protocol:
  Client → Server:
    { "type": "run",          "content": "<initial prompt>" }
    { "type": "user_message", "content": "<next turn>",
      "block_id"?: "<workbench block id>"                    }
    { "type": "stop"                                         }
    { "type": "ping"                                         }

  Server → Client:
    All events from the session event log (see models.make_event).
    Existing events are replayed on connect; new events are streamed live.
    Every log event carries an integer `seq` (strictly increasing per
    session). The log is compacted (token events are removed once their
    message completes), so positions in it are not stable — clients are
    tracked by the last seq they were sent, never by list index.
"""
from __future__ import annotations

import asyncio
import bisect
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from caribou.server.models import SessionStatus
from caribou.server.session_manager import session_manager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/sessions/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    session = session_manager.get_session(session_id)
    if not session:
        # Close code 4004 is the "session not found" signal; no frame is sent
        # because every frame on this socket is a session log event with seq.
        await websocket.close(code=4004)
        return

    # Send full event history so the client can rebuild state on reconnect.
    # Snapshot first: the log can be appended to or compacted while we await.
    history = list(session.events)
    for event in history:
        try:
            await websocket.send_json(event)
        except Exception:
            return
    last_seq = history[-1]["seq"] if history else 0

    # Stream new events as they arrive; also handle inbound messages
    send_task = asyncio.create_task(_stream_events(websocket, session, last_seq))
    recv_task = asyncio.create_task(_receive_messages(websocket, session))

    try:
        done, pending = await asyncio.wait(
            {send_task, recv_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        send_task.cancel()
        recv_task.cancel()
    except Exception:
        pass


def _events_after(session, last_seq: int) -> list:
    """Snapshot of the session's log events with seq > last_seq."""
    start = bisect.bisect_right(session.events, last_seq, key=lambda e: e["seq"])
    return session.events[start:]


async def _stream_events(websocket: WebSocket, session, last_seq: int) -> None:
    """Wait for events with seq > last_seq and forward them to the WebSocket."""
    while True:
        async with session.event_condition:
            pending = _events_after(session, last_seq)
            while not pending:
                if session.status in (SessionStatus.stopped, SessionStatus.error):
                    return
                await session.event_condition.wait()
                pending = _events_after(session, last_seq)

            for event in pending:
                last_seq = event["seq"]
                try:
                    await websocket.send_json(event)
                except Exception:
                    return

        if session.status in (SessionStatus.stopped, SessionStatus.error):
            return


async def _receive_messages(websocket: WebSocket, session) -> None:
    """Handle inbound WebSocket messages from the client."""
    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            return
        except Exception:
            return

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type")

        if msg_type == "ping":
            await websocket.send_json({
                "type": "pong",
                "session_id": session.id,
                "turn": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {},
            })

        elif msg_type == "run":
            content = msg.get("content", "")
            if not content:
                continue
            await session_manager.start_run(session.id, content)

        elif msg_type == "user_message":
            content = msg.get("content", "")
            if content:
                # The manager validates block_id (type and existence) and logs
                # a refusal as an error event; an absent or null id means the
                # message is not about a block.
                await session_manager.send_user_message(
                    session.id, content, block_id=msg.get("block_id")
                )

        elif msg_type == "stop":
            await session_manager.stop_session(session.id)

        elif msg_type == "cancel_response":
            await session_manager.cancel_response(session.id)
