from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from caribou.core.python_environments import PythonEnvironmentError
from caribou.execution.evaluation import EvaluationContextTooLarge
from caribou.execution.work_items import WorkItemConflict, WorkItemNotFound
from caribou.server.models import (
    ArtifactRecord,
    BlockRecord,
    BlocksResponse,
    BriefDecisionRequest,
    CodeEventRecord,
    EvaluationResult,
    EvaluatorModelState,
    EvaluatorModelUpdateRequest,
    MessageRecord,
    SessionCreateRequest,
    SessionForkRequest,
    SessionResumeRequest,
    SessionResponse,
    WorkItemCreateRequest,
    WorkItemDetail,
    WorkItemHumanReviewRequest,
    WorkItemReviewResult,
    WorkItemSummary,
)
from caribou.server.session_manager import UnknownWorkItemOwner, session_manager


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(body: SessionCreateRequest) -> SessionResponse:
    try:
        return await session_manager.create_session(body)
    except PythonEnvironmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _lifecycle_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(404, str(exc).strip("'"))
    if isinstance(exc, PythonEnvironmentError):
        return HTTPException(400, str(exc))
    return HTTPException(409, str(exc))


@router.post("/{session_id}/resume", response_model=SessionResponse, status_code=202)
async def resume_session(
    session_id: str, body: SessionResumeRequest
) -> SessionResponse:
    try:
        return await session_manager.resume_session(session_id, body)
    except (KeyError, ValueError) as exc:
        raise _lifecycle_error(exc) from exc


@router.post("/{session_id}/fork", response_model=SessionResponse, status_code=202)
async def fork_session(session_id: str, body: SessionForkRequest) -> SessionResponse:
    try:
        return await session_manager.fork_session(session_id, body)
    except (KeyError, ValueError) as exc:
        raise _lifecycle_error(exc) from exc


@router.post(
    "/{session_id}/recovery/retry", response_model=SessionResponse, status_code=202
)
async def retry_recovery(
    session_id: str, body: SessionResumeRequest
) -> SessionResponse:
    try:
        return await session_manager.retry_recovery(session_id, body)
    except (KeyError, ValueError) as exc:
        raise _lifecycle_error(exc) from exc


@router.post("/{session_id}/recovery/accept-partial", response_model=SessionResponse)
async def accept_partial_recovery(session_id: str) -> SessionResponse:
    try:
        return await session_manager.accept_partial_recovery(session_id)
    except (KeyError, ValueError) as exc:
        raise _lifecycle_error(exc) from exc


@router.get("", response_model=List[SessionResponse])
async def list_sessions() -> List[SessionResponse]:
    return session_manager.list_sessions()


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s.to_response()


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    await session_manager.delete_session(session_id)


@router.get("/{session_id}/messages", response_model=List[MessageRecord])
async def get_messages(
    session_id: str, offset: int = 0, limit: int = 500
) -> List[MessageRecord]:
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s.messages[offset : offset + limit]


@router.get("/{session_id}/notebook")
async def download_notebook(session_id: str):
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")

    from caribou.core.io_helpers import chat_history_to_notebook

    history = [
        {"role": message.role, "content": message.content}
        for message in s.messages
        if message.role in ("user", "assistant")
    ]
    notebook = chat_history_to_notebook(history)
    filename = f"caribou-session-{session_id[:8]}.ipynb"
    return Response(
        content=json.dumps(notebook, indent=2),
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{session_id}/artifacts", response_model=List[ArtifactRecord])
async def get_artifacts(session_id: str) -> List[ArtifactRecord]:
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s.artifacts


@router.get("/{session_id}/artifacts/{artifact_id}/download")
async def download_artifact(session_id: str, artifact_id: str):
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")

    artifact = next((a for a in s.artifacts if a.id == artifact_id), None)
    if not artifact:
        raise HTTPException(404, "Artifact not found")

    path = Path(artifact.local_path)
    if not path.exists():
        raise HTTPException(404, "Artifact file not found on disk")

    return FileResponse(
        path=str(path),
        media_type=artifact.mime_type,
        filename=artifact.filename,
    )


@router.get("/{session_id}/code_events", response_model=List[CodeEventRecord])
async def get_code_events(session_id: str) -> List[CodeEventRecord]:
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s.code_events


@router.get("/{session_id}/brief")
async def get_brief_state(session_id: str) -> dict:
    """The session's current phase and frozen brief (null until accepted)."""
    try:
        return session_manager.get_brief_state(session_id)
    except KeyError as exc:
        raise HTTPException(404, "Session not found") from exc


@router.post("/{session_id}/brief/decision", status_code=204)
async def submit_brief_decision(
    session_id: str, request: BriefDecisionRequest
) -> None:
    """Accept, reject, or edit the agent's current brief draft during the
    briefing phase (WS-5). The running session's briefing loop is blocked
    waiting on this decision — see `_Session.brief_decision_queue`."""
    try:
        session_manager.submit_brief_decision(
            session_id,
            decision=request.decision,
            reason=request.reason,
            brief=request.brief,
        )
    except KeyError as exc:
        raise HTTPException(404, "Session not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{session_id}/work-items", response_model=List[WorkItemSummary])
async def get_work_items(session_id: str) -> List[WorkItemSummary]:
    try:
        return [
            WorkItemSummary.model_validate(item)
            for item in session_manager.list_work_items(session_id)
        ]
    except KeyError as exc:
        raise HTTPException(404, "Session not found") from exc


@router.post(
    "/{session_id}/work-items", response_model=WorkItemDetail, status_code=201
)
async def create_work_item(
    session_id: str, body: WorkItemCreateRequest
) -> WorkItemDetail:
    """Open a human ticket for one of the session's agents."""
    try:
        item = await session_manager.open_human_work_item(
            session_id,
            title=body.title,
            body=body.body,
            owner=body.owner,
            anchor=body.anchor.model_dump() if body.anchor is not None else None,
        )
    except KeyError as exc:
        raise HTTPException(404, "Session not found") from exc
    except UnknownWorkItemOwner as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        # WorkItemConflict (empty title/body, bad anchor) or a stopped session.
        raise HTTPException(409, str(exc)) from exc
    return WorkItemDetail.model_validate(item)


@router.post(
    "/{session_id}/work-items/{item_id}/human-review", response_model=WorkItemDetail
)
async def human_review_work_item(
    session_id: str, item_id: int, body: WorkItemHumanReviewRequest
) -> WorkItemDetail:
    """Record a person's approve/reject; optional-QC rejects reopen a Done item."""
    try:
        item = await session_manager.record_human_review(
            session_id,
            item_id,
            verdict=body.verdict,
            assessment=body.assessment,
        )
    except KeyError as exc:
        raise HTTPException(404, "Session not found") from exc
    except WorkItemNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except WorkItemConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        # The session is not running, so its blueprint policy is unknown.
        raise HTTPException(409, str(exc)) from exc
    return WorkItemDetail.model_validate(item)


@router.get("/{session_id}/blocks", response_model=BlocksResponse)
async def get_blocks(session_id: str) -> BlocksResponse:
    # Only the unknown-session lookup maps to 404; a malformed blocks.json
    # raises out of read_blocks and surfaces as a 500, never as [].
    if session_manager.get_session(session_id) is None:
        raise HTTPException(404, "Session not found")
    index = session_manager.read_blocks(session_id)
    if index is None:
        return BlocksResponse(recorded=False, blocks=[])
    return BlocksResponse(
        recorded=True,
        blocks=[BlockRecord.model_validate(block) for block in index["blocks"]],
    )


@router.get("/{session_id}/work-items/{item_id}", response_model=WorkItemDetail)
async def get_work_item(session_id: str, item_id: int) -> WorkItemDetail:
    try:
        return WorkItemDetail.model_validate(
            session_manager.read_work_item(session_id, item_id)
        )
    except KeyError as exc:
        raise HTTPException(404, "Session not found") from exc
    except WorkItemNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post(
    "/{session_id}/work-items/{item_id}/review",
    response_model=WorkItemReviewResult,
)
async def review_work_item(session_id: str, item_id: int) -> WorkItemReviewResult:
    try:
        return WorkItemReviewResult.model_validate(
            await session_manager.review_work_item(session_id, item_id)
        )
    except KeyError as exc:
        raise HTTPException(404, "Session not found") from exc
    except WorkItemNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Evaluator provider failed: {exc}") from exc


@router.get("/{session_id}/memory")
async def get_memory_state(session_id: str) -> dict:
    """Return the current memory state and context breakdown of the session."""
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    state = session_manager.get_context_breakdown(session_id)
    return state


@router.post("/{session_id}/evaluate", response_model=EvaluationResult)
async def evaluate_session(session_id: str) -> EvaluationResult:
    """Send this session's full transcript to an evaluator agent for review."""
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if s.evaluator_llm_client is None or s.agent_system is None:
        raise HTTPException(
            400,
            "Session is not running — start (or restart) the run before evaluating it.",
        )
    try:
        return await session_manager.evaluate_session(session_id)
    except EvaluationContextTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(500, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{session_id}/evaluator-model", response_model=EvaluatorModelState)
async def get_evaluator_model(session_id: str) -> EvaluatorModelState:
    try:
        return session_manager.get_evaluator_model(session_id)
    except KeyError as exc:
        raise HTTPException(404, "Session not found") from exc


@router.patch("/{session_id}/evaluator-model", response_model=EvaluatorModelState)
async def update_evaluator_model(
    session_id: str, body: EvaluatorModelUpdateRequest
) -> EvaluatorModelState:
    try:
        return await session_manager.update_evaluator_model(session_id, body)
    except KeyError as exc:
        raise HTTPException(404, "Session not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
