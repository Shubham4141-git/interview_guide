"""Endpoints for managing interview sessions."""

from fastapi import APIRouter, HTTPException, Query, status

from ..schemas.sessions import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionSummary,
    SessionListResponse,
    SessionQuestionsResponse,
    SessionQuestionsItem,
)
from ..services.session_service import (
    create_session,
    list_user_sessions,
    list_session_questions,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post(
    "",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new interview session",
)
async def create(request: SessionCreateRequest) -> SessionCreateResponse:
    """Create a session tied to a user, optionally storing the JD used to start it."""
    try:
        session_data = create_session(request.user_id, request.jd_text)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    summary = SessionSummary(
        id=int(session_data.get("id")),
        user_id=session_data.get("user_id", "anon"),
        created_at=session_data.get("created_at", ""),
        jd_text=session_data.get("jd_text"),
    )
    return SessionCreateResponse(session=summary)


@router.get(
    "",
    response_model=SessionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List recent sessions for a user",
)
async def list_sessions_endpoint(
    user_id: str = Query(..., description="User identifier to filter sessions."),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of sessions to return."),
) -> SessionListResponse:
    """Return recent sessions for the given user."""
    try:
        sessions = list_user_sessions(user_id=user_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    items = [
        SessionSummary(
            id=int(item.get("id")),
            user_id=item.get("user_id", user_id),
            created_at=item.get("created_at", ""),
            jd_text=item.get("jd_text"),
        )
        for item in sessions
    ]
    return SessionListResponse(sessions=items)


@router.get(
    "/{session_id}/questions",
    response_model=SessionQuestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get questions stored for a session",
)
async def session_questions(session_id: int) -> SessionQuestionsResponse:
    """Return questions associated with the session (if any were stored)."""
    try:
        questions = list_session_questions(session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    items = [
        SessionQuestionsItem(
            id=int(q.get("id")),
            text=q.get("text", ""),
            topic=q.get("topic"),
            difficulty=q.get("difficulty"),
        )
        for q in questions
    ]
    return SessionQuestionsResponse(session_id=session_id, questions=items)
