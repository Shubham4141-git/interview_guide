"""Endpoint that runs the LangGraph supervisor flow."""

from fastapi import APIRouter, HTTPException, status

from ..schemas.agent import AgentRequest, AgentResponse, AgentState
from ..services.agent_service import execute_agent

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/execute",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a query through the supervisor graph",
)
async def agent_execute(request: AgentRequest) -> AgentResponse:
    """Route the user's query through the orchestrated agent and return the resulting state."""
    try:
        preview = (request.query or "").strip().replace("\n", " ")
        preview = (preview[:120] + "...") if len(preview) > 120 else preview
        print(
            f"[agent_execute] user_id={request.user_id!r} session_id={request.session_id!r} "
            f"intent={request.intent!r} query_len={len(request.query or '')} preview={preview!r}",
            flush=True,
        )
        state = execute_agent(
            query=request.query.strip(),
            user_id=request.user_id,
            session_id=request.session_id,
            intent=request.intent,
            slots=request.slots,
        )
    except Exception as exc:
        print(f"[agent_execute] error={exc!r}", flush=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return AgentResponse(state=AgentState(**state))
