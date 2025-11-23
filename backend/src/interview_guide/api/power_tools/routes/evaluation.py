"""Endpoints for evaluating user answers."""

from fastapi import APIRouter, HTTPException, status

from ..schemas.evaluation import EvaluationRequest, EvaluationResponse, EvaluationItem
from ..services.evaluation_service import evaluate_answers

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post(
    "",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate answers against a topic",
)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    """Return scores and feedback for submitted answers."""
    try:
        results, session_id = evaluate_answers(
            answers=request.answers,
            topic=request.topic.strip() if request.topic else None,
            session_id=request.session_id,
            question_texts=request.question_texts,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    items = [
        EvaluationItem(
            answer=entry.get("answer", ""),
            score=int(entry.get("score", 0)),
            feedback=entry.get("feedback", ""),
            confidence=float(entry.get("confidence", 0.0)),
            skill=entry.get("skill"),
        )
        for entry in results
        if entry
    ]
    return EvaluationResponse(results=items, session_id=session_id)
