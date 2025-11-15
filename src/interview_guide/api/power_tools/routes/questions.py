"""Endpoints for generating interview questions."""

from fastapi import APIRouter, HTTPException, status

from ..schemas.questions import QuestionRequest, QuestionResponse, QuestionItem
from ..services.questions_service import generate_questions

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post(
    "",
    response_model=QuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate interview questions for a topic",
)
async def create_questions(request: QuestionRequest) -> QuestionResponse:
    """Generate questions for the requested topic via the question agent."""
    normalized_topic = request.topic.strip()
    try:
        generated, session_id = generate_questions(
            topic=normalized_topic,
            count=request.count,
            session_id=request.session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    questions = [
        QuestionItem(
            text=item["text"],
            topic=item.get("topic", normalized_topic),
            difficulty=item.get("difficulty", "auto"),
        )
        for item in generated
        if item.get("text")
    ]
    return QuestionResponse(session_id=session_id, questions=questions)
