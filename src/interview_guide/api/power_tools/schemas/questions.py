"""Request and response models for question generation."""

from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200)
    count: int = Field(5, ge=1, le=25, description="Desired number of questions.")
    session_id: Optional[int] = Field(
        None, description="Optional session id to persist generated questions."
    )


class QuestionItem(BaseModel):
    text: str = Field(..., min_length=3)
    topic: str
    difficulty: str


class QuestionResponse(BaseModel):
    session_id: Optional[int] = None
    questions: List[QuestionItem] = Field(default_factory=list)
