"""Schemas for answer evaluation endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    answers: List[str] = Field(..., min_length=1, description="User answers to evaluate.")
    topic: Optional[str] = Field(None, description="Optional topic context for the answers.")
    session_id: Optional[int] = Field(None, description="Session id to persist evaluation results.")
    question_texts: Optional[List[str]] = Field(
        None,
        description="Optional list of question texts aligned with the answers for storage.",
    )


class EvaluationItem(BaseModel):
    answer: str
    score: int = Field(..., ge=0, le=5)
    feedback: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    skill: Optional[str] = None


class EvaluationResponse(BaseModel):
    results: List[EvaluationItem] = Field(default_factory=list)
    session_id: Optional[int] = None
