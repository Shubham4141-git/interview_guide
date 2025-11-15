"""Schemas for session management endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User identifier associated with the session.")
    jd_text: Optional[str] = Field(None, description="Optional job description text captured at session start.")


class SessionSummary(BaseModel):
    id: int
    user_id: str
    created_at: str
    jd_text: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session: SessionSummary


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary] = Field(default_factory=list)


class SessionQuestionsItem(BaseModel):
    id: int
    text: str
    topic: Optional[str] = None
    difficulty: Optional[str] = None


class SessionQuestionsResponse(BaseModel):
    session_id: int
    questions: List[SessionQuestionsItem] = Field(default_factory=list)
