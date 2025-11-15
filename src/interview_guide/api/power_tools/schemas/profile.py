"""Schemas for profile-related endpoints."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProfileSummaryRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User identifier whose profile to summarize.")
    session_id: Optional[int] = Field(None, description="Specific session to summarize.")
    session_selector: Optional[str] = Field(
        None,
        description="Optional selector such as 'latest' or 'all' to control session scope.",
    )


class SkillStat(BaseModel):
    count: int = 0
    avg: float = 0.0


class RecentEntry(BaseModel):
    id: Optional[int] = None
    session_id: Optional[int] = None
    question_text: Optional[str] = None
    answer: Optional[str] = None
    score: Optional[int] = None
    feedback: Optional[str] = None
    confidence: Optional[float] = None
    skill: Optional[str] = None
    created_at: Optional[str] = None


class ProfileSummary(BaseModel):
    total_answers: int = 0
    avg_score: float = 0.0
    distribution: Dict[str, int] = Field(default_factory=dict)
    strength_skills: List[str] = Field(default_factory=list)
    weakness_skills: List[str] = Field(default_factory=list)
    skill_stats: Dict[str, SkillStat] = Field(default_factory=dict)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recent: List[RecentEntry] = Field(default_factory=list)


class ProfileSummaryResponse(BaseModel):
    summary: ProfileSummary
