"""Schemas for user directory endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field

from .profile import ProfileSummary
from .recommendations import RecommendationItem


class UserListResponse(BaseModel):
    users: List[str] = Field(default_factory=list)


class UserOverviewResponse(BaseModel):
    user_id: str
    focus_skill: Optional[str] = None
    profile: ProfileSummary
    resources: List[RecommendationItem] = Field(default_factory=list)
