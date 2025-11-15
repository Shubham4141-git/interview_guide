"""Schemas for resource recommendation endpoints."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendationPrefs(BaseModel):
    free_only: Optional[bool] = Field(None, description="Limit to free resources when true.")
    sources: Optional[List[str]] = Field(None, description="Preferred content sources, e.g. ['YouTube'].")


class RecommendationRequest(BaseModel):
    skill: Optional[str] = Field(None, description="Primary skill to target.")
    topic: Optional[str] = Field(None, description="Alternative topic if skill is missing.")
    max_results: int = Field(5, ge=1, le=10)
    prefs: Optional[RecommendationPrefs] = None


class RecommendationItem(BaseModel):
    title: str
    url: str
    snippet: str
    score: float
    source: str


class RecommendationResponse(BaseModel):
    query: str
    results: List[RecommendationItem] = Field(default_factory=list)
