"""Request and response models for JD-related endpoints."""

from typing import Optional, Dict, List, Any

from pydantic import BaseModel, Field, HttpUrl


class JDFetchRequest(BaseModel):
    url: HttpUrl
    max_chars: int = Field(4000, ge=500, le=10000)


class JDFetchResponse(BaseModel):
    text: str


class JDProfileRequest(BaseModel):
    jd_text: str = Field(..., min_length=50, max_length=10000)


class JDProfile(BaseModel):
    role: Optional[str] = None
    subroles: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    weights_by_theme: Dict[str, float] = Field(default_factory=dict)
    tasks: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    contexts: List[str] = Field(default_factory=list)


class JDProfileResponse(BaseModel):
    profile: JDProfile
    source: str = Field(default="jd_parser", description="Origin of the profile data.")
    raw: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw profile payload as returned by the existing agent layer.",
    )
