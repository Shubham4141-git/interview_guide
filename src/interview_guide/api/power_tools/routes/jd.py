"""Job description ingestion endpoints."""

from fastapi import APIRouter, HTTPException, status

from ..schemas.jd import (
    JDFetchRequest,
    JDFetchResponse,
    JDProfile,
    JDProfileRequest,
    JDProfileResponse,
)
from ..services.jd_service import build_jd_profile, fetch_jd_text

router = APIRouter(prefix="/jd", tags=["job-description"])


@router.post(
    "/fetch",
    response_model=JDFetchResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch JD text from a public job URL",
)
async def fetch_jd(request: JDFetchRequest) -> JDFetchResponse:
    """Fetch and clean job description text from a URL using the existing agent."""
    try:
        text = fetch_jd_text(url=str(request.url), max_chars=request.max_chars)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return JDFetchResponse(text=text)


@router.post(
    "/profile",
    response_model=JDProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate structured JD profile from raw text",
)
async def profile(request: JDProfileRequest) -> JDProfileResponse:
    """Derive a structured job description profile using the JD parser agent."""
    try:
        raw_profile = build_jd_profile(request.jd_text)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    profile_model = JDProfile(
        role=raw_profile.get("role"),
        subroles=list(raw_profile.get("subroles") or []),
        themes=list(raw_profile.get("themes") or []),
        weights_by_theme=dict(raw_profile.get("weights_by_theme") or {}),
        tasks=list(raw_profile.get("tasks") or []),
        skills=list(raw_profile.get("skills") or []),
        contexts=list(raw_profile.get("contexts") or []),
    )

    return JDProfileResponse(profile=profile_model, raw=raw_profile)
