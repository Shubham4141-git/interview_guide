"""Endpoints for retrieving profile summaries."""

from fastapi import APIRouter, HTTPException, status

from ..schemas.profile import (
    ProfileSummaryRequest,
    ProfileSummaryResponse,
    ProfileSummary,
    SkillStat,
    RecentEntry,
)
from ..services.profile_service import get_profile_summary

router = APIRouter(prefix="/profile", tags=["profile"])


@router.post(
    "/summary",
    response_model=ProfileSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize user or session performance",
)
async def summarize(request: ProfileSummaryRequest) -> ProfileSummaryResponse:
    """Return aggregated strengths and weaknesses for a user or session."""
    try:
        summary_data = get_profile_summary(
            user_id=request.user_id,
            session_id=request.session_id,
            session_selector=request.session_selector,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    skill_stats = {
        key: SkillStat(**value) if isinstance(value, dict) else SkillStat()
        for key, value in (summary_data.get("skill_stats") or {}).items()
    }
    recent_items = [
        RecentEntry(**entry) if isinstance(entry, dict) else RecentEntry()
        for entry in (summary_data.get("recent") or [])
    ]

    profile_summary = ProfileSummary(
        total_answers=int(summary_data.get("total_answers", 0)),
        avg_score=float(summary_data.get("avg_score", 0.0)),
        distribution=dict(summary_data.get("distribution") or {}),
        strength_skills=list(summary_data.get("strength_skills") or []),
        weakness_skills=list(summary_data.get("weakness_skills") or []),
        skill_stats=skill_stats,
        strengths=list(summary_data.get("strengths") or []),
        weaknesses=list(summary_data.get("weaknesses") or []),
        recent=recent_items,
    )

    return ProfileSummaryResponse(summary=profile_summary)
