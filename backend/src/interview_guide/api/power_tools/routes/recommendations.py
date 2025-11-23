"""Endpoints for resource recommendations."""

from fastapi import APIRouter, HTTPException, status

from ..schemas.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
)
from ..services.recommendation_service import recommend_resources

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post(
    "",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend study resources for a skill or topic",
)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """Return learning resources tailored to the requested skill/topic."""
    if not (request.skill or request.topic):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either 'skill' or 'topic' must be provided.",
        )

    prefs_dict = request.prefs.model_dump(exclude_none=True) if request.prefs else {}

    try:
        results = recommend_resources(
            skill=request.skill,
            topic=request.topic,
            prefs=prefs_dict,
            max_results=request.max_results,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    items = [
        RecommendationItem(
            title=item.get("title", "Untitled"),
            url=item.get("url", ""),
            snippet=item.get("snippet", ""),
            score=float(item.get("score", 0.0) or 0.0),
            source=item.get("source", "tavily"),
        )
        for item in results
        if item.get("url")
    ]

    return RecommendationResponse(
        query=request.skill or request.topic or "",
        results=items,
    )
