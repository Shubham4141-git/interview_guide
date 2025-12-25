"""User directory endpoints."""

from fastapi import APIRouter, HTTPException, status

from ..schemas.users import UserListResponse, UserOverviewResponse
from ..services.users_service import build_user_overview, fetch_user_ids

router = APIRouter(prefix="/users", tags=["users"])


@router.get(
    "",
    response_model=UserListResponse,
    status_code=status.HTTP_200_OK,
    summary="List known user IDs",
)
async def list_users() -> UserListResponse:
    users = fetch_user_ids()
    return UserListResponse(users=users)


@router.get(
    "/{user_id}",
    response_model=UserOverviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch profile summary and resources for a user",
)
async def user_overview(user_id: str) -> UserOverviewResponse:
    user_id = user_id.strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing user_id")
    overview = build_user_overview(user_id)
    return UserOverviewResponse(**overview)
