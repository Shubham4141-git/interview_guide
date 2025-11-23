"""Health and readiness probe endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/ping", summary="Basic liveness check")
async def ping() -> dict[str, str]:
    return {"status": "ok"}
