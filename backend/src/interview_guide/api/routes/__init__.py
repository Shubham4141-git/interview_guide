"""Core application routers (non power tools)."""

from fastapi import APIRouter

from . import agent, health, evals

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(agent.router)
api_router.include_router(evals.router)
