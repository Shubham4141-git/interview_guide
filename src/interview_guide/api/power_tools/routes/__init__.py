"""Routers that expose auxiliary power tools functionality."""

from fastapi import APIRouter

from . import evaluation, jd, profile, questions, recommendations, sessions

power_tools_router = APIRouter()
power_tools_router.include_router(jd.router)
power_tools_router.include_router(questions.router)
power_tools_router.include_router(evaluation.router)
power_tools_router.include_router(profile.router)
power_tools_router.include_router(recommendations.router)
power_tools_router.include_router(sessions.router)
