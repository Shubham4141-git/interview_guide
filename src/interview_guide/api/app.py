"""Application factory for the Interview Guide FastAPI backend."""

from fastapi import FastAPI

from .routes import api_router as core_router
from .power_tools.routes import power_tools_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Interview Guide API",
        description="Backend services for the Interview Guide platform.",
        version="0.1.0",
    )

    app.include_router(core_router, prefix="/api")
    app.include_router(power_tools_router, prefix="/api")

    return app
