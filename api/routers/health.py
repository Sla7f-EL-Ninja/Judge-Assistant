"""
health.py

GET /api/v1/health -- service health and dependency status.
"""

import logging

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings
from api.dependencies import get_db, get_settings
from api.schemas.health import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the overall service health status and connectivity information "
        "for each dependency (MongoDB, Chroma vector store). No authentication required."
    ),
    response_description="Health status and dependency connectivity",
)
async def health_check(
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Return service health including MongoDB and Chroma status."""
    deps = {}

    # MongoDB
    try:
        await db.command("ping")
        deps["mongodb"] = "connected"
    except Exception as exc:
        logger.warning("MongoDB health check failed: %s", exc)
        deps["mongodb"] = "disconnected"

    # Chroma -- lightweight check: try to import and verify persist dir exists
    try:
        import os

        if os.path.isdir(settings.chroma_persist_dir):
            deps["chroma"] = "connected"
        else:
            deps["chroma"] = "directory_missing"
    except Exception:
        deps["chroma"] = "disconnected"

    overall = "healthy" if all(v == "connected" for v in deps.values()) else "degraded"

    return {
        "status": overall,
        "version": settings.app_version,
        "dependencies": deps,
    }
