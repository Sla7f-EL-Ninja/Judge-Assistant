"""
health.py

GET /api/v1/health -- service health and dependency status.

Checks connectivity to all production databases:
- MongoDB
- Qdrant
- Redis
- MinIO
- PostgreSQL
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
        "for each dependency (MongoDB, Qdrant, Redis, MinIO, PostgreSQL). "
        "No authentication required."
    ),
    response_description="Health status and dependency connectivity",
)
async def health_check(
    db: AsyncIOMotorDatabase = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Return service health including all database statuses."""
    deps = {}

    # MongoDB
    try:
        await db.command("ping")
        deps["mongodb"] = "connected"
    except Exception as exc:
        logger.warning("MongoDB health check failed: %s", exc)
        deps["mongodb"] = "disconnected"

    # Qdrant
    try:
        from api.db.qdrant import get_qdrant_client

        client = get_qdrant_client()
        collections = client.get_collections()
        deps["qdrant"] = "connected"
    except RuntimeError:
        deps["qdrant"] = "disconnected"
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        deps["qdrant"] = "disconnected"

    # Redis
    try:
        from api.db.redis import get_redis

        redis_client = get_redis()
        if redis_client is not None:
            await redis_client.ping()
            deps["redis"] = "connected"
        else:
            deps["redis"] = "disconnected"
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        deps["redis"] = "disconnected"

    # MinIO
    try:
        from api.db.minio_client import get_minio, get_bucket

        minio_client = get_minio()
        if minio_client is not None:
            bucket = get_bucket()
            minio_client.bucket_exists(bucket)
            deps["minio"] = "connected"
        else:
            deps["minio"] = "disconnected"
    except RuntimeError:
        deps["minio"] = "disconnected"
    except Exception as exc:
        logger.warning("MinIO health check failed: %s", exc)
        deps["minio"] = "disconnected"

    # PostgreSQL
    try:
        from api.db.postgres import _engine

        if _engine is not None:
            async with _engine.connect() as conn:
                from sqlalchemy import text

                await conn.execute(text("SELECT 1"))
            deps["postgresql"] = "connected"
        else:
            deps["postgresql"] = "disconnected"
    except Exception as exc:
        logger.warning("PostgreSQL health check failed: %s", exc)
        deps["postgresql"] = "disconnected"

    # Core services are MongoDB and Qdrant; others degrade gracefully
    core_healthy = deps.get("mongodb") == "connected" and deps.get("qdrant") == "connected"
    overall = "healthy" if core_healthy else "degraded"

    return {
        "status": overall,
        "version": settings.app_version,
        "dependencies": deps,
    }


@router.get(
    "/ready",
    summary="Liveness probe",
    description=(
        "Lightweight liveness check — returns 200 immediately if the process is running. "
        "Does NOT check database connectivity. Use /health for readiness. "
        "No authentication required."
    ),
)
async def liveness_check():
    """Return 200 if the process is alive (no dependency checks)."""
    return {"status": "alive"}
