"""
app.py

FastAPI application factory.

Creates the app with CORS middleware, lifespan handler for DB connections,
error handlers, and all routers mounted.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.db.mongodb import close_mongo, connect_mongo

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown: connect and disconnect MongoDB."""
    settings = get_settings()
    logger.info("Connecting to MongoDB at %s ...", settings.mongo_uri)
    await connect_mongo(settings)
    logger.info("MongoDB connected (db=%s)", settings.mongo_db)
    yield
    logger.info("Shutting down -- closing MongoDB connection")
    await close_mongo()


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # -- CORS -----------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # -- Error handlers -------------------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    # -- Routers --------------------------------------------------------------
    from api.routers.cases import router as cases_router
    from api.routers.conversations import router as conversations_router
    from api.routers.documents import router as documents_router
    from api.routers.files import router as files_router
    from api.routers.health import router as health_router
    from api.routers.query import router as query_router
    from api.routers.summaries import router as summaries_router

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(files_router)
    app.include_router(cases_router)
    app.include_router(documents_router)
    app.include_router(summaries_router)
    app.include_router(conversations_router)

    return app
