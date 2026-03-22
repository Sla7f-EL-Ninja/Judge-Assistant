"""
app.py

FastAPI application factory.

Creates the app with CORS middleware, lifespan handler for DB connections,
structured error handlers, and all routers mounted.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.api import get_settings
from api.db.mongodb import close_mongo, connect_mongo
from api.errors import INTERNAL_ERROR, UNAUTHORIZED, VALIDATION_ERROR
from api.schemas.common import ErrorDetail, ErrorEnvelope

logger = logging.getLogger(__name__)


def _error_response(status_code: int, code: str, detail: str) -> JSONResponse:
    """Build a JSONResponse using the standard ErrorEnvelope shape."""
    envelope = ErrorEnvelope(
        error=ErrorDetail(code=code, detail=detail, status=status_code)
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(),
    )


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


# -- OpenAPI tag descriptions -------------------------------------------------
OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Service health and dependency connectivity checks.",
    },
    {
        "name": "Cases",
        "description": "CRUD operations for case management. Cases group documents, conversations, and summaries.",
    },
    {
        "name": "Files",
        "description": "File upload endpoint. Uploaded files are stored on disk and referenced by file_id.",
    },
    {
        "name": "Documents",
        "description": "Document ingestion pipeline: OCR, classification, and vector indexing into a case.",
    },
    {
        "name": "Query",
        "description": "Supervisor query endpoint with SSE streaming. Runs the full LangGraph workflow.",
    },
    {
        "name": "Conversations",
        "description": "Conversation history management. Each query exchange is stored as a turn.",
    },
    {
        "name": "Summaries",
        "description": "Retrieve auto-generated case summaries produced by the summarization agent.",
    },
]


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "The Judge Assistant API provides AI-powered legal research and case "
            "management capabilities for judicial professionals. It supports document "
            "ingestion with OCR, case management, conversational queries backed by a "
            "multi-agent LangGraph supervisor, and automatic case summarisation.\n\n"
            "All endpoints (except /health) require a JWT Bearer token with a `user_id` claim."
        ),
        debug=settings.debug,
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        contact={
            "name": "Judge Assistant Team",
        },
        license_info={
            "name": "Proprietary",
        },
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
        code = UNAUTHORIZED if exc.status_code == 401 else INTERNAL_ERROR
        # Try to infer a better error code from the detail text
        detail_lower = (exc.detail or "").lower() if isinstance(exc.detail, str) else ""
        if "not found" in detail_lower:
            if "case" in detail_lower:
                code = "CASE_NOT_FOUND"
            elif "conversation" in detail_lower:
                code = "CONVERSATION_NOT_FOUND"
            elif "file" in detail_lower:
                code = "FILE_NOT_FOUND"
            elif "summary" in detail_lower:
                code = "SUMMARY_NOT_FOUND"
            else:
                code = "NOT_FOUND"  # generic not-found fallback
        elif "not allowed" in detail_lower or "mime" in detail_lower:
            code = "INVALID_MIME_TYPE"
        elif "exceeds" in detail_lower or "too large" in detail_lower:
            code = "FILE_TOO_LARGE"
        elif "no fields" in detail_lower:
            code = "NO_FIELDS_TO_UPDATE"
        elif exc.status_code == 422:
            code = VALIDATION_ERROR
        elif exc.status_code == 400:
            code = VALIDATION_ERROR

        detail_str = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return _error_response(exc.status_code, code, detail_str)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        """Return a structured error for Pydantic / FastAPI validation failures."""
        errors = exc.errors()
        # Build a human-readable summary
        parts = []
        for err in errors:
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "")
            parts.append(f"{loc}: {msg}" if loc else msg)
        detail = "; ".join(parts) if parts else "Request validation failed"
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY, VALIDATION_ERROR, detail
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return _error_response(
            status.HTTP_400_BAD_REQUEST, VALIDATION_ERROR, str(exc)
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            INTERNAL_ERROR,
            "Internal server error",
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
