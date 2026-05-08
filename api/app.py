"""
app.py

FastAPI application factory.

Creates the app with CORS middleware, lifespan handler for DB connections,
structured error handlers, and all routers mounted.
"""

import logging
import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.api import get_settings
from api.db.mongodb import close_mongo, connect_mongo
from api.db.qdrant import close_qdrant, connect_qdrant
from api.db.redis import close_redis, connect_redis
from api.db.minio_client import close_minio, connect_minio
from api.errors import (
    INTERNAL_ERROR, UNAUTHORIZED, VALIDATION_ERROR,
    NOT_FOUND, CASE_NOT_FOUND, CONVERSATION_NOT_FOUND, FILE_NOT_FOUND,
    SUMMARY_NOT_FOUND, DOCUMENT_NOT_FOUND, CASE_REASONING_NOT_FOUND,
    REPORT_NOT_FOUND, INVALID_MIME_TYPE, FILE_TOO_LARGE, NO_FIELDS_TO_UPDATE,
)
from api.schemas.common import ErrorDetail, ErrorEnvelope


def _configure_logging(debug: bool = False) -> None:
    """Wire structlog for JSON structured logging in production, console in dev."""
    log_level = logging.DEBUG if debug else logging.INFO

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if debug:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _configure_langsmith() -> None:
    """Activate LangSmith tracing if LANGCHAIN_API_KEY is set (P1.7.3)."""
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGCHAIN_PROJECT", "hakim"))
    logging.getLogger(__name__).info("LangSmith tracing enabled (project=%s)", os.environ["LANGCHAIN_PROJECT"])


def _configure_sentry() -> None:
    """Wire Sentry error aggregator if DSN is configured."""
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        return
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=dsn,
            integrations=[StarletteIntegration(), FastApiIntegration()],
            traces_sample_rate=0.1,
            send_default_pii=False,
        )
        logging.getLogger(__name__).info("Sentry initialized")
    except Exception as exc:
        logging.getLogger(__name__).warning("Sentry init failed (non-fatal): %s", exc)


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
    """Manage startup/shutdown: connect and disconnect all databases."""
    settings = get_settings()

    # -- MongoDB (required) ---------------------------------------------------
    logger.info("Connecting to MongoDB at %s ...", settings.mongo_uri)
    await connect_mongo(settings)
    logger.info("MongoDB connected (db=%s)", settings.mongo_db)

    # -- Qdrant (required) ----------------------------------------------------
    try:
        logger.info("Connecting to Qdrant at %s:%s ...", settings.qdrant_host, settings.qdrant_port)
        connect_qdrant(settings)
        logger.info("Qdrant connected (collection=%s)", settings.qdrant_collection)
    except Exception as exc:
        logger.warning("Qdrant connection failed (non-fatal): %s", exc)

    # -- Redis (optional -- degrades gracefully) ------------------------------
    try:
        logger.info("Connecting to Redis at %s ...", settings.redis_url)
        await connect_redis(settings)
        logger.info("Redis connected")
    except Exception as exc:
        logger.warning("Redis connection failed (non-fatal, caching disabled): %s", exc)

    # -- MinIO (optional -- falls back to local disk) -------------------------
    try:
        logger.info("Connecting to MinIO at %s ...", settings.minio_endpoint)
        connect_minio(settings)
        logger.info("MinIO connected (bucket=%s)", settings.minio_bucket)
    except Exception as exc:
        logger.warning("MinIO connection failed (non-fatal, using local disk): %s", exc)

    # -- Civil Law RAG: ensure corpus is indexed (fast no-op if already present)
    try:
        from RAG.civil_law_rag.indexing.indexer import ensure_civil_law_indexed
        ensure_civil_law_indexed()
    except Exception as exc:
        logger.warning("Civil Law RAG indexing check failed (non-fatal): %s", exc)

    # -- BGE-M3 warm-up (single load, GPU) ------------------------------------
    # _get_vectorstore() loads BGE-M3 onto cuda:0 once. We then expose the
    # loaded embeddings object over HTTP on :8080 so the two MCP child
    # processes (legal_rag_server, case_doc_server) can call it instead of
    # each loading their own copy of the model.
    try:
        import asyncio
        from DocumentProcessor.pipeline import _get_vectorstore
        logger.info("Warming up BGE-M3 embedding model and Qdrant vectorstore ...")
        vs = await asyncio.get_event_loop().run_in_executor(None, _get_vectorstore)
        logger.info("BGE-M3 + Qdrant vectorstore ready")

        from api.embedding_server import start_embedding_server
        start_embedding_server(vs.embeddings, port=8080)
        logger.info("Local TEI-compatible embedding server ready on :8080")
    except Exception as exc:
        logger.warning("Vectorstore warm-up failed (non-fatal): %s", exc)

    # -- MCP servers (legal_rag + case_doc_rag) --------------------------------
    # Children probe :8080 first; since it's now up they skip their own model
    # load entirely — startup is ~20s faster and uses no extra VRAM/RAM.
    try:
        import asyncio
        from mcp_servers.lifecycle import start_mcp_servers
        logger.info("Spawning MCP child processes (legal_rag, case_doc_rag) ...")
        await asyncio.get_event_loop().run_in_executor(None, start_mcp_servers)
        logger.info("MCP servers ready")
    except Exception as exc:
        logger.warning("MCP server startup failed (non-fatal): %s", exc)

    yield

    # -- Shutdown all connections ----------------------------------------------
    logger.info("Shutting down -- closing all database connections")
    await close_mongo()
    close_qdrant()
    await close_redis()
    close_minio()

    # -- Shutdown MCP child processes -----------------------------------------
    try:
        from mcp_servers.lifecycle import get_client
        for name in ("legal_rag", "case_doc_rag"):
            try:
                get_client(name).shutdown()
            except Exception:
                pass
        logger.info("MCP servers shut down")
    except Exception as exc:
        logger.warning("MCP shutdown error (non-fatal): %s", exc)

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
    {
        "name": "CaseReasoning",
        "description": "Retrieve persisted case reasoning results produced by the Case Reasoner.",
    },
    {
        "name": "Reports",
        "description": "Async report generation: runs summarizer + case reasoner, returns job ID for polling.",
    },
    {
        "name": "LegalSearch",
        "description": "Direct legal corpus search without invoking the supervisor graph.",
    },
]


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    _configure_logging(debug=settings.debug)
    _configure_sentry()
    _configure_langsmith()

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

    # -- Prometheus metrics ---------------------------------------------------
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception as exc:
        logger.warning("Prometheus instrumentation failed (non-fatal): %s", exc)

    # -- Auth middleware (runs before body parsing so 401 always beats 422) ---
    _PUBLIC_PATHS = {"/api/v1/health", "/health", "/api/v1/ready", "/docs", "/openapi.json", "/redoc", "/metrics"}

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            envelope = ErrorEnvelope(
                error=ErrorDetail(
                    code=UNAUTHORIZED,
                    detail="Not authenticated",
                    status=status.HTTP_401_UNAUTHORIZED,
                )
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=envelope.model_dump(),
            )
        return await call_next(request)

    # -- Error handlers -------------------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        # Prefer structured detail: {"code": "...", "message": "..."}
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            code = exc.detail["code"]
            detail_str = exc.detail.get("message", str(exc.detail))
            return _error_response(exc.status_code, code, detail_str)

        # Legacy string detail — infer code from status + text
        detail_str = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        detail_lower = detail_str.lower()

        if exc.status_code == 401:
            code = UNAUTHORIZED
        elif "not found" in detail_lower:
            if "case" in detail_lower:
                code = CASE_NOT_FOUND
            elif "conversation" in detail_lower:
                code = CONVERSATION_NOT_FOUND
            elif "file" in detail_lower:
                code = FILE_NOT_FOUND
            elif "summary" in detail_lower:
                code = SUMMARY_NOT_FOUND
            elif "document" in detail_lower:
                code = DOCUMENT_NOT_FOUND
            elif "report" in detail_lower:
                code = REPORT_NOT_FOUND
            elif "reasoning" in detail_lower:
                code = CASE_REASONING_NOT_FOUND
            else:
                code = NOT_FOUND
        elif "not allowed" in detail_lower or "mime" in detail_lower:
            code = INVALID_MIME_TYPE
        elif "exceeds" in detail_lower or "too large" in detail_lower:
            code = FILE_TOO_LARGE
        elif "no fields" in detail_lower:
            code = NO_FIELDS_TO_UPDATE
        elif exc.status_code in (400, 422):
            code = VALIDATION_ERROR
        else:
            code = INTERNAL_ERROR

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
    from api.routers.case_reasoning import router as case_reasoning_router
    from api.routers.reports import router as reports_router
    from api.routers.legal_search import router as legal_search_router

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(files_router)
    app.include_router(cases_router)
    app.include_router(documents_router)
    app.include_router(summaries_router)
    app.include_router(conversations_router)
    app.include_router(case_reasoning_router)
    app.include_router(reports_router)
    app.include_router(legal_search_router)

    return app