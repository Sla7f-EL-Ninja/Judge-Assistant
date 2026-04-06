"""
config.api
----------
Pydantic Settings class for the FastAPI application.

Moved from ``api/config.py`` during config consolidation.
Defaults are sourced from the centralized ``config`` module so that
``config/settings.yaml`` is the single source of truth.
"""

from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

from config import cfg


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Defaults are sourced from the centralized config module so that
    ``config/settings.yaml`` is the single source of truth.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Server ---------------------------------------------------------------
    app_name: str = cfg.api.get("app_name", "Judge Assistant API")
    app_version: str = cfg.api.get("app_version", "0.1.0")
    debug: bool = cfg.api.get("debug", False)

    # -- CORS -----------------------------------------------------------------
    cors_origins: str = cfg.api.get("cors_origins", "*")

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # -- JWT ------------------------------------------------------------------
    jwt_secret: str = cfg.api.get("jwt_secret", "change-me-in-production")
    jwt_algorithm: str = cfg.api.get("jwt_algorithm", "HS256")

    # -- File Uploads ---------------------------------------------------------
    max_upload_bytes: int = cfg.api.get("max_upload_bytes", 20_971_520)
    allowed_mime_types: str = cfg.api.get(
        "allowed_mime_types",
        "application/pdf,image/png,image/jpeg,image/tiff,image/bmp,image/webp",
    )

    @property
    def allowed_mime_type_list(self) -> List[str]:
        return [m.strip() for m in self.allowed_mime_types.split(",") if m.strip()]

    # -- LangGraph ------------------------------------------------------------
    langgraph_module: str = cfg.api.get("langgraph_module", "Supervisor.graph")

    # -- LLM ------------------------------------------------------------------
    supervisor_llm_model: str = cfg.llm.get("high", {}).get("model", "gemini-2.5-flash")
    supervisor_llm_temperature: float = cfg.llm.get("high", {}).get("temperature", 0.0)

    # -- MongoDB --------------------------------------------------------------
    mongo_uri: str = cfg.mongodb.get("uri", "mongodb://localhost:27017/")
    mongo_db: str = cfg.mongodb.get("database", "Rag")
    mongo_collection: str = cfg.mongodb.get("collection", "Document Storage")
    mongo_min_pool_size: int = cfg.mongodb.get("min_pool_size", 5)
    mongo_max_pool_size: int = cfg.mongodb.get("max_pool_size", 50)
    mongo_server_selection_timeout_ms: int = cfg.mongodb.get("server_selection_timeout_ms", 5000)

    # -- Vector Store (Qdrant) ------------------------------------------------
    embedding_model: str = cfg.embedding.get("model", "BAAI/bge-m3")
    qdrant_host: str = cfg.qdrant.get("host", "localhost")
    qdrant_port: int = cfg.qdrant.get("port", 6333)
    qdrant_grpc_port: int = cfg.qdrant.get("grpc_port", 6334)
    qdrant_collection: str = cfg.qdrant.get("collection", "judicial_docs")
    qdrant_vector_size: int = cfg.qdrant.get("vector_size", 1024)
    qdrant_prefer_grpc: bool = cfg.qdrant.get("prefer_grpc", True)

    # -- Redis ----------------------------------------------------------------
    redis_url: str = cfg.redis.get("url", "redis://localhost:6379/0")
    redis_max_connections: int = cfg.redis.get("max_connections", 20)
    redis_cache_ttl_seconds: int = cfg.redis.get("cache_ttl_seconds", 3600)
    redis_rate_limit_requests: int = cfg.redis.get("rate_limit_requests", 100)
    redis_rate_limit_window_seconds: int = cfg.redis.get("rate_limit_window_seconds", 60)

    # -- MinIO ----------------------------------------------------------------
    minio_endpoint: str = cfg.minio.get("endpoint", "localhost:9000")
    minio_access_key: str = cfg.minio.get("access_key", "minioadmin")
    minio_secret_key: str = cfg.minio.get("secret_key", "minioadmin")
    minio_bucket: str = cfg.minio.get("bucket", "judge-assistant-files")
    minio_secure: bool = cfg.minio.get("secure", False)

    # -- PostgreSQL -----------------------------------------------------------
    postgresql_url: str = cfg.postgresql.get("url", "postgresql+asyncpg://postgres:postgres@localhost:5432/judge_assistant")
    postgresql_sync_url: str = cfg.postgresql.get("sync_url", "postgresql://postgres:postgres@localhost:5432/judge_assistant")
    postgresql_pool_size: int = cfg.postgresql.get("pool_size", 10)
    postgresql_max_overflow: int = cfg.postgresql.get("max_overflow", 20)

    # -- Upload directory (legacy fallback, MinIO preferred) -------------------
    upload_dir: str = cfg.api.get("upload_dir", "./uploads")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings singleton."""
    return Settings()
