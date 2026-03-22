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
    supervisor_llm_model: str = cfg.llm.get("high", {}).get("model", "gemini-1.5-flash")
    supervisor_llm_temperature: float = cfg.llm.get("high", {}).get("temperature", 0.0)

    # -- MongoDB --------------------------------------------------------------
    mongo_uri: str = cfg.mongodb.get("uri", "mongodb://localhost:27017/")
    mongo_db: str = cfg.mongodb.get("database", "Rag")
    mongo_collection: str = cfg.mongodb.get("collection", "Document Storage")

    # -- Vector Store (Chroma) ------------------------------------------------
    embedding_model: str = cfg.embedding.get("model", "BAAI/bge-m3")
    chroma_collection: str = cfg.chroma.get("collection", "judicial_docs")
    chroma_persist_dir: str = cfg.chroma.get("persist_dir", "./chroma_data")

    # -- Upload directory -----------------------------------------------------
    upload_dir: str = cfg.api.get("upload_dir", "./uploads")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings singleton."""
    return Settings()
