"""
config.py

Centralised configuration via pydantic-settings.

All tuneable parameters are read from environment variables (or ``.env``).
The ``Settings`` singleton is used throughout the API layer so that
nothing is hard-coded in routers or services.
"""

from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Server ---------------------------------------------------------------
    app_name: str = "Judge Assistant API"
    app_version: str = "0.1.0"
    debug: bool = False

    # -- CORS -----------------------------------------------------------------
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # -- JWT ------------------------------------------------------------------
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"

    # -- File Uploads ---------------------------------------------------------
    max_upload_bytes: int = 20_971_520  # 20 MB
    allowed_mime_types: str = (
        "application/pdf,image/png,image/jpeg,image/tiff,image/bmp,image/webp"
    )

    @property
    def allowed_mime_type_list(self) -> List[str]:
        return [m.strip() for m in self.allowed_mime_types.split(",") if m.strip()]

    # -- LangGraph ------------------------------------------------------------
    langgraph_module: str = "Supervisor.graph"

    # -- LLM ------------------------------------------------------------------
    supervisor_llm_model: str = "gemini-1.5-flash"
    supervisor_llm_temperature: float = 0

    # -- MongoDB --------------------------------------------------------------
    mongo_uri: str = "mongodb://localhost:27017/"
    mongo_db: str = "Rag"
    mongo_collection: str = "Document Storage"

    # -- Vector Store (Chroma) ------------------------------------------------
    embedding_model: str = "BAAI/bge-m3"
    chroma_collection: str = "judicial_docs"
    chroma_persist_dir: str = "./chroma_data"

    # -- Upload directory -----------------------------------------------------
    upload_dir: str = "./uploads"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings singleton."""
    return Settings()
