"""
config.supervisor
-----------------
Supervisor agent constants.

Moved from ``Supervisor/config.py`` during config consolidation.
All values are sourced from the centralized ``config`` module.
"""

from config import cfg

# ---------------------------------------------------------------------------
# LLM configuration -- now driven by the tier system in config.get_llm()
# These are retained only for any legacy code that reads them directly.
# ---------------------------------------------------------------------------
LLM_MODEL: str = cfg.llm.get("high", {}).get("model", "gemini-2.5-flash")
LLM_TEMPERATURE: float = cfg.llm.get("high", {}).get("temperature", 0.0)

# ---------------------------------------------------------------------------
# Retry / validation
# ---------------------------------------------------------------------------
MAX_RETRIES: int = cfg.supervisor.get("max_retries", 3)
MAX_QUERY_CHARS = 4000

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
MAX_CONVERSATION_TURNS: int = cfg.supervisor.get("max_conversation_turns", 20)

# ---------------------------------------------------------------------------
# Agent registry -- canonical names used in target_agents lists
# ---------------------------------------------------------------------------
AGENT_NAMES = cfg.supervisor.get("agent_names", [
    "civil_law_rag",
    "case_doc_rag",
    "reason",
])

# ---------------------------------------------------------------------------
# Valid intents the classifier may return
# ---------------------------------------------------------------------------
VALID_INTENTS = AGENT_NAMES + ["multi", "off_topic"]

# ---------------------------------------------------------------------------
# MongoDB configuration
# ---------------------------------------------------------------------------
MONGO_URI: str = cfg.mongodb.get("uri", "mongodb://localhost:27017/")
MONGO_DB: str = cfg.mongodb.get("database", "Rag")
MONGO_COLLECTION: str = cfg.mongodb.get("collection", "Document Storage")

# ---------------------------------------------------------------------------
# Vector store (Qdrant) configuration -- shared with Case Doc RAG
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = cfg.embedding.get("model", "BAAI/bge-m3")
QDRANT_HOST: str = cfg.qdrant.get("host", "localhost")
QDRANT_PORT: int = cfg.qdrant.get("port", 6333)
QDRANT_GRPC_PORT: int = cfg.qdrant.get("grpc_port", 6334)
QDRANT_PREFER_GRPC: bool = cfg.qdrant.get("prefer_grpc", True)
QDRANT_COLLECTION: str = cfg.qdrant.get("collection", "judicial_docs")
QDRANT_COLLECTION_CASE: str = cfg.qdrant.get("case_collection", "case_docs")


# Legacy aliases for backward compatibility
CHROMA_COLLECTION: str = QDRANT_COLLECTION
CHROMA_PERSIST_DIR: str = ""

# ---------------------------------------------------------------------------
# P1.10 — Observability constants
# ---------------------------------------------------------------------------
import os as _os

SENTRY_DSN: str = _os.getenv("SENTRY_DSN", "")
LANGSMITH_PROJECT: str = cfg.supervisor.get("langsmith_project", "hakim-supervisor")
LOG_FORMAT: str = cfg.supervisor.get("log_format", "text")          # "text" | "json"
PROMETHEUS_ENABLED: bool = cfg.supervisor.get("prometheus_enabled", True)

# ---------------------------------------------------------------------------
# P1.6.1 — Configurable external-module directories (env var overrides)
# ---------------------------------------------------------------------------
HAKIM_OCR_DIR: str = _os.getenv("HAKIM_OCR_DIR", "")
HAKIM_REASONER_DIR: str = _os.getenv("HAKIM_REASONER_DIR", "")
