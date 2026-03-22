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
LLM_MODEL: str = cfg.llm.get("high", {}).get("model", "gemini-1.5-flash")
LLM_TEMPERATURE: float = cfg.llm.get("high", {}).get("temperature", 0.0)

# ---------------------------------------------------------------------------
# Retry / validation
# ---------------------------------------------------------------------------
MAX_RETRIES: int = cfg.supervisor.get("max_retries", 2)

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
MAX_CONVERSATION_TURNS: int = cfg.supervisor.get("max_conversation_turns", 20)

# ---------------------------------------------------------------------------
# Agent registry -- canonical names used in target_agents lists
# ---------------------------------------------------------------------------
AGENT_NAMES = cfg.supervisor.get("agent_names", [
    "ocr",
    "summarize",
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
# Vector store (Chroma) configuration -- shared with Case Doc RAG
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = cfg.embedding.get("model", "BAAI/bge-m3")
CHROMA_COLLECTION: str = cfg.chroma.get("collection", "judicial_docs")
CHROMA_PERSIST_DIR: str = cfg.chroma.get("persist_dir", "./chroma_data")
