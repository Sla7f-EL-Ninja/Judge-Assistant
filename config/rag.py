"""
config.rag
----------
Civil Law RAG constants and default state template.

Moved from ``RAG/Civil Law RAG/config.py`` during config consolidation.
All values are sourced from the centralized ``config`` module.
"""

import os

from config import cfg

# -----------------------------
# Paths -- resolve relative to project root
# -----------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(_PROJECT_ROOT, "RAG", "Civil Law RAG", "docs", "civil_law_clean.txt")
DB_DIR = os.path.join(_PROJECT_ROOT, "RAG", "Civil Law RAG", "db", "chroma_db")

# -----------------------------
# Initializations -- from central config
# -----------------------------
EMBEDDING_MODEL = cfg.embedding.get("model", "BAAI/bge-m3")
BATCH_SIZE = 50
LLM_MODEL = cfg.llm.get("high", {}).get("model", "llama-3.3-70b-versatile")

# -----------------------------
# Default State Template
# -----------------------------
default_state_template = {
    "last_query": None,
    "last_results": [],
    "last_answer": None,
    "current_book": None,
    "current_part": None,
    "current_chapter": None,
    "current_article": None,
    "filter_type": "",
    "k": 8,
    "books_in_scope": [],
    "query_history": [],
    "retrieval_history": [],
    "retry_count": 0,
    "max_retries": 2,
    "answer_history": [],
    "db_initialized": True,
    "db": None,  # to be initialized in main or graph
    "split_config": {},
    "rewritten_question": None,
    "classification": None,
    "retrieval_confidence": None,
    "refined_query": None,
    "grade": None,
    "llm_pass": None,
    "failure_reason": None,
    "proceedToGenerate": None,
    "retrieval_attempts": 0,
    "final_answer": None
}

# -----------------------------
# Graph Constants
# -----------------------------
START = "__start__"
END = "__end__"
