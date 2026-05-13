"""
legal_rag_server.py
-------------------
FastMCP stdio server: one tool, corpus-parameterised.

Warmup at import: compiles the LangGraph graph for every registered corpus
before accepting any request, so the cold-start cost is paid at child-process
spawn time rather than inside the first request.

Spawn: python -m mcp_servers.legal_rag_server
"""
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # prevent LangSmith from hanging stdio pipes

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from RAG.legal_rag.civil_law_rag.corpus import CIVIL_LAW_CORPUS
from RAG.legal_rag.evidence_rag.corpus import EVIDENCE_CORPUS
from RAG.legal_rag.procedures_rag.corpus import PROCEDURES_CORPUS
from RAG.legal_rag.errors import (
    GenerationError,
    LLMBudgetExceededError,
    LLMTimeoutError,
    QueryValidationError,
    RetrievalError,
)
from RAG.legal_rag.graph import build_graph
from RAG.legal_rag.retrieval.embeddings import get_client as _get_embeddings
from RAG.legal_rag.retrieval.vectorstore import load_vectorstore as _load_vectorstore
from RAG.legal_rag.retrieval.reranker import _probe_reranker
from mcp_servers.errors import ErrorCode, raise_tool_error
from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)

mcp = FastMCP("legal-rag-server")

# ---------------------------------------------------------------------------
# Module-level warmup — forces build_graph() once per corpus at child boot.
# ---------------------------------------------------------------------------

_REGISTERED_CORPORA = [CIVIL_LAW_CORPUS, EVIDENCE_CORPUS, PROCEDURES_CORPUS]
_CORPUS_MAP = {}

for _c in _REGISTERED_CORPORA:
    _CORPUS_MAP[_c.name] = _c

# Call build_graph once (it's a singleton now)
build_graph()
logger.info("Warmed unified legal_rag graph")

_get_embeddings()
logger.info("Embedding client ready")

# AFTER
for _c in _REGISTERED_CORPORA:
    try:
        _load_vectorstore(_c.collection_name)
        logger.info("Warmed vectorstore: corpus=%s", _c.name)
    except Exception as _vs_err:
        logger.warning("Vectorstore warm-up failed for corpus=%s (non-fatal): %s", _c.name, _vs_err)

try:
    _probe_reranker()
    logger.info("Reranker ready")
except Exception as _rr_err:
    logger.warning("Reranker probe failed (non-fatal): %s", _rr_err)

_probe_reranker()
logger.info("Reranker ready")

_SERVICE_ERROR_PREFIXES = ("حدث خطأ", "تعذّر", "تعذر", "لم يتمكن", "خطأ في")

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_legal_corpus(
    query: str,
    corpus: str,
    # FIX-1: scope_fallback is injected by executor on retry attempts.
    #   None          → normal two-stage scoping (first attempt)
    #   "section"     → skip section classification, filter by chapter only
    #   "chapter"     → skip all scoping, full corpus search
    scope_fallback: Optional[str] = None,
) -> str:
    """Search a legal corpus and return a structured JSON answer."""
    import anyio

    corpus_config = _CORPUS_MAP.get(corpus)
    if corpus_config is None:
        raise_tool_error(
            ErrorCode.INVALID_ARG,
            f"Unknown corpus '{corpus}'. Valid corpora: {list(_CORPUS_MAP)}",
        )

    if scope_fallback not in (None, "section", "chapter"):
        raise_tool_error(
            ErrorCode.INVALID_ARG,
            f"scope_fallback must be None, 'section', or 'chapter'; got '{scope_fallback}'",
        )

    try:
        from RAG.legal_rag.service import ask_question
        result = await anyio.to_thread.run_sync(
            lambda: ask_question(query, scope_fallback=scope_fallback),
            abandon_on_cancel=True,
        )
    except QueryValidationError as e:
        raise_tool_error(ErrorCode.QUERY_VALIDATION, str(e))
    except RetrievalError as e:
        raise_tool_error(ErrorCode.RETRIEVAL, str(e))
    except GenerationError as e:
        raise_tool_error(ErrorCode.GENERATION, str(e))
    except LLMBudgetExceededError as e:
        raise_tool_error(ErrorCode.LLM_BUDGET, str(e))
    except LLMTimeoutError as e:
        raise_tool_error(ErrorCode.LM_TIMEOUT, str(e))
    except Exception as e:
        raise_tool_error(ErrorCode.INTERNAL, f"Unexpected error: {e}")

    if not result.from_cache and any(
        result.answer.startswith(p) for p in _SERVICE_ERROR_PREFIXES
    ):
        raise_tool_error(
            ErrorCode.INTERNAL,
            "Service returned error string as answer (swallowed exception)",
            answer=result.answer[:200],
        )

    return json.dumps({
        "answer":               result.answer,
        "sources":              result.sources,
        "classification":       result.classification,
        "retrieval_confidence": result.retrieval_confidence,
        "citation_integrity":   result.citation_integrity,
        "from_cache":           result.from_cache,
        "corpus":               result.corpus,
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")