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

from mcp.server.fastmcp import FastMCP

from RAG.legal_rag.civil_law_rag.corpus import CIVIL_LAW_CORPUS
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
from RAG.legal_rag.retrieval.reranker import _probe_reranker, _get_cross_encoder
from mcp_servers.errors import ErrorCode, raise_tool_error
from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)

mcp = FastMCP("legal-rag-server")

# ---------------------------------------------------------------------------
# Module-level warmup — forces build_graph() once per corpus at child boot.
# legal_rag.graph.build_graph already memoises per corpus name; this call just
# ensures the compiled graph is resident before the first tool call.
# ---------------------------------------------------------------------------

_REGISTERED_CORPORA = [CIVIL_LAW_CORPUS]
_CORPUS_MAP = {}

for _c in _REGISTERED_CORPORA:
    _CORPUS_MAP[_c.name] = _c
    build_graph(_c)
    logger.info("Warmed legal_rag graph: corpus=%s", _c.name)

_get_embeddings()
logger.info("Embedding client ready")

for _c in _REGISTERED_CORPORA:
    _load_vectorstore(_c.collection_name)
    logger.info("Warmed vectorstore: corpus=%s", _c.name)

_probe_reranker()
_get_cross_encoder()
logger.info("Reranker ready")

_SERVICE_ERROR_PREFIXES = ("حدث خطأ", "تعذّر", "تعذر", "لم يتمكن", "خطأ في")

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_legal_corpus(query: str, corpus: str) -> str:
    """Search a legal corpus and return a structured JSON answer."""
    import anyio

    corpus_config = _CORPUS_MAP.get(corpus)
    if corpus_config is None:
        raise_tool_error(
            ErrorCode.INVALID_ARG,
            f"Unknown corpus '{corpus}'. Valid corpora: {list(_CORPUS_MAP)}",
        )

    try:
        from RAG.legal_rag.service import ask_question
        result = await anyio.to_thread.run_sync(
            lambda: ask_question(query, corpus_config),
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
        raise_tool_error(ErrorCode.LLM_TIMEOUT, str(e))
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
        "answer": result.answer,
        "sources": result.sources,
        "classification": result.classification,
        "retrieval_confidence": result.retrieval_confidence,
        "citation_integrity": result.citation_integrity,
        "from_cache": result.from_cache,
        "corpus": result.corpus,
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")